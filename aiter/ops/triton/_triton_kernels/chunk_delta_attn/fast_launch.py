# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Launch a Triton kernel without redoing the part that only depends on shapes.

``JITFunction.run`` binds the arguments, specializes each one, builds a cache
key and looks the kernel up, and it does all of that on every call. On the
flash_kda pipeline that is ~51us per dispatch against ~3us for the launch
itself, which is enough to decide a design question the wrong way: the
segmented recurrence issues 98us of GPU work through five launches, so an eager
caller waits on the host and the faster pipeline measures as the slower one.
The HIP implementation does not pay this because its kernels sit behind one
extension call rather than one Python call each.

Nothing in that preamble depends on the argument *values*, so a caller that has
seen the same shapes before can go straight to the launch. What it must not
skip is the reason the preamble exists. Triton compiles under assumptions --
that pointers are 16-byte aligned, that an integer argument is divisible by 16
or equal to 1, and on AMD that a buffer is addressable in 32 bits -- and reusing
a kernel for arguments that break one is a miscompile rather than an error. So
the key below asks exactly those questions again, plus the value of everything
that is not a tensor, which covers constexprs and the autotuner's choice of
config. It leaves ~10us of the ~51us in place, and
``test_fast_launch.py`` checks a matrix of shapes reaches the same kernel and
the same bytes as the ordinary path.

Wrap a kernel once and call it as before::

    _kernel = fast_launch(_kernel)
    _kernel[grid](arg=..., ...)
"""

import contextlib

import torch
from triton.runtime import driver
from triton.runtime.jit import JITFunction

_MAX_INT32 = 2**31 - 1

# Set to record every key miss, which a test uses to prove the guard is what
# separates two shapes rather than luck.
_MISSES: list | None = None

# Set while a test wants the ordinary path, to compare this one against.
_BYPASS = False


@contextlib.contextmanager
def bypassed():
    """Route every launch the ordinary way, for tests to compare against."""
    global _BYPASS
    _BYPASS, was = True, _BYPASS
    try:
        yield
    finally:
        _BYPASS = was


@contextlib.contextmanager
def recording():
    """Collect ``(kernel, key)`` for each key this block compiles for."""
    global _MISSES
    _MISSES, was = [], _MISSES
    try:
        yield _MISSES
    finally:
        _MISSES = was


def _tensor_key(t: torch.Tensor):
    """The three things Triton compiled against, and nothing else.

    Shape is deliberately absent: a tensor reaches the kernel as a pointer, so
    two shapes that agree here produce the same machine code. Anything that
    does vary with shape arrives as an integer argument and is keyed by value.
    """
    return (
        t.dtype,
        t.data_ptr() % 16 == 0,
        t.untyped_storage().size() <= _MAX_INT32,
    )


class _FastLaunch:
    """A kernel that remembers what it compiled for a given set of shapes."""

    def __init__(self, kernel):
        self._kernel = kernel
        self._cache = {}
        # The autotuner and any other decorator wrap the JITFunction that owns
        # the argument binder, which is the only piece of it needed here.
        jit = kernel
        while not isinstance(jit, JITFunction):
            jit = jit.fn
        self._jit = jit
        self._disabled = False

    def __getitem__(self, grid):
        def launch(**kwargs):
            if self._disabled or _BYPASS:
                return self._kernel[grid](**kwargs)
            try:
                key = (
                    tuple(kwargs.keys()),
                    tuple(
                        _tensor_key(v) if isinstance(v, torch.Tensor) else v
                        for v in kwargs.values()
                    ),
                )
                entry = self._cache.get(key)
            except TypeError:
                # An argument that cannot be hashed cannot be guarded, and a
                # guard that silently omits an argument is the failure mode
                # this whole file exists to avoid.
                self._disabled = True
                return self._kernel[grid](**kwargs)

            if entry is None:
                return self._capture(grid, kwargs, key)

            compiled, names, frozen, meta = entry
            g = grid(meta) if callable(grid) else grid
            device = driver.active.get_current_device()
            # Not hoisted: capture redirects work onto its own stream, so the
            # answer changes under torch.cuda.graph.
            stream = driver.active.get_current_stream(device)
            compiled.run(
                g[0],
                g[1] if len(g) > 1 else 1,
                g[2] if len(g) > 2 else 1,
                stream,
                compiled.function,
                compiled.packed_metadata,
                None,
                None,
                None,
                *[kwargs[n] if n in kwargs else frozen[n] for n in names],
            )
            return compiled

        return launch

    def _capture(self, grid, kwargs, key):
        """Launch the ordinary way, and keep what it worked out.

        The arguments are read back through a run hook rather than rebuilt.
        Every decorator in the chain contributes some -- heuristics supply the
        predicates, the autotuner supplies the config -- and reconstructing
        that from the outside means tracking what each one does, which is
        exactly the coupling this cache should not have.
        """
        if _MISSES is not None:
            _MISSES.append((self._jit.__name__, key))

        seen = []

        def hook(*args, **kw):
            seen.append((args, kw))

        self._jit.pre_run_hooks.append(hook)
        try:
            compiled = self._kernel[grid](**kwargs)
        finally:
            self._jit.pre_run_hooks.remove(hook)
        if compiled is None or not seen:
            return compiled

        # Autotuning launches once per candidate config; the last one is the
        # launch that used the config it settled on.
        args, kw = seen[-1]
        device = driver.active.get_current_device()
        binder = self._jit.device_caches[device][4]
        bound_args, _, _ = binder(*args, **kw)

        names = list(bound_args.keys())
        # Arguments the call site does not pass -- defaults, and whatever the
        # autotuner added. The key pins every non-tensor value, so these are
        # constant for as long as this entry is the one selected.
        frozen = {n: v for n, v in bound_args.items() if n not in kwargs}
        self._cache[key] = (compiled, names, frozen, bound_args)
        return compiled


def fast_launch(kernel):
    return _FastLaunch(kernel)
