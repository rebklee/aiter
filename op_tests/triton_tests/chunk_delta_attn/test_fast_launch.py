# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for the shape cache in front of the flash_kda kernel launches.

The cache skips Triton's per-call preamble by reusing the kernel it compiled
for a set of shapes. That is only sound while its key asks everything Triton
specialized on, and getting it wrong does not raise -- it runs a kernel
compiled under assumptions the arguments no longer meet, and returns whatever
that produces. So the tests here are about the key, not about speed:

* the cached path returns the same bytes as the ordinary one, across shapes,
  varlen, bias, initial state, and both segmented and not;
* two shapes that need different kernels get different entries, rather than
  the first one's kernel being handed to the second;
* an argument whose alignment differs is treated as different, since 16-byte
  alignment is one of the things the compiler was told it could assume.
"""

import pytest
import torch

from aiter.ops.triton._triton_kernels.chunk_delta_attn import fast_launch
from aiter.ops.triton._triton_kernels.chunk_delta_attn.flash_kda import flash_kda_fwd

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="flash_kda needs a GPU"
)

device = "cuda"
dtype = torch.bfloat16
K_DIM = 128
LOWER_BOUND = -5.0


def make_inputs(B, T, H, seed=0, bias=False, state=False, varlen=False):
    g = torch.Generator(device=device).manual_seed(seed)

    def rnd(*shape, dt=dtype):
        return torch.randn(*shape, generator=g, device=device, dtype=dt)

    args = {
        "q": rnd(B, T, H, K_DIM),
        "k": rnd(B, T, H, K_DIM),
        "v": rnd(B, T, H, K_DIM),
        "g": rnd(B, T, H, K_DIM, dt=torch.float32),
        "beta": rnd(B, T, H, dt=torch.float32),
        "A_log": rnd(H, K_DIM, dt=torch.float32),
        "dt_bias": rnd(H, K_DIM, dt=torch.float32) if bias else None,
        "initial_state": (
            rnd(B, H, K_DIM, K_DIM, dt=torch.float32) if state else None
        ),
    }
    if varlen:
        # One packed sequence per batch entry, which is how the varlen path is
        # reached: B collapses to 1 and the bounds carry the split.
        args["q"], args["k"], args["v"], args["g"] = (
            args[n].reshape(1, B * T, H, -1) for n in ("q", "k", "v", "g")
        )
        args["beta"] = args["beta"].reshape(1, B * T, H)
        args["cu_seqlens"] = torch.arange(
            0, B * T + 1, T, device=device, dtype=torch.int32
        )
        if args["initial_state"] is not None:
            args["initial_state"] = args["initial_state"][:B]
    return args


def run(args, **kw):
    return flash_kda_fwd(
        **args,
        scale=K_DIM**-0.5,
        lower_bound=LOWER_BOUND,
        output_final_state=True,
        **kw,
    )


@pytest.mark.parametrize("B,T,H", [(1, 512, 12), (1, 4096, 12), (2, 1024, 4)])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("state", [False, True])
def test_matches_ordinary_path(B, T, H, bias, state):
    args = make_inputs(B, T, H, bias=bias, state=state)
    with fast_launch.bypassed():
        want_o, want_s = run(args)
    got_o, got_s = run(args)
    assert torch.equal(got_o, want_o)
    assert torch.equal(got_s, want_s)


@pytest.mark.parametrize("chunks_per_seg", [0, 4])
def test_matches_ordinary_path_when_segmented(chunks_per_seg):
    args = make_inputs(1, 4096, 12)
    with fast_launch.bypassed():
        want_o, want_s = run(args, chunks_per_seg=chunks_per_seg)
    got_o, got_s = run(args, chunks_per_seg=chunks_per_seg)
    assert torch.equal(got_o, want_o)
    assert torch.equal(got_s, want_s)


def test_matches_ordinary_path_varlen():
    args = make_inputs(3, 512, 12, varlen=True)
    with fast_launch.bypassed():
        want_o, want_s = run(args)
    got_o, got_s = run(args)
    assert torch.equal(got_o, want_o)
    assert torch.equal(got_s, want_s)


def test_a_second_shape_does_not_reuse_the_first_entry():
    small, large = make_inputs(1, 512, 12), make_inputs(1, 1024, 12)
    run(small)  # whatever either shape needs is compiled and cached by now
    run(large)
    with fast_launch.recording() as misses:
        run(small)
        run(large)
    assert misses == [], "a warm shape should not be compiling"

    other = make_inputs(1, 2048, 12)
    with fast_launch.recording() as misses:
        run(other)
    assert misses, "a shape not seen before has to miss, not reuse an entry"


def test_alignment_is_part_of_the_key():
    """A tensor off a 16-byte boundary must not reach a kernel told otherwise."""
    B, T, H = 1, 512, 12
    args = make_inputs(B, T, H)
    run(args)

    # Same shape and dtype, one element into its storage, so the pointer moves
    # by 2 bytes and the divisibility Triton specialized on no longer holds.
    wide = torch.randn(B * T * H * K_DIM + 1, device=device, dtype=dtype)
    assert wide.data_ptr() % 16 == 0
    skewed = wide[1:].view(B, T, H, K_DIM)
    assert skewed.data_ptr() % 16 != 0

    misaligned = dict(args, q=skewed)
    with fast_launch.recording() as misses:
        run(misaligned)
    assert misses, "a differently aligned argument has to miss"

    with fast_launch.bypassed():
        want_o, want_s = run(misaligned)
    got_o, got_s = run(misaligned)
    assert torch.equal(got_o, want_o)
    assert torch.equal(got_s, want_s)
