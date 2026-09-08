"""K2, the delta-rule recurrence, with the state resident in MFMA registers.

The state tile is at once the accumulator of ``dot(kr^T, U)`` and the B operand
of ``dot(kd, h)``. On CDNA those two register distributions coincide for a
``[16, 16]`` instruction with ``transposed=False``, so the state can hold both
roles without moving; the Triton kernel cannot see that and writes all four
state tiles to LDS and reads them back every iteration.

That alone buys nothing measurable -- unfused, this lands within a few percent
of the Triton kernel either way. What it buys is the *fusion*: pass A's two
launches differ only in how they are seeded, so once the state is in registers
one launch can carry both recurrences, share every operand load, and hand the
scheduler two independent MFMA chains to interleave against the serial
dependence each one has on its own. That is ``k2_ab_fused_gluon``, it is worth
1.3-1.7x on those two launches, and it is the only kernel here that
``flash_kda_fwd`` routes to.

``k2_segment_gluon`` is the faithful drop-in for ``_flash_kda_segment_kernel``,
same arguments and same ABI, and it is what establishes the comparison above.
It is deliberately not on the default path. Pass C has nothing to fuse with, and
there the register-resident state is a liability rather than a win: it cannot be
warp-split along K, because K is the contraction axis of ``dot(kd, h)`` and a B
operand is replicated over those warps rather than split. So its wave count is
capped at ``W / 16 * num_segs * H``, and on an unsegmented shape -- which is
what aiter picks below 256 chunks -- that is half the parallelism the Triton
kernel gets from its LDS round-trip, and it measures 2x slower.
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton._triton_kernels.chunk_delta_attn.fast_launch import fast_launch

# BW -> num_warps. The state is the B operand of dot(kd, h), whose contraction
# axis is the state's K axis, so the warps can only split BW, and BW has to be a
# multiple of 16 * num_warps for that split to divide rather than duplicate.
# BW = 16 is left out: it forces one warp, and single-wave workgroups lose more
# to per-workgroup overhead than the extra blocks return.
BW_WARPS = {32: 2, 64: 4}
KW = 8
KW_BIG = 8


def build_layouts(nw, kw=KW, kw_big=KW_BIG):
    """The layout set, derived from one decision: the state stays in registers.

    ``instr_shape[0:2] = [16, 16]`` with ``transposed=False`` is what makes an
    MFMA accumulator a legal B operand, so the state can be the accumulator of
    ``dot(kr^T, U)`` and the B operand of ``dot(kd, h)`` without a round trip.
    ``kw`` names the dots that contract over C and ``kw_big`` the one that
    contracts over K; the two accumulator distributions are the same, since an
    accumulator's layout is set by M and N and not by K.
    """
    mma = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 4 * kw], transposed=False,
        warps_per_cta=[1, nw],
    )
    mma_b = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 4 * kw_big], transposed=False,
        warps_per_cta=[1, nw],
    )
    return {
        "MMA": mma,
        "A_OP": gl.DotOperandLayout(0, mma, kw),
        "B_OP": gl.DotOperandLayout(1, mma, kw),
        "MMA_B": mma_b,
        "A_OP_B": gl.DotOperandLayout(0, mma_b, kw_big),
        "B_OP_B": gl.DotOperandLayout(1, mma_b, kw_big),
        "BLK": gl.BlockedLayout([1, 8], [4, 16], [nw, 1], [1, 0]),
        "SH_KR": gl.SwizzledSharedLayout(8, 1, 16, [0, 1]),
    }


@gluon.jit
def _sigmoid(x):
    return gl.extra.libdevice.fast_dividef(1.0, 1.0 + gl.exp(-x.to(gl.float32)))


@gluon.jit
def _recur(
    h,
    kd_a,
    inv_a,
    kr_a,
    gt,
    beta,
    v,
    m_c,
    C: gl.constexpr,
    BW: gl.constexpr,
    MMA: gl.constexpr,
    B_OP: gl.constexpr,
    MMA_B: gl.constexpr,
    B_OP_B: gl.constexpr,
    INV_TY: gl.constexpr,
    HAS_V: gl.constexpr,
):
    """One chunk of the delta-rule recurrence, entirely in registers.

    Returns the state before the update alongside the new one, since the output
    pass contracts against the incoming state and against U.

    ``h * gt[:, None]`` is handed to the last mfma as its accumulator rather
    than added afterwards, so the decay rides in the MFMA chain's initial value.
    """
    h_op = gl.convert_layout(h.to(gl.bfloat16), B_OP_B)
    tmp = gl.convert_layout(
        gl.amd.cdna4.mfma(kd_a, h_op, gl.zeros([C, BW], gl.float32, MMA_B)), MMA
    )
    if HAS_V:
        u = (v - tmp) * beta[:, None]
    else:
        u = (-tmp) * beta[:, None]
    # Tail rows must stay zero: U feeds the state update, which sums over all C
    # rows regardless of how many are real.
    u = gl.where(m_c[:, None], u, 0.0)
    big_u = gl.amd.cdna4.mfma(
        inv_a, gl.convert_layout(u.to(INV_TY), B_OP), gl.zeros([C, BW], gl.float32, MMA)
    )
    h_next = gl.amd.cdna4.mfma(
        kr_a, gl.convert_layout(big_u.to(gl.bfloat16), B_OP), h * gt[:, None]
    )
    return h_next, big_u, h_op


@gluon.jit
def _kr_operand(
    kr_raw,
    K: gl.constexpr,
    C: gl.constexpr,
    SH_KR: gl.constexpr,
    A_OP: gl.constexpr,
):
    """The one transpose in the kernel.

    kr contracts over C and ws_kr has C strided, so it comes in row-major and
    coalesced, goes to LDS in that same element order, and the transpose happens
    on the read, where gfx950 has ``ds_read_b64_tr_b16``. ``permute`` only swaps
    the layout's basis axes, so nothing moves for it.
    """
    return gl.allocate_shared_memory(
        gl.bfloat16, [K, C], SH_KR, gl.permute(kr_raw, 1, 0)
    ).load(A_OP)


@gluon.jit
def k2_segment_gluon(
    ws_kd,
    ws_qd,
    ws_kr,
    ws_gt,
    ws_inv_mqk,
    v_input,
    beta_raw,
    out,
    h_in,
    h_out,
    final_state,
    seg_chunk_base,
    seg_nchunks,
    seg_tok_base,
    seg_tok_end,
    seg_seq,
    seg_is_last,
    TOTAL_TILES,
    H: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    W: gl.constexpr,
    C: gl.constexpr,
    BW: gl.constexpr,
    MMA: gl.constexpr,
    A_OP: gl.constexpr,
    B_OP: gl.constexpr,
    MMA_B: gl.constexpr,
    A_OP_B: gl.constexpr,
    B_OP_B: gl.constexpr,
    BLK: gl.constexpr,
    SH_KR: gl.constexpr,
    INIT_IDENTITY: gl.constexpr,
    HAS_H_IN: gl.constexpr,
    HAS_V: gl.constexpr,
    COMPUTE_OUTPUT: gl.constexpr,
    STORE_H_OUT: gl.constexpr,
    STORE_FINAL: gl.constexpr,
    STATE_V_FIRST: gl.constexpr,
):
    """Delta-rule recurrence over one segment, the same three passes as Triton's.

    Drop-in for ``_flash_kda_segment_kernel``: same arguments, same ABI, and the
    same seeding/storing flags select the same pass.
    """
    gl.static_assert(W % BW == 0)

    i_w = gl.program_id(0).to(gl.int64)
    i_sh = gl.program_id(1).to(gl.int64)
    i_seg = i_sh // H
    i_h = i_sh % H

    chunk_base = gl.load(seg_chunk_base + i_seg).to(gl.int64)
    n_chunks = gl.load(seg_nchunks + i_seg)
    tok_base = gl.load(seg_tok_base + i_seg).to(gl.int64)
    tok_end = gl.load(seg_tok_end + i_seg).to(gl.int64)

    o_c_ab = gl.arange(0, C, layout=gl.SliceLayout(1, A_OP_B))
    o_k_ab = gl.arange(0, K, layout=gl.SliceLayout(0, A_OP_B))
    o_c_a = gl.arange(0, C, layout=gl.SliceLayout(1, A_OP))
    o_cc_a = gl.arange(0, C, layout=gl.SliceLayout(0, A_OP))
    o_c_s = gl.arange(0, C, layout=gl.SliceLayout(1, BLK))
    o_k_s = gl.arange(0, K, layout=gl.SliceLayout(0, BLK))
    o_c_m = gl.arange(0, C, layout=gl.SliceLayout(1, MMA))
    o_k_m = gl.arange(0, K, layout=gl.SliceLayout(1, MMA))
    o_w_m = i_w * BW + gl.arange(0, BW, layout=gl.SliceLayout(0, MMA))

    # Each base pointer carries its tile's scalar displacement and only the
    # intra-tile part stays here, which keeps these loop-invariant.
    kd_off = (o_c_ab[:, None] * K + o_k_ab[None, :]).to(gl.int32)
    inv_off = (o_c_a[:, None] * C + o_cc_a[None, :]).to(gl.int32)
    kr_off = (o_c_s[:, None] * K + o_k_s[None, :]).to(gl.int32)
    gt_off = o_k_m.to(gl.int32)
    beta_off = (i_h + o_c_m * H).to(gl.int32)
    vo_off = (i_h * V + o_c_m[:, None] * (H * V) + o_w_m[None, :]).to(gl.int32)
    st_off = (o_k_m[:, None] * W + o_w_m[None, :]).to(gl.int32)

    if INIT_IDENTITY:
        b_h = gl.where(o_k_m[:, None] == o_w_m[None, :], 1.0, 0.0)
    elif HAS_H_IN:
        b_h = gl.amd.cdna4.buffer_load(
            ptr=h_in + (i_seg * H + i_h) * (K * W), offsets=st_off
        ).to(gl.float32)
    else:
        b_h = gl.zeros([K, BW], gl.float32, MMA)

    ws0 = i_h * TOTAL_TILES + chunk_base
    inv_ty: gl.constexpr = ws_inv_mqk.dtype.element_ty

    for j in range(n_chunks):
        ws_idx = ws0 + j
        ck = ws_idx * (C * K)
        cc = ws_idx * (2 * C * C)
        t0 = tok_base + j * C
        tb = t0 * H
        m_c = (t0 + o_c_m) < tok_end

        kd_a = gl.amd.cdna4.buffer_load(ptr=ws_kd + ck, offsets=kd_off)
        inv_a = gl.amd.cdna4.buffer_load(ptr=ws_inv_mqk + cc, offsets=inv_off)
        gt = gl.amd.cdna4.buffer_load(ptr=ws_gt + ws_idx * K, offsets=gt_off)
        kr_raw = gl.amd.cdna4.buffer_load(ptr=ws_kr + ck, offsets=kr_off)
        beta = _sigmoid(
            gl.amd.cdna4.buffer_load(
                ptr=beta_raw + tb, offsets=beta_off, mask=m_c, other=0.0
            )
        )
        if HAS_V:
            b_v = gl.amd.cdna4.buffer_load(
                ptr=v_input + tb * V, offsets=vo_off, mask=m_c[:, None], other=0.0
            ).to(gl.float32)
        else:
            b_v = gl.zeros([C, BW], gl.float32, MMA)

        kr_a = _kr_operand(kr_raw, K, C, SH_KR, A_OP)
        b_h_next, big_u, h_op = _recur(
            b_h, kd_a, inv_a, kr_a, gt, beta, b_v, m_c,
            C, BW, MMA, B_OP, MMA_B, B_OP_B, inv_ty, HAS_V,
        )  # fmt: skip

        if COMPUTE_OUTPUT:
            qd_a = gl.amd.cdna4.buffer_load(ptr=ws_qd + ck, offsets=kd_off)
            mqk_a = gl.amd.cdna4.buffer_load(
                ptr=ws_inv_mqk + cc + C * C, offsets=inv_off
            )
            b_o = gl.convert_layout(
                gl.amd.cdna4.mfma(qd_a, h_op, gl.zeros([C, BW], gl.float32, MMA_B)),
                MMA,
            )
            b_o = gl.amd.cdna4.mfma(
                mqk_a, gl.convert_layout(big_u.to(inv_ty), B_OP), b_o
            )
            gl.amd.cdna4.buffer_store(
                b_o.to(out.dtype.element_ty), out + tb * V, vo_off, mask=m_c[:, None]
            )

        b_h = b_h_next

    if STORE_H_OUT:
        gl.amd.cdna4.buffer_store(
            b_h.to(h_out.dtype.element_ty),
            h_out + (i_seg * H + i_h) * (K * W),
            st_off,
        )

    # Not merged into one condition: the outer test is a constexpr, and when it
    # is false final_state is a null pointer and seg_is_last must not be read.
    if STORE_FINAL:  # noqa: SIM102
        if gl.load(seg_is_last + i_seg) == 1:
            i_n = gl.load(seg_seq + i_seg).to(gl.int64)
            if STATE_V_FIRST:
                gl.amd.cdna4.buffer_store(
                    b_h,
                    final_state + (i_n * H + i_h) * (V * K),
                    (o_w_m[None, :] * K + o_k_m[:, None]).to(gl.int32),
                )
            else:
                gl.amd.cdna4.buffer_store(
                    b_h,
                    final_state + (i_n * H + i_h) * (K * V),
                    (o_k_m[:, None] * V + o_w_m[None, :]).to(gl.int32),
                )


@gluon.jit
def k2_ab_fused_gluon(
    ws_kd,
    ws_kr,
    ws_gt,
    ws_inv_mqk,
    v_input,
    beta_raw,
    h_out_b,
    h_out_a,
    seg_chunk_base,
    seg_nchunks,
    seg_tok_base,
    seg_tok_end,
    TOTAL_TILES,
    H: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    C: gl.constexpr,
    BW: gl.constexpr,
    MMA: gl.constexpr,
    A_OP: gl.constexpr,
    B_OP: gl.constexpr,
    MMA_B: gl.constexpr,
    A_OP_B: gl.constexpr,
    B_OP_B: gl.constexpr,
    BLK: gl.constexpr,
    SH_KR: gl.constexpr,
):
    """Both pass-A recurrences in one launch, sharing every operand load.

    The two differ only in seeding -- zero with the real v against the identity
    with v = 0 -- so they read the same chunk and can share it. Register
    residency is what makes the sharing worth having: the two chains are
    independent, so the scheduler interleaves them, and that is what covers the
    serial dependence each one has on its own. Requires K == V, since the two
    states are then the same width and one program covers a column block of
    both.
    """
    i_w = gl.program_id(0).to(gl.int64)
    i_sh = gl.program_id(1).to(gl.int64)
    i_seg = i_sh // H
    i_h = i_sh % H

    chunk_base = gl.load(seg_chunk_base + i_seg).to(gl.int64)
    n_chunks = gl.load(seg_nchunks + i_seg)
    tok_base = gl.load(seg_tok_base + i_seg).to(gl.int64)
    tok_end = gl.load(seg_tok_end + i_seg).to(gl.int64)

    o_c_ab = gl.arange(0, C, layout=gl.SliceLayout(1, A_OP_B))
    o_k_ab = gl.arange(0, K, layout=gl.SliceLayout(0, A_OP_B))
    o_c_a = gl.arange(0, C, layout=gl.SliceLayout(1, A_OP))
    o_cc_a = gl.arange(0, C, layout=gl.SliceLayout(0, A_OP))
    o_c_s = gl.arange(0, C, layout=gl.SliceLayout(1, BLK))
    o_k_s = gl.arange(0, K, layout=gl.SliceLayout(0, BLK))
    o_c_m = gl.arange(0, C, layout=gl.SliceLayout(1, MMA))
    o_k_m = gl.arange(0, K, layout=gl.SliceLayout(1, MMA))
    o_w_m = i_w * BW + gl.arange(0, BW, layout=gl.SliceLayout(0, MMA))

    kd_off = (o_c_ab[:, None] * K + o_k_ab[None, :]).to(gl.int32)
    inv_off = (o_c_a[:, None] * C + o_cc_a[None, :]).to(gl.int32)
    kr_off = (o_c_s[:, None] * K + o_k_s[None, :]).to(gl.int32)
    gt_off = o_k_m.to(gl.int32)
    beta_off = (i_h + o_c_m * H).to(gl.int32)
    v_off = (i_h * V + o_c_m[:, None] * (H * V) + o_w_m[None, :]).to(gl.int32)

    h_b = gl.zeros([K, BW], gl.float32, MMA)
    h_a = gl.where(o_k_m[:, None] == o_w_m[None, :], 1.0, 0.0)

    ws0 = i_h * TOTAL_TILES + chunk_base
    inv_ty: gl.constexpr = ws_inv_mqk.dtype.element_ty

    for j in range(n_chunks):
        ws_idx = ws0 + j
        ck = ws_idx * (C * K)
        t0 = tok_base + j * C
        tb = t0 * H
        m_c = (t0 + o_c_m) < tok_end

        kd_a = gl.amd.cdna4.buffer_load(ptr=ws_kd + ck, offsets=kd_off)
        inv_a = gl.amd.cdna4.buffer_load(
            ptr=ws_inv_mqk + ws_idx * (2 * C * C), offsets=inv_off
        )
        gt = gl.amd.cdna4.buffer_load(ptr=ws_gt + ws_idx * K, offsets=gt_off)
        kr_raw = gl.amd.cdna4.buffer_load(ptr=ws_kr + ck, offsets=kr_off)
        beta = _sigmoid(
            gl.amd.cdna4.buffer_load(
                ptr=beta_raw + tb, offsets=beta_off, mask=m_c, other=0.0
            )
        )
        b_v = gl.amd.cdna4.buffer_load(
            ptr=v_input + tb * V, offsets=v_off, mask=m_c[:, None], other=0.0
        ).to(gl.float32)

        kr_a = _kr_operand(kr_raw, K, C, SH_KR, A_OP)
        # Written one after the other so the scheduler has two independent MFMA
        # chains to interleave, which is what covers the serial dependence.
        h_b, _u, _h = _recur(h_b, kd_a, inv_a, kr_a, gt, beta, b_v, m_c, C, BW,
                             MMA, B_OP, MMA_B, B_OP_B, inv_ty, True)  # fmt: skip
        h_a, _u, _h = _recur(h_a, kd_a, inv_a, kr_a, gt, beta, b_v, m_c, C, BW,
                             MMA, B_OP, MMA_B, B_OP_B, inv_ty, False)  # fmt: skip

    s_base = (i_seg * H + i_h) * (K * V)
    s_off = (o_k_m[:, None] * V + o_w_m[None, :]).to(gl.int32)
    gl.amd.cdna4.buffer_store(h_b.to(h_out_b.dtype.element_ty), h_out_b + s_base, s_off)
    gl.amd.cdna4.buffer_store(h_a.to(h_out_a.dtype.element_ty), h_out_a + s_base, s_off)


# Launched through the shape cache; see fast_launch.py for why.
k2_ab_fused_fast = fast_launch(k2_ab_fused_gluon)
