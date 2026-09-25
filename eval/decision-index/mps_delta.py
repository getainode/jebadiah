"""MPS-only speedup for Qwen3.5's Gated DeltaNet prefill, used when the CUDA kernels
(flash-linear-attention) are unavailable.

transformers' reference `torch_chunk_gated_delta_rule` solves two unit lower-triangular systems per
chunk with torch.linalg.solve_triangular, which on MPS costs about 39 ms per call and dominates the
forward pass. Here the inverse of the unit lower-triangular chunk matrix is formed in fp32 by 2x2 block
recursion down to 16x16 blocks, which are inverted with transformers' own forward-substitution loop,
and applied by matmul. Same algebra, same inputs and outputs; everything else is the
reference function. Installed by patching the module attribute the attention layer calls.
"""
import functools

import torch


def _forward_substitution(a):
    """Inverse of a small unit lower-triangular block, by transformers' own forward-substitution
    loop (the branch its reference function uses when exporting)."""
    n = a.shape[-1]
    t = -a.tril(-1)
    for i in range(1, n):
        row = t[..., i, :i].clone()
        sub = t[..., :i, :i].clone()
        t[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    return t + torch.eye(n, dtype=a.dtype, device=a.device)


def unit_lower_inverse(a, base=16):
    """Exact inverse of a unit lower-triangular matrix by 2x2 block recursion:
    inv([[A11, 0], [A21, A22]]) = [[B11, 0], [-B22 A21 B11, B22]], with Bii = inv(Aii)."""
    n = a.shape[-1]
    if n <= base:
        return _forward_substitution(a)
    h = n // 2
    b11 = unit_lower_inverse(a[..., :h, :h], base)
    b22 = unit_lower_inverse(a[..., h:, h:], base)
    b21 = -(b22 @ a[..., h:, :h] @ b11)
    top = torch.cat([b11, torch.zeros_like(a[..., :h, h:])], dim=-1)
    return torch.cat([top, torch.cat([b21, b22], dim=-1)], dim=-2)


def install():
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m

    reference = m.torch_chunk_gated_delta_rule
    original_solve = torch.linalg.solve_triangular

    def fast_solve(a, b, *, upper, left=True, unitriangular=False, out=None):
        if upper or not unitriangular or not left or out is not None or a.device.type != "mps":
            return original_solve(a, b, upper=upper, left=left, unitriangular=unitriangular, out=out)
        return unit_lower_inverse(a.tril(-1) + torch.eye(a.shape[-1], dtype=a.dtype, device=a.device)) @ b

    @functools.wraps(reference)
    def patched(*args, **kwargs):
        torch.linalg.solve_triangular = fast_solve
        try:
            return reference(*args, **kwargs)
        finally:
            torch.linalg.solve_triangular = original_solve

    m.torch_chunk_gated_delta_rule = patched
    return "unit-lower-triangular inverse by block recursion (16x16 forward substitution) in place of solve_triangular (MPS only)"
