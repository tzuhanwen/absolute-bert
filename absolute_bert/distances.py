import torch
from jaxtyping import Complex
from torch import Tensor


def pairwise_distance(
    a: Complex[Tensor, "M D"], b: Complex[Tensor, "N D"], metric="l2", eps=1e-8
) -> Complex[Tensor, "M N"]:
    """
    A: M x D
    B: N x D
    distance: 'cos', 'l2'

    output: M x N
    """
    if metric == "cos":
        product = torch.einsum("md,nd->mn", a, b.conj())
        norms = a.norm(dim=1)[..., None] * b.norm(dim=1)[None, ...] + eps
        return 1 - (product / norms).real

    if metric == "l2":
        a_t = a[..., None]
        b_t = b.T[None, ...]
        # return (
        #     (a_t.real**2+a_t.imag**2).sum(dim=0) +
        #     (b_t.real**2+b_t.imag**2).sum(dim=-1) -
        #     torch.einsum('DM...,...ND->MN', a_t, b_t.conj())
        # )
        return (a_t - b_t).norm(dim=1)


def paired_distance(a: Complex[Tensor, "M D"], b: Complex[Tensor, "N D"], metric="l2", eps=1e-8):
    """
    A: [M, D]
    B: [M, D]
    distance: 'cos', 'l2'

    output: [M]
    """
    if metric == "l2":
        return (a - b).norm(dim=1)

    if metric == "cos":
        product = torch.einsum("md,md->m", a, b.conj())
        norms = a.norm(dim=1) * b.norm(dim=1) + eps
        return 1 - (product / norms).real
