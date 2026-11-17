"""Output-feature shards of a replicated BF16 projection (the Muse Spark DFlash draft fc) across TP ranks.

Each rank multiplies the whole input by its contiguous 1 / tp_size of the weight's rows and the caller all-gathers
the parts in rank order. ``F.linear`` on a slice picks its own hipBLASLt solution, which can split K differently
from the replicated GEMM, so a few elements may round differently; that is accepted for the draft's context.
"""

import torch
import torch.nn.functional as F


def fc_shard_applies(x: torch.Tensor, weight: torch.Tensor, tp_size: int) -> bool:
    """Whether ``x @ weight.T`` splits into ``tp_size`` contiguous output-feature slices."""
    return (
        x.dim() == 2
        and x.shape[0] > 0
        and x.dtype == weight.dtype == torch.bfloat16
        and x.is_contiguous()
        and weight.is_contiguous()
        and x.shape[1] == weight.shape[1]
        and weight.shape[0] % tp_size == 0
    )


def fc_shard_gemm(
    x: torch.Tensor, weight: torch.Tensor, *, tp_rank: int, tp_size: int
) -> torch.Tensor:
    """This rank's contiguous 1 / tp_size of ``x @ weight.T``'s output features."""
    rows = weight.shape[0] // tp_size
    return F.linear(x, weight.narrow(0, tp_rank * rows, rows))
