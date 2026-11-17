"""DFlash context projection sharded by output features over TP ranks (SGLANG_ROCM_MUSE_DFLASH_FC_SHARD)."""

import types
import unittest
from unittest import mock

import torch
import torch.nn.functional as F
from torch import nn

from sglang.kernels.ops.gemm.muse_spark_fc_shard import fc_shard_applies, fc_shard_gemm
from sglang.srt.models import dflash
from sglang.srt.models.dflash import DFlash2DraftModel
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=60, suite="base-b-test-1-gpu-small-amd-mi35x")

_OUT, _IN = 64, 5 * 32


class _RowRMSNorm(nn.Module):
    # Depends on the whole row, so a norm applied before the gather changes the result.
    def forward(self, x):
        x32 = x.float()
        return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + 1e-5)).to(x.dtype)


class _Group:
    """One TP group simulated rank by rank: ``all_gather`` records each rank's input
    on the first pass and returns the rank-order concatenation on the second."""

    def __init__(self, tp_size):
        self.parts = [None] * tp_size
        self.rank = 0

    def all_gather(self, x, dim=-1):
        self.parts[self.rank] = x.clone()
        if any(p is None for p in self.parts):
            return torch.zeros(
                *x.shape[:-1], x.shape[-1] * len(self.parts), dtype=x.dtype
            )
        return torch.cat(self.parts, dim=dim)


class _NoGather:
    def all_gather(self, x, dim=-1):
        raise AssertionError("the replicated path must not all-gather")


def _draft(fc):
    model = DFlash2DraftModel.__new__(DFlash2DraftModel)
    nn.Module.__init__(model)
    model.fc = fc
    model.hidden_norm = _RowRMSNorm()
    model.shard_fc = True
    model.fc_shard_enabled = True
    model.quantizable = False
    # The base class's own fc splits are off, so a switched-off shard runs the replicated fc.
    model.fc_tp_columns = None
    model.shards_context_fc = False
    model.num_context_features = 5
    model.config = types.SimpleNamespace(hidden_size=fc.in_features // 5)
    return model


def _run_ranks(model, hidden, tp_size):
    group = _Group(tp_size)
    outs = []
    for _ in range(2):
        outs = []
        for rank in range(tp_size):
            group.rank = rank
            parallel = types.SimpleNamespace(
                tp_rank=rank, tp_size=tp_size, tp_group=group
            )
            with mock.patch.object(dflash, "get_parallel", return_value=parallel):
                outs.append(model.project_target_hidden(hidden))
    return outs


def _fc():
    torch.manual_seed(0)
    return nn.Linear(_IN, _OUT, bias=False).to(torch.bfloat16)


@mock.patch.object(dflash, "fc_shard_applies", fc_shard_applies, create=True)
@mock.patch.object(dflash, "fc_shard_gemm", fc_shard_gemm, create=True)
class TestDFlashFcShard(unittest.TestCase):
    def test_applies_to_any_row_count_that_splits(self):
        weight = _fc().weight
        hidden = torch.randn(7, _IN).to(torch.bfloat16)
        self.assertTrue(fc_shard_applies(hidden, weight, 4))
        self.assertTrue(fc_shard_applies(hidden[:1], weight, 8))
        self.assertFalse(fc_shard_applies(hidden[:0], weight, 4))
        self.assertFalse(fc_shard_applies(hidden, weight, 3))
        self.assertFalse(fc_shard_applies(hidden.float(), weight, 4))
        self.assertFalse(fc_shard_applies(hidden.t().contiguous().t(), weight, 4))
        self.assertFalse(fc_shard_applies(hidden[:, :-32].contiguous(), weight, 4))

    def test_every_rank_gets_the_rank_order_gather_then_the_norm(self):
        fc = _fc()
        model = _draft(fc)
        hidden = torch.randn(101, _IN).to(torch.bfloat16)
        parts = [
            fc_shard_gemm(hidden, fc.weight, tp_rank=r, tp_size=4) for r in range(4)
        ]
        expected = model.hidden_norm(torch.cat(parts, dim=-1))
        for out in _run_ranks(model, hidden, 4):
            self.assertTrue(torch.equal(out, expected))
        torch.testing.assert_close(expected, model.hidden_norm(fc(hidden)))

    def test_switch_off_non_muse_target_and_unsplittable_run_the_replicated_fc(self):
        fc = _fc()
        model = _draft(fc)
        # (switch on, Muse Spark target, tp)
        for shard_fc, muse_target, tp_size in [
            (False, True, 4),
            (True, False, 4),
            (True, True, 3),
        ]:
            hidden = torch.randn(101, _IN).to(torch.bfloat16)
            model.shard_fc = shard_fc
            model.fc_shard_enabled = muse_target
            parallel = types.SimpleNamespace(
                tp_rank=1, tp_size=tp_size, tp_group=_NoGather()
            )
            with mock.patch.object(dflash, "get_parallel", return_value=parallel):
                out = model.project_target_hidden(hidden)
            self.assertTrue(torch.equal(out, model.hidden_norm(fc(hidden))))


class TestDFlashFcShardOnDevice(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
    def test_rank_slices_match_the_replicated_projection(self):
        # Muse Spark's fc at TP4: verify-sized and prefill-sized row counts.
        torch.manual_seed(0)
        out_f, in_f = 8192, 5 * 8192
        weight = (torch.randn(out_f, in_f, device="cuda") * in_f**-0.5).to(
            torch.bfloat16
        )
        for rows in (8, 192, 4800, 4801):
            hidden = torch.randn(rows, in_f, device="cuda").to(torch.bfloat16)
            self.assertTrue(fc_shard_applies(hidden, weight, 4))
            parts = [
                fc_shard_gemm(hidden, weight, tp_rank=r, tp_size=4) for r in range(4)
            ]
            torch.testing.assert_close(
                torch.cat(parts, dim=-1), F.linear(hidden, weight)
            )


if __name__ == "__main__":
    unittest.main()
