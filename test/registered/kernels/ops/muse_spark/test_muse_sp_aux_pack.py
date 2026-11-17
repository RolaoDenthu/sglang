"""Muse Spark sequence-parallel prefill: packed DFlash aux captures.

A sequence-parallel prefill casts each captured residual stream straight into its slot of one
packed [shard rows, captures x hidden] BF16 block and all-gathers that block once, instead of casting, gathering and
later concatenating every capture separately. This test runs the real ``MuseSpark1xModel.forward`` on CPU with stub
decoder layers and a TP group simulated by one thread per rank (the real ``sp_shard`` / ``sp_gather`` run unchanged).
It checks, on every rank, that:
- the packed gather is byte-identical to the captures of the same forward without sequence parallelism;
- the final hidden states do not change.
"""

import threading
import types
import unittest
from unittest import mock

import torch
from torch import nn

from sglang.srt.rocm_muse_spark import model as muse_model
from sglang.srt.rocm_muse_spark import sequence_parallel
from sglang.srt.rocm_muse_spark.sequence_parallel import SPShape
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=30, suite="base-b-test-1-gpu-small-amd-mi35x")

_HIDDEN = 32
_LAYERS = 8
_CAPTURE = [0, 2, 3, 5, 7]


def _row_norm(x):
    x32 = x.float()
    return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + 1e-6)).to(
        torch.bfloat16
    )


class _Layer(nn.Module):
    """Row-local stand-in for a decoder layer: a fresh FP32 stream with full-precision low bits, as the real layer
    hands on."""

    def __init__(self, index):
        super().__init__()
        self.index = index

    def self_attn_input_norm(self, stream):
        return _row_norm(stream)

    def forward(
        self,
        positions,
        stream,
        forward_batch,
        *,
        attn_input=None,
        ffn_boundary=None,
        pre_norm_dtype=None,
        sp=None,
        attn_input_fp8=None,
    ):
        new = (
            stream * (1.0 + 0.0123 * (self.index + 1))
            + torch.sin(attn_input.float()) * 0.37
        )
        return new, _row_norm(new), None


class _ThreadGroup:
    """A TP group whose ranks are threads: ``all_gather_into_tensor`` concatenates every rank's input in rank
    order."""

    def __init__(self, world_size):
        self.world_size = world_size
        self._local = threading.local()
        self._barrier = threading.Barrier(world_size)
        self._parts = [None] * world_size

    @property
    def rank_in_group(self):
        return self._local.rank

    def set_rank(self, rank):
        self._local.rank = rank

    def all_gather_into_tensor(self, out, inp):
        self._parts[self.rank_in_group] = inp.clone()
        self._barrier.wait()
        out.copy_(torch.cat(self._parts))
        self._barrier.wait()  # nobody overwrites a part before every rank has read it


def _model():
    model = muse_model.MuseSpark1xModel.__new__(muse_model.MuseSpark1xModel)
    nn.Module.__init__(model)
    model.residual_dtype = torch.float32
    model.config = types.SimpleNamespace(hidden_size=_HIDDEN)
    model.embed_tokens = None
    model.embed_norm = None
    model.layers = nn.ModuleList([_Layer(i) for i in range(_LAYERS)])
    model._ffn_boundaries = [None] * _LAYERS
    model.layers_to_capture = list(_CAPTURE)
    model.dflash_context_projector = None  # no DFlash draft projecting the shards
    return model


def _forward_batch():
    return types.SimpleNamespace(
        forward_mode=types.SimpleNamespace(is_target_verify=lambda: False),
        aux_hidden_states_buffer=None,
    )


def _embeds(tokens):
    torch.manual_seed(tokens)
    x = torch.randn(tokens, _HIDDEN) * 3
    # Rows with BF16 round-to-nearest-even ties and near-ties, so a cast that is not RNE shows.
    ties = torch.tensor([1 + 2**-8, 1 + 3 * 2**-8, -(1 + 2**-8), 2**-8 * (1 + 2**-8)])
    x[0, :4] = ties
    return x


def _no_tbo(*args, **kwargs):
    return False


def _run(model, embeds, *, world_size, use_sp=True):
    """Return [(hidden_states, aux_hidden_states)] per rank (one entry without sequence parallelism)."""
    tokens = embeds.shape[0]
    group = _ThreadGroup(world_size)
    shape = SPShape(tokens, -(-tokens // world_size) * world_size) if use_sp else None
    patches = [
        mock.patch.object(
            muse_model, "sp_shape", lambda forward_batch, num_tokens, row_bytes: shape
        ),
        mock.patch.object(
            muse_model, "get_parallel", lambda: types.SimpleNamespace(tp_group=group)
        ),
        mock.patch.object(
            sequence_parallel,
            "get_parallel",
            lambda: types.SimpleNamespace(tp_group=group),
        ),
        mock.patch.object(muse_model, "tbo_begin", _no_tbo, create=True),
        mock.patch.object(muse_model, "tbo_end", _no_tbo, create=True),
    ]
    for p in patches:
        p.start()
    try:
        ranks = world_size if use_sp else 1
        results = [None] * ranks
        errors = []

        def rank_main(rank):
            try:
                group.set_rank(rank)
                positions = torch.arange(tokens)
                results[rank] = model.forward(
                    None, positions, _forward_batch(), input_embeds=embeds
                )
            except (
                BaseException
            ) as e:  # surfaced below; a dead rank would leave the others at the barrier
                errors.append(e)
                group._barrier.abort()

        threads = [threading.Thread(target=rank_main, args=(r,)) for r in range(ranks)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=120)
        if errors:
            raise errors[0]
        return results
    finally:
        for p in reversed(patches):
            p.stop()


def _bits(t):
    return t.contiguous().view(torch.int16)


class TestMuseSpAuxPack(unittest.TestCase):
    CASES = [(4, 64), (4, 61), (8, 61), (4, 3), (2, 37)]

    def test_packed_gather_is_byte_identical_to_the_unsharded_captures(self):
        for world_size, tokens in self.CASES:
            with self.subTest(world_size=world_size, tokens=tokens):
                embeds = _embeds(tokens)
                ref_hidden, ref_aux = _run(
                    _model(), embeds, world_size=world_size, use_sp=False
                )[0]
                ref = torch.cat(ref_aux, dim=1)
                self.assertEqual(ref.shape, (tokens, len(_CAPTURE) * _HIDDEN))
                packed = _run(_model(), embeds, world_size=world_size)
                for rank in range(world_size):
                    hidden_p, aux_p = packed[rank]
                    self.assertIsInstance(
                        aux_p, torch.Tensor
                    )  # one packed block, no concatenation left
                    self.assertEqual(aux_p.dtype, torch.bfloat16)
                    self.assertEqual(aux_p.shape, ref.shape)
                    self.assertTrue(torch.equal(_bits(aux_p), _bits(ref)), rank)
                    self.assertTrue(
                        torch.equal(_bits(hidden_p), _bits(ref_hidden)), rank
                    )

    def test_no_sequence_parallelism_keeps_the_capture_list(self):
        embeds = _embeds(61)
        _, aux = _run(_model(), embeds, world_size=4, use_sp=False)[0]
        self.assertIsInstance(aux, list)
        self.assertEqual(len(aux), len(_CAPTURE))


if __name__ == "__main__":
    unittest.main()
