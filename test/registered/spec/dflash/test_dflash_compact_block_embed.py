"""DFlash compact block embedding: the draft block embedding from its distinct rows (bs bonus ids plus the
mask id) must equal embed_module(block_ids) bit for bit, for blocks laid out as the draft prepares them (bonus
token, then mask tokens). A vocab-sharded table is emulated by summing per-shard masked lookups, as the
all-reduce does. The worker module needs a visible GPU to import, so that test runs where one is.

The compact path is a Muse Spark fast path: DFlashWorkerV2.__init__ enables it only for a Muse Spark target
(``_muse_target``, from the target's hf_config.architectures[0] through is_muse_spark_architecture).
That gate is checked on CPU from the worker's source:
the ``_compact_block_embed`` expression and ``_is_muse_target`` are evaluated against stand-ins, without importing
the module.
"""

import ast
import importlib.util
import pathlib
import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_amd_ci, register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_amd_ci(est_time=10, suite="base-b-test-1-gpu-small-amd-mi35x")

VOCAB, HIDDEN, MASK_ID = 1000, 64, 997


def _has_gpu():
    try:
        torch.cuda.get_device_properties(0)
        return True
    except (AssertionError, RuntimeError):
        return False


class _ShardedEmbedding(torch.nn.Module):
    def __init__(self, weight, shards):
        super().__init__()
        self.weight, self.shards = weight, shards

    def forward(self, ids):
        out = torch.zeros(*ids.shape, self.weight.shape[1], dtype=self.weight.dtype)
        bounds = torch.linspace(0, VOCAB, self.shards + 1).long()
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            local = torch.where((ids >= lo) & (ids < hi), ids, 0)
            part = torch.nn.functional.embedding(local, self.weight)
            out = out + part.masked_fill(((ids < lo) | (ids >= hi))[..., None], 0)
        return out


@pytest.mark.skipif(
    not _has_gpu(), reason="dflash_worker_v2 imports need a visible GPU"
)
@pytest.mark.parametrize(
    "bs,block_size,shards", [(1, 8, 4), (28, 8, 4), (40, 8, 8), (5, 16, 1)]
)
def test_compact_block_embed_matches_full_lookup(bs, block_size, shards):
    from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

    gen = torch.Generator().manual_seed(bs)
    weight = torch.randn(VOCAB, HIDDEN, generator=gen).to(torch.bfloat16)
    weight[3] = -0.0  # signed zeros must survive as the full lookup leaves them
    embed = _ShardedEmbedding(weight, shards)
    block_ids = torch.full((bs, block_size), MASK_ID, dtype=torch.int64)
    block_ids[:, 0] = torch.randint(0, VOCAB, (bs,), generator=gen)
    block_ids[0, 0] = 3
    worker = SimpleNamespace(_mask_token_id=MASK_ID)
    compact = DFlashWorkerV2._embed_block_compact(
        worker, embed_module=embed, block_ids=block_ids
    )
    full = embed(block_ids)
    assert compact.shape == full.shape
    assert torch.equal(compact.view(torch.int16), full.view(torch.int16))


def _worker_class_ast():
    src = pathlib.Path(
        importlib.util.find_spec("sglang.srt.speculative.dflash_worker_v2").origin
    ).read_text()
    tree = ast.parse(src)
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "DFlashWorkerV2"
    )
    return src, cls


def _compact_gate_expr():
    """The expression DFlashWorkerV2.__init__ assigns to self._compact_block_embed (exactly one assignment)."""
    src, cls = _worker_class_ast()
    init = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    )
    found = [
        n.value
        for n in ast.walk(init)
        if isinstance(n, ast.Assign)
        and len(n.targets) == 1
        and isinstance(n.targets[0], ast.Attribute)
        and n.targets[0].attr == "_compact_block_embed"
    ]
    assert len(found) == 1, (
        f"expected one assignment of self._compact_block_embed, found {len(found)}"
    )
    return compile(ast.Expression(found[0]), "<compact_gate>", "eval")


def _eval_gate(muse_target, hip=True, selector=object()):
    self = SimpleNamespace(_muse_target=muse_target, selector=selector)
    return bool(eval(_compact_gate_expr(), {"is_hip": lambda: hip, "self": self}))


def test_compact_block_embed_off_for_non_muse_target():
    # everything else on: only the target decides
    assert _eval_gate(muse_target=True) is True
    assert _eval_gate(muse_target=False) is False
    # and the platform / selector still gate a Muse target
    assert _eval_gate(muse_target=True, hip=False) is False
    assert _eval_gate(muse_target=True, selector=None) is False


def test_muse_target_from_model_type():
    _, cls = _worker_class_ast()
    fn = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "_is_muse_target"
    )
    fn.decorator_list = []
    from sglang.srt.layers.layernorm_sp import is_muse_spark_architecture

    ns = {"is_muse_spark_architecture": is_muse_spark_architecture}
    exec(compile(ast.Module([fn], type_ignores=[]), "<is_muse_target>", "exec"), ns)

    def cfg(model_type, architectures):
        return SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type=model_type, architectures=architectures
            )
        )

    assert (
        ns["_is_muse_target"](cfg("muse_spark", ["MuseSparkForConditionalGeneration"]))
        is True
    )
    for other, archs in (
        ("llama", ["LlamaForCausalLM"]),
        ("qwen3_moe", ["Qwen3MoeForCausalLM"]),
        ("qwen3", ["Qwen3ForCausalLM"]),  # SP-supported, not Muse
        ("deepseek_v3", ["DeepseekV3ForCausalLM"]),
        ("muse_spark_draft", ["DFlash2DraftModel"]),
        ("", None),
        ("", []),
    ):
        assert ns["_is_muse_target"](cfg(other, archs)) is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
