"""The DFlash worker's Muse Spark fast paths stay off for every other target (CPU, no GPU import):
- DFlashWorkerV2._is_muse_target is true for the Muse Spark architecture only (the tree's
  is_muse_spark_architecture on hf_config.architectures[0]);
- in DFlashWorkerV2.__init__ every Muse fast-path switch is gated by self._muse_target: the compact block embedding
  (_compact_block_embed) and the draft's fc sharding (draft_model.fc_shard_enabled), so a non-Muse target
  (``_muse_target`` False) keeps the existing paths.
The source is parsed, not imported, so the test runs without the GPU stack; when the module does import, the static
method is also called."""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from sglang.test.ci.ci_register import register_amd_ci

# Stand-in targets: the architecture each checkpoint names (DFlashWorkerV2._is_muse_target reads architectures[0]).
_ARCH = {
    "muse_spark": "MuseSparkForConditionalGeneration",
    "kimi_k2": "DeepseekV3ForCausalLM",
    "qwen3": "Qwen3ForCausalLM",
    "llama": "LlamaForCausalLM",
    "deepseek_v3": "DeepseekV3ForCausalLM",
    "qwen3_5": "Qwen3_5ForConditionalGeneration",
    "glm4_moe": "Glm4MoeForCausalLM",
}

register_amd_ci(est_time=5, suite="base-b-test-1-gpu-small-amd-mi35x")

GATED = ("_compact_block_embed", "fc_shard_enabled")


def _worker_class():
    spec = importlib.util.find_spec("sglang.srt.speculative.dflash_worker_v2")
    tree = ast.parse(Path(spec.origin).read_text())
    return next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "DFlashWorkerV2"
    )


def _method(cls, name):
    return next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name
    )


def _uses_muse_target(expr) -> bool:
    """``self._muse_target`` itself, or an ``and`` chain containing it (so False forces the whole value False)."""
    if isinstance(expr, ast.Attribute) and expr.attr == "_muse_target":
        return True
    if isinstance(expr, ast.BoolOp) and isinstance(expr.op, ast.And):
        return any(_uses_muse_target(v) for v in expr.values)
    return False


def test_every_muse_fast_path_is_gated_by_the_target():
    init = _method(_worker_class(), "__init__")
    found = {}
    for node in ast.walk(init):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Attribute) and target.attr in GATED:
                    found.setdefault(target.attr, []).append(node.value)
    assert set(found) == set(GATED), f"missing assignments: {set(GATED) - set(found)}"
    for name, values in found.items():
        for value in values:
            assert _uses_muse_target(value), (
                f"{name} is assigned without the self._muse_target gate"
            )


def test_muse_target_is_the_muse_spark_model_type():
    fn = _method(_worker_class(), "_is_muse_target")
    src = ast.unparse(fn)
    assert "is_muse_spark_architecture(" in src and "model_type" not in src
    try:
        from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
    except (
        Exception
    ):  # the GPU stack is not importable here; the parsed check above stands
        pytest.skip("dflash_worker_v2 needs the GPU stack to import")

    def config(model_type):
        # the gate reads the checkpoint's architecture (hf_config.architectures[0], is_muse_spark_architecture)
        return SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type=model_type, architectures=[_ARCH[model_type]]
            )
        )

    assert DFlashWorkerV2._is_muse_target(config("muse_spark"))
    for other in ("kimi_k2", "qwen3", "llama", "deepseek_v3", "qwen3_5", "glm4_moe"):
        assert not DFlashWorkerV2._is_muse_target(config(other))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
