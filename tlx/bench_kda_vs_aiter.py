"""Kimi-K3 KDA kernels on gfx950: sglang Triton vs TLX vs aiter, one timing method.

run_bench_kda_vs_aiter.sh runs all three steps. By hand, on the same idle GPU:

    python bench_kda_vs_aiter.py sglang     --out OUT   # /opt/venv with uTLX
    UTLX_NO_AUTOREGISTER=1 \
    python bench_kda_vs_aiter.py aiter-main --out OUT --aiter-main /path/to/aiter-main
    python bench_kda_vs_aiter.py report     --out OUT   # -> OUT/compare.md

  sglang      TritonKDAKernel and TlxKDAKernel. Saves the Triton outputs and final
              states as the reference for the aiter side.
  aiter-main  aiter's Gluon fused_recurrent_kda (decode), FlashKDA and Gluon
              chunk_kda (prefill) from an upstream aiter checkout. Runs in its own
              process because the installed aiter and the checkout share the
              package name; only aiter's Triton/Gluon modules are imported, not
              aiter/__init__.py (which loads the installed C++ extensions).

K3 per-rank shape (TP8): 12 heads x 128, q/k/v as views of one packed qkv, raw
gate, A_log / dt_bias / lower_bound=-5, fp32 V-major state pool updated in place.
Every provider gets the same inputs (fixed seed), is called the way serving calls
it, and is timed with sglang's marker.do_bench (median; flush sized for the 256 MB
Infinity Cache). Decode runs in a CUDA graph (as served), prefill eagerly.
"""

import argparse
import importlib
import json
import os
import pathlib
import subprocess
import sys
import types

import torch
import torch.nn.functional as F

from sglang.kernels.jit.benchmark import marker

# MI355X keeps a 256 MB Infinity Cache behind L2 that L2_cache_size (4 MB) does
# not report; size marker's flush buffer and input rotation to cover it.
_reported_l2 = marker._get_l2_cache_size
marker._get_l2_cache_size = lambda: max(_reported_l2(), 256 << 20)

H, D, LB = 12, 128, -5.0
W = H * D
SCALE = D**-0.5
DECODE_BATCHES = (1, 4, 16, 32, 64, 128, 256)
# Total tokens per case stay within --chunked-prefill-size 16384.
PREFILL_CASES = {
    "1x512": [512],
    "1x1024": [1024],
    "1x2048": [2048],
    "1x4096": [4096],
    "1x8192": [8192],
    "1x16384": [16384],
    "2x8192": [8192] * 2,
    "4x4096": [4096] * 4,
    "16x1024": [1024] * 16,
}
TEST_FILE = (
    pathlib.Path(__file__).resolve().parents[1]
    / "test/registered/kernels/ops/attention/kda_tlx/test_kimi_k3_kda_tlx.py"
)


def median_us(fn, args, graph):
    res = marker.do_bench(
        fn, input_args=args, use_cuda_graph=graph, estimated_time_ms=200.0,
        disable_log_bandwidth=True,
    )
    return res.times[0] * 1e6


def make_case(lens, decode, seed=20261006):
    """Prefill: gate [1, T, H, D]. Decode: flat gate [T, H*D] (forward_decode). Beta raw bf16."""
    gen = torch.Generator(device="cuda").manual_seed(seed + sum(lens) + (7 if decode else 0))
    T = sum(lens)

    def r(*shape, dtype=torch.bfloat16, scale=1.0):
        return scale * torch.randn(shape, dtype=dtype, device="cuda", generator=gen)

    slots = len(lens) + 3
    idx = (torch.randperm(slots - 1, device="cuda", generator=gen)[: len(lens)] + 1).int()
    return dict(
        qkv=r(T, 3 * W),
        a=r(T, W) if decode else r(1, T, H, D),
        b=r(1, T, H),
        A_log=-4.0 + r(H, dtype=torch.float32, scale=0.1),
        dt_bias=r(W, dtype=torch.float32, scale=0.1),
        pool=r(slots, H, D, D, dtype=torch.float32, scale=0.02),
        idx=idx,
        qsl=F.pad(torch.tensor(lens, device="cuda").cumsum(0), (1, 0)).int(),
    )


def split_qkv(qkv):
    return tuple(x.unflatten(-1, (H, D)).unsqueeze(0) for x in qkv.split(W, dim=-1))


def rel_err(ref, x):
    d = x.detach().float().cpu() - ref.float()
    return float(d.square().mean().sqrt() / (ref.float().square().mean().sqrt() + 1e-8))


def import_aiter_main(root):
    """kda, chunk_kda and chunk_delta_attn modules from the aiter checkout at root."""
    assert "aiter" not in sys.modules, "the installed aiter is already imported in this process"
    for name, rel in (("aiter", "aiter"), ("aiter.ops", "aiter/ops"), ("aiter.ops.triton", "aiter/ops/triton")):
        pkg = types.ModuleType(name)
        pkg.__path__ = [f"{root}/{rel}"]
        sys.modules[name] = pkg
    return tuple(
        importlib.import_module(f"aiter.ops.triton.{m}")
        for m in ("attention.kda", "attention.chunk_kda", "kimi_delta_attn.chunk_delta_attn")
    )


def run_sglang(out_dir):
    import importlib.util

    import triton

    spec = importlib.util.spec_from_file_location("kda_test", TEST_FILE)
    t = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(t)
    assert t._gfx950_tlx_available() and t.tlx_kernels.is_prefill_available(), "needs gfx950 + uTLX"

    kernels = {"triton": t.TritonKDAKernel(), "tlx": t.TlxKDAKernel()}
    results, errors, refs = {}, {}, {}

    for bs in DECODE_BATCHES:
        c = make_case([1] * bs, decode=True)
        key = f"decode bs={bs}"
        outs, states = {}, {}
        for name, kernel in kernels.items():
            def step(qkv, a, b, pool, kernel=kernel):
                q, k, v = split_qkv(qkv)
                return kernel.decode(
                    q, k, v, a, b, A_log=c["A_log"], dt_bias=c["dt_bias"], ssm_states=pool,
                    cache_indices=c["idx"], query_start_loc=c["qsl"], lower_bound=LB,
                )

            results[f"{key} | {name}"] = median_us(step, (c["qkv"], c["a"], c["b"], c["pool"].clone()), True)
            pool = c["pool"].clone()
            outs[name] = step(c["qkv"].clone(), c["a"], c["b"], pool).cpu()
            states[name] = pool[c["idx"].long()].cpu()
        errors[f"{key} | tlx"] = (rel_err(outs["triton"], outs["tlx"]), rel_err(states["triton"], states["tlx"]))
        refs[key] = {"out": outs["triton"], "state": states["triton"]}
        print(key, {k.split(" | ")[1]: round(v, 1) for k, v in results.items() if k.startswith(key + " |")}, flush=True)

    for label, lens in PREFILL_CASES.items():
        c = make_case(lens, decode=False)
        beta = c["b"].float().sigmoid()  # kimi_k3 forward_extend pre-sigmoids beta
        key = f"prefill {label}"
        outs, states = {}, {}
        for name, kernel in kernels.items():
            def step(qkv, a, b, pool, kernel=kernel):
                q, k, v = split_qkv(qkv)
                return kernel.extend(
                    q, k, v, a, b, ssm_states=pool, cache_indices=c["idx"],
                    query_start_loc=c["qsl"], A_log=c["A_log"], dt_bias=c["dt_bias"],
                    lower_bound=LB, beta_is_raw=False,
                )

            results[f"{key} | {name}"] = median_us(step, (c["qkv"], c["a"], beta, c["pool"].clone()), False)
            pool = c["pool"].clone()
            outs[name] = step(c["qkv"].clone(), c["a"], beta, pool).cpu()
            states[name] = pool[c["idx"].long()].cpu()
        errors[f"{key} | tlx"] = (rel_err(outs["triton"], outs["tlx"]), rel_err(states["triton"], states["tlx"]))
        refs[key] = {"out": outs["triton"], "state": states["triton"]}
        print(key, {k.split(" | ")[1]: round(v, 1) for k, v in results.items() if k.startswith(key + " |")}, flush=True)
        del c, outs, states
        torch.cuda.empty_cache()

    torch.save(refs, pathlib.Path(out_dir) / "refs.pt")
    return {"triton": triton.__version__, "results": results, "errors": errors}


def run_aiter_main(out_dir, root):
    import triton

    kda, ck, cda = import_aiter_main(root)
    refs = torch.load(pathlib.Path(out_dir) / "refs.pt")
    results, errors = {}, {}

    for bs in DECODE_BATCHES:
        c = make_case([1] * bs, decode=True)
        key = f"decode bs={bs}"
        cu = c["qsl"].long()
        idx2 = c["idx"].view(bs, 1)
        out = torch.empty(1, bs, H, D, dtype=torch.bfloat16, device="cuda")

        def step(qkv, a, b, pool):
            q, k, v = split_qkv(qkv)
            kda.fused_recurrent_kda(
                q, k, v, a.view(1, bs, H, D), b, A_log=c["A_log"], dt_bias=c["dt_bias"],
                initial_state=pool, inplace_final_state=True, state_v_first=True,
                cu_seqlens=cu, ssm_state_indices=idx2, use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True, lower_bound=LB,
                scale=SCALE, out=out,
            )
            return out

        name = "aiter_gluon_decode"
        results[f"{key} | {name}"] = median_us(step, (c["qkv"], c["a"], c["b"], c["pool"].clone()), True)
        pool = c["pool"].clone()
        o = step(c["qkv"], c["a"], c["b"], pool)
        errors[f"{key} | {name}"] = (rel_err(refs[key]["out"], o), rel_err(refs[key]["state"], pool[c["idx"].long()]))
        print(key, round(results[f"{key} | {name}"], 1), flush=True)

    for label, lens in PREFILL_CASES.items():
        c = make_case(lens, decode=False)
        key = f"prefill {label}"
        T = sum(lens)
        has = torch.ones(len(lens), dtype=torch.bool, device="cuda")
        out = torch.empty(1, T, H, D, dtype=torch.bfloat16, device="cuda")
        cu = c["qsl"].long()
        # once per batch in serving (host sync), shared by all KDA layers
        ci, co = ck.prepare_chunk_kda_metadata(c["qsl"])

        def flash_kda(qkv, a, b, pool):
            q, k, v = split_qkv(qkv)
            cda.chunk_kimi_delta_attn(
                q, k, v, a, b, A_log=c["A_log"], dt_bias=c["dt_bias"], scale=SCALE,
                use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True, safe_gate=True, lower_bound=LB,
                state_v_first=True, cu_seqlens=cu, out=out, state_cache=pool,
                state_indices=c["idx"], has_initial_state=has,
            )
            return out

        def chunk_kda(qkv, a, b, pool):
            q, k, v = split_qkv(qkv)
            ck.chunk_kda(
                q, k, v, a, b, c["A_log"], c["dt_bias"], LB, c["qsl"], chunk_indices=ci,
                chunk_offsets=co, scale=SCALE, out=out, state_cache=pool,
                state_indices=c["idx"], has_initial_state=has,
            )
            return out

        for name, fn in (("aiter_flash_kda", flash_kda), ("aiter_chunk_kda", chunk_kda)):
            results[f"{key} | {name}"] = median_us(fn, (c["qkv"], c["a"], c["b"], c["pool"].clone()), False)
            pool = c["pool"].clone()
            o = fn(c["qkv"], c["a"], c["b"], pool)
            errors[f"{key} | {name}"] = (rel_err(refs[key]["out"], o), rel_err(refs[key]["state"], pool[c["idx"].long()]))
        print(key, {k.split(" | ")[1]: round(v, 1) for k, v in results.items() if k.startswith(key + " |")}, flush=True)
        del c, out
        torch.cuda.empty_cache()

    commit = subprocess.run(["git", "-C", root, "log", "-1", "--format=%h %cd", "--date=short"],
                            capture_output=True, text=True).stdout.strip()
    return {"triton": triton.__version__, "aiter": f"{root} @ {commit}", "results": results, "errors": errors}


def report(out_dir):
    a = json.load(open(pathlib.Path(out_dir) / "sglang.json"))
    b = json.load(open(pathlib.Path(out_dir) / "aiter_main.json"))
    res, err = {**a["results"], **b["results"]}, {**a["errors"], **b["errors"]}
    lines = [
        f"- {a['gpu']}; triton {a['triton']} + uTLX; aiter {b['aiter']}",
        "- us per KDA layer call (marker.do_bench median)",
        "",
        "## Decode (CUDA graph)",
        "",
        "| batch | Triton | TLX | aiter Gluon decode |",
        "|---|---|---|---|",
    ]
    for bs in DECODE_BATCHES:
        g = lambda k: res[f"decode bs={bs} | {k}"]
        lines.append(f"| {bs} | {g('triton'):.1f} | {g('tlx'):.1f} | {g('aiter_gluon_decode'):.1f} |")
    lines += [
        "",
        "## Prefill (eager)",
        "",
        "| case | Triton | TLX | aiter FlashKDA | aiter chunk_kda |",
        "|---|---|---|---|---|",
    ]
    for label in PREFILL_CASES:
        g = lambda k: res[f"prefill {label} | {k}"]
        lines.append(f"| {label} | {g('triton'):.1f} | {g('tlx'):.1f} | {g('aiter_flash_kda'):.1f} "
                     f"| {g('aiter_chunk_kda'):.1f} |")
    lines += ["", "## Max relative error vs sglang Triton over all cases (output / final state)", "",
              "| kernel | error |", "|---|---|"]
    for phase, names in (("decode", ["tlx", "aiter_gluon_decode"]),
                         ("prefill", ["tlx", "aiter_flash_kda", "aiter_chunk_kda"])):
        for n in names:
            vals = [v for k, v in err.items() if k.startswith(phase) and k.endswith(f"| {n}")]
            lines.append(f"| {phase} {n} | {max(v[0] for v in vals):.1e} / {max(v[1] for v in vals):.1e} |")
    text = "\n".join(lines) + "\n"
    (pathlib.Path(out_dir) / "compare.md").write_text(text)
    print(text)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["sglang", "aiter-main", "report"])
    p.add_argument("--out", required=True)
    p.add_argument("--aiter-main", default=os.environ.get("AITER_MAIN", "/sgl-workspace/aiter-main"))
    args = p.parse_args()
    pathlib.Path(args.out).mkdir(parents=True, exist_ok=True)
    if args.mode == "report":
        report(args.out)
    else:
        torch.cuda.set_device(0)
        data = run_sglang(args.out) if args.mode == "sglang" else run_aiter_main(args.out, args.aiter_main)
        data["gpu"] = torch.cuda.get_device_properties(0).name
        name = "sglang.json" if args.mode == "sglang" else "aiter_main.json"
        json.dump(data, open(pathlib.Path(args.out) / name, "w"), indent=1)
        print("wrote", pathlib.Path(args.out) / name)
