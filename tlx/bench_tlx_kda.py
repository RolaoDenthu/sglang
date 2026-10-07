"""Time the gfx950 TLX KDA kernels under whichever triton this interpreter imports.

run_bench_tlx_kda.sh runs it under both environments and prints the comparison.
By hand, run once per environment on the same idle GPU, then compare:

    python bench_tlx_kda.py run  --label utlx  --out utlx.json
    python bench_tlx_kda.py compare utlx.json fbtriton.json ...

Inputs follow test_kimi_k3_kda_tlx.py (K3 per-rank shape: 12 heads, dim 128,
split views of one packed qkv, fp32 state pool) with lower_bound=-5. Every
tensor argument is passed as the packed qkv so marker's input rotation keeps
the serving layout of the q/k/v views.
"""

import argparse
import importlib.util
import json
import pathlib
import sys

import torch

from sglang.kernels.jit.benchmark import marker

# MI355X keeps a 256 MB Infinity Cache behind L2 that L2_cache_size (4 MB) does
# not report; size marker's flush buffer and input rotation to cover it.
_reported_l2 = marker._get_l2_cache_size
marker._get_l2_cache_size = lambda: max(_reported_l2(), 256 << 20)

LB = -5.0
DECODE_BATCHES = (1, 4, 16, 32, 64, 128, 256)
# Total tokens per case stay within --chunked-prefill-size 16384.
PREFILL_CASES = {
    "1x512": [512],
    "1x1024": [1024],
    "1x2048": [2048],
    "1x4096": [4096],
    "1x8192": [8192],
    "1x16384": [16384],
    "4x4096": [4096] * 4,
    "2x8192": [8192, 8192],
    "16x1024": [1024] * 16,
}


TEST_FILE = (
    pathlib.Path(__file__).resolve().parents[1]
    / "test/registered/kernels/ops/attention/kda_tlx/test_kimi_k3_kda_tlx.py"
)


def load_test_module():
    spec = importlib.util.spec_from_file_location("kda_test", TEST_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def median_us(fn, args, graph):
    res = marker.do_bench(
        fn,
        input_args=args,
        use_cuda_graph=graph,
        estimated_time_ms=200.0,
        disable_log_bandwidth=True,
    )
    return res.times[0] * 1e6


def checksum(*tensors):
    return [float(x.double().sum()) for x in tensors] + [float(x.double().abs().sum()) for x in tensors]


def run(label, out_path):
    import triton

    from sglang.kernels.ops.kimi_k3 import tlx as tlx_kernels
    from sglang.kernels.ops.kimi_k3.tlx.kimi_k3_kda_prepare import prepare_kda_inputs

    t = load_test_module()
    H, D = t._HEADS, t._DIM
    scale = D**-0.5
    kernel = t.TlxKDAKernel()
    tlx_mod = tlx_kernels._tlx_module()
    results, checks = {}, {}

    for bs in DECODE_BATCHES:
        c = t._make_case([1] * bs, decode=True)

        def prep(qkv, a, b):
            q, k, _ = t._split_qkv(qkv)
            return prepare_kda_inputs(
                q, k, a, b, c.A_log, c.dt_bias, num_heads=H, head_dim=D,
                lower_bound=LB, sigmoid_beta=True,
            )

        q_n, k_n, g, beta = prep(c.qkv, c.a, c.b)

        def core(q_n, k_n, qkv, g, beta, pool):
            return tlx_kernels.kda_recurrent_decode(
                q_n, k_n, t._split_qkv(qkv)[2], g, beta, scale=scale, state_pool=pool,
                read_indices=c.cache_indices, write_indices=c.cache_indices,
                cu_seqlens=c.query_start_loc,
            )

        def total(qkv, a, b, pool):
            q, k, v = t._split_qkv(qkv)
            return kernel.decode(
                q, k, v, a, b, A_log=c.A_log, dt_bias=c.dt_bias, ssm_states=pool,
                cache_indices=c.cache_indices, query_start_loc=c.query_start_loc,
                lower_bound=LB,
            )

        key = f"decode bs={bs}"
        results[f"{key} | prepare"] = median_us(prep, (c.qkv, c.a, c.b), True)
        results[f"{key} | tlx core"] = median_us(core, (q_n, k_n, c.qkv, g, beta, c.pool.clone()), True)
        results[f"{key} | total"] = median_us(total, (c.qkv, c.a, c.b, c.pool.clone()), True)
        pool = c.pool.clone()
        out = total(c.qkv, c.a, c.b, pool)
        checks[key] = checksum(out, pool[c.cache_indices.long()])
        print(label, key, {k.split(" | ")[1]: round(v, 2) for k, v in results.items() if k.startswith(key + " |")}, flush=True)

    for name, lens in PREFILL_CASES.items():
        c = t._make_case(lens, decode=False, beta_is_raw=False)

        def prep(qkv, a, b):
            q, k, _ = t._split_qkv(qkv)
            return prepare_kda_inputs(
                q, k, a, b, c.A_log, c.dt_bias, num_heads=H, head_dim=D,
                lower_bound=LB, sigmoid_beta=False,
            )

        q_n, k_n, log_g, beta_act = prep(c.qkv, c.a, c.b)
        init = c.pool[c.cache_indices.long()].contiguous()

        def core(q_n, k_n, qkv, log_g, beta_act, init):
            return tlx_kernels.kda_paged_prefill(
                q_n, k_n, t._split_qkv(qkv)[2], log_g, beta_act, scale=scale,
                initial_state=init, cu_seqlens=c.query_start_loc,
            )

        def total(qkv, a, b, pool):
            q, k, v = t._split_qkv(qkv)
            return kernel.extend(
                q, k, v, a, b, ssm_states=pool, cache_indices=c.cache_indices,
                query_start_loc=c.query_start_loc, A_log=c.A_log, dt_bias=c.dt_bias,
                lower_bound=LB, beta_is_raw=False,
            )

        key = f"prefill {name}"
        for graph, mode in ((True, "graph"), (False, "eager")):
            results[f"{key} | prepare | {mode}"] = median_us(prep, (c.qkv, c.a, c.b), graph)
            results[f"{key} | tlx core | {mode}"] = median_us(
                core, (q_n, k_n, c.qkv, log_g, beta_act, init), graph
            )
            results[f"{key} | total | {mode}"] = median_us(total, (c.qkv, c.a, c.b, c.pool.clone()), graph)
        pool = c.pool.clone()
        out = total(c.qkv, c.a, c.b, pool)
        checks[key] = checksum(out, pool[c.cache_indices.long()])
        print(label, key, {k.split(" | ", 1)[1]: round(v, 1) for k, v in results.items() if k.startswith(key + " |")}, flush=True)
        del c, q_n, k_n, log_g, beta_act, init, pool, out
        torch.cuda.empty_cache()

    json.dump(
        {
            "label": label,
            "triton": triton.__version__,
            "triton_file": triton.__file__,
            "tlx_file": getattr(tlx_mod, "__file__", None),
            "gpu": torch.cuda.get_device_properties(0).name,
            "results": results,
            "checks": checks,
        },
        open(out_path, "w"),
        indent=1,
    )
    print("wrote", out_path)


def compare(paths):
    runs = [json.load(open(p)) for p in paths]
    base = runs[0]
    for r in runs:
        print(f"- {r['label']}: triton {r['triton']}, tlx from {r['tlx_file']}")
    print()

    worst = {}
    for r in runs[1:]:
        for key, ref in base["checks"].items():
            got = r["checks"][key]
            rel = max(abs(a - b) / (abs(b) + 1e-12) for a, b in zip(got, ref))
            worst[r["label"]] = max(worst.get(r["label"], 0.0), rel)
    print("max relative checksum difference vs", base["label"], worst, "\n")

    labels = [r["label"] for r in runs]
    print("| case | part | " + " | ".join(f"{x} (us)" for x in labels) + " | "
          + " | ".join(f"{x} / {base['label']}" for x in labels[1:]) + " |")
    print("|---" * (2 + 2 * len(labels) - 1) + "|")
    for key in base["results"]:
        parts = key.split(" | ")
        vals = [r["results"][key] for r in runs]
        ratios = [v / vals[0] for v in vals[1:]]
        print(f"| {parts[0]} | {' '.join(parts[1:])} | " + " | ".join(f"{v:.1f}" for v in vals) + " | "
              + " | ".join(f"{x:.2f}" for x in ratios) + " |")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--label", required=True)
    r.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("paths", nargs="+")
    a = p.parse_args()
    run(a.label, a.out) if a.cmd == "run" else compare(a.paths)
