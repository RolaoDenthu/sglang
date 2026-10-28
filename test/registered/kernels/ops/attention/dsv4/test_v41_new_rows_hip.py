"""V41_NEW pools on gfx950: the layout is only selected with fp8 compressed caches
and aiter's mla_sparse_prefill_v41 and only serves V4.1's compress ratios, the c1 /
c2 compressed writes hold the same rows as in V41, the CSR and query-row kernels
match their references, and the HIP backend's mla_sparse_prefill_v41 attention
matches a torch softmax-with-sink over the dequantized rows, with the split count
deterministic inference pins. The DSpark draft that hands attention the rows of q
still feeds wo_b its activation dtype."""

import math
import sys
import types
import unittest
from unittest import mock

import torch

from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, stage="stage-b", runner_config="1-gpu-small-amd-mi35x")

HEAD_DIM, ROPE_DIM, FULL_PAGE, HEADS, SWA_WIDTH, TOPK = 512, 64, 256, 64, 128, 512


def _has_v41_prefill() -> bool:
    if not (is_hip() and is_gfx95_supported()):
        return False
    try:
        from aiter.ops.mla_sparse_prefill import mla_sparse_prefill_v41  # noqa: F401
    except ImportError:
        return False
    return True


def _freqs(max_pos: int = 8192) -> torch.Tensor:
    angles = torch.rand(max_pos, ROPE_DIM // 2, device="cuda") * 2 * math.pi
    polar = torch.polar(torch.ones_like(angles), angles)
    return torch.view_as_real(polar).flatten(-2)


def _dequant(values: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    groups = values.float().unflatten(-1, (-1, 32))
    return (groups * torch.exp2(scales.float() - 127)[..., None]).flatten(-2)


class TestV41NewLayoutSelection(CustomTestCase):
    def _select(self, option, aiter_module=None):
        from sglang.srt.environ import envs
        from sglang.srt.mem_cache import deepseek_v4_memory_pool as pool

        modules = {}
        if aiter_module is not None:
            modules["aiter.ops.mla_sparse_prefill"] = aiter_module
        with (
            envs.SGLANG_DSV4_KV_LAYOUT.override("v41_new"),
            envs.SGLANG_DSV4_COMPRESSED_KV_LAYOUT.override(option),
            mock.patch.object(pool, "_is_hip", True),
            mock.patch.object(pool, "is_gfx95_supported", return_value=True),
            mock.patch.object(
                pool, "resolve_hip_flashmla_backend", return_value="aiter_sparse"
            ),
            mock.patch.object(
                pool,
                "get_exec",
                return_value=types.SimpleNamespace(
                    kernel=types.SimpleNamespace(dsv4_attn_backend="flashmla")
                ),
            ),
            mock.patch.dict(sys.modules, modules),
        ):
            return pool.select_dsv4_kv_layout()

    def test_v41_new_keeps_every_compressed_cache_fp8(self):
        from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout

        with_kernel = types.ModuleType("aiter.ops.mla_sparse_prefill")
        with_kernel.mla_sparse_prefill_v41 = object()
        for option in ("auto", "fp8"):
            with self.subTest(option=option):
                self.assertEqual(
                    self._select(option, with_kernel), (KVLayout.V41_NEW, "fp8")
                )
        with self.assertRaisesRegex(ValueError, "fp8"):
            self._select("fp4", with_kernel)

    def test_v41_new_needs_the_aiter_kernel(self):
        without_kernel = types.ModuleType("aiter.ops.mla_sparse_prefill")
        with self.assertRaisesRegex(ValueError, "mla_sparse_prefill_v41"):
            self._select("auto", without_kernel)

    def test_v41_new_attention_rejects_other_ratios_at_startup(self):
        """A V4 model's ratios 4 / 128 fail when the backend starts, not at the
        first forward of one of their layers."""
        from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
        from sglang.srt.layers.attention import deepseek_v4_backend_hip_radix as hip

        def pool(layout, ratios):
            return types.SimpleNamespace(
                get_swa_key_layout=lambda: layout, present_ratios=ratios
            )

        for layout, ratios in [
            (KVLayout.V41_NEW, (1, 2)),
            # the draft pool, whose layers are all ratio 0
            (KVLayout.V41_NEW, ()),
            (KVLayout.V41, (4, 128)),
            (KVLayout.V4, (4, 128)),
        ]:
            with self.subTest(layout=layout, ratios=ratios):
                hip._check_v41_new_ratios(pool(layout, ratios))
        with self.assertRaisesRegex(ValueError, r"v41_new.*\[4, 128\]"):
            hip._check_v41_new_ratios(pool(KVLayout.V41_NEW, (1, 4, 128)))
        with mock.patch(
            "sglang.srt.layers.attention.dsv4.unified_kv_gate.is_unified_kv_triton",
            return_value=True,
        ):
            hip._check_v41_new_ratios(pool(KVLayout.V41_NEW, (4, 128)))


class TestV41NewRowsSplitCount(CustomTestCase):
    def _splits(self, num_tokens, deterministic):
        """num_kv_splits the backend hands mla_sparse_prefill_v41 for one step."""
        from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
        from sglang.srt.layers.attention import deepseek_v4_backend_hip_radix as hip
        from sglang.srt.layers.attention import hip_flash_mla

        layout = KVLayout.V41_NEW
        kernel = mock.Mock(
            return_value=torch.zeros(num_tokens, HEADS, HEAD_DIM, dtype=torch.bfloat16)
        )
        aiter_module = types.ModuleType("aiter.ops.mla_sparse_prefill")
        aiter_module.mla_sparse_prefill_v41 = kernel
        csr = (
            torch.zeros(0, dtype=torch.int32),
            torch.zeros(num_tokens + 1, dtype=torch.int32),
        )
        backend = types.SimpleNamespace(
            token_to_kv_pool=types.SimpleNamespace(
                swa_page_size=FULL_PAGE,
                get_swa_key_buffer_radix=lambda layer_id: torch.zeros(
                    1, layout.page_bytes(FULL_PAGE), dtype=torch.uint8
                ),
            ),
            softmax_scale=HEAD_DIM**-0.5,
            _v41_rows_csr=lambda *args: types.SimpleNamespace(rows=csr, no_rows=csr),
        )
        core = types.SimpleNamespace(swa_page_indices=None, swa_topk_lengths=None)
        q = torch.zeros(num_tokens, HEADS, layout.bytes_per_token, dtype=torch.uint8)
        exec_config = types.SimpleNamespace(
            deterministic=types.SimpleNamespace(
                enable_deterministic_inference=deterministic
            )
        )
        with (
            mock.patch.dict(
                sys.modules, {"aiter.ops.mla_sparse_prefill": aiter_module}
            ),
            mock.patch.object(hip_flash_mla, "get_exec", return_value=exec_config),
        ):
            attend = hip.DeepseekV4HipRadixBackend._forward_attention_v41_rows
            attend(backend, q, 0, 0, core, None, None)
        return kernel.call_args.kwargs["num_kv_splits"]

    def test_deterministic_inference_pins_the_split_count(self):
        """aiter's own count follows the step's token count, which would reduce one
        request in a different order in each batch it lands in."""
        for num_tokens in (1, 6, 300):
            with self.subTest(num_tokens=num_tokens):
                self.assertEqual(self._splits(num_tokens, deterministic=True), 4)
                self.assertIsNone(self._splits(num_tokens, deterministic=False))
        # prefill-sized steps run unsplit, as on the other gfx950 sparse kernels
        for deterministic in (False, True):
            with self.subTest(deterministic=deterministic):
                self.assertEqual(self._splits(1024, deterministic), 1)


class TestV41NewDSparkDraft(CustomTestCase):
    def test_reference_wo_a_keeps_the_activation_dtype(self):
        """With SGLANG_DSPARK_FAST_KERNEL=0 the unpadded draft still hands attention
        the uint8 V41_NEW rows of q; its torch wo_a must not take their dtype."""
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.models import deepseek_v4_dspark as dspark

        tokens, groups, rank = 3, 8, 16
        q_rows = torch.zeros(tokens, HEADS, 528, dtype=torch.uint8)
        o = torch.randn(tokens, HEADS, HEAD_DIM).to(torch.bfloat16)
        wo_a = torch.randn(groups * rank, HEADS * HEAD_DIM // groups)
        backend = mock.Mock()
        backend.forward.return_value = o.clone()

        attn = object.__new__(dspark.DSparkAttention)
        torch.nn.Module.__init__(attn)
        attn.n_local_heads, attn.head_dim = HEADS, HEAD_DIM
        attn.rope_head_dim = ROPE_DIM
        attn.n_local_groups, attn.o_lora_rank = groups, rank
        attn.q_head_norm, attn._use_fast_kernel, attn.alt_streams = False, False, None
        attn.attn = object()
        attn.wq_a = lambda x: (x, None)
        attn.q_norm = lambda x: x
        attn.wq_b = lambda x: (x.new_zeros(x.shape[0], HEADS * HEAD_DIM), None)
        attn.kv_proj_only = lambda x: x.new_zeros(x.shape[0], HEAD_DIM)
        attn._store_block_kv = lambda **kwargs: None
        attn._local_attn_sink = lambda: None
        # rotations by zero, so the inverse RoPE leaves o as attention returned it
        attn.freqs_cis = torch.ones(8, ROPE_DIM // 2, dtype=torch.complex64)
        attn.wo_a = types.SimpleNamespace(weight=wo_a.to(torch.bfloat16))
        attn.wo_b = lambda x, **kwargs: (x, None)
        with (
            mock.patch.object(dspark, "_is_hip", True),
            mock.patch.object(dspark, "_is_npu", False),
            mock.patch.object(
                dspark,
                "_hip",
                types.SimpleNamespace(rope_q_as_v41_rows=lambda *args: q_rows),
                create=True,
            ),
            mock.patch.object(
                dspark,
                "_resolve_dspark_pool",
                return_value=types.SimpleNamespace(uniform_fp8=False),
            ),
            mock.patch(
                "sglang.srt.model_executor.forward_context.get_attn_backend",
                return_value=backend,
            ),
        ):
            out = attn.forward(
                torch.zeros(tokens, dtype=torch.int64),
                torch.randn(tokens, 32).to(torch.bfloat16),
                types.SimpleNamespace(forward_mode=ForwardMode.DECODE),
            )
        self.assertIs(backend.forward.call_args.kwargs["q"], q_rows)
        want = torch.einsum(
            "bgd,grd->bgr",
            o.view(tokens, groups, -1).float(),
            wo_a.to(torch.bfloat16).view(groups, rank, -1).float(),
        )
        self.assertEqual(out.dtype, torch.bfloat16)
        torch.testing.assert_close(out, want.to(torch.bfloat16).reshape(tokens, -1))


@unittest.skipUnless(
    is_hip() and is_gfx95_supported(), "the V4.1 writers are gfx950 JIT"
)
class TestV41NewCompressedWrites(CustomTestCase):
    def _pool(self, layout, page_size, pages):
        return torch.zeros(
            pages, layout.page_bytes(page_size), device="cuda", dtype=torch.uint8
        )

    def test_c1_matches_block_split(self):
        from sglang.kernels.ops.attention.dsv4.dequant_k_cache import (
            dequantize_k_cache_paged_v41,
        )
        from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
        from sglang.kernels.ops.attention.dsv4.low_ratio_compress import (
            c1_decode_norm_rope_store,
        )

        torch.manual_seed(0)
        num_tokens, pages = 777, 8
        kv = torch.randn(num_tokens, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        weight = (1 + 0.1 * torch.randn(HEAD_DIM, device="cuda")).to(torch.bfloat16)
        positions = torch.randint(0, 8192, (num_tokens,), device="cuda").int()
        # slot 0 marks a padded graph row
        out_loc = torch.randperm(pages * FULL_PAGE - 1, device="cuda")[:num_tokens] + 1
        freqs = _freqs()
        rows, latents = [], []
        for layout in (KVLayout.V41, KVLayout.V41_NEW):
            cache = self._pool(layout, FULL_PAGE, pages)
            latents.append(
                c1_decode_norm_rope_store(
                    kv,
                    weight,
                    positions,
                    out_loc,
                    1e-6,
                    freqs,
                    cache,
                    page_size=FULL_PAGE,
                    layout=layout,
                )
            )
            rows.append(
                dequantize_k_cache_paged_v41(
                    cache, out_loc.int(), FULL_PAGE, layout=layout
                )
            )
        self.assertTrue(torch.equal(latents[0], latents[1]))
        self.assertTrue(torch.equal(rows[0], rows[1]))
        self.assertGreater(rows[0].abs().sum().item(), 0)

    def test_c2_prefill_matches_block_split(self):
        from sglang.kernels.ops.attention.dsv4.dequant_k_cache import (
            dequantize_k_cache_paged_v41,
        )
        from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
        from sglang.kernels.ops.attention.dsv4.low_ratio_compress import (
            c2_prefill_norm_rope_store,
        )

        torch.manual_seed(1)
        lens, ring_size, c2_page = [9, 6, 1, 33], 8, FULL_PAGE // 2
        num_tokens = sum(lens)
        kv = torch.randn(num_tokens, 2 * HEAD_DIM, device="cuda")
        weight = (1 + 0.1 * torch.randn(HEAD_DIM, device="cuda")).to(torch.bfloat16)
        req = torch.cat([torch.full((n,), r) for r, n in enumerate(lens)]).cuda()
        positions = torch.cat([torch.arange(n) for n in lens]).int().cuda()
        # each request's FULL slots start on an even base, so a pair shares a c2 slot
        raw_out_loc = torch.cat(
            [torch.arange(n) + 2 * FULL_PAGE * (r + 1) for r, n in enumerate(lens)]
        ).cuda()
        starts = torch.tensor([0, *torch.tensor(lens).cumsum(0)[:-1]]).int().cuda()
        seq_lens = torch.tensor(lens).int().cuda()
        written = raw_out_loc[positions % 2 == 1] // 2
        freqs = _freqs()
        rows, latents, states = [], [], []
        for layout in (KVLayout.V41, KVLayout.V41_NEW):
            cache = self._pool(layout, c2_page, 16)
            state = torch.zeros(
                len(lens) * ring_size + ring_size + 1, 2 * HEAD_DIM, device="cuda"
            )
            state[:, HEAD_DIM:] = float("-inf")
            latent = c2_prefill_norm_rope_store(
                kv,
                state,
                weight,
                positions,
                req,
                raw_out_loc,
                1e-6,
                freqs,
                cache,
                starts,
                seq_lens,
                page_size=c2_page,
                ring_size=ring_size,
                layout=layout,
            )
            latents.append(latent[positions % 2 == 1])
            states.append(state)
            rows.append(
                dequantize_k_cache_paged_v41(
                    cache, written.int(), c2_page, layout=layout
                )
            )
        self.assertTrue(torch.equal(latents[0], latents[1]))
        self.assertTrue(torch.equal(states[0], states[1]))
        self.assertTrue(torch.equal(rows[0], rows[1]))
        self.assertGreater(rows[0].abs().sum().item(), 0)


@unittest.skipUnless(is_hip() and is_gfx95_supported(), "gfx950 Triton kernels")
class TestV41RowsKernels(CustomTestCase):
    def test_q_to_v41_rows_matches_rope_then_quant(self):
        from sglang.kernels.ops.attention.dsv4.rope import (
            apply_rotary_emb_triton,
            set_batched_rope,
        )
        from sglang.kernels.ops.attention.dsv4.v41_q_rows_hip import (
            q_to_v41_rows,
            v41_rows_split,
        )
        from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import (
            mxfp8_e4m3_quantize,
        )

        set_batched_rope(True)
        torch.manual_seed(0)
        angles = torch.rand(8192, ROPE_DIM // 2, device="cuda") * 2 * math.pi
        freqs_cis = torch.polar(torch.ones_like(angles), angles)
        for num_tokens in (0, 1, 7, 300):
            for rope in (False, True):
                with self.subTest(num_tokens=num_tokens, rope=rope):
                    q = torch.randn(num_tokens, HEADS, HEAD_DIM, device="cuda") * 3
                    q = q.to(torch.bfloat16)
                    positions = torch.randint(0, 8192, (num_tokens,), device="cuda")
                    want = q.clone()
                    if rope:
                        apply_rotary_emb_triton(
                            want[..., -ROPE_DIM:], freqs_cis, positions=positions
                        )
                    want_v, want_s = mxfp8_e4m3_quantize(want.reshape(-1, HEAD_DIM))
                    args = (freqs_cis, positions) if rope else ()
                    got_v, got_s = v41_rows_split(q_to_v41_rows(q, *args))
                    self.assertTrue(
                        torch.equal(
                            got_v.reshape(-1, HEAD_DIM).view(torch.uint8),
                            want_v.view(torch.uint8),
                        )
                    )
                    self.assertTrue(torch.equal(got_s.reshape(want_s.shape), want_s))

    def test_rows_to_csr_matches_reference(self):
        from sglang.kernels.ops.attention.dsv4.opus_sparse_prefill_hip import (
            rows_to_csr,
        )

        torch.manual_seed(0)
        # (source rows, tokens, width, pool rows); 5000 tokens takes the cumsum branch
        for src, tokens, width, num_rows in [
            (24, 24, 128, 5000),
            (4, 20, 192, 1000),
            (30, 24, 512, 300),
            (5000, 5000, 128, 70000),
        ]:
            with self.subTest(src=src, tokens=tokens, width=width):
                idx = torch.randint(
                    -1, num_rows + 5, (src, width), device="cuda", dtype=torch.int32
                )
                lens = torch.randint(
                    -2, width + 3, (src,), device="cuda", dtype=torch.int32
                )
                (out, indptr), (no_idx, no_indptr) = rows_to_csr(
                    idx, lens, tokens, num_rows, with_empty=True
                )
                want, offsets = [], [0]
                for t in range(tokens):
                    row = idx[t, : max(int(lens[t]), 0)] if t < src else idx[0, :0]
                    row = row[(row >= 0) & (row < num_rows)]
                    want.append(row)
                    offsets.append(offsets[-1] + row.numel())
                self.assertEqual(indptr.tolist(), offsets)
                self.assertTrue(torch.equal(out[: offsets[-1]], torch.cat(want)))
                self.assertEqual(no_idx.numel(), 0)
                self.assertEqual(no_indptr.tolist(), [0] * (tokens + 1))


@unittest.skipUnless(
    _has_v41_prefill(), "needs gfx950 and aiter's mla_sparse_prefill_v41"
)
class TestV41NewRowsAttention(CustomTestCase):
    def _pool(self, page_size, pages, scale):
        from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
        from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import (
            mxfp8_e4m3_quantize,
        )

        layout = KVLayout.V41_NEW
        x = (torch.randn(pages * page_size, HEAD_DIM, device="cuda") * scale).to(
            torch.bfloat16
        )
        values, scales = mxfp8_e4m3_quantize(x)
        pool = torch.empty(
            pages, layout.page_bytes(page_size), device="cuda", dtype=torch.uint8
        )
        rows = pool.view(-1, layout.bytes_per_token)
        rows[:, : layout.data_bytes] = values.view(torch.uint8)
        rows[:, layout.data_bytes :] = scales
        return pool

    @staticmethod
    def _lists(num_tokens, width, num_rows, hole_rate):
        idx = torch.randint(0, num_rows, (num_tokens, width), device="cuda").int()
        idx[torch.rand(num_tokens, width, device="cuda") < hole_rate] = -1
        lens = torch.randint(0, width + 1, (num_tokens,), device="cuda").int()
        return idx, lens

    @staticmethod
    def _reference(q, sources, sink, scale):
        from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import (
            mxfp8_e4m3_quantize,
        )

        q_values, q_scales = mxfp8_e4m3_quantize(q.reshape(-1, HEAD_DIM))
        qf = _dequant(q_values, q_scales).view(q.shape)
        out = torch.zeros(q.shape, device="cuda")
        for t in range(q.shape[0]):
            keys = []
            for (values, scales), idx, lens in sources:
                sel = idx[t, : lens[t]]
                sel = sel[sel >= 0].long()
                keys.append(_dequant(values[sel], scales[sel]))
            k = torch.cat(keys)
            s = (qf[t] @ k.T) * scale
            m = torch.maximum(s.max(-1).values, sink) if k.shape[0] else sink
            p = torch.exp(s - m[:, None])
            out[t] = (p @ k) / (p.sum(-1) + torch.exp(sink - m))[:, None]
        return out

    def _run(self, num_tokens, compress_ratio):
        from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
        from sglang.srt.layers.attention import deepseek_v4_backend_hip_radix as hip

        torch.manual_seed(num_tokens * 3 + compress_ratio)
        c_page = FULL_PAGE // max(compress_ratio, 1)
        swa_pool = self._pool(FULL_PAGE, 8, 1.0)
        c_pool = self._pool(c_page, 8, 0.5) if compress_ratio else None
        swa_rows = hip._v41_new_rows(swa_pool, FULL_PAGE)
        swa_idx, swa_len = self._lists(
            num_tokens, SWA_WIDTH, swa_rows[0].shape[0], 0.05
        )
        sources, sparse = [(swa_rows, swa_idx, swa_len)], {}
        if compress_ratio:
            c_rows = hip._v41_new_rows(c_pool, c_page)
            c_idx, c_len = self._lists(num_tokens, TOPK, c_rows[0].shape[0], 0.1)
            sparse[compress_ratio] = (c_idx, c_len)
            sources.append((c_rows, c_idx, c_len))
        backend = types.SimpleNamespace(
            token_to_kv_pool=types.SimpleNamespace(
                swa_page_size=FULL_PAGE,
                get_swa_key_buffer_radix=lambda layer_id: swa_pool,
                get_extra_key_buffer=lambda layer_id: c_pool,
                get_extra_key_page_size=lambda layer_id: c_page,
                get_extra_key_layout=lambda layer_id: KVLayout.V41_NEW,
            ),
            forward_metadata=object(),
            softmax_scale=HEAD_DIM**-0.5,
            _v41_csr_cache={},
        )
        backend._v41_rows_csr = types.MethodType(
            hip.DeepseekV4HipRadixBackend._v41_rows_csr, backend
        )
        core = types.SimpleNamespace(
            swa_page_indices=swa_idx,
            swa_topk_lengths=swa_len,
            sparse_page_indices=lambda r: sparse[r][0],
            sparse_topk_lengths=lambda r: sparse[r][1],
        )
        q = (torch.randn(num_tokens, HEADS, HEAD_DIM, device="cuda") * 2).to(
            torch.bfloat16
        )
        sink = torch.randn(HEADS, device="cuda")
        attend = hip.DeepseekV4HipRadixBackend._forward_attention_v41_rows
        got = attend(backend, q, 0, compress_ratio, core, sink, None)
        want = self._reference(q, sources, sink, backend.softmax_scale)
        rel = ((got.float() - want).norm() / want.norm()).item()
        self.assertLess(rel, 1e-2)
        # a later layer of the same forward reuses the cached CSRs, and the model's
        # V41_NEW rows of q are what the backend quantizes the bf16 q into
        from sglang.kernels.ops.attention.dsv4.v41_q_rows_hip import q_to_v41_rows

        again = attend(backend, q_to_v41_rows(q), 1, compress_ratio, core, sink, None)
        self.assertTrue(torch.equal(again, got))

    def test_csrs_are_rebuilt_every_forward_of_one_graph_bucket(self):
        """The decode graph runner preps the bucket metadata once, then runs warmup
        and capture forwards on that same object; each must build its own CSRs so
        the capture records the build."""
        from sglang.srt.layers.attention import deepseek_v4_backend_hip_radix as hip

        backend = object.__new__(hip.DeepseekV4HipRadixBackend)
        backend.forward_metadata = object()
        backend.token_to_kv_pool = types.SimpleNamespace(request_window=None)
        backend.enable_deepseek_v4_fp4_indexer = False
        backend._fp4_c4_rowgroup_rows = lambda metadata, forward_batch: None
        forward_batch = types.SimpleNamespace(
            out_cache_loc=None,
            forward_mode=types.SimpleNamespace(
                is_decode=lambda: True, is_target_verify=lambda: False
            ),
        )
        idx, lens = self._lists(6, SWA_WIDTH, 1000, 0.0)
        built = []
        for _ in range(3):
            backend.init_forward_metadata_in_graph(forward_batch)
            first = backend._v41_rows_csr(0, idx, lens, 6, 1000)
            # later layers of the same forward share it
            self.assertIs(backend._v41_rows_csr(0, idx, lens, 6, 1000), first)
            built.append(first)
        self.assertEqual(len({id(entry) for entry in built}), 3)

    def test_matches_reference(self):
        for compress_ratio in (0, 1, 2):
            for num_tokens in (1, 6, 48, 384):
                with self.subTest(compress_ratio=compress_ratio, num_tokens=num_tokens):
                    self._run(num_tokens, compress_ratio)


if __name__ == "__main__":
    unittest.main()
