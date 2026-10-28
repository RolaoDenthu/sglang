"""V41_NEW rows on gfx950: the CSR and query-row kernels match their references."""

import math
import unittest

import torch

from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, stage="stage-b", runner_config="1-gpu-small-amd-mi35x")

HEAD_DIM, ROPE_DIM, FULL_PAGE, HEADS, SWA_WIDTH, TOPK = 512, 64, 256, 64, 128, 512


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


if __name__ == "__main__":
    unittest.main()
