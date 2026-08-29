import unittest
from unittest.mock import patch

from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDeepseekV4Fp4IndexerValidation(unittest.TestCase):
    def _server_args(self, **kwargs):
        return ServerArgs(
            model_path="dummy", enable_deepseek_v4_fp4_indexer=True, **kwargs
        )

    @patch.object(ServerArgs, "_handle_multimodal_feature_transport")
    @patch("sglang.srt.server_args.is_gfx95_supported", return_value=False)
    @patch("sglang.srt.server_args.is_hip", return_value=False)
    def test_cuda_sm100_and_sm120_are_accepted(
        self, _mock_is_hip, _mock_gfx95, _mock_multimodal
    ):
        for sm100, sm120 in ((True, False), (False, True)):
            with (
                self.subTest(sm100=sm100, sm120=sm120),
                patch(
                    "sglang.srt.server_args.is_sm100_supported",
                    return_value=sm100,
                ),
                patch(
                    "sglang.srt.server_args.is_sm120_supported",
                    return_value=sm120,
                ),
            ):
                self._server_args()._handle_environment_variables()

    @patch.object(ServerArgs, "_handle_multimodal_feature_transport")
    @patch("sglang.srt.server_args.is_gfx95_supported", return_value=False)
    @patch("sglang.srt.server_args.is_sm120_supported", return_value=False)
    @patch("sglang.srt.server_args.is_sm100_supported", return_value=False)
    @patch("sglang.srt.server_args.is_hip", return_value=False)
    def test_cuda_other_capabilities_are_rejected(
        self, _mock_is_hip, _mock_sm100, _mock_sm120, _mock_gfx95, _mock_multimodal
    ):
        with self.assertRaisesRegex(ValueError, "requires SM100 or SM120"):
            self._server_args()._handle_environment_variables()


if __name__ == "__main__":
    unittest.main()
