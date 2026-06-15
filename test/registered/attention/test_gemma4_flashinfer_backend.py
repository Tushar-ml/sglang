"""Smoke test for FlashInfer attention backend on Gemma4 models.

Gemma4 has a mixed attention pattern: alternating sliding-window (SWA) and
full-attention layers, each potentially with different num_kv_heads. This test
verifies that --attention-backend flashinfer works end-to-end with Gemma4
(both 31B dense and 26B-A4B MoE variants).
"""

import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=120, stage="base-b", runner_config="2-gpu-large")

PROMPT = (
    "Question: Janet's ducks lay 16 eggs per day. She eats three for breakfast "
    "every morning and bakes muffins for her friends every day with four. She "
    "sells the remainder at the farmers' market daily for $2 per fresh duck "
    "egg. How much in dollars does she make every day at the farmers' market?\n"
    "Answer:"
)


class _GemmaFlashInferBase(CustomTestCase):
    model: str = ""
    tp_size: int = 2

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp-size",
                str(cls.tp_size),
                "--attention-backend",
                "flashinfer",
                "--dtype",
                "bfloat16",
                "--mem-fraction-static",
                "0.55",
                "--context-length",
                "2048",
                "--skip-server-warmup",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def _generate(self, max_tokens: int = 64):
        r = requests.post(
            self.base_url + "/v1/completions",
            json={
                "model": self.model,
                "prompt": PROMPT,
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "top_k": 1,
            },
            timeout=120,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["text"]

    def test_prefill_and_decode(self):
        """Ensure FlashInfer can run a full prefill + decode pass for Gemma4."""
        out = self._generate(max_tokens=32)
        self.assertIsInstance(out, str)
        self.assertGreater(len(out), 0)

    def test_cached_prefix(self):
        """Run twice with the same prompt to exercise the prefix-cache path."""
        out1 = self._generate(max_tokens=16)
        out2 = self._generate(max_tokens=16)
        self.assertEqual(out1, out2)


class TestGemma4_31B_FlashInfer(_GemmaFlashInferBase):
    model = "google/gemma-4-27b-it"
    tp_size = 2


class TestGemma4_26BA4B_FlashInfer(_GemmaFlashInferBase):
    model = "google/gemma-4-26B-A4B-it"
    tp_size = 2


if __name__ == "__main__":
    unittest.main()
