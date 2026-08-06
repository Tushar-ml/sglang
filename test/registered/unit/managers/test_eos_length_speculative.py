"""Regression: speculative multi-token commits that both hit max_new_tokens
and contain an EOS/stop token (e.g. GLM user-role token 154827) must finish as
a stop at the EOS, not as FINISH_LENGTH with the EOS left in the emitted
prefix. Voice/tool requests set skip_special_tokens=False, so a leaked EOS
shows up as a literal role marker in streamed content.
"""

import unittest
from array import array

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

EOS_ID = 154827  # GLM user-role EOS token
AFTER_EOS_ID = 44785  # e.g. "assume" hallucinated user continuation


class _FakeTokenizer:
    eos_token_id = -1
    additional_stop_token_ids = None

    def decode(self, ids):
        return "".join(f"t{int(i)}" for i in ids)


def _make_req(output_ids, *, max_new_tokens, eos_token_ids=None, stop_token_ids=None):
    sp = SamplingParams(max_new_tokens=max_new_tokens, stop_token_ids=stop_token_ids)
    sp.normalize(tokenizer=_FakeTokenizer())
    req = Req(
        rid="eos-len-spec",
        origin_input_text="",
        origin_input_ids=array("q", [0]),
        sampling_params=sp,
        eos_token_ids=set(eos_token_ids or []),
        vocab_size=200_000,
    )
    req.tokenizer = _FakeTokenizer()
    req.output_ids = array("q", output_ids)
    return req


class TestEosLengthSpeculative(unittest.TestCase):
    def test_eos_midchunk_wins_over_length(self):
        # 90 tokens already emitted; spec accepts 8 more including EOS then junk.
        # max_new_tokens=96 so the chunk also crosses the length cap.
        prefix = list(range(100, 190))  # 90 tokens
        chunk = [190, 191, EOS_ID, AFTER_EOS_ID, 192, 193, 194, 195]  # 8 tokens
        req = _make_req(
            prefix + chunk, max_new_tokens=96, eos_token_ids={EOS_ID}
        )
        req.update_finish_state(new_accepted_len=8)
        self.assertTrue(req.finished())
        self.assertEqual(req.finished_reason.matched, EOS_ID)
        # EOS is at index 92; include it, drop post-EOS junk and length overshoot.
        self.assertEqual(req.finished_len, 93)
        self.assertEqual(list(req.output_ids_through_stop)[-1], EOS_ID)
        self.assertNotIn(AFTER_EOS_ID, list(req.output_ids_through_stop))

    def test_length_still_wins_when_no_eos_in_visible_prefix(self):
        prefix = list(range(100, 190))
        chunk = list(range(190, 198))
        req = _make_req(prefix + chunk, max_new_tokens=96, eos_token_ids={EOS_ID})
        req.update_finish_state(new_accepted_len=8)
        self.assertTrue(req.finished())
        self.assertIsNone(getattr(req.finished_reason, "matched", None))
        self.assertEqual(req.finished_len, 96)

    def test_eos_after_max_new_tokens_is_ignored(self):
        # EOS only appears past the length cap — client must not see past-cap tokens.
        prefix = list(range(100, 190))  # 90
        chunk = [190, 191, 192, 193, 194, 195, EOS_ID, AFTER_EOS_ID]
        # indices 90..97; max=96 means visible is 90..95 (no EOS)
        req = _make_req(
            prefix + chunk, max_new_tokens=96, eos_token_ids={EOS_ID}
        )
        req.update_finish_state(new_accepted_len=8)
        self.assertTrue(req.finished())
        self.assertEqual(req.finished_len, 96)
        self.assertNotIn(EOS_ID, list(req.output_ids_through_stop))

    def test_stop_token_id_midchunk_wins_over_length(self):
        prefix = list(range(100, 190))
        stop_id = 17
        chunk = [190, stop_id, 191, 192, 193, 194, 195, 196]
        req = _make_req(
            prefix + chunk,
            max_new_tokens=96,
            stop_token_ids=[stop_id],
        )
        req.update_finish_state(new_accepted_len=8)
        self.assertTrue(req.finished())
        self.assertEqual(req.finished_reason.matched, stop_id)
        self.assertEqual(req.finished_len, 92)


if __name__ == "__main__":
    unittest.main()
