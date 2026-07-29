import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.managers.schedule_policy import AddReqResult
from sglang.srt.managers.scheduler import Scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeReq:
    def __init__(self, *, return_logprob: bool = False, input_embeds=None):
        self.return_logprob = return_logprob
        self.input_embeds = input_embeds
        self.lora_id = None

    def init_next_round_input(self, _tree_cache):
        return None


class _FakeBatch:
    def __init__(self, reqs, *, return_logprob: bool = False):
        self.reqs = list(reqs)
        self.return_logprob = return_logprob
        self.batch_is_full = False

    def is_empty(self):
        return len(self.reqs) == 0


class _CaptureAdder:
    last_init = None

    def __init__(self, *args, **kwargs):
        _CaptureAdder.last_init = (args, kwargs)
        self.can_run_list = []
        self.preempt_list = []
        self.new_chunked_req = None

    def add_one_req(self, _req, **_kwargs):
        return AddReqResult.CONTINUE

    def add_chunked_req(self, req):
        return req

    def preempt_to_schedule(self, _req, _server_args):
        return False


class TestDecodeFirstScheduling(CustomTestCase):
    def setUp(self):
        _CaptureAdder.last_init = None

    def _build_scheduler(
        self,
        *,
        waiting_reqs,
        max_prefill_tokens=100,
        chunked_prefill_size=90,
        draft_stride=5,
    ):
        s = Scheduler.__new__(Scheduler)
        s.grammar_manager = SimpleNamespace(
            has_waiting_grammars=lambda: False, get_ready_grammar_requests=lambda: []
        )
        s._add_request_to_queue = MagicMock()
        s.enable_hierarchical_cache = False
        s.server_args = SimpleNamespace(
            enable_flexkv=False,
            prefill_max_requests=None,
            speculative_num_draft_tokens=draft_stride,
        )
        s.tree_cache = MagicMock()
        s.enable_priority_preemption = False
        s.is_hybrid_swa = False
        s.chunked_req = None
        s.min_free_slots_delayer = None
        s.get_num_allocatable_reqs = lambda _running_bs: 8
        s.policy = MagicMock()
        s.policy.calc_priority = MagicMock()
        s.enable_dynamic_chunking = False
        s.chunked_prefill_size = chunked_prefill_size
        s.page_size = 1
        s.token_to_kv_pool_allocator = MagicMock()
        s.new_token_ratio_tracker = SimpleNamespace(current=1.0)
        s.max_prefill_tokens = max_prefill_tokens
        s.priority_scheduling_preemption_threshold = 0
        s.max_prefill_bs = 0
        s.max_running_requests = 32
        s.dllm_config = None
        s.waiting_queue = list(waiting_reqs)
        s.enable_lora = False
        s.disaggregation_mode = None
        s.enable_hicache_storage = False
        s.truncation_align_size = 0
        s.enable_decode_first_schedule = True
        s.is_mixed_chunk = False
        return s

    def test_decode_first_uses_residual_budget_for_prefill_adder(self):
        scheduler = self._build_scheduler(waiting_reqs=[_FakeReq()])
        running_batch = _FakeBatch([_FakeReq() for _ in range(4)], return_logprob=False)

        with patch(
            "sglang.srt.managers.scheduler.PrefillAdder",
            _CaptureAdder,
        ):
            batch, _ = Scheduler._get_new_batch_prefill_raw(
                scheduler,
                prefill_delayer_single_pass=None,
                running_batch=running_batch,
            )

        self.assertIsNone(batch)
        args, _kwargs = _CaptureAdder.last_init
        rem_input_tokens = args[5]
        rem_chunk_tokens = args[6]
        num_mixed_decode_tokens = args[7]
        self.assertEqual(rem_input_tokens, 100)
        self.assertEqual(num_mixed_decode_tokens, 20)  # 4 reqs * draft stride 5
        self.assertEqual(rem_chunk_tokens, 80)  # min(90, 100 - 20)

    def test_decode_first_skips_prefill_when_running_is_logprob(self):
        scheduler = self._build_scheduler(waiting_reqs=[_FakeReq()])
        running_batch = _FakeBatch([_FakeReq()], return_logprob=True)

        with patch(
            "sglang.srt.managers.scheduler.PrefillAdder",
            _CaptureAdder,
        ):
            batch, same_running = Scheduler._get_new_batch_prefill_raw(
                scheduler,
                prefill_delayer_single_pass=None,
                running_batch=running_batch,
            )

        self.assertIsNone(batch)
        self.assertIs(same_running, running_batch)
        self.assertIsNone(_CaptureAdder.last_init)

    def test_decode_first_does_not_subtract_decode_budget_when_running_empty(self):
        scheduler = self._build_scheduler(waiting_reqs=[_FakeReq()])
        running_batch = _FakeBatch([], return_logprob=False)

        with patch(
            "sglang.srt.managers.scheduler.PrefillAdder",
            _CaptureAdder,
        ):
            batch, _ = Scheduler._get_new_batch_prefill_raw(
                scheduler,
                prefill_delayer_single_pass=None,
                running_batch=running_batch,
            )

        self.assertIsNone(batch)
        args, _kwargs = _CaptureAdder.last_init
        rem_input_tokens = args[5]
        rem_chunk_tokens = args[6]
        num_mixed_decode_tokens = args[7]
        self.assertEqual(rem_input_tokens, 100)
        self.assertEqual(num_mixed_decode_tokens, 0)
        self.assertEqual(rem_chunk_tokens, 90)


if __name__ == "__main__":
    unittest.main()
