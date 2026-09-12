import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
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
        self.beam_group = None
        self.rid = "fake"
        self.kv = SimpleNamespace(
            mamba_cow_src_index=None,
            mamba_needs_clear=False,
            holds_mamba=False,
            mamba_pool_idx=None,
        )

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

    def preempt_to_schedule(self, _req):
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
        s.enable_unified_cache_external_linker = False
        s.tree_cache = MagicMock()
        s.enable_priority_preemption = False
        s.is_hybrid_swa = False
        s.chunked_req = None
        s.min_free_slots_delayer = None
        s.get_num_allocatable_reqs = lambda running_bs, beam_width=None, running_batch=None: 8
        s.policy = MagicMock()
        s.policy.calc_priority = MagicMock()
        s.dynamic_chunk_sizer = None
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
        s.is_mixed_chunk = True
        s.req_to_token_pool = SimpleNamespace(mamba_allocator=None, available_size=lambda: 64)
        s.tp_worker = SimpleNamespace(
            model_runner=SimpleNamespace(attn_backend=SimpleNamespace())
        )
        s._get_decode_first_stride = lambda: draft_stride
        return s

    def _run_prefill(self, scheduler, running_batch):
        with (
            patch("sglang.srt.managers.scheduler.PrefillAdder", _CaptureAdder),
            patch(
                "sglang.srt.managers.scheduler.get_memory",
                return_value=SimpleNamespace(enable_flexkv=False),
            ),
            patch(
                "sglang.srt.managers.scheduler.get_schedule",
                return_value=SimpleNamespace(prefill_max_requests=None),
            ),
            patch("sglang.srt.managers.scheduler.TEST_RETRACT", False),
        ):
            return Scheduler._get_new_batch_prefill_raw(
                scheduler,
                prefill_delayer_single_pass=None,
                running_batch=running_batch,
            )

    def test_decode_first_uses_residual_budget_for_prefill_adder(self):
        scheduler = self._build_scheduler(waiting_reqs=[_FakeReq()])
        running_batch = _FakeBatch([_FakeReq() for _ in range(4)], return_logprob=False)

        batch, _ = self._run_prefill(scheduler, running_batch)

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

        batch, same_running = self._run_prefill(scheduler, running_batch)

        self.assertIsNone(batch)
        self.assertIs(same_running, running_batch)
        self.assertIsNone(_CaptureAdder.last_init)

    def test_decode_first_does_not_subtract_decode_budget_when_running_empty(self):
        scheduler = self._build_scheduler(waiting_reqs=[_FakeReq()])
        running_batch = _FakeBatch([], return_logprob=False)

        batch, _ = self._run_prefill(scheduler, running_batch)

        self.assertIsNone(batch)
        args, _kwargs = _CaptureAdder.last_init
        rem_input_tokens = args[5]
        rem_chunk_tokens = args[6]
        num_mixed_decode_tokens = args[7]
        self.assertEqual(rem_input_tokens, 100)
        self.assertEqual(num_mixed_decode_tokens, 0)
        self.assertEqual(rem_chunk_tokens, 90)

    def test_mix_decode_spec_never_allocates_below_live_seq_len(self):
        req = SimpleNamespace(
            kv=SimpleNamespace(kv_allocated_len=64, kv_committed_len=70),
            decode_batch_idx=0,
        )
        batch = SimpleNamespace(
            sampling_info=SimpleNamespace(
                penalizer_orchestrator=SimpleNamespace(is_required=False)
            ),
            model_config=SimpleNamespace(is_encoder_decoder=False),
            token_to_kv_pool_allocator=SimpleNamespace(page_size=64),
            reqs=[req],
            seq_lens_cpu=torch.tensor([70], dtype=torch.int32),
            seq_lens=torch.tensor([70], dtype=torch.int64),
            orig_seq_lens=torch.tensor([70], dtype=torch.int64),
            device=torch.device("cpu"),
            tree_cache=SimpleNamespace(page_size=64),
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.arange(0, 512, dtype=torch.int64).view(1, 512)
            ),
            req_pool_indices=torch.tensor([0], dtype=torch.int64),
            req_pool_indices_cpu=torch.tensor([0], dtype=torch.int32),
            hisparse_coordinator=None,
            spec_info=object(),
            enable_overlap=False,
        )

        with (
            patch(
                "sglang.srt.managers.schedule_batch.alloc_for_spec_decode"
            ) as alloc_mock,
            patch(
                "sglang.srt.managers.schedule_batch.mamba_extra_buffer_enabled",
                return_value=False,
            ),
        ):

            def _fake_alloc(*_args, **kwargs):
                for r, nxt in zip(kwargs["reqs"], kwargs["nxt_kv_lens_cpu"].tolist()):
                    r.kv.kv_allocated_len = max(r.kv.kv_allocated_len, int(nxt))

            alloc_mock.side_effect = _fake_alloc
            ScheduleBatch._prepare_for_mix_decode_reusing_spec_kv(batch)

        _, kwargs = alloc_mock.call_args
        # Start from live seq_lens clock (70), not stale kv_allocated_len=64.
        self.assertEqual(int(kwargs["cur_kv_lens_cpu"][0].item()), 70)
        self.assertEqual(int(kwargs["nxt_kv_lens_cpu"][0].item()), 128)
        self.assertEqual(req.kv.kv_allocated_len, 128)
        self.assertEqual(req.kv.kv_committed_len, 71)


if __name__ == "__main__":
    unittest.main()
