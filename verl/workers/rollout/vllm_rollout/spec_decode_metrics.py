# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Eagle3 speculative-decoding acceptance metrics for the vLLM rollout.

A custom vLLM v1 ``StatLogger`` accumulates per-iteration spec-decode counters
(emitted by the engine in ``scheduler_stats.spec_decoding_stats``) into a
process-local accumulator. The accumulator lives in the same process as the
``AsyncLLM`` frontend (i.e. the ``vLLMHttpServer`` Ray actor), so the server can
read+reset it and surface acceptance into the trainer's per-step metrics.

acceptance_rate       = num_accepted_tokens / num_draft_tokens
mean_acceptance_length = 1 + num_accepted_tokens / num_drafts   (incl. bonus token)
"""

from __future__ import annotations

import threading

# Process-local accumulator (summed across all local engine indices).
_LOCK = threading.Lock()
_ACCUM = {"num_drafts": 0, "num_draft_tokens": 0, "num_accepted_tokens": 0}


def pop_spec_decode_accumulator() -> dict[str, int]:
    """Return the accumulated spec-decode counts and reset them."""
    with _LOCK:
        out = dict(_ACCUM)
        _ACCUM["num_drafts"] = 0
        _ACCUM["num_draft_tokens"] = 0
        _ACCUM["num_accepted_tokens"] = 0
    return out


def _observe(stats) -> None:
    if stats is None:
        return
    with _LOCK:
        _ACCUM["num_drafts"] += int(getattr(stats, "num_drafts", 0) or 0)
        _ACCUM["num_draft_tokens"] += int(getattr(stats, "num_draft_tokens", 0) or 0)
        _ACCUM["num_accepted_tokens"] += int(getattr(stats, "num_accepted_tokens", 0) or 0)


def make_spec_decode_stat_logger():
    """Return a vLLM v1 StatLogger factory that feeds the accumulator.

    Returns None if the vLLM StatLogger base class is unavailable (so callers can
    skip wiring it).
    """
    try:
        from vllm.v1.metrics.loggers import StatLoggerBase
    except Exception:
        return None

    class SpecDecodeStatLogger(StatLoggerBase):
        def __init__(self, vllm_config, engine_index: int = 0):
            self.engine_index = engine_index

        def record(self, scheduler_stats, iteration_stats, mm_cache_stats=None, engine_idx: int = 0):
            if scheduler_stats is not None:
                _observe(getattr(scheduler_stats, "spec_decoding_stats", None))

        def log_engine_initialized(self):
            pass

        def log(self):
            pass

    return SpecDecodeStatLogger
