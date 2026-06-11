# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
"""EAGLE3 online draft-model training for verl (Megatron backend).

Reference: NVIDIA-NeMo/RL PR #2078 (nemo_rl/models/megatron/draft/*), re-implemented
without the nvidia-modelopt dependency so it runs on both CUDA and Ascend NPU
(torch_npu) stacks.

Components:
  - EagleDraftModel:      self-contained mcore-compatible EAGLE3 draft module
  - HiddenStateCapture:   forward hooks capturing aux hidden states + input embeds
  - draft loss utilities: forward-KL (soft CE) against detached policy logits
  - weight utilities:     HF <-> trainer load/export, d2t handling, lm_head init
"""

from .config import DraftModelConfig, validate_draft_config
from .eagle import EagleDraftModel
from .hidden_capture import HiddenStateCapture, get_eagle3_aux_hidden_state_layers
from .loss import (
    DraftLossState,
    draft_soft_ce_loss,
    gather_teacher_logits_for_draft_vocab,
    roll_left_seq,
)
from .weight_utils import (
    export_eagle_weights_to_hf,
    init_draft_lm_head_from_policy,
    load_hf_weights_to_eagle,
)

__all__ = [
    "DraftModelConfig",
    "validate_draft_config",
    "EagleDraftModel",
    "HiddenStateCapture",
    "get_eagle3_aux_hidden_state_layers",
    "DraftLossState",
    "draft_soft_ce_loss",
    "gather_teacher_logits_for_draft_vocab",
    "roll_left_seq",
    "load_hf_weights_to_eagle",
    "export_eagle_weights_to_hf",
    "init_draft_lm_head_from_policy",
]
