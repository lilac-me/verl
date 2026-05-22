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

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EagleDraftOptimizerConfig:
    """Optimizer settings for the Eagle draft model (separate from policy optimizer)."""

    # If None, draft model shares the policy optimizer's lr/wd
    lr: Optional[float] = None
    weight_decay: Optional[float] = None


@dataclass
class EagleDraftConfig:
    """Configuration for Eagle3 online draft model training.

    Enables joint training of an Eagle3 draft model alongside the RL policy.
    The draft model learns to imitate the policy's token distribution via
    soft cross-entropy distillation, keeping the speculative decoding drafter
    synchronized as the policy evolves during RL training.

    Requires:
      - vLLM rollout with speculative_config.method = "eagle3"
      - Draft model checkpoint compatible with the policy architecture
      - Pipeline parallelism NOT supported (pp_size must be 1)

    Example YAML config::

        actor_rollout_ref:
          model:
            eagle_draft:
              enabled: true
              model_path: "AngelSlim/Qwen3-1.7B_eagle3"
              loss_weight: 0.1
          rollout:
            speculative_method: eagle3
            num_speculative_tokens: 3
    """

    # Master switch
    enabled: bool = False

    # HuggingFace repo or local path to the Eagle3 draft checkpoint.
    # Must be loaded with trust_remote_code=True.
    model_path: Optional[str] = None

    # Weight of draft distillation loss relative to policy loss.
    # L_total = L_policy + loss_weight * L_draft
    loss_weight: float = 0.1

    # Indices of policy decoder layers whose output hidden states are captured
    # and concatenated as input to the draft model.
    # None → auto-select (layer 1, mid-1, last-4) matching Eagle3 defaults.
    aux_layer_indices: Optional[List[int]] = None

    # Optional separate optimizer settings for the draft model.
    # If not set, draft params share the same lr/wd as the policy.
    optimizer: EagleDraftOptimizerConfig = field(default_factory=EagleDraftOptimizerConfig)

    def __post_init__(self):
        if self.enabled and self.model_path is None:
            raise ValueError("EagleDraftConfig.model_path must be set when enabled=True")
        if self.loss_weight < 0:
            raise ValueError(f"EagleDraftConfig.loss_weight must be >= 0, got {self.loss_weight}")
