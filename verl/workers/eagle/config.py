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

"""Eagle3 online draft-model training configuration.

The Eagle3 draft model is a lightweight transformer that runs alongside the
policy during RL training.  It is trained via distillation from the policy's
LM-head logits so that the draft's token distribution tracks the evolving
policy, keeping the acceptance rate high during speculative decoding rollout.

Example config (YAML):
    actor_rollout_ref:
      model:
        eagle_draft:
          enabled: true
          model_path: /path/to/eagle3-draft
          loss_weight: 0.1         # lambda that scales L_draft relative to L_policy
          aux_layer_indices: null  # null → auto (1, num_layers//2-1, num_layers-4)
          optimizer:
            lr: 1.0e-4
            weight_decay: 0.0
"""

from dataclasses import dataclass, field
from typing import List, Optional

from verl.base_config import BaseConfig


@dataclass
class EagleDraftOptimizerConfig(BaseConfig):
    lr: float = 1e-4
    weight_decay: float = 0.0


@dataclass
class EagleDraftConfig(BaseConfig):
    """Configuration for Eagle3 online draft-model training.

    Attributes:
        enabled:           Whether to enable Eagle3 draft training.
        model_path:        Path to the pretrained Eagle3 draft model checkpoint.
        loss_weight:       Scaling factor λ applied to L_draft before adding to L_policy.
        aux_layer_indices: Indices of policy layers whose hidden states are fed to the
                           draft model.  null → auto-select using the Eagle3 heuristic
                           (1, num_layers//2-1, num_layers-4).
        optimizer:         AdamW settings for the standalone draft-model optimizer.
    """

    enabled: bool = False

    # Path to a pretrained HuggingFace Eagle3 checkpoint.
    # If None, the draft model is built from the policy's own components
    # (feature-fusion fc + shallow copy of policy layers + frozen lm_head).
    model_path: Optional[str] = None

    # ---- "build from policy" options (only used when model_path is None) ----
    # Number of transformer decoder layers in the built draft model.
    num_draft_layers: int = 1

    # Scaling factor λ: total_loss = policy_loss + λ * draft_loss
    loss_weight: float = 0.1
    # null → auto Eagle3 heuristic (1, num_layers//2-1, num_layers-4)
    aux_layer_indices: Optional[List[int]] = None
    optimizer: EagleDraftOptimizerConfig = field(default_factory=EagleDraftOptimizerConfig)
