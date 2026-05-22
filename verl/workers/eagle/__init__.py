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

"""Eagle3 online draft model training support for verl.

Provides joint RL + draft distillation training so the Eagle3 speculative
decoding drafter stays synchronized with the evolving RL policy.

Typical usage in a custom trainer::

    from verl.workers.eagle import (
        EagleDraftConfig,
        EagleDraftModelWrapper,
        load_eagle_draft_model,
        get_capture_context,
        compute_eagle_draft_loss_with_alignment,
    )
"""

from verl.workers.eagle.config import EagleDraftConfig, EagleDraftOptimizerConfig
from verl.workers.eagle.manager import EagleDraftManager, EagleLossWrapper
from verl.workers.eagle.draft_model import (
    EagleDraftModelWrapper,
    get_draft_state_dict_for_vllm,
    load_eagle_draft_model,
    maybe_wrap_fsdp,
)
from verl.workers.eagle.hidden_capture import (
    CapturedStates,
    HiddenStateCapture,
    get_capture_context,
    get_eagle3_aux_layer_indices,
    roll_inputs_embeds,
)
from verl.workers.eagle.losses import (
    compute_eagle_draft_loss_with_alignment,
    eagle_draft_loss,
    roll_for_eagle_alignment,
)

__all__ = [
    # config
    "EagleDraftConfig",
    "EagleDraftOptimizerConfig",
    # manager / loss wrapper
    "EagleDraftManager",
    "EagleLossWrapper",
    "EagleDraftOptimizerConfig",
    # draft model
    "EagleDraftModelWrapper",
    "load_eagle_draft_model",
    "maybe_wrap_fsdp",
    "get_draft_state_dict_for_vllm",
    # hidden capture
    "CapturedStates",
    "HiddenStateCapture",
    "get_capture_context",
    "get_eagle3_aux_layer_indices",
    "roll_inputs_embeds",
    # losses
    "eagle_draft_loss",
    "roll_for_eagle_alignment",
    "compute_eagle_draft_loss_with_alignment",
]
