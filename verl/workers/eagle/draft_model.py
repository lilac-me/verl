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

"""Eagle3 draft model wrapper and weight utilities.

The draft model is a HuggingFace EAGLE3Model.  During RL training it lives
on the same device as the policy model and is updated by a separate optimizer
after each training step.

Weight export for vLLM follows the HuggingFace state-dict convention so that
the draft-model parameters can be loaded directly into vLLM's Eagle3 proposer
without any key renaming.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterator, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class EagleDraftModelWrapper(nn.Module):
    """Thin wrapper around a HuggingFace Eagle3 draft model.

    Handles the draft-model forward pass: given the policy's captured hidden
    states and the rolled input embeddings (Eagle3 time-step alignment), it
    returns draft logits over the vocabulary.

    Args:
        model: The underlying HuggingFace EAGLE3 model instance.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the Eagle3 draft model forward pass.

        Args:
            hidden_states:  Concatenated aux-layer hidden states from the policy.
                            Shape [batch, seq, N_aux * hidden_size].
            inputs_embeds:  Token embeddings shifted left by one (Eagle3 time-step
                            alignment: position t sees embedding of token t+1).
                            Shape [batch, seq, hidden_size].
            attention_mask: Optional bool mask (True = valid token).

        Returns:
            draft_logits: Shape [batch, seq, vocab_size].
        """
        outputs = self.model(
            hidden_states=hidden_states,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        # EAGLE3Model returns logits directly or as outputs.logits
        if isinstance(outputs, torch.Tensor):
            return outputs
        return outputs.logits


def load_eagle_draft_model(
    model_path: str,
    torch_dtype: torch.dtype = torch.bfloat16,
    device: Optional[torch.device] = None,
) -> EagleDraftModelWrapper:
    """Load an Eagle3 draft model from a HuggingFace checkpoint.

    The draft model is placed on *device* (defaults to CUDA device 0 if not
    specified) and set to training mode so that gradients flow during
    distillation.
    """
    from transformers import AutoModelForCausalLM

    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())

    logger.info(f"Loading Eagle3 draft model from {model_path}")

    # Eagle3 draft models are registered under their own AutoModel class in
    # recent transformers, but AutoModelForCausalLM works as a fallback.
    try:
        from transformers import AutoModel

        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )

    model = model.to(device)
    model.train()

    return EagleDraftModelWrapper(model)


def get_draft_state_dict_for_vllm(
    draft_model: EagleDraftModelWrapper,
) -> Iterator[Tuple[str, torch.Tensor]]:
    """Yield (name, tensor) pairs from the draft model for loading into vLLM.

    Parameters are yielded on CPU in float32 to minimise VRAM pressure during
    the synchronisation step.
    """
    inner = draft_model.model
    for name, param in inner.state_dict().items():
        yield name, param.detach().cpu().float()
