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

"""Eagle3 draft model manager: coordinates hidden-state capture, draft forward pass,
distillation loss, and draft optimizer updates alongside the verl training loop.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
from tensordict import TensorDict

from verl.workers.eagle.config import EagleDraftConfig
from verl.workers.eagle.draft_model import EagleDraftModelWrapper, get_draft_state_dict_for_vllm
from verl.workers.eagle.hidden_capture import HiddenStateCapture, roll_inputs_embeds
from verl.workers.eagle.losses import compute_eagle_draft_loss_with_alignment

logger = logging.getLogger(__name__)


class EagleDraftManager:
    """Owns the Eagle3 draft model, its optimizer, and the hidden-state capture hooks.

    Lifecycle::

        manager = EagleDraftManager.build(policy_model, eagle_config)

        # Before each training step, activate persistent capture hooks
        # (hooks stay registered; they fire on every policy forward)

        # After each training step (engine.train_batch completes):
        manager.optimizer_step()

    The draft model's parameters receive gradients via the combined loss returned
    by ``EagleLossWrapper``.  The draft optimizer is stepped separately to avoid
    interfering with the policy FSDP optimizer.
    """

    def __init__(
        self,
        draft_model: EagleDraftModelWrapper,
        capture: HiddenStateCapture,
        config: EagleDraftConfig,
        optimizer: torch.optim.Optimizer,
    ):
        self.draft_model = draft_model
        self.capture = capture
        self.config = config
        self.optimizer = optimizer

        # Register persistent hooks once; they remain active for the lifetime
        # of the manager.  Captured state is cleared in EagleLossWrapper after
        # each call so stale data is never reused.
        self.capture._register_hooks()
        logger.info("Eagle3 draft hooks registered on policy model.")

    @classmethod
    def build(
        cls,
        policy_model: nn.Module,
        eagle_config: EagleDraftConfig,
        torch_dtype: torch.dtype = torch.bfloat16,
        device_id: int = 0,
    ) -> "EagleDraftManager":
        """Factory method: load draft model, set up hooks, and create optimizer."""
        from verl.workers.eagle.draft_model import load_eagle_draft_model, maybe_wrap_fsdp

        draft_model = load_eagle_draft_model(
            model_path=eagle_config.model_path,
            torch_dtype=torch_dtype,
        )
        draft_model = maybe_wrap_fsdp(draft_model, device_id=device_id, use_fsdp=False)

        aux_layer_indices = (
            tuple(eagle_config.aux_layer_indices)
            if eagle_config.aux_layer_indices is not None
            else None
        )
        capture = HiddenStateCapture(
            model=policy_model,
            aux_layer_indices=aux_layer_indices,
            capture_logits=True,
        )

        # Build draft optimizer
        optimizer = cls._build_optimizer(draft_model, eagle_config)

        return cls(
            draft_model=draft_model,
            capture=capture,
            config=eagle_config,
            optimizer=optimizer,
        )

    @staticmethod
    def _build_optimizer(
        draft_model: nn.Module,
        config: EagleDraftConfig,
    ) -> torch.optim.Optimizer:
        lr = config.optimizer.lr if config.optimizer.lr is not None else 1e-4
        wd = config.optimizer.weight_decay if config.optimizer.weight_decay is not None else 0.0
        params = [p for p in draft_model.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)

    def make_loss_wrapper(
        self,
        base_loss_fn: Callable,
    ) -> "EagleLossWrapper":
        """Wrap a base policy loss function with Eagle draft distillation."""
        return EagleLossWrapper(base_loss_fn=base_loss_fn, manager=self)

    def optimizer_step(self) -> None:
        """Step and zero the draft model optimizer (called after engine.train_batch)."""
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.draft_model.parameters() if p.requires_grad],
            max_norm=1.0,
        )
        self.optimizer.step()
        self.optimizer.zero_grad()

    def state_dict_for_vllm(self) -> Dict[str, torch.Tensor]:
        """Return draft model weights suitable for loading into vLLM."""
        return get_draft_state_dict_for_vllm(self.draft_model)

    def save_pretrained(self, path: str) -> None:
        """Save draft model in HuggingFace format for checkpointing."""
        inner = self.draft_model
        if hasattr(inner, "module"):
            inner = inner.module
        if hasattr(inner, "model"):
            inner = inner.model
        if hasattr(inner, "save_pretrained"):
            inner.save_pretrained(path)
        else:
            torch.save(inner.state_dict(), path)
        logger.info(f"Eagle3 draft model saved to {path}")


class EagleLossWrapper:
    """Wraps a base policy-loss callable with Eagle3 draft distillation.

    This is the callable passed to ``TrainingWorker.set_loss_fn()``.  When
    invoked by the FSDP engine inside ``forward_step``, it:

    1. Reads the hidden states and LM head logits captured by the persistent hooks
       during the policy's forward pass (the hooks fired moments earlier in the
       same stack frame).
    2. Runs the Eagle3 draft model forward.
    3. Computes soft-target cross-entropy distillation loss.
    4. Returns ``L_total = L_policy + λ * L_draft``.

    The draft model's parameters accumulate gradients via autograd; the
    corresponding optimizer is stepped in ``EagleDraftManager.optimizer_step()``
    after the engine completes ``train_batch``.
    """

    def __init__(
        self,
        base_loss_fn: Callable,
        manager: EagleDraftManager,
    ):
        self.base_loss_fn = base_loss_fn
        self.manager = manager

    def __call__(
        self,
        model_output: dict,
        data: TensorDict,
        dp_group=None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, dict]:
        # ------------------------------------------------------------------ #
        # 1. Standard policy gradient loss                                    #
        # ------------------------------------------------------------------ #
        policy_loss, metrics = self.base_loss_fn(
            model_output=model_output, data=data, dp_group=dp_group, **kwargs
        )

        # ------------------------------------------------------------------ #
        # 2. Collect captured states from the just-completed policy forward   #
        # ------------------------------------------------------------------ #
        captured = self.manager.capture.get_captured_states()
        # Clear immediately so stale state is never reused for the next micro-batch
        self.manager.capture._captured.clear()

        if (
            captured.hidden_states is None
            or captured.inputs_embeds is None
            or captured.lm_head_logits is None
        ):
            logger.debug("Eagle: missing captured states; skipping draft loss this step.")
            return policy_loss, metrics

        # ------------------------------------------------------------------ #
        # 3. Roll embeddings for Eagle3 time-step alignment                  #
        # ------------------------------------------------------------------ #
        rolled_embeds = roll_inputs_embeds(captured.inputs_embeds)

        # ------------------------------------------------------------------ #
        # 4. Draft model forward pass                                         #
        # ------------------------------------------------------------------ #
        # Build attention mask from the response mask if available
        response_mask = data.get("response_mask", None)
        if response_mask is not None and hasattr(response_mask, "to_padded_tensor"):
            response_mask_t: Optional[torch.Tensor] = response_mask.to_padded_tensor().bool()
        elif isinstance(response_mask, torch.Tensor):
            response_mask_t = response_mask.bool()
        else:
            response_mask_t = None

        draft_logits: torch.Tensor = self.manager.draft_model(
            hidden_states=captured.hidden_states,
            inputs_embeds=rolled_embeds,
            attention_mask=response_mask_t,
        )

        # ------------------------------------------------------------------ #
        # 5. Draft distillation loss (with Eagle3 time-step alignment)        #
        # ------------------------------------------------------------------ #
        teacher_logits = captured.lm_head_logits  # already detached + float32

        if response_mask_t is None:
            # Fall back: treat all tokens as valid
            response_mask_t = torch.ones(
                draft_logits.shape[:2], dtype=torch.bool, device=draft_logits.device
            )

        draft_loss = compute_eagle_draft_loss_with_alignment(
            draft_logits=draft_logits.float(),
            teacher_logits=teacher_logits,
            response_mask=response_mask_t,
            loss_weight=self.manager.config.loss_weight,
        )

        # ------------------------------------------------------------------ #
        # 6. Combine losses                                                   #
        # ------------------------------------------------------------------ #
        total_loss = policy_loss + draft_loss
        metrics["actor/draft_loss"] = draft_loss.detach().item()

        return total_loss, metrics
