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

"""Eagle3 draft-model manager and loss wrapper.

EagleDraftManager
-----------------
Owns the Eagle3 draft model (Megatron-native), its optimizer, and the
hidden-state capture hooks.  Built once per training process and kept alive
for the full training run.

EagleLossWrapper
----------------
A callable that wraps the base policy-loss function.  Injected as the
``loss_fn`` of the Megatron TrainingWorker, invoked once per micro-batch:

    total_loss, metrics = loss_wrapper(model_output, data, dp_group)

After each optimizer step the manager calls ``sync_lm_head`` to copy the
current policy LM-head shard into the draft output layer.  This keeps the
draft's logit space aligned with the policy as the policy evolves.

Tensor unpacking
----------------
Megatron (thd): 2-D ``[total_tokens_padded, feat]`` — sequences are padded
to a multiple of ``TP × CP × 2`` tokens.  Offsets are recomputed from
Megatron's parallel state to correctly slice each sequence.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Iterator, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from tensordict import TensorDict

from verl.workers.eagle.config import EagleDraftConfig
from verl.workers.eagle.draft_model import EagleDraftModel
from verl.workers.eagle.draft_utils import (
    copy_policy_lm_head,
    gather_vocab_parallel_logits,
    get_draft_state_dict_for_vllm,
    load_eagle_draft_model,
)
from verl.workers.eagle.hidden_capture import HiddenStateCapture
from verl.workers.utils.losses import eagle_draft_loss

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tensor unpack helpers
# ---------------------------------------------------------------------------

def _unpack_megatron_thd(
    packed: Optional[torch.Tensor],
    seq_lens: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Unpack a Megatron thd-format [total_tokens_padded, *feat] → [batch, max_seq, *feat]."""
    if packed is None:
        return None

    from megatron.core import parallel_state as mpu
    tp = mpu.get_tensor_model_parallel_world_size()
    cp = mpu.get_context_parallel_world_size()
    align = max(tp * cp * 2 if cp > 1 else tp, 1)

    batch = seq_lens.shape[0]
    max_seq = int(seq_lens.max().item())
    feat = packed.shape[1:]
    out = packed.new_zeros(batch, max_seq, *feat)

    offset = 0
    for i, length in enumerate(seq_lens.tolist()):
        length = int(length)
        pad = (align - length % align) % align
        out[i, :length] = packed[offset : offset + length]
        offset += length + pad

    return out


# ---------------------------------------------------------------------------
# EagleDraftManager
# ---------------------------------------------------------------------------

class EagleDraftManager:
    """Owns the Eagle3 draft model, optimizer, and hidden-state capture hooks.

    Lifecycle::

        manager = EagleDraftManager.build(policy_model, eagle_config)
        # hooks fire on every policy forward pass

        manager.optimizer_step()   # after engine.train_batch() completes
        manager.sync_lm_head()     # after policy optimizer step, before next rollout
        manager.state_dict_for_vllm()  # after policy weights synced to vLLM
    """

    def __init__(
        self,
        draft_model: EagleDraftModel,
        capture: HiddenStateCapture,
        config: EagleDraftConfig,
        optimizer: torch.optim.Optimizer,
        policy_model: nn.Module,
    ):
        self.config = config
        self.capture = capture
        self.draft_model = draft_model
        self.optimizer = optimizer
        self._policy_model = policy_model  # for lm_head extraction

        self.capture.register_hooks()
        logger.info("Eagle3 draft hooks registered on policy model.")

    @classmethod
    def build(
        cls,
        policy_model: nn.Module,
        eagle_config: EagleDraftConfig,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> "EagleDraftManager":
        """Load a pretrained Eagle3 checkpoint and register hooks on the policy.

        Args:
            policy_model: The (unwrapped) Megatron policy model.
            eagle_config: EagleDraftConfig with at least ``model_path`` set.
            torch_dtype: Dtype for draft model parameters.
            device: Target device; defaults to current CUDA device.
        """
        if eagle_config.model_path is None:
            raise ValueError(
                "EagleDraftConfig.model_path must be set. "
                "Provide the path to a pretrained HuggingFace Eagle3 checkpoint."
            )

        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())

        # Extract policy lm_head shard for initial sync into draft output layer.
        policy_lm_head_weight = cls._extract_policy_lm_head(policy_model)

        draft_model = load_eagle_draft_model(
            eagle_config=eagle_config,
            policy_lm_head_weight=policy_lm_head_weight,
            device=device,
            torch_dtype=torch_dtype,
        )

        aux_layer_indices = (
            tuple(eagle_config.aux_layer_indices)
            if eagle_config.aux_layer_indices is not None
            else None
        )
        capture = HiddenStateCapture(model=policy_model, aux_layer_indices=aux_layer_indices)
        optimizer = cls._build_optimizer(draft_model, eagle_config)

        return cls(
            draft_model=draft_model,
            capture=capture,
            config=eagle_config,
            optimizer=optimizer,
            policy_model=policy_model,
        )

    @staticmethod
    def _extract_policy_lm_head(policy_model: nn.Module) -> torch.Tensor:
        """Return the local TP shard of the policy output layer weight."""
        from megatron.training.utils import unwrap_model
        unwrapped_model = unwrap_model(policy_model)
        if getattr(unwrapped_model, "share_embeddings_and_output_weights", False):
            return unwrapped_model.shared_embedding_or_output_weight().detach()
        return unwrapped_model.output_layer.weight.detach()

    @staticmethod
    def _build_optimizer(
        draft_model: EagleDraftModel,
        config: EagleDraftConfig,
    ) -> torch.optim.Optimizer:
        # Output layer is frozen; only train the rest of the draft model.
        params = [p for p in draft_model.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=config.optimizer.lr, weight_decay=config.optimizer.weight_decay)

    def make_loss_wrapper(self, base_loss_fn: Callable) -> "EagleLossWrapper":
        """Wrap a base policy-loss function with Eagle3 draft distillation."""
        return EagleLossWrapper(base_loss_fn=base_loss_fn, manager=self)

    def sync_lm_head(self) -> None:
        """Copy the current policy LM-head shard into the draft output layer.

        Call this after each policy optimizer step so the draft's token-space
        stays aligned with the evolving policy.  No inter-rank communication
        is needed — both tensors live on the same TP rank.
        """
        lm_head_weight = self._extract_policy_lm_head(self._policy_model)
        copy_policy_lm_head(self.draft_model, lm_head_weight)

    def optimizer_step(self) -> None:
        """All-reduce draft gradients across DP ranks, clip, and step."""
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            from megatron.core import parallel_state as mpu
            dp_group = mpu.get_data_parallel_group()
            dp_world_size = mpu.get_data_parallel_world_size()
            for p in self.draft_model.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=dp_group)
                    p.grad.div_(dp_world_size)

        torch.nn.utils.clip_grad_norm_(
            [p for p in self.draft_model.parameters() if p.requires_grad],
            max_norm=1.0,
        )
        self.optimizer.step()
        self.optimizer.zero_grad()

    def state_dict_for_vllm(self) -> Iterator[Tuple[str, torch.Tensor]]:
        """Yield (HF name, cpu_float32_tensor) pairs for loading into vLLM."""
        return get_draft_state_dict_for_vllm(self.draft_model)

    def save_pretrained(self, path: str) -> None:
        """Export draft weights to a directory in HF format for checkpointing."""
        import os
        os.makedirs(path, exist_ok=True)
        hf_items = list(get_draft_state_dict_for_vllm(self.draft_model))
        state_dict = {k: v for k, v in hf_items}
        try:
            from safetensors.torch import save_file
            save_file(state_dict, os.path.join(path, "model.safetensors"))
        except ImportError:
            torch.save(state_dict, os.path.join(path, "pytorch_model.bin"))
        logger.info(f"Eagle3 draft model saved to {path}")


# ---------------------------------------------------------------------------
# EagleLossWrapper
# ---------------------------------------------------------------------------

class EagleLossWrapper:
    """Wraps a policy-loss callable with Eagle3 draft distillation.

    Called once per micro-batch inside the engine's forward-backward loop::

        total_loss, metrics = wrapper(model_output, data, dp_group)
    """

    def __init__(self, base_loss_fn: Callable, manager: EagleDraftManager):
        self.base_loss_fn = base_loss_fn
        self.manager = manager

    def __call__(
        self,
        model_output: dict,
        data: TensorDict,
        dp_group=None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, dict]:
        # 1. Standard policy loss
        policy_loss, metrics = self.base_loss_fn(
            model_output=model_output, data=data, dp_group=dp_group, **kwargs
        )

        # 2. Collect captured states
        captured = self.manager.capture.get_captured_states()
        self.manager.capture._captured.clear()

        if captured.hidden_states is None or captured.inputs_embeds is None:
            logger.debug("Eagle3: missing captured states; skipping draft loss this step.")
            return policy_loss, metrics

        teacher_logits = model_output.get("logits", None)
        if teacher_logits is None:
            logger.debug("Eagle3: logits not found in model_output; skipping draft loss.")
            return policy_loss, metrics
        teacher_logits = teacher_logits.detach().float()

        # 3. Unpack Megatron thd packed tensors
        input_ids = data.get("input_ids", None)
        if (
            input_ids is not None
            and isinstance(input_ids, torch.Tensor)
            and input_ids.is_nested
        ):
            seq_lens = input_ids.offsets().diff()
            hidden_states = _unpack_megatron_thd(captured.hidden_states, seq_lens)
            inputs_embeds = _unpack_megatron_thd(captured.inputs_embeds, seq_lens)
            teacher_logits = _unpack_megatron_thd(teacher_logits, seq_lens)
        else:
            hidden_states = captured.hidden_states
            inputs_embeds = captured.inputs_embeds

        assert hidden_states is not None and inputs_embeds is not None

        # 4. Eagle3 time-step alignment roll
        rolled_embeds = torch.roll(inputs_embeds, shifts=-1, dims=0)

        # 5. Draft model forward
        response_mask = data.get("response_mask", None)
        if response_mask is not None and hasattr(response_mask, "to_padded_tensor"):
            response_mask_t: Optional[torch.Tensor] = response_mask.to_padded_tensor().bool()
        elif isinstance(response_mask, torch.Tensor):
            response_mask_t = response_mask.bool()
        else:
            response_mask_t = None

        draft_logits: torch.Tensor = self.manager.draft_model(
            hidden_states=hidden_states,
            inputs_embeds=rolled_embeds,
            attention_mask=None,
        )

        if response_mask_t is None:
            # draft_logits is always [B, S, vocab] — use its shape to get correct (B, S)
            response_mask_t = torch.ones(
                draft_logits.shape[0], draft_logits.shape[1],
                dtype=torch.bool, device=draft_logits.device,
            )
        # eagle_output_layer uses gather_output=False → vocab-parallel [B, S, vocab/TP].
        # eagle_draft_loss needs full-vocab softmax, so gather across TP ranks.
        draft_logits = gather_vocab_parallel_logits(draft_logits)

        # 6. Distillation loss
        draft_loss = eagle_draft_loss(
            draft_logits=draft_logits.float(),
            teacher_logits=teacher_logits,
            response_mask=response_mask_t,
        )

        total_loss = policy_loss + self.manager.config.loss_weight * draft_loss
        metrics["actor/eagle_draft_loss"] = draft_loss.detach().item()

        return total_loss, metrics
