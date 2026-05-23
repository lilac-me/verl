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
Owns the Eagle3 draft model, its optimizer, and the hidden-state capture
hooks.  Built once per training process and kept alive for the full training
run.

EagleLossWrapper
----------------
A callable that wraps the base policy-loss function.  It is injected as the
``loss_fn`` of the FSDP / Megatron ``TrainingWorker``, so it is invoked once
per micro-batch inside the engine's forward-backward loop:

    total_loss, metrics = loss_wrapper(model_output, data, dp_group)

Tensor unpacking
----------------
verl supports two packed (remove_padding) tensor formats:

* FSDP:    3-D ``[1, total_nnz, feat]`` — detected by ``packed.shape[0] == 1``
* Megatron (thd): 2-D ``[total_tokens_padded, feat]`` — sequences are padded
  to a multiple of ``TP_size × CP_size (× 2 if CP > 1)`` tokens before
  concatenation.  The padded offsets are recomputed from Megatron's parallel
  state to correctly slice each sequence.

Both cases are triggered when ``data["input_ids"]`` is a nested tensor
(sequences have variable lengths stored via ``.offsets()``).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from tensordict import TensorDict

from verl.workers.eagle.config import EagleDraftConfig
from verl.workers.eagle.draft_model import (
    Eagle3DraftModel,
    EagleDraftModelWrapper,
    build_eagle3_from_policy,
    get_draft_state_dict_for_vllm,
    load_eagle_draft_model,
)
from verl.workers.eagle.hidden_capture import HiddenStateCapture, get_eagle3_aux_layer_indices, roll_inputs_embeds
from verl.workers.utils.losses import eagle_draft_loss

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tensor unpack helpers
# ---------------------------------------------------------------------------

def _unpack_fsdp_packed(
    packed: Optional[torch.Tensor],
    seq_lens: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Unpack a densely packed [1, total_nnz, *feat] tensor → [batch, max_seq, *feat].

    FSDP remove_padding mode concatenates sequences without any inter-sequence
    padding, so we can advance the offset by exactly ``seq_len`` each time.
    """
    if packed is None:
        return None
    if not (packed.dim() >= 2 and packed.shape[0] == 1):
        return packed  # already padded batch-first format
    batch = seq_lens.shape[0]
    max_seq = int(seq_lens.max().item())
    feat = packed.shape[2:]
    out = packed.new_zeros(batch, max_seq, *feat)
    offset = 0
    for i, length in enumerate(seq_lens.tolist()):
        length = int(length)
        out[i, :length] = packed[0, offset : offset + length]
        offset += length
    return out


def _unpack_megatron_thd(
    packed: Optional[torch.Tensor],
    seq_lens: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Unpack a Megatron thd-format [total_tokens_padded, *feat] tensor → [batch, max_seq, *feat].

    In Megatron's remove_padding (thd) mode each sequence is zero-padded to a
    multiple of ``align = TP × CP × 2`` (or just TP when CP=1) before being
    concatenated.  We recompute the per-sequence padded length to correctly
    advance the offset between sequences.

    Falls back to FSDP-style unpack for 3-D tensors.
    """
    if packed is None:
        return None
    if packed.dim() != 2:
        # Unexpected shape or 3-D [1, total, feat] — delegate to FSDP unpacker
        return _unpack_fsdp_packed(packed, seq_lens)

    # Compute alignment from Megatron parallel state (mirrors preprocess_thd_engine)
    try:
        from megatron.core import parallel_state as mpu
        tp = mpu.get_tensor_model_parallel_world_size()
        cp = mpu.get_context_parallel_world_size()
    except ImportError:
        tp, cp = 1, 1
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
        # Hooks are registered once in build(); they fire on every forward pass.

        # After engine.train_batch() completes:
        manager.optimizer_step()   # all-reduce grads across DP, then step

        # After policy weights sync to vLLM:
        state_dict = manager.state_dict_for_vllm()   # yield (name, tensor)
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

        # Register hooks once — they remain active for the training lifetime
        self.capture.register_hooks()
        logger.info("Eagle3 draft hooks registered on policy model.")

    @classmethod
    def build(
        cls,
        policy_model: nn.Module,
        eagle_config: EagleDraftConfig,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
        hf_config=None,
    ) -> "EagleDraftManager":
        """Factory: load or build draft model, register hooks, build optimizer.

        Path A (``eagle_config.model_path`` is set): loads a pretrained HF
        Eagle3 checkpoint via ``load_eagle_draft_model``.

        Path B (``eagle_config.model_path`` is None): assembles an Eagle3 draft
        from the policy's own components via ``build_eagle3_from_policy``.
        Requires ``hf_config`` to be provided.
        """
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())

        aux_layer_indices = (
            tuple(eagle_config.aux_layer_indices)
            if eagle_config.aux_layer_indices is not None
            else None
        )

        if eagle_config.model_path is not None:
            # Path A: load pretrained HF Eagle3 checkpoint
            draft_model = load_eagle_draft_model(
                model_path=eagle_config.model_path,
                torch_dtype=torch_dtype,
                device=device,
            )
        else:
            # Path B: build from policy components (nemo-rl style)
            if hf_config is None:
                raise ValueError(
                    "Eagle3 Path B (build from policy) requires hf_config. "
                    "Pass the HuggingFace model config to EagleDraftManager.build()."
                )

            # Determine n_aux from whichever aux_layer_indices we end up using.
            # auto-select uses get_eagle3_aux_layer_indices → 3 layers by default.
            if aux_layer_indices is not None:
                n_aux = len(aux_layer_indices)
            else:
                n_aux = len(get_eagle3_aux_layer_indices(hf_config.num_hidden_layers))

            draft_model = build_eagle3_from_policy(
                policy_model=policy_model,
                hf_config=hf_config,
                n_aux=n_aux,
                num_draft_layers=eagle_config.num_draft_layers,
                torch_dtype=torch_dtype,
                device=device,
            )

        capture = HiddenStateCapture(
            model=policy_model,
            aux_layer_indices=aux_layer_indices,
            capture_logits=True,
        )

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
        lr = config.optimizer.lr
        wd = config.optimizer.weight_decay
        params = [p for p in draft_model.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)

    def make_loss_wrapper(self, base_loss_fn: Callable) -> "EagleLossWrapper":
        """Wrap a base policy-loss function with Eagle3 draft distillation."""
        return EagleLossWrapper(base_loss_fn=base_loss_fn, manager=self)

    def optimizer_step(self) -> None:
        """All-reduce draft gradients across DP ranks, clip, and step.

        The draft model is not wrapped by FSDP/DDP so gradients from different
        data-parallel ranks are local only.  Explicit all-reduce ensures every
        rank applies the same full-dataset gradient (equivalent to DDP's
        gradient hook).
        """
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            world_size = dist.get_world_size()
            for p in self.draft_model.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                    p.grad.div_(world_size)

        torch.nn.utils.clip_grad_norm_(
            [p for p in self.draft_model.parameters() if p.requires_grad],
            max_norm=1.0,
        )
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Path B: keep frozen lm_head aligned with the policy's updated lm_head.
        if isinstance(self.draft_model, Eagle3DraftModel):
            self.draft_model.sync_lm_head()

    def state_dict_for_vllm(self) -> Iterator[Tuple[str, torch.Tensor]]:
        """Yield (name, cpu_float32_tensor) pairs for loading into vLLM."""
        return get_draft_state_dict_for_vllm(self.draft_model)

    def save_pretrained(self, path: str) -> None:
        """Save the draft model in HuggingFace format for checkpointing."""
        inner = self.draft_model.model
        if hasattr(inner, "save_pretrained"):
            inner.save_pretrained(path)
        else:
            torch.save(inner.state_dict(), path)
        logger.info(f"Eagle3 draft model saved to {path}")


# ---------------------------------------------------------------------------
# EagleLossWrapper
# ---------------------------------------------------------------------------

class EagleLossWrapper:
    """Wraps a policy-loss callable with Eagle3 draft distillation.

    Injected as the ``loss_fn`` of the FSDP / Megatron TrainingWorker.
    Called once per micro-batch inside the engine's forward-backward loop:

        total_loss, metrics = wrapper(model_output, data, dp_group)

    Steps:
    1. Compute the base policy loss via ``base_loss_fn``.
    2. Read hidden states / embeddings / LM-head logits captured by the hooks.
    3. Unpack packed tensors (remove_padding mode) to [batch, max_seq, feat].
    4. Run the Eagle3 draft model forward.
    5. Compute distillation loss (soft cross-entropy with Eagle3 alignment).
    6. Return ``L_total = L_policy + λ × L_draft`` and merged metrics.
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
        # 1. Standard policy loss ------------------------------------------------
        policy_loss, metrics = self.base_loss_fn(
            model_output=model_output, data=data, dp_group=dp_group, **kwargs
        )

        # 2. Collect captured states ---------------------------------------------
        captured = self.manager.capture.get_captured_states()
        # Clear immediately so stale state never leaks to the next micro-batch
        self.manager.capture._captured.clear()

        if (
            captured.hidden_states is None
            or captured.inputs_embeds is None
            or captured.lm_head_logits is None
        ):
            logger.debug("Eagle3: missing captured states; skipping draft loss this step.")
            return policy_loss, metrics

        # 3. Unpack packed tensors -----------------------------------------------
        # Both FSDP and Megatron remove_padding modes mark input_ids as a nested
        # tensor, with per-sequence lengths accessible via .offsets().
        input_ids = data.get("input_ids", None)
        if (
            input_ids is not None
            and isinstance(input_ids, torch.Tensor)
            and input_ids.is_nested
        ):
            seq_lens = input_ids.offsets().diff()  # [batch]

            # Dispatch: 2-D packed → Megatron thd; 3-D [1, total, feat] → FSDP
            hs = captured.hidden_states
            _unpack = _unpack_megatron_thd if (hs is not None and hs.dim() == 2) else _unpack_fsdp_packed

            hidden_states = _unpack(captured.hidden_states, seq_lens)
            inputs_embeds = _unpack(captured.inputs_embeds, seq_lens)
            lm_head_logits = _unpack(captured.lm_head_logits, seq_lens)
        else:
            # No packing: tensors are already [batch, seq, feat]
            hidden_states = captured.hidden_states
            inputs_embeds = captured.inputs_embeds
            lm_head_logits = captured.lm_head_logits

        # 4. Eagle3 time-step alignment roll ------------------------------------
        rolled_embeds = roll_inputs_embeds(inputs_embeds)

        # 5. Draft model forward ------------------------------------------------
        response_mask = data.get("response_mask", None)
        if response_mask is not None and hasattr(response_mask, "to_padded_tensor"):
            response_mask_t: Optional[torch.Tensor] = response_mask.to_padded_tensor().bool()
        elif isinstance(response_mask, torch.Tensor):
            response_mask_t = response_mask.bool()
        else:
            # Fall back to all-valid mask
            response_mask_t = torch.ones(
                hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device
            )

        draft_logits: torch.Tensor = self.manager.draft_model(
            hidden_states=hidden_states,
            inputs_embeds=rolled_embeds,
            attention_mask=response_mask_t,
        )

        # 6. Distillation loss (with Eagle3 alignment roll) ----------------------
        draft_loss = eagle_draft_loss(
            draft_logits=draft_logits.float(),
            teacher_logits=lm_head_logits,          # already detached + float32
            response_mask=response_mask_t,
            loss_weight=self.manager.config.loss_weight,
        )

        total_loss = policy_loss + draft_loss
        metrics["actor/eagle_draft_loss"] = draft_loss.detach().item()

        return total_loss, metrics
