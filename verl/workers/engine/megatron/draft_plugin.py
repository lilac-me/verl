# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
"""Engine-side plugin for EAGLE3 online draft training (Megatron backend).

Design decision — STANDALONE draft model, not attached to policy chunks:
verl's `make_megatron_module` wraps model chunks with Megatron DDP (grad
buffers built at wrap time) before `initialize()` can attach anything, so a
post-hoc attached submodule would silently miss DP grad reduction and the
distributed optimizer. NeMo-RL solved this with a pre-wrap hook; verl has
none. Instead the plugin owns:
  * a standalone EagleDraftModel on the last PP stage (TP-replicated),
  * its own torch.optim.AdamW,
  * explicit grad all-reduce (TP replica AVG, then DP AVG) before step(),
  * its own checkpoint files under <ckpt>/draft_model/.
This also keeps `draft_model.*` keys out of the policy state dict, so the
mcore->HF bridge export and dist-ckpt need no filtering at all.

Gradients still reach the draft params through the normal pipeline backward:
postprocess returns `policy_loss + loss_weight * draft_loss`, and autograd
traverses the (separately built) draft graph when Megatron backprops the
microbatch loss on the last stage.

Hook points in verl/workers/engine/megatron/transformer_impl.py (see the
accompanying patch):
    initialize()                  -> plugin.setup(...)
    optimizer_zero_grad()         -> plugin.zero_grad()
    optimizer_step()              -> plugin.step()
    forward_step()                -> plugin.capture_context()/begin_microbatch()/
                                     stash_teacher_from_logits_processor()/
                                     run_draft_forward()
    postprocess_micro_batch_func()-> plugin.add_draft_loss(...)
    get_per_tensor_param()        -> plugin.chain_draft_export(...)
    save/load_checkpoint()        -> plugin.save_checkpoint()/load_checkpoint()
"""

from __future__ import annotations

import logging
import os
from contextlib import nullcontext
from typing import Iterator, Optional

import torch
import torch.distributed as dist
from megatron.core import parallel_state as mpu
from torch import Tensor

from verl.models.mcore.draft.config import DraftModelConfig
from verl.models.mcore.draft.eagle import EagleDraftModel
from verl.models.mcore.draft.hidden_capture import HiddenStateCapture
from verl.models.mcore.draft.loss import (
    DraftLossState,
    gather_teacher_logits_for_draft_vocab,
    roll_left_seq,
)
from verl.models.mcore.draft.weight_utils import (
    DRAFT_WEIGHT_PREFIX,
    export_eagle_weights_to_hf,
    init_draft_lm_head_from_policy,
    load_hf_weights_to_eagle,
)

logger = logging.getLogger(__name__)


class MegatronDraftTrainerPlugin:
    def __init__(self, draft_cfg: DraftModelConfig):
        self.cfg = draft_cfg
        self.enabled = draft_cfg.enable
        self.draft_model: Optional[EagleDraftModel] = None
        self.draft_optimizer: Optional[torch.optim.Optimizer] = None
        self.capture: Optional[HiddenStateCapture] = None
        self._is_owner_stage = False
        # one in-flight microbatch state; with non-interleaved schedules,
        # forward -> postprocess for a microbatch completes before the next
        # forward on the same rank. TODO(pp>1 interleaved): key by mb id.
        self._state: Optional[DraftLossState] = None
        self._pending_loss_mask: Optional[Tensor] = None

    # ------------------------------------------------------------------ #
    # setup
    # ------------------------------------------------------------------ #
    def setup(self, module_chunks: list, target_hf_config, params_dtype=torch.bfloat16):
        if not self.enabled:
            return
        self._is_owner_stage = mpu.is_pipeline_last_stage(ignore_virtual=True)
        num_layers = target_hf_config.num_hidden_layers

        self.capture = HiddenStateCapture(
            model_chunks=module_chunks,
            num_layers=num_layers,
            aux_layer_indices=self.cfg.aux_layer_indices,
            detach=self.cfg.detach_hidden_states,
        )

        if not self._is_owner_stage:
            logger.info("[draft] rank participates in capture gather only (not last PP stage).")
            return

        draft_hf_config = self._load_draft_hf_config(target_hf_config)
        self.draft_model = EagleDraftModel(
            hf_config=draft_hf_config,
            draft_cfg=self.cfg,
            target_hidden_size=target_hf_config.hidden_size,
            params_dtype=params_dtype,
        ).to(self._device())

        if self.cfg.model_path:
            missing, unexpected = load_hf_weights_to_eagle(self.draft_model, self.cfg.model_path)
            if unexpected:
                logger.warning("[draft] unexpected ckpt keys ignored: %s", unexpected[:8])
            if any(m.startswith("lm_head") for m in missing):
                self._init_lm_head_from_policy(module_chunks)
            real_missing = [m for m in missing if not m.startswith("lm_head") and m != "d2t"]
            assert not real_missing, f"[draft] missing checkpoint keys: {real_missing}"
        else:
            logger.warning(
                "[draft] model_path is null: RANDOM INIT drafter (debug only); "
                "early rollouts will have ~0 acceptance."
            )
            self._init_lm_head_from_policy(module_chunks)

        # fp32 master-less AdamW over bf16 params is acceptable for a 1-layer
        # model at lr~1e-4; TODO(optional): fp32 master weights if loss
        # plateaus suspiciously early on NPU bf16.
        self.draft_optimizer = torch.optim.AdamW(
            self.draft_model.parameters(),
            lr=self.cfg.lr,
            betas=tuple(self.cfg.betas),
            weight_decay=self.cfg.weight_decay,
        )
        logger.info(
            "[draft] standalone EagleDraftModel ready (draft_vocab=%d, aux_layers=%s, params=%s).",
            self.draft_model.draft_vocab_size,
            self.capture.aux_layer_indices,
            f"{sum(p.numel() for p in self.draft_model.parameters()):,}",
        )

    @staticmethod
    def _device():
        from verl.utils.device import get_device_id  # cuda/npu agnostic

        return get_device_id()

    def _load_draft_hf_config(self, target_hf_config):
        if self.cfg.model_path:
            from transformers import AutoConfig

            return AutoConfig.from_pretrained(self.cfg.model_path, trust_remote_code=True)
        return target_hf_config  # random init: clone salient fields from target

    @torch.no_grad()
    def _init_lm_head_from_policy(self, module_chunks):
        from megatron.core.utils import unwrap_model

        chunk = unwrap_model(module_chunks[-1])
        out_layer = getattr(chunk, "output_layer", None)
        if out_layer is None:
            raise RuntimeError("[draft] last-stage chunk has no output_layer; cannot init draft lm_head.")
        init_draft_lm_head_from_policy(
            self.draft_model, out_layer.weight.data, tp_group=mpu.get_tensor_model_parallel_group()
        )
        logger.info("[draft] initialized draft lm_head from policy output layer (d2t-selected).")

    # ------------------------------------------------------------------ #
    # forward-path hooks
    # ------------------------------------------------------------------ #
    def capture_context(self):
        if not self.enabled or self.capture is None:
            return nullcontext()
        return self.capture.capture_context()

    def begin_microbatch(self, loss_mask: Optional[Tensor], global_valid_toks: Optional[Tensor]):
        if not self.enabled:
            return
        self._state = DraftLossState(loss_weight=self.cfg.loss_weight, global_valid_toks=global_valid_toks)
        self._pending_loss_mask = loss_mask

    def stash_teacher_from_logits_processor(self, vp_logits: Tensor, seq_dim: int):
        """Call inside the engine's logits_processor on the last stage.

        Must run BEFORE the in-place temperature `logits.div_()` unless
        cfg.temperature_scaled_teacher. vp_logits: vocab-parallel policy
        logits; only the d2t-mapped draft-vocab columns are gathered
        ([*, draft_vocab] fp32), rolled left, last position zeroed.
        """
        if not self.enabled or self._state is None or self.draft_model is None:
            return
        teacher = gather_teacher_logits_for_draft_vocab(
            vp_logits.detach(),
            self.draft_model.d2t,
            tp_rank=mpu.get_tensor_model_parallel_rank(),
            tp_world_size=mpu.get_tensor_model_parallel_world_size(),
            tp_group=mpu.get_tensor_model_parallel_group(),
        )
        teacher = roll_left_seq(teacher, seq_dim=seq_dim)
        teacher.select(seq_dim, teacher.shape[seq_dim] - 1).zero_()
        self._state.teacher_logits = teacher

    def run_draft_forward(self, position_ids: Optional[Tensor] = None):
        """Call on ALL PP ranks right after the policy forward returns — the
        capture gather is a PP collective. Owner stage runs the draft forward
        ([s, b, draft_vocab]) and builds the shifted loss mask.
        """
        if not self.enabled or self.capture is None or self._state is None:
            return
        states = self.capture.get_captured_states()
        if not self._is_owner_stage or states.hidden_states is None:
            return
        shifted_embeds = roll_left_seq(states.inputs_embeds, seq_dim=0)  # mcore [s,b,h]
        student = self.draft_model(
            hidden_states=states.hidden_states,
            input_embeds=shifted_embeds,
            position_ids=position_ids,
        )
        self._state.student_logits = student

        s = student.shape[0]
        lm = self._pending_loss_mask
        if lm is None:
            lm = torch.ones(s, student.shape[1], device=student.device)
        else:
            lm = lm.to(student.device)
            if lm.dim() == 2 and lm.shape[0] != s and lm.shape[1] == s:
                lm = lm.transpose(0, 1)  # [b,s] -> [s,b]
            if lm.shape[0] != s:
                # loss_mask covers response tokens only: left-pad prompt
                # region with zeros => draft trains on response positions.
                # TODO(verify): full-seq mask incl. prompt may help acceptance
                # on long prompts; revisit after bring-up.
                pad = torch.zeros(s - lm.shape[0], lm.shape[1], device=lm.device, dtype=lm.dtype)
                lm = torch.cat([pad, lm], dim=0)
        lm = roll_left_seq(lm, seq_dim=0)
        lm.select(0, lm.shape[0] - 1).zero_()
        self._state.loss_mask = lm

    def add_draft_loss(self, loss: Tensor, metrics: dict) -> Tensor:
        if not self.enabled or self._state is None:
            return loss
        draft_loss = self._state.compute()
        if draft_loss is not None:
            loss = loss + draft_loss
            metrics.update(self._state.metrics)
        self._state.clear()
        self._state = None
        return loss

    # ------------------------------------------------------------------ #
    # optimizer hooks
    # ------------------------------------------------------------------ #
    def zero_grad(self):
        if self.enabled and self.draft_optimizer is not None:
            self.draft_optimizer.zero_grad(set_to_none=True)

    def step(self) -> Optional[float]:
        """All-reduce grads (TP replica AVG, then DP AVG), clip, step.

        Grad accumulation across microbatches happened naturally via autograd
        (.grad +=); AVG reductions commute with the accumulation sum.
        CP=1 is enforced by validate_draft_config, so the DP group with
        context parallel folded in equals the plain DP group.
        """
        if not self.enabled or self.draft_model is None or self.draft_optimizer is None:
            return None
        tp_group = mpu.get_tensor_model_parallel_group()
        dp_group = mpu.get_data_parallel_group(with_context_parallel=True)
        grads = [p.grad for p in self.draft_model.parameters() if p.grad is not None]
        if not grads:
            return None
        flat = torch._utils._flatten_dense_tensors([g.float() for g in grads])
        if dist.get_world_size(group=tp_group) > 1:
            dist.all_reduce(flat, op=dist.ReduceOp.AVG, group=tp_group)
        if dist.get_world_size(group=dp_group) > 1:
            dist.all_reduce(flat, op=dist.ReduceOp.AVG, group=dp_group)
        for g, synced in zip(grads, torch._utils._unflatten_dense_tensors(flat, grads)):
            g.copy_(synced.to(g.dtype))
        grad_norm = torch.nn.utils.clip_grad_norm_(self.draft_model.parameters(), self.cfg.grad_clip)
        self.draft_optimizer.step()
        return float(grad_norm)

    # ------------------------------------------------------------------ #
    # checkpoint
    # ------------------------------------------------------------------ #
    _CKPT_SUBDIR = "draft_model"

    def save_checkpoint(self, local_path: str):
        if not (self.enabled and self._is_owner_stage and self.draft_model is not None):
            return
        # params are TP/DP-replicated on the last stage: one writer suffices.
        if mpu.get_tensor_model_parallel_rank() != 0:
            return
        if mpu.get_data_parallel_rank(with_context_parallel=True) != 0:
            return
        path = os.path.join(local_path, self._CKPT_SUBDIR)
        os.makedirs(path, exist_ok=True)
        torch.save(
            {
                "model": self.draft_model.state_dict(),
                "optimizer": self.draft_optimizer.state_dict() if self.draft_optimizer else None,
            },
            os.path.join(path, "draft.pt"),
        )
        logger.info("[draft] checkpoint saved to %s", path)

    def load_checkpoint(self, local_path: str):
        if not (self.enabled and self._is_owner_stage and self.draft_model is not None):
            return
        f = os.path.join(local_path, self._CKPT_SUBDIR, "draft.pt")
        if not os.path.exists(f):
            logger.warning("[draft] no draft checkpoint at %s; keeping current init.", f)
            return
        state = torch.load(f, map_location="cpu", weights_only=True)
        self.draft_model.load_state_dict(state["model"])
        if self.draft_optimizer is not None and state.get("optimizer") is not None:
            self.draft_optimizer.load_state_dict(state["optimizer"])
        logger.info("[draft] checkpoint restored from %s", f)

    # ------------------------------------------------------------------ #
    # refit export
    # ------------------------------------------------------------------ #
    def chain_draft_export(self, per_tensor_param: Iterator) -> Iterator:
        """Append 'draft.*' tensors to the policy per-tensor refit stream.

        Yields on every rank for collective-consumption safety: the bridge's
        per-tensor export is consumed identically on all ranks of a refit
        group, and draft tensors are bit-identical replicas after step()'s
        all-reduce. TODO(verify): if the bucketed sender dedups by rank,
        gate on (last PP stage, tp_rank 0, dp_rank 0) instead.
        """
        if not self.enabled:
            return per_tensor_param

        def gen():
            yield from per_tensor_param
            if self.draft_model is not None:
                for name, tensor in export_eagle_weights_to_hf(self.draft_model):
                    yield DRAFT_WEIGHT_PREFIX + name, tensor

        return gen()
