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
import os
from typing import Any, Callable, Iterator, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from tensordict import TensorDict

from verl.workers.eagle.config import EagleDraftConfig
from verl.workers.eagle.draft_model import EagleDraftModel
from megatron.core import parallel_state
from verl.workers.eagle.draft_utils import (
    all_reduce_sum_grad,
    assert_capture_preconditions,
    build_vocab_parallel_select_plan,
    copy_policy_lm_head_to_draft,
    get_draft_state_dict_for_vllm,
    load_eagle_draft_model,
)
from verl.workers.eagle.hidden_capture import HiddenStateCapture

logger = logging.getLogger(__name__)


# EAGLE3 train/inference time-step alignment. teacher_logits[t] is the policy's
# distribution for token t+1 (standard causal LM), and at inference the vLLM Eagle3
# proposer pairs hidden_t with embed(token_{t+1}) (see vllm v1/spec_decode/eagle.py:
# `input_ids[:n-1] = target_token_ids[1:]`, hidden_states unshifted). So the draft at
# position t must see embed rolled -1 and be trained against teacher rolled -1 (with the
# wrap-around edge masked); hidden is NOT rolled. This matches the documented convention
# in verl/workers/utils/losses.roll_for_eagle_alignment.
#
# The previous 0/0/0 default pairs hidden_t with embed_t and targets teacher_t — a
# DEGENERATE objective: hidden_t (aux layers incl. ~last) already determines teacher_t
# via the target's own LM head, so the draft reaches low TRAINING loss (~3.6) by copying
# instead of learning to predict the future token, and acceptance stays ~0 at inference.
# Overridable via EAGLE_EROLL / EAGLE_HROLL / EAGLE_TROLL for diagnosis; the chosen
# convention should be validated against measured acceptance (test_vllm_eagle3_acceptance.py).
_EAGLE_DEFAULT_EROLL = -1
_EAGLE_DEFAULT_HROLL = 0
_EAGLE_DEFAULT_TROLL = -1


def eagle3_alignment_rolls() -> Tuple[int, int, int]:
    """Return (embed_roll, hidden_roll, teacher_roll) for EAGLE3 distillation.

    Defaults to the inference-consistent convention (-1, 0, -1); env vars override.
    """
    return (
        int(os.environ.get("EAGLE_EROLL", str(_EAGLE_DEFAULT_EROLL))),
        int(os.environ.get("EAGLE_HROLL", str(_EAGLE_DEFAULT_HROLL))),
        int(os.environ.get("EAGLE_TROLL", str(_EAGLE_DEFAULT_TROLL))),
    )


def eagle3_distill_loss(
    draft_pr: torch.Tensor,
    teacher_pr: torch.Tensor,
    mask_f: torch.Tensor,
    draft_vocab_size: int,
    sel_idx: Optional[torch.Tensor],
    token_budget: int = 1024,
    t_roll: int = 0,
) -> torch.Tensor:
    """Chunked forward-KL distillation loss — fully VOCAB-PARALLEL (no full-vocab gather).

    Keeps the draft logits SHARDED ``[B,S,draft_vocab/TP]`` and NEVER gathers the full draft or
    teacher vocab. Uses the soft-CE identity (per token, since Σ_v p_T[v] = 1)::

        loss_tok = -Σ_v p_T[v]·log_softmax_D[v] = LSE_D − Σ_v p_T[v]·D[v]

    ``LSE_D`` (draft logsumexp over the full draft vocab) and the cross term ``Σ p_T·D`` are
    computed with per-token-SCALAR TP all-reduces (MAX for stability — detached; SUM for the
    value). The SUM all-reduces are DIFFERENTIABLE (``all_reduce_sum_grad``), so the gradient
    ``∂loss/∂D_r[v] = softmax_D[v] − p_T[v]`` flows to each rank's draft shard with no full-vocab
    materialization. The teacher (detached) is selected to the draft vocab via a vocab-parallel
    scatter + ``all_reduce(SUM)`` (``build_vocab_parallel_select_plan``). At TP=1 every collective
    is a no-op and this reduces EXACTLY to a plain full-vocab-gather soft-CE (the test ground
    truth ``_eagle3_distill_loss_gather`` in tests/test_eagle3_cpu.py). Per-chunk grad-bearing
    memory drops from ``draft_vocab`` to ``draft_vocab/TP`` and the draft-side comm from an
    all-gather to a few scalar all-reduces.

    Args:
        draft_pr:   ``[B, S, draft_vocab/TP]`` per-rank draft logits (carries grad, stays sharded).
        teacher_pr: ``[B, S, vocab/TP]`` per-rank teacher logits (detached, UNROLLED).
        mask_f:     ``[B, S]`` float mask.
        draft_vocab_size: total draft vocab.
        sel_idx:    ``[draft_vocab_size]`` teacher column per draft token (``i + d2t[i]``);
                    None → identity ``arange(draft_vocab_size)``.
        token_budget: per-chunk token budget (chunk = token_budget // B).
        t_roll: EAGLE teacher time-shift ``rolled[:,j] = teacher[:,(j - t_roll) % S]``, per chunk (0 = none).
    """
    tp_world = parallel_state.get_tensor_model_parallel_world_size()
    tp_group = parallel_state.get_tensor_model_parallel_group() if tp_world > 1 else None
    tp_rank = parallel_state.get_tensor_model_parallel_rank()

    B, S = draft_pr.shape[0], draft_pr.shape[1]
    dpv = draft_pr.shape[-1]            # draft_vocab/TP — this rank's draft shard width
    tshard = teacher_pr.shape[-1]       # padded_vocab/TP — this rank's teacher shard width
    draft_lo = tp_rank * dpv            # this rank owns draft columns [draft_lo, draft_lo + dpv)

    if sel_idx is None:
        sel_idx = torch.arange(draft_vocab_size, device=draft_pr.device)
    owns, local_cols = build_vocab_parallel_select_plan(sel_idx, tshard)   # precompute once

    chunk = max(1, token_budget // max(B, 1))
    loss_sum = draft_pr.new_zeros((), dtype=torch.float32)
    for start in range(0, S, chunk):
        end = min(start + chunk, S)
        m_chunk = mask_f[:, start:end]                                      # [B, c]
        if float(m_chunk.sum()) == 0:
            continue
        D_r = draft_pr[:, start:end].float()                               # [B, c, dpv] (grad)

        # teacher chunk: per-chunk EAGLE roll (seq dim), then vocab-parallel select to draft vocab
        if t_roll:
            idx = (torch.arange(start, end, device=teacher_pr.device) - t_roll) % S
            t_src = teacher_pr.index_select(1, idx)                         # [B, c, tshard]
        else:
            t_src = teacher_pr[:, start:end]                               # [B, c, tshard]
        tsel = t_src.new_zeros(t_src.shape[0], t_src.shape[1], draft_vocab_size)
        tsel[:, :, owns] = t_src[:, :, local_cols]
        if tp_world > 1:
            dist.all_reduce(tsel, op=dist.ReduceOp.SUM, group=tp_group)     # teacher is detached
        p_T = torch.softmax(tsel.float(), dim=-1)                          # [B, c, draft_v]
        p_T_r = p_T[:, :, draft_lo:draft_lo + dpv]                         # [B, c, dpv] (this rank)

        # draft logsumexp over the FULL draft vocab — vocab-parallel + differentiable
        M = D_r.max(dim=-1, keepdim=True).values.detach()                 # [B, c, 1] stability const
        if tp_world > 1:
            dist.all_reduce(M, op=dist.ReduceOp.MAX, group=tp_group)
        se_r = torch.exp(D_r - M).sum(dim=-1)                              # [B, c] (grad)
        lse_D = M.squeeze(-1) + torch.log(all_reduce_sum_grad(se_r))       # [B, c]

        # cross term Σ_v p_T[v]·D[v] over the full draft vocab — vocab-parallel + differentiable
        ct = all_reduce_sum_grad((p_T_r * D_r).sum(dim=-1))               # [B, c]

        per_tok = lse_D - ct                                               # [B, c] == -Σ p_T·log_softmax_D
        loss_sum = loss_sum + (per_tok * m_chunk).sum()
    return loss_sum / mask_f.sum().clamp(min=1.0)


# ---------------------------------------------------------------------------
# Tensor unpack helpers
# ---------------------------------------------------------------------------

def _unpack_megatron_thd(
    packed: Optional[torch.Tensor],
    attention_mask: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Unpack Megatron thd-packed activations → ``[batch, seq, feat]``.

    Mirrors ``verl/models/mcore/util.py`` packing exactly: each sequence is
    padded to ``align = tp*cp*2 (cp>1) | tp`` and laid out at
    ``cu_seqlens_padded`` offsets. The valid length of sequence ``i`` is
    ``attention_mask[i].sum()``. Using the attention mask (not the RL data's
    input_ids) is essential — the packed layout follows these padded offsets.

    Args:
        packed: thd activations, seq-first with batch dim 1: ``[packed_len, 1, feat]``
                (also accepts ``[packed_len, feat]``).
        attention_mask: ``[batch, seq]``, 1 for valid tokens.

    Returns:
        ``[batch, seq, feat]`` (CP=1 only; SP must be disabled so the captured
        activations are full-length, not sequence-parallel scattered).
    """
    if packed is None:
        return None

    from megatron.core import parallel_state as mpu
    tp = mpu.get_tensor_model_parallel_world_size()
    cp = mpu.get_context_parallel_world_size()
    align = max(tp * cp * 2 if cp > 1 else tp, 1)

    # Collapse the singleton batch dim of thd activations to [packed_len, feat].
    # Megatron captures different layouts: decoder-layer outputs are seq-first
    # [packed_len, 1, feat], the embedding output is batch-first [1, packed_len, feat].
    if packed.dim() == 3:
        if packed.shape[0] == 1:
            packed = packed.squeeze(0)        # [1, packed_len, feat] -> [packed_len, feat]
        elif packed.shape[1] == 1:
            packed = packed.squeeze(1)        # [packed_len, 1, feat] -> [packed_len, feat]

    batch, seq = attention_mask.shape
    seqlens = attention_mask.sum(dim=-1).to(torch.int64)        # valid length per sequence
    seqlens_padded = seqlens + (align - seqlens % align) % align
    cu_padded = torch.zeros(batch + 1, dtype=torch.int64)
    cu_padded[1:] = torch.cumsum(seqlens_padded.cpu(), dim=0)
    cu_padded = cu_padded.tolist()
    seqlens_cpu = seqlens.tolist()

    feat = packed.shape[1:]
    out = packed.new_zeros(batch, seq, *feat)
    am_bool = attention_mask.bool()
    for i in range(batch):
        s = int(seqlens_cpu[i])
        start = int(cu_padded[i])
        if s > 0:
            # SCATTER the s packed tokens back to their ORIGINAL masked positions —
            # exactly as verl/models/mcore/util.py postprocess_packed_seqs does
            # (`output_new[i, attention_mask[i]] = packed[start:start+s]`). The earlier
            # `out[i, :s] = ...` LEFT-ALIGNED the tokens, which mismatched the original
            # padded [B,S] layout that response_mask/attention_mask use → the draft loss
            # was computed on padding and sat ~17 (above uniform) regardless of roll.
            out[i, am_bool[i]] = packed[start : start + s].to(out.dtype)

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
        master_pairs: Optional[list] = None,
    ):
        self.config = config
        self.capture = capture
        self.draft_model = draft_model
        self.optimizer = optimizer
        self._policy_model = policy_model  # for lm_head extraction
        # (bf16_param, fp32_master) pairs — the optimizer steps the fp32 masters and we
        # copy them back into the bf16 model. Without fp32 masters, AdamW updates (~lr)
        # are smaller than the bf16 representable step at the params' magnitude and ROUND
        # TO ZERO (norm weights ~1.0 never move even at lr=1e-3), so the draft never
        # trains. See the bf16-rounding diagnosis.
        self._master_pairs = master_pairs or []
        # Per-layer draft grad norms from the last optimizer_step (merged into the step
        # metrics by engine_workers → logged to wandb/console).
        self.last_grad_metrics: dict = {}

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

        # Fail fast if SP/CP would break full-length hidden capture (before loading the ckpt).
        assert_capture_preconditions(policy_model)

        if device is None:
            from verl.utils.device import get_device_name, get_device_id
            device = torch.device(get_device_name(), get_device_id())

        draft_model = load_eagle_draft_model(
            eagle_config=eagle_config,
            policy_model=policy_model,
            device=device,
            torch_dtype=torch_dtype,
        )

        eagle_aux_hidden_state_layer_ids = eagle_config.eagle_aux_hidden_state_layer_ids
        capture = HiddenStateCapture(model=policy_model, eagle_aux_hidden_state_layer_ids=eagle_aux_hidden_state_layer_ids)
        optimizer, master_pairs = cls._build_optimizer(draft_model, eagle_config)

        return cls(
            draft_model=draft_model,
            capture=capture,
            config=eagle_config,
            optimizer=optimizer,
            policy_model=policy_model,
            master_pairs=master_pairs,
        )

    @staticmethod
    def _build_optimizer(
        draft_model: EagleDraftModel,
        config: EagleDraftConfig,
    ) -> Tuple[torch.optim.Optimizer, list]:
        # Only train params with requires_grad (the output layer may be frozen).
        # fp32 MASTER weights: the model runs in bf16 (fast, vLLM-compatible) but AdamW
        # must step fp32 copies — otherwise the ~lr-sized updates round to zero in bf16
        # and the draft never trains. optimizer_step() copies grads bf16->fp32 and
        # writes the updated fp32 masters back into the bf16 model. Names are kept so
        # per-layer grad norms can be logged.
        master_pairs = []
        masters = []
        for name, p in draft_model.named_parameters():
            if not p.requires_grad:
                continue
            if p.dtype == torch.float32:
                m = p                                   # already fp32 — optimize in place
            else:
                m = p.detach().clone().float().requires_grad_(True)
            masters.append(m)
            master_pairs.append((name, p, m))
        optimizer = torch.optim.AdamW(
            masters, lr=config.optimizer.lr, weight_decay=config.optimizer.weight_decay
        )
        return optimizer, master_pairs

    def make_loss_wrapper(self, base_loss_fn: Callable) -> "EagleLossWrapper":
        """Wrap a base policy-loss function with Eagle3 draft distillation."""
        return EagleLossWrapper(base_loss_fn=base_loss_fn, manager=self)

    def sync_lm_head(self) -> None:
        """Copy the current policy LM-head shard into the draft output layer.

        Only for tied-head drafts (no own lm_head in the checkpoint). Eagle3 drafts
        with their own trained lm_head (e.g. Tengyunw, cosine≈0 vs target) must NOT
        be overwritten — doing so pins the draft KL at ~14.5 and kills acceptance.
        """
        if getattr(self.draft_model, "_eagle_has_own_lm_head", False):
            return
        copy_policy_lm_head_to_draft(self.draft_model, self._policy_model)

    def optimizer_step(self) -> None:
        """Copy bf16 grads → fp32 masters, all-reduce across DP, clip, step, copy back.

        The model params are bf16; the optimizer holds fp32 master copies. We move the
        autograd grads (on the bf16 params) onto the fp32 masters, step the masters in
        fp32 (so updates don't round away), then write the updated masters back into the
        bf16 model. Without this the draft does not learn at all.
        """
        # 1. bf16 param.grad -> fp32 master.grad
        for _name, p, m in self._master_pairs:
            if p.grad is not None:
                m.grad = p.grad.detach().float()
            else:
                m.grad = None

        # 1b. Per-layer grad norms (post-DP-reduce, pre-clip) for logging to wandb.
        #     Stored in self.last_grad_metrics; engine_workers merges it into the step
        #     metrics. This is what makes the draft gradient flow visible.
        self.last_grad_metrics = {}

        # 2. all-reduce master grads across DP ranks
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            from megatron.core import parallel_state as mpu
            dp_group = mpu.get_data_parallel_group()
            dp_world_size = mpu.get_data_parallel_world_size()
            for _name, _p, m in self._master_pairs:
                if m.grad is not None:
                    dist.all_reduce(m.grad, op=dist.ReduceOp.SUM, group=dp_group)
                    m.grad.div_(dp_world_size)

        # record per-layer + total grad norm (this rank's shard) for monitoring
        total_sq = 0.0
        for name, _p, m in self._master_pairs:
            if m.grad is not None:
                gn = float(m.grad.detach().norm())
                self.last_grad_metrics[f"eagle_grad/{name}"] = gn
                total_sq += gn * gn
        self.last_grad_metrics["eagle_grad/total"] = float(total_sq ** 0.5)
        n_with_grad = sum(1 for _n, p, _m in self._master_pairs if p.grad is not None)
        self.last_grad_metrics["eagle_grad/num_params_with_grad"] = float(n_with_grad)

        # 3. clip + step the fp32 masters
        torch.nn.utils.clip_grad_norm_(
            [m for _n, _p, m in self._master_pairs if m.grad is not None], max_norm=1.0
        )
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        # 4. write updated fp32 masters back into the bf16 model + clear bf16 grads
        with torch.no_grad():
            for _name, p, m in self._master_pairs:
                if p is not m:
                    p.data.copy_(m.data)
                p.grad = None

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
        self.eagle_trainer = manager

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

        # 2. Collect captured states (hidden_states, inputs_embeds, logits all from hooks)
        captured = self.eagle_trainer.capture.get_captured_states()
        self.eagle_trainer.capture._captured.clear()

        if captured.hidden_states is None or captured.inputs_embeds is None:
            logger.debug("Eagle3: missing captured hidden states; skipping draft loss this step.")
            return policy_loss, metrics

        # captured.logits: [S, B, vocab/TP] vocab-parallel seq-first, captured from output_layer hook
        if captured.logits is None:
            logger.debug("Eagle3: missing captured logits; skipping draft loss this step.")
            return policy_loss, metrics

        # 3. Unpack Megatron thd-packed activations using the attention mask.
        # With use_remove_padding=True the captured hidden_states / inputs_embeds are
        # thd-packed [packed_len, 1, *]; _unpack_megatron_thd reconstructs [B, S, *]
        # via the same cu_seqlens_padded layout the policy forward used. We then
        # transpose to seq-first [S, B, *] for the Megatron-native draft model.
        # If no attention_mask is available (non-packed path) the captured tensors
        # are already [S, B, *] and used directly.
        attention_mask = data.get("attention_mask", None)
        if not getattr(EagleLossWrapper, "_logged_shapes", False):
            EagleLossWrapper._logged_shapes = True
            am_shape = (
                tuple(attention_mask.shape) if isinstance(attention_mask, torch.Tensor) else None
            )
            am_nested = isinstance(attention_mask, torch.Tensor) and attention_mask.is_nested
            logger.info(
                "[eagle][shapes] data keys=%s | attention_mask shape=%s nested=%s | "
                "captured hidden=%s embeds=%s logits=%s",
                list(data.keys()) if hasattr(data, "keys") else "?",
                am_shape, am_nested,
                tuple(captured.hidden_states.shape),
                tuple(captured.inputs_embeds.shape),
                tuple(captured.logits.shape),
            )
        if isinstance(attention_mask, torch.Tensor):
            am = (
                attention_mask.to_padded_tensor(0)
                if attention_mask.is_nested
                else attention_mask
            ).to(captured.hidden_states.device)
            # Sanity guard: the implied dense tensor must be plausible. If the data
            # contract differs from our assumption (am not [B,S]), skip the draft
            # loss this step instead of OOMing, and log the shapes once.
            packed_len = captured.hidden_states.shape[0]
            if am.dim() != 2 or int(am.sum().item()) > packed_len * max(am.shape[0], 1) or am.numel() > 4 * packed_len:
                logger.warning(
                    "[eagle] attention_mask shape %s inconsistent with packed_len=%d; "
                    "skipping draft loss this step.", tuple(am.shape), packed_len
                )
                return policy_loss, metrics
            hidden_states = _unpack_megatron_thd(captured.hidden_states, am).transpose(0, 1)
            inputs_embeds = _unpack_megatron_thd(captured.inputs_embeds, am).transpose(0, 1)
        else:
            hidden_states = captured.hidden_states   # [S, B, N_aux*H]
            inputs_embeds = captured.inputs_embeds   # [S, B, H]

        assert hidden_states is not None and inputs_embeds is not None

        # 4. Eagle3 time-step alignment (inference-consistent convention; see the module
        # docstring on eagle3_alignment_rolls). At inference the draft pairs target hidden
        # h_t with the NEXT token's embedding e_{t+1} to predict the following token, whose
        # teacher target is logits_{t+1}. So: embed rolled -1, teacher rolled -1, hidden not
        # rolled, with the wrap-around edge masked. Default (-1,0,-1); overridable via
        # EAGLE_EROLL/EAGLE_HROLL/EAGLE_TROLL for diagnosis. NOTE: 0/0/0 is degenerate
        # (the draft copies the target's own output from h_t instead of predicting the
        # future) — low training loss, ~0 acceptance. Validate any change against measured
        # acceptance (test_vllm_eagle3_acceptance.py).
        e_roll, h_roll, t_roll = eagle3_alignment_rolls()
        if h_roll:
            hidden_states = torch.roll(hidden_states, shifts=h_roll, dims=0)
        rolled_embeds = torch.roll(inputs_embeds, shifts=e_roll, dims=0) if e_roll else inputs_embeds

        # 5. Draft model forward → [B, S, vocab/TP]
        # Loss mask. The draft logits are full-sequence [B, S] (unpacked from thd),
        # so we need a [B, S] mask. response_mask is response-only and not trivially
        # alignable to the full sequence, so when we unpacked via the attention mask
        # we use that (all valid tokens) as the distillation mask. Otherwise fall back
        # to response_mask (non-packed path, already [B, S]).
        response_mask = data.get("response_mask", None)
        if isinstance(attention_mask, torch.Tensor):
            response_mask_t: Optional[torch.Tensor] = am.bool()
        elif isinstance(response_mask, torch.Tensor) and response_mask.is_nested:
            response_mask_t = response_mask.to_padded_tensor(0).bool()
        elif isinstance(response_mask, torch.Tensor):
            response_mask_t = response_mask.bool()
        else:
            response_mask_t = None

        draft_logits: torch.Tensor = self.eagle_trainer.draft_model(
            hidden_states=hidden_states,
            inputs_embeds=rolled_embeds,
            attention_mask=None,
        )

        if response_mask_t is None:
            response_mask_t = torch.ones(
                draft_logits.shape[0], draft_logits.shape[1],
                dtype=torch.bool, device=draft_logits.device,
            )

        # 6. Distillation loss — fully VOCAB-PARALLEL + chunked over the sequence dim. The
        # draft logits stay sharded [B,S,draft_vocab/TP]; the full vocab is never gathered
        # (only per-token scalars are all-reduced). See eagle3_distill_loss.
        draft_pr = draft_logits                                   # [B, S, draft_vocab/TP] (per-rank)
        S = draft_pr.shape[1]                                      # (B is derived inside eagle3_distill_loss)
        draft_v = int(self.eagle_trainer.draft_model.config.draft_vocab_size)

        # Keep the teacher in its captured dtype (bf16) — the loss upcasts to fp32 per chunk,
        # so materializing the full unpacked [B,S,vocab/TP] in fp32 here would just waste memory.
        teacher_pr = captured.logits.detach()
        if isinstance(attention_mask, torch.Tensor):
            teacher_pr = _unpack_megatron_thd(teacher_pr, am)    # [B, S, vocab/TP]
        else:
            teacher_pr = teacher_pr.transpose(0, 1).contiguous()  # [B, S, vocab/TP]

        # Eagle3 alignment (see step 4): the teacher roll is applied per-chunk inside
        # eagle3_distill_loss (via t_roll); here we only roll the small [B,S] mask to match
        # and zero the wrap-around edge.
        mask_f = response_mask_t.to(draft_pr.device).clone()
        if t_roll:
            mask_f = torch.roll(mask_f, shifts=t_roll, dims=1)
            if t_roll < 0:
                mask_f[:, t_roll:] = 0
            else:
                mask_f[:, :t_roll] = 0
        # Mask the embed-roll wrap-around edge too (position t uses embed_{t-e_roll};
        # the wrapped positions are invalid). For the default (-1,0,-1) this coincides
        # with the teacher-roll edge above, but keep it explicit so any roll is safe.
        if e_roll:
            if e_roll < 0:
                mask_f[:, e_roll:] = 0
            else:
                mask_f[:, :e_roll] = 0
        mask_f = mask_f.float()

        d2t = getattr(self.eagle_trainer.draft_model.eagle_module, "d2t", None)
        use_d2t = d2t is not None and d2t.numel() == draft_v and int(d2t.abs().sum()) > 0
        sel_idx = (
            (torch.arange(draft_v, device=draft_pr.device) + d2t.to(draft_pr.device))
            if use_d2t else None
        )

        # token_budget caps the per-chunk sequence length (chunk = token_budget // B), bounding
        # the transient teacher-select buffer [B, chunk, draft_vocab]. 1024 is safe; override with
        # EAGLE_LOSS_TOKEN_BUDGET. The soft-CE itself lives in eagle3_distill_loss (unit-tested).
        token_budget = int(os.environ.get("EAGLE_LOSS_TOKEN_BUDGET", "1024"))
        draft_loss = eagle3_distill_loss(
            draft_pr, teacher_pr, mask_f, draft_v, sel_idx, token_budget, t_roll
        )

        total_loss = policy_loss + self.eagle_trainer.config.loss_weight * draft_loss
        metrics["actor/eagle_draft_loss"] = draft_loss.detach().item()

        return total_loss, metrics
