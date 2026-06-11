# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
"""EAGLE3 draft loss for verl Megatron engine.

Loss (per NeMo-RL PR #2078 docs, identical alignment):
    draft position t consumes (hidden_t, embed_{t+1}) and is trained to match
    the DETACHED teacher distribution for position t+1:

        L_draft = E_t[ -sum_v softmax(z_teacher[t+1])_v * log_softmax(z_draft[t])_v ]
        L_total = L_policy + loss_weight * L_draft

    Implementation detail: teacher and student are first restricted to the
    draft vocabulary via the d2t offset map (target_id = draft_id +
    d2t[draft_id]) and the teacher softmax is taken over those columns —
    matching how vLLM's eagle3 drafter scores tokens at inference time.

TP handling: the policy's logits are vocab-parallel [*, V/tp]. We gather only
the d2t-mapped columns (draft_vocab of them) via a masked local scatter +
all_reduce(SUM) over the TP group: memory cost s*b*draft_vocab, no full-vocab
gather. The student lm_head is TP-replicated (see eagle.py), so the student
side needs no communication.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor


def roll_left_seq(t: Tensor, seq_dim: int = 0) -> Tensor:
    """Shift left by one along the sequence dim; last position becomes garbage
    and MUST be masked by the caller. CP=1 only (phase 1); for CP>1 use
    megatron.core.transformer.multi_token_prediction.roll_tensor with cp_group.
    """
    return torch.roll(t, shifts=-1, dims=seq_dim)


@torch.no_grad()
def gather_teacher_logits_for_draft_vocab(
    vp_logits: Tensor,            # [s, b, V_local] vocab-parallel teacher logits (detached upstream)
    d2t: Tensor,                  # [draft_vocab] long, offsets: target_id = i + d2t[i]
    tp_rank: int,
    tp_world_size: int,
    tp_group: Optional[dist.ProcessGroup],
) -> Tensor:
    """Return [s, b, draft_vocab] teacher logits at the d2t-mapped target columns.

    Each TP rank owns target-vocab columns [tp_rank*V_local, (tp_rank+1)*V_local).
    NOTE on padded vocab: Megatron pads V to a multiple of tp*divisor; padded
    columns never appear in d2t targets (d2t targets < true target vocab), so
    they are naturally excluded.
    """
    s, b, v_local = vp_logits.shape
    draft_vocab = d2t.numel()
    device = vp_logits.device

    target_ids = torch.arange(draft_vocab, device=device, dtype=torch.long) + d2t.to(device)
    lo = tp_rank * v_local
    hi = lo + v_local
    local_mask = (target_ids >= lo) & (target_ids < hi)

    out = torch.zeros(s, b, draft_vocab, dtype=torch.float32, device=device)
    if local_mask.any():
        local_cols = (target_ids[local_mask] - lo)
        out[..., local_mask] = vp_logits[..., local_cols].to(torch.float32)

    if tp_world_size > 1 and tp_group is not None:
        dist.all_reduce(out, op=dist.ReduceOp.SUM, group=tp_group)
    return out


def draft_soft_ce_loss(
    student_logits: Tensor,       # [s, b, draft_vocab], requires_grad -> draft params
    teacher_logits: Tensor,       # [s, b, draft_vocab], fp32, detached
    loss_mask: Tensor,            # [s, b] bool/float — valid NEXT-TOKEN positions, already shifted
    global_valid_toks: Optional[Tensor] = None,
) -> Tensor:
    """Forward-KL-equivalent soft cross-entropy, token-mean normalized.

    Normalization: divide by global_valid_toks when provided (matches verl's
    global-token loss normalization so draft loss scales identically to the
    policy loss across DP/mbs); otherwise local masked mean.
    """
    teacher_logits = teacher_logits.detach()
    log_q = F.log_softmax(student_logits.to(torch.float32), dim=-1)
    p = F.softmax(teacher_logits, dim=-1)
    per_tok = -(p * log_q).sum(dim=-1)                  # [s, b]
    mask = loss_mask.to(per_tok.dtype)
    masked_sum = (per_tok * mask).sum()
    if global_valid_toks is not None:
        return masked_sum / torch.clamp(global_valid_toks.to(per_tok.dtype), min=1.0)
    return masked_sum / torch.clamp(mask.sum(), min=1.0)


@dataclass
class DraftLossState:
    """Per-microbatch carrier between forward_step and postprocess.

    forward_step:
      1. logits_processor stashes `teacher_logits_draft_vocab` (detached,
         d2t-gathered, rolled left, last position zeroed).
      2. after forward_fn returns: capture-gather (ALL PP ranks), draft
         forward on the owner rank, stash `student_logits`.
    postprocess_micro_batch_func:
      3. compute() adds loss_weight * draft_soft_ce to the policy loss.
    """

    loss_weight: float = 1.0
    student_logits: Optional[Tensor] = None
    teacher_logits: Optional[Tensor] = None     # already rolled + d2t-restricted
    loss_mask: Optional[Tensor] = None          # already rolled
    global_valid_toks: Optional[Tensor] = None
    metrics: dict = field(default_factory=dict)

    def ready(self) -> bool:
        return self.student_logits is not None and self.teacher_logits is not None

    def compute(self) -> Optional[Tensor]:
        if not self.ready():
            return None
        teacher = self.teacher_logits
        # auto-align: student is [s, b, Vd] (mcore layout); teacher may arrive
        # [b, s, Vd] from the engine's logits_processor (bshd batches).
        if teacher.shape[:2] != self.student_logits.shape[:2] and \
                teacher.shape[:2] == self.student_logits.shape[:2][::-1]:
            teacher = teacher.transpose(0, 1)
        loss = draft_soft_ce_loss(
            self.student_logits, teacher, self.loss_mask, self.global_valid_toks
        )
        self.metrics["actor/draft_loss"] = loss.detach().item()
        return self.loss_weight * loss

    def clear(self):
        self.student_logits = None
        self.teacher_logits = None
        self.loss_mask = None
