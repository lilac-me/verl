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

"""Eagle3 draft model distillation loss.

Implements forward-KL soft cross-entropy:
  L_draft = -E[Σ_v p_teacher(v|x) log p_draft(v|x)]

where p_teacher is the RL policy (teacher/target) and p_draft is the Eagle
draft model (student).  Gradients flow only through the draft model — the
teacher logits are detached.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def eagle_draft_loss(
    draft_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    response_mask: torch.Tensor,
    loss_weight: float = 1.0,
) -> torch.Tensor:
    """Compute the soft cross-entropy distillation loss for the Eagle draft model.

    Implements the forward-KL objective:
        L = -Σ_t mask[t] · Σ_v softmax(teacher[t,v]) · log_softmax(draft[t,v])

    normalized by the number of valid tokens.

    The one-token time-step shift (Eagle3 alignment) must be applied to
    ``draft_logits`` and ``teacher_logits`` BEFORE calling this function:
    both should already be rolled so that position t predicts position t+1.

    Args:
        draft_logits:   [batch, seq, vocab]  — draft model output logits.
        teacher_logits: [batch, seq, vocab]  — policy logits (will be detached).
        response_mask:  [batch, seq]         — 1 for valid response tokens, 0 for padding/prompt.
        loss_weight:    Scalar multiplier applied to the returned loss.

    Returns:
        Scalar loss tensor (already multiplied by loss_weight).
    """
    teacher_logits = teacher_logits.detach()

    # Compute per-token forward-KL
    teacher_probs = F.softmax(teacher_logits, dim=-1)       # [batch, seq, vocab]
    student_log_probs = F.log_softmax(draft_logits, dim=-1)  # [batch, seq, vocab]
    per_token_loss = -(teacher_probs * student_log_probs).sum(dim=-1)  # [batch, seq]

    # Mask padding + prompt tokens, then average over valid response tokens
    mask = response_mask.bool()
    num_valid = mask.sum().clamp(min=1)
    loss = (per_token_loss * mask).sum() / num_valid

    return loss_weight * loss


def roll_for_eagle_alignment(
    draft_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    response_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply Eagle3 time-step alignment by rolling all tensors left by 1.

    Eagle3 trains position t of the draft to predict the policy's distribution
    at position t+1.  Rolling left by 1 achieves this while keeping tensor
    shapes identical.

    The last token position after rolling contains stale/wrap-around data and
    is excluded by zeroing the last column of the mask.

    Args:
        draft_logits:   [batch, seq, vocab]
        teacher_logits: [batch, seq, vocab]
        response_mask:  [batch, seq]

    Returns:
        Rolled (draft_logits, teacher_logits, response_mask) with the same shapes.
    """
    draft_logits = torch.roll(draft_logits, shifts=-1, dims=1)
    teacher_logits = torch.roll(teacher_logits, shifts=-1, dims=1)
    response_mask = torch.roll(response_mask, shifts=-1, dims=1)

    # Mask out the wrap-around position to prevent it from contributing to loss
    response_mask = response_mask.clone()
    response_mask[:, -1] = 0

    return draft_logits, teacher_logits, response_mask


def compute_eagle_draft_loss_with_alignment(
    draft_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    response_mask: torch.Tensor,
    loss_weight: float = 1.0,
) -> torch.Tensor:
    """Full Eagle3 draft loss with built-in time-step alignment.

    Convenience wrapper: applies ``roll_for_eagle_alignment`` then
    ``eagle_draft_loss``.

    Args:
        draft_logits:   [batch, seq, vocab]  — draft model raw logits.
        teacher_logits: [batch, seq, vocab]  — policy raw logits.
        response_mask:  [batch, seq]         — valid response token mask.
        loss_weight:    Multiplier for the returned scalar.

    Returns:
        Scalar draft distillation loss.
    """
    draft_logits, teacher_logits, response_mask = roll_for_eagle_alignment(
        draft_logits, teacher_logits, response_mask
    )
    return eagle_draft_loss(draft_logits, teacher_logits, response_mask, loss_weight)
