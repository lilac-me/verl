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

"""Eagle3 draft-model distillation loss functions.

The training objective is to minimise the forward KL divergence between the
draft model's output distribution and the current policy's distribution:

    L_draft = -Σ_t mask[t] · Σ_v π_policy(v|t) · log π_draft(v|t)
              / Σ_t mask[t]

where π_policy comes from the policy's LM-head logits (detached — no gradient
flows back to the policy) and π_draft comes from the Eagle3 draft model.

Eagle3 time-step alignment
--------------------------
The draft model at position t is trained to predict the policy's distribution
at position t+1.  This is achieved by:

1. ``roll_inputs_embeds``: shift input embeddings left by one before the draft
   forward so that position t sees embedding[t+1].
2. ``roll_for_eagle_alignment``: after the draft forward, shift draft logits,
   teacher logits, and the response mask left by one, then zero the last mask
   position to exclude the wrap-around artifact.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def roll_for_eagle_alignment(
    draft_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    response_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Left-shift logits and mask by one for Eagle3 time-step alignment.

    After rolling, position t of draft_logits aligns with position t of
    teacher_logits — both now representing predictions for token t+1.
    The last position receives mask=0 to exclude the wrap-around artifact.

    Args:
        draft_logits:   [batch, seq, vocab]
        teacher_logits: [batch, seq, vocab]
        response_mask:  [batch, seq] — bool or float, 1 for valid response tokens

    Returns:
        Aligned (draft_logits, teacher_logits, response_mask) with the same shapes.
    """
    draft_logits = torch.roll(draft_logits, shifts=-1, dims=1)
    teacher_logits = torch.roll(teacher_logits, shifts=-1, dims=1)
    response_mask = torch.roll(response_mask, shifts=-1, dims=1).clone()
    response_mask[:, -1] = 0  # zero out wrap-around
    return draft_logits, teacher_logits, response_mask


def compute_eagle_draft_loss(
    draft_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute soft cross-entropy (forward KL) distillation loss.

    Args:
        draft_logits:   [batch, seq, vocab]  — from draft model, requires_grad=True
        teacher_logits: [batch, seq, vocab]  — from policy LM head, already detached
        response_mask:  [batch, seq]         — 1 for response tokens, 0 for prompt/pad

    Returns:
        Scalar loss tensor.
    """
    teacher_probs = F.softmax(teacher_logits, dim=-1)          # stop-gradient
    student_log_probs = F.log_softmax(draft_logits, dim=-1)
    per_token = -(teacher_probs * student_log_probs).sum(dim=-1)  # [batch, seq]

    mask = response_mask.float()
    num_valid = mask.sum().clamp(min=1)
    return (per_token * mask).sum() / num_valid


def compute_eagle_draft_loss_with_alignment(
    draft_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    response_mask: torch.Tensor,
    loss_weight: float = 0.1,
) -> torch.Tensor:
    """Apply Eagle3 alignment roll then compute the distillation loss.

    Combines ``roll_for_eagle_alignment`` and ``compute_eagle_draft_loss``
    into a single convenience function called from ``EagleLossWrapper``.

    Args:
        draft_logits:   [batch, seq, vocab]  — draft model output, in float32
        teacher_logits: [batch, seq, vocab]  — policy LM-head output, detached
        response_mask:  [batch, seq]
        loss_weight:    λ applied to the returned loss (does not affect gradients)

    Returns:
        Scaled scalar draft loss: λ × L_draft.
    """
    d, t, m = roll_for_eagle_alignment(draft_logits, teacher_logits, response_mask)
    loss = compute_eagle_draft_loss(d, t, m)
    return loss_weight * loss
