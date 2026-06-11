# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
"""HF <-> trainer weight mapping for the EAGLE3 draft model.

Because EagleDraftModel parameter names mirror the public HF EAGLE3 layout
(fc / midlayer.* / norm / lm_head / d2t), load and export are near-identity:
only key normalization (optional "model." prefix, "layers.0." aliases) and
dtype handling are needed. Export feeds vLLM's LlamaForCausalLMEagle3
load_weights() directly via the `draft.` prefix in the refit stream.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator

import torch
from torch import Tensor


# ---------------------------------------------------------------------- #
# checkpoint loading
# ---------------------------------------------------------------------- #
def _load_state_dict_from_dir(ckpt_dir: str) -> dict[str, Tensor]:
    p = Path(ckpt_dir)
    index = p / "model.safetensors.index.json"
    single_st = p / "model.safetensors"
    single_pt = p / "pytorch_model.bin"
    state: dict[str, Tensor] = {}
    if index.exists():
        from safetensors.torch import load_file

        with open(index) as f:
            weight_map = json.load(f)["weight_map"]
        for shard in sorted(set(weight_map.values())):
            state.update(load_file(str(p / shard)))
    elif single_st.exists():
        from safetensors.torch import load_file

        state = load_file(str(single_st))
    elif single_pt.exists():
        state = torch.load(str(single_pt), map_location="cpu", weights_only=True)
    else:
        # some EAGLE releases ship a bare .pt / .bin with another name
        cands = list(p.glob("*.safetensors")) + list(p.glob("*.bin")) + list(p.glob("*.pt"))
        if not cands:
            raise FileNotFoundError(f"No checkpoint files found under {ckpt_dir}")
        if cands[0].suffix == ".safetensors":
            from safetensors.torch import load_file

            state = load_file(str(cands[0]))
        else:
            state = torch.load(str(cands[0]), map_location="cpu", weights_only=True)
    return state


def _normalize_key(k: str) -> str:
    """Map the known EAGLE3 checkpoint dialects onto our module names."""
    k = k.removeprefix("model.")
    # SpecForge / vLLM-internal dialect: layers.0.* == midlayer.*
    k = k.replace("layers.0.", "midlayer.")
    # some checkpoints name the fusion norm differently
    k = k.replace("midlayer.hidden_layernorm.", "midlayer.hidden_norm.")
    return k


def load_hf_weights_to_eagle(draft_model, ckpt_dir: str, strict_d2t: bool = True) -> tuple[list[str], list[str]]:
    """Load an HF EAGLE3 checkpoint into EagleDraftModel.

    Returns (missing_keys, unexpected_keys). `lm_head.weight` may legitimately
    be missing — the caller should then run init_draft_lm_head_from_policy().
    `embed_tokens` in the checkpoint is IGNORED: the trainer uses the policy's
    captured embeddings, and vLLM ties the drafter's embeds to the target.
    """
    raw = _load_state_dict_from_dir(ckpt_dir)
    state = {}
    for k, v in raw.items():
        nk = _normalize_key(k)
        if nk.startswith("embed_tokens"):
            continue
        state[nk] = v

    if "d2t" in state:
        state["d2t"] = state["d2t"].to(torch.long)
    elif strict_d2t and draft_model.draft_vocab_size != draft_model.target_vocab_size:
        raise ValueError(
            f"Checkpoint at {ckpt_dir} has draft_vocab={draft_model.draft_vocab_size} != "
            f"target_vocab={draft_model.target_vocab_size} but no d2t buffer."
        )
    # t2d is inference-side only; drop if present.
    state.pop("t2d", None)

    missing, unexpected = draft_model.load_state_dict(state, strict=False)
    missing = [m for m in missing]
    unexpected = [u for u in unexpected]
    return missing, unexpected


# ---------------------------------------------------------------------- #
# lm_head init from policy (when checkpoint lacks lm_head.weight)
# ---------------------------------------------------------------------- #
@torch.no_grad()
def init_draft_lm_head_from_policy(draft_model, policy_output_layer_weight_local: Tensor,
                                   tp_group=None) -> None:
    """Initialize the draft lm_head from the policy's output layer.

    policy_output_layer_weight_local: the LOCAL vocab-parallel shard
    [V_padded/tp, h] of the policy lm_head (mcore output_layer.weight).
    Gathers full rows across TP, drops vocab padding, then d2t-selects the
    draft-vocab rows.
    """
    import torch.distributed as dist

    w_local = policy_output_layer_weight_local
    if tp_group is not None and dist.get_world_size(group=tp_group) > 1:
        shards = [torch.empty_like(w_local) for _ in range(dist.get_world_size(group=tp_group))]
        dist.all_gather(shards, w_local.contiguous(), group=tp_group)
        w_full = torch.cat(shards, dim=0)  # [V_padded, h]
    else:
        w_full = w_local

    d2t = draft_model.d2t
    target_ids = torch.arange(d2t.numel(), device=w_full.device) + d2t.to(w_full.device)
    assert int(target_ids.max()) < w_full.shape[0], "d2t maps outside policy vocab (padding mismatch?)"
    draft_model.lm_head.weight.copy_(w_full[target_ids].to(draft_model.lm_head.weight.dtype))


# ---------------------------------------------------------------------- #
# export for vLLM refit
# ---------------------------------------------------------------------- #
DRAFT_WEIGHT_PREFIX = "draft."


def export_eagle_weights_to_hf(draft_model) -> Iterator[tuple[str, Tensor]]:
    """Yield (hf_name, tensor) for the vLLM eagle3 drafter, WITHOUT prefix.

    The caller wraps names with DRAFT_WEIGHT_PREFIX before merging into the
    policy per-tensor stream. embed_tokens is intentionally not exported
    (vLLM resolves it from the target model). Only TP rank 0's copy needs to
    be sent (params are TP-replicated); the engine integration handles
    rank gating.
    """
    for name, param in draft_model.state_dict().items():
        if name.startswith("embed_tokens"):
            continue
        yield name, param.detach()


def split_draft_weights(weights: list[tuple[str, Tensor]]) -> tuple[list, list]:
    """Partition a received weight list into (policy_weights, draft_weights),
    stripping the draft prefix. vLLM-worker side helper."""
    policy, draft = [], []
    for name, t in weights:
        if name.startswith(DRAFT_WEIGHT_PREFIX):
            draft.append((name[len(DRAFT_WEIGHT_PREFIX):], t))
        else:
            policy.append((name, t))
    return policy, draft
