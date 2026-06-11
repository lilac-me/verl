# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
"""vLLM-side EAGLE3 utilities for verl rollout workers.

Sits next to verl/workers/rollout/vllm_rollout/utils.py, which already contains
the MTP drafter-sync machinery (_iter_all_models / _use_mtp_drafter_weight_sync).
EAGLE3 differs from MTP: the drafter has its OWN trainer-owned weights arriving
under a "draft." prefix, instead of receiving a copy of the actor weights.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any

import torch

logger = logging.getLogger(__name__)

DRAFT_WEIGHT_PREFIX = "draft."


# ---------------------------------------------------------------------- #
# engine speculative_config (mirror of build_mtp_speculative_config)
# ---------------------------------------------------------------------- #
def build_eagle3_speculative_config(
    model: str,
    num_speculative_tokens: int,
    draft_tensor_parallel_size: int = 1,
    engine_speculative_config: Any = None,
) -> dict:
    """Build vLLM's eagle3 speculative config, applying engine_kwargs overrides."""
    if engine_speculative_config is None:
        engine_speculative_config = {}
    if isinstance(engine_speculative_config, str):
        engine_speculative_config = json.loads(engine_speculative_config)
    if not isinstance(engine_speculative_config, Mapping):
        raise TypeError(
            "rollout.engine_kwargs.vllm.speculative_config must be a mapping when eagle3 rollout is enabled"
        )
    cfg = {
        "method": "eagle3",
        "model": model,
        "num_speculative_tokens": num_speculative_tokens,
        "draft_tensor_parallel_size": draft_tensor_parallel_size,
        **{k: v for k, v in engine_speculative_config.items() if v is not None},
    }
    if cfg["model"] is None:
        raise ValueError(
            "eagle3 speculative decoding requires a draft checkpoint path at startup "
            "(rollout.speculative.model). With online training, point it at the same "
            "checkpoint as actor.draft.model_path."
        )
    return cfg


# ---------------------------------------------------------------------- #
# weight splitting / trimming / loading  (call from _update_weights)
# ---------------------------------------------------------------------- #
def split_draft_weights(weights: list[tuple[str, torch.Tensor]]) -> tuple[list, list]:
    policy, draft = [], []
    for name, t in weights:
        if name.startswith(DRAFT_WEIGHT_PREFIX):
            draft.append((name[len(DRAFT_WEIGHT_PREFIX):], t))
        else:
            policy.append((name, t))
    return policy, draft


def _get_eagle3_drafter_model(model_runner):
    spec = model_runner.vllm_config.speculative_config
    if spec is None or getattr(spec, "method", None) != "eagle3":
        return None
    drafter = getattr(model_runner, "drafter", None)
    return getattr(drafter, "model", None) if drafter is not None else None


def trim_vocab_padding(
    draft_model: torch.nn.Module, draft_weights: list[tuple[str, torch.Tensor]]
) -> list[tuple[str, torch.Tensor]]:
    """Trim trainer-side padded vocab rows to match vLLM drafter shapes.

    Megatron pads vocab dims for divisibility; vLLM's drafter lm_head uses the
    exact draft_vocab_size. Match by destination parameter shape (port of
    NeMo-RL _trim_vocab_padding).
    """
    dst_shapes = {name: p.shape for name, p in draft_model.named_parameters()}
    # vLLM remaps midlayer. -> model.layers.0. internally; try both keys.
    out = []
    for key, tensor in draft_weights:
        dst = dst_shapes.get(key) or dst_shapes.get(key.replace("midlayer.", "model.layers.0."))
        if dst is not None and tensor.dim() >= 1 and tensor.shape[0] > dst[0] and tensor.shape[1:] == dst[1:]:
            tensor = tensor[: dst[0]]
        out.append((key, tensor))
    return out


def load_draft_weights(model_runner, draft_weights: list[tuple[str, torch.Tensor]]) -> int:
    """Load trainer-owned eagle3 weights into the vLLM drafter. Returns count."""
    if not draft_weights:
        return 0
    draft_model = _get_eagle3_drafter_model(model_runner)
    if draft_model is None:
        logger.warning("[eagle3] received %d draft weights but no eagle3 drafter is active; skipped.",
                       len(draft_weights))
        return 0
    draft_weights = trim_vocab_padding(draft_model, draft_weights)
    # buffers (d2t) are not parameters; load_weights handles named buffers in
    # recent vLLM eagle3 modules. TODO(verify): on your pinned vllm/vllm-ascend
    # version, confirm "d2t" appears in the drafter's load_weights mapping;
    # if not, set it manually:
    #   draft_model.get_buffer("d2t").copy_(...)
    loaded = draft_model.load_weights(weights=iter(draft_weights))
    n = len(loaded) if loaded is not None else len(draft_weights)
    logger.info("[eagle3] drafter refit: loaded %d tensors.", n)
    return n


# ---------------------------------------------------------------------- #
# lm_head ownership patch (version-gated, runtime monkey-patch)
# ---------------------------------------------------------------------- #
def maybe_patch_eagle3_lm_head_ownership() -> None:
    """Ensure the vLLM eagle3 drafter owns a loadable lm_head.

    Some vLLM versions tie the drafter lm_head to the target model when
    draft_vocab_size == vocab_size and skip loading a drafter-owned head,
    which breaks online refit (NeMo-RL patches llama_eagle3.py on disk; we
    patch the class at runtime instead, consistent with VLLMHijack style).

    TODO(verify): inspect your pinned vllm-ascend's eagle3 module — vllm-ascend
    forks model files, so this patch may be unnecessary or need an
    ascend-specific target class.
    """
    try:
        from vllm.model_executor.models import llama_eagle3  # noqa: F401
    except ImportError:
        logger.info("[eagle3] llama_eagle3 module not found; skipping lm_head patch.")
        return

    cls = getattr(llama_eagle3, "Eagle3LlamaForCausalLM", None) or getattr(
        llama_eagle3, "LlamaForCausalLMEagle3", None
    )
    if cls is None:
        logger.warning("[eagle3] could not locate eagle3 model class; lm_head patch skipped.")
        return

    if getattr(cls, "_verl_eagle3_lm_head_patch", False):
        return

    orig_load_weights = cls.load_weights

    def load_weights(self, weights):
        # Force drafter-owned lm_head loading even when vocab sizes match.
        # Strategy: intercept and stage lm_head.weight, let the rest pass
        # through, then copy into the (possibly tied) head parameter.
        staged_lm_head = None

        def gen():
            nonlocal staged_lm_head
            for name, w in weights:
                if name == "lm_head.weight":
                    staged_lm_head = w
                    continue
                yield name, w

        loaded = orig_load_weights(self, gen())
        if staged_lm_head is not None:
            head = self.lm_head.weight if hasattr(self, "lm_head") else None
            if head is not None:
                rows = min(head.shape[0], staged_lm_head.shape[0])
                head.data[:rows].copy_(staged_lm_head[:rows].to(head.dtype))
                if isinstance(loaded, set):
                    loaded.add("lm_head.weight")
        return loaded

    cls.load_weights = load_weights
    cls._verl_eagle3_lm_head_patch = True
    logger.info("[eagle3] applied runtime lm_head ownership patch to %s.", cls.__name__)
