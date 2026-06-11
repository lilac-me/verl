# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
"""Configuration for EAGLE3 online draft training.

Wire-up (omegaconf):

    actor_rollout_ref:
      actor:
        draft:
          enable: false
          model_path: null          # HF-format EAGLE3 ckpt dir; null => random init
          loss_weight: 1.0
          aux_layer_indices: null   # null => (1, L//2-1, L-4), vLLM convention
          temperature_scaled_teacher: false
          detach_hidden_states: true
      rollout:
        speculative:
          method: eagle3
          model: ${actor_rollout_ref.actor.draft.model_path}
          num_speculative_tokens: 3
          draft_tensor_parallel_size: 1
        update_draft_weights: true
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DraftModelConfig:
    enable: bool = False
    model_path: Optional[str] = None
    loss_weight: float = 1.0
    # None => default (1, num_layers // 2 - 1, num_layers - 4)
    aux_layer_indices: Optional[tuple[int, ...]] = None
    # If True, the teacher distribution uses the same temperature scaling as
    # the policy loss / rollout sampling. NeMo-RL uses unscaled logits
    # (False). Keep False for parity; flip for ablation.
    temperature_scaled_teacher: bool = False
    # Must stay True in production: guarantees the draft loss has zero
    # gradient into the policy backbone. Exposed only for ablations.
    detach_hidden_states: bool = True
    # Draft optimizer (separate from the Megatron distributed optimizer:
    # the draft model is built standalone so it never enters the policy's DDP
    # grad buffers; grads are manually all-reduced over TP+DP by the plugin).
    lr: float = 1.0e-4
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    # Architecture overrides; normally inferred from the HF checkpoint config.
    num_layers: int = 1
    draft_vocab_size: Optional[int] = None
    # Internal/derived
    target_hidden_size: Optional[int] = field(default=None)


def validate_draft_config(draft_cfg: DraftModelConfig, *, strategy: str, engine_config, rollout_cfg) -> None:
    """Fail fast on unsupported combinations (phase-1 restrictions).

    Mirrors the restrictions of NeMo-RL PR #2078 plus verl specifics.
    Call from the engine/worker init before building models.
    """
    if not draft_cfg.enable:
        return

    if strategy not in ("megatron",):
        raise ValueError(f"actor.draft.enable=true requires actor.strategy=megatron, got {strategy!r}")

    # Phase 1: the per-position roll/shift in the draft loss assumes one
    # contiguous sequence per row. Packed/rmpad layouts need a segment-wise
    # roll over cu_seqlens (phase 2).
    if getattr(engine_config, "use_remove_padding", False):
        raise ValueError(
            "actor.draft.enable=true does not support use_remove_padding/sequence packing yet "
            "(roll-by-one must become per-segment over cu_seqlens). Disable remove_padding."
        )

    # Phase 1: the bring-up EagleDecoderLayer implements plain causal SDPA
    # which is wrong under CP sharding. Either set CP=1 or swap in the
    # mcore-TransformerLayer implementation (see eagle.py TODO).
    cp = getattr(engine_config, "context_parallel_size", 1) or 1
    if cp > 1:
        raise ValueError("actor.draft.enable=true requires context_parallel_size=1 in phase 1.")

    if rollout_cfg is not None:
        spec = getattr(rollout_cfg, "speculative", None)
        update_draft = getattr(rollout_cfg, "update_draft_weights", False)
        if update_draft:
            if spec is None or getattr(spec, "method", None) != "eagle3":
                raise ValueError(
                    "rollout.update_draft_weights=true requires rollout.speculative.method=eagle3"
                )
        if spec is not None and getattr(spec, "method", None) == "eagle3":
            if spec.model is None and not draft_cfg.enable:
                raise ValueError(
                    "rollout.speculative.model is null and no trainer-owned draft model is enabled; "
                    "vLLM would start with an uninitialized drafter."
                )

    # use_fused_kernels / linear-CE fused logprob paths never materialize the
    # full logits tensor that the draft teacher needs.
    if getattr(engine_config, "use_fused_kernels", False):
        raise ValueError(
            "actor.draft.enable=true is incompatible with use_fused_kernels=true: "
            "the fused path does not materialize teacher logits. "
            "(Phase 2: fused d2t-column gather.)"
        )
