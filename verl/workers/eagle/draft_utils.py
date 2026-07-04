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

"""Eagle3 draft-model utilities: weight loading, export, and LM-head sync."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

from megatron.core import parallel_state
from megatron.core.transformer import MegatronModule, TransformerConfig
from megatron.training.utils import unwrap_model
from verl.workers.eagle.draft_model import EagleDraftModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def _load_hf_checkpoint(model_path: str) -> dict[str, torch.Tensor]:
    p = Path(model_path)
    if p.is_file():
        return _load_file(p)
    if p.is_dir():
        for name in ("model.safetensors", "pytorch_model.bin"):
            if (p / name).exists():
                return _load_file(p / name)
    raise FileNotFoundError(f"[eagle] No checkpoint found at '{model_path}'")


def _load_file(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(str(path))
    obj = torch.load(str(path), map_location="cpu", weights_only=True)
    if isinstance(obj, dict) and "state_dict" in obj:
        obj = obj["state_dict"]
    return {k: v for k, v in obj.items() if isinstance(v, torch.Tensor)}


def _all_gather_tp_shards(local_weight: torch.Tensor) -> List[torch.Tensor]:
    tp_group = parallel_state.get_tensor_model_parallel_group()
    tp_world_size = parallel_state.get_tensor_model_parallel_world_size()
    if tp_world_size == 1:
        return [local_weight]
    gathered = [torch.empty_like(local_weight) for _ in range(tp_world_size)]
    dist.all_gather(gathered, local_weight.contiguous(), group=tp_group)
    return gathered


def _gather_tp_weight(local_weight: torch.Tensor, dim: int) -> torch.Tensor:
    gathered = _all_gather_tp_shards(local_weight)
    if len(gathered) == 1:
        return local_weight
    return torch.cat(gathered, dim=dim).contiguous()


def _shard_tp_weight(local_tensor: torch.Tensor, dim: int) -> torch.Tensor:
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    tp_world_size = parallel_state.get_tensor_model_parallel_world_size()
    if tp_world_size == 1:
        return local_tensor
    return torch.chunk(local_tensor, tp_world_size, dim=dim)[tp_rank].contiguous()


class _AllReduceSumG(torch.autograd.Function):
    """Differentiable all-reduce(SUM) over the TP group (Megatron's "g" operator).

    forward:  y = Σ_ranks x_r   (replicated on every rank)
    backward: grad_x_r = grad_y  (identity, no communication)

    Correct because the distillation loss is REPLICATED across TP ranks: every rank
    computes the same scalar loss, so grad_y is identical on all ranks and each rank's
    partial x_r simply receives it. This lets a vocab-parallel soft-CE pass gradient back
    to each rank's draft-logit shard WITHOUT gathering the full vocab.
    """

    @staticmethod
    def forward(ctx, x, tp_world_size, tp_group):
        if tp_world_size == 1:
            return x
        out = x.contiguous().clone()
        dist.all_reduce(out, op=dist.ReduceOp.SUM, group=tp_group)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out, None, None


def all_reduce_sum_grad(x: torch.Tensor) -> torch.Tensor:
    """Differentiable TP all-reduce(SUM) (see _AllReduceSumG). Identity at TP=1."""
    tp_world_size = parallel_state.get_tensor_model_parallel_world_size()
    if tp_world_size == 1:
        return x
    tp_group = parallel_state.get_tensor_model_parallel_group()
    return _AllReduceSumG.apply(x, tp_world_size, tp_group)


def build_vocab_parallel_select_plan(
    sel_idx: torch.Tensor, teacher_shard_width: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Plan to select teacher columns ``teacher_full[sel_idx]`` under vocab parallelism.

    Under TP, each rank owns the full-vocab columns ``[rank*W, (rank+1)*W)`` where
    ``W = teacher_shard_width = padded_vocab/TP``. For THIS rank, returns ``(owns, local_cols)``:
    ``owns`` is a bool mask over the draft vocab marking the draft columns ``j`` whose teacher
    target ``sel_idx[j]`` is owned by this rank, and ``local_cols`` the corresponding LOCAL teacher
    indices (``sel_idx[j] - rank*W``). Caller does ``buf[..., owns] = teacher_shard[..., local_cols]``
    then ``all_reduce(SUM)`` across TP to assemble the full ``[..., draft_vocab]`` selected teacher
    (each draft column is contributed by exactly its one owning rank). Plan depends only on
    ``sel_idx`` + this rank → precompute once, reuse across chunks/steps.
    """
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    lo = tp_rank * teacher_shard_width
    owns = (sel_idx >= lo) & (sel_idx < lo + teacher_shard_width)
    local_cols = sel_idx[owns] - lo
    return owns, local_cols


def _fuse_qkv_for_megatron(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_attention_heads: int,
    num_query_groups: int,
    head_dim: int,
) -> torch.Tensor:
    """Fuse separate q/k/v projections into Megatron's ``linear_qkv`` layout.

    Megatron's ``SelfAttention`` reshapes ``linear_qkv`` as
    ``[ng, (np // ng + 2) * hd, in]`` and reads q/k/v PER QUERY GROUP, so the rows
    must be interleaved ``[q_group0, k0, v0, q_group1, k1, v1, ...]`` — exactly the
    layout the mcore loader builds (see ``verl/models/mcore/loader.py``).

    A flat ``cat([all_q, all_k, all_v])`` is WRONG: with GQA (np != ng) it places
    query heads into the wrong group's k/v slots, and even with MHA (ng == np, one
    head per group) the per-group order differs. The total size matches either way,
    so the mistake loads silently and only corrupts attention numerically.

    Returns the FULL (un-sharded) fused weight ``[ng * (np//ng + 2) * hd, in]``.
    """
    if num_attention_heads % num_query_groups != 0:
        raise RuntimeError(
            f"[eagle] num_attention_heads={num_attention_heads} not divisible by "
            f"num_query_groups={num_query_groups}"
        )
    r = num_attention_heads // num_query_groups
    in_dim = q.shape[1]
    q_g = q.reshape(num_query_groups, r * head_dim, in_dim)
    k_g = k.reshape(num_query_groups, head_dim, in_dim)
    v_g = v.reshape(num_query_groups, head_dim, in_dim)
    fused = torch.cat([q_g, k_g, v_g], dim=1)  # [ng, (r+2)*hd, in]
    return fused.reshape(num_query_groups * (r + 2) * head_dim, in_dim).contiguous()


def _split_megatron_qkv(
    fused: torch.Tensor,
    num_attention_heads: int,
    num_query_groups: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Inverse of :func:`_fuse_qkv_for_megatron`.

    Recover ``(q, k, v)`` in head order from the FULL (TP-gathered) interleaved
    ``linear_qkv`` weight ``[ng * (np//ng + 2) * hd, in]``.
    """
    r = num_attention_heads // num_query_groups
    in_dim = fused.shape[1]
    grouped = fused.reshape(num_query_groups, (r + 2) * head_dim, in_dim)
    q = grouped[:, : r * head_dim, :].reshape(num_attention_heads * head_dim, in_dim).contiguous()
    k = grouped[:, r * head_dim : (r + 1) * head_dim, :].reshape(num_query_groups * head_dim, in_dim).contiguous()
    v = grouped[:, (r + 1) * head_dim : (r + 2) * head_dim, :].reshape(num_query_groups * head_dim, in_dim).contiguous()
    return q, k, v


def _fuse_and_shard_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_attention_heads: int,
    num_query_groups: int,
    head_dim: int,
    target: Optional[torch.Tensor],
) -> torch.Tensor:
    """Build the interleaved Megatron QKV weight and shard it for this TP rank.

    The interleaved layout places whole query groups contiguously, so the
    column-parallel shard is a plain chunk along dim 0 — provided ``num_query_groups``
    is divisible by the TP world size (kv-head replication for ng < TP is not
    supported by the draft and would corrupt the layout, so we fail loudly).
    """
    tp_world_size = parallel_state.get_tensor_model_parallel_world_size()
    if tp_world_size > 1 and num_query_groups % tp_world_size != 0:
        raise RuntimeError(
            f"[eagle] num_query_groups={num_query_groups} is not divisible by "
            f"TP={tp_world_size}; kv-head replication (num_query_groups < TP) is not "
            "supported for the Eagle3 draft QKV fusion."
        )
    fused_full = _fuse_qkv_for_megatron(q, k, v, num_attention_heads, num_query_groups, head_dim)
    fused = _shard_tp_weight(fused_full, dim=0)
    return fused.to(dtype=target.dtype) if target is not None else fused


def _get_tp_qkv_weight(
    local_fused_weight: torch.Tensor,
    q_dim: int,
    kv_dim: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather the interleaved ``linear_qkv`` across TP and de-interleave to q/k/v.

    Inverse of :func:`_fuse_and_shard_qkv`: ``_gather_tp_weight`` concatenates the
    per-rank group blocks back into the full interleaved weight (rank i holds the
    contiguous groups ``[i*ng/tp : (i+1)*ng/tp]``), then :func:`_split_megatron_qkv`
    recovers the per-head q/k/v that the unfused (Tengyunw/vLLM) checkpoint expects.
    """
    full = _gather_tp_weight(local_fused_weight, dim=0)
    num_attention_heads = q_dim // head_dim
    num_query_groups = kv_dim // head_dim
    return _split_megatron_qkv(full, num_attention_heads, num_query_groups, head_dim)


def _get_tp_gate_up_weight(
    local_fused_weight: torch.Tensor,
    ffn_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    shards = _all_gather_tp_shards(local_fused_weight)
    if len(shards) == 1 and local_fused_weight.shape[0] == 2 * ffn_size:
        return local_fused_weight.split([ffn_size, ffn_size], dim=0)

    tp_world_size = len(shards)
    if ffn_size % tp_world_size != 0:
        raise RuntimeError("ffn_size is not divisible by the tensor-parallel world size.")

    local_ffn_size = ffn_size // tp_world_size
    gate_shards, up_shards = [], []
    for shard in shards:
        gate_local, up_local = shard.split([local_ffn_size, local_ffn_size], dim=0)
        gate_shards.append(gate_local)
        up_shards.append(up_local)

    return (
        torch.cat(gate_shards, dim=0).contiguous(),
        torch.cat(up_shards, dim=0).contiguous(),
    )


# ---------------------------------------------------------------------------
# HF -> Megatron key mapping
# ---------------------------------------------------------------------------

_NO_TP_KEYS = {"eagle_module.fc.weight", "eagle_module.enorm.weight"}
_COL_PARALLEL_RE = re.compile(
    r"(self_attention\.linear_qkv|mlp\.linear_fc1|eagle_output_layer)\.weight$"
)
_ROW_PARALLEL_RE = re.compile(
    r"(self_attention\.linear_proj|mlp\.linear_fc2)\.weight$"
)


def _normalize_key(key: str) -> str:
    for prefix in ("draft.", "module.", "eagle_module."):
        while key.startswith(prefix):
            key = key[len(prefix):]
    return key


def _shard_for_tp(megatron_key: str, tensor: torch.Tensor, model_state: dict) -> torch.Tensor:
    target = model_state.get(megatron_key)
    if target is None or tensor.shape == target.shape or megatron_key in _NO_TP_KEYS:
        return tensor.to(dtype=target.dtype) if target is not None else tensor
    if _COL_PARALLEL_RE.search(megatron_key):
        return _shard_tp_weight(tensor, dim=0).to(dtype=target.dtype)
    if _ROW_PARALLEL_RE.search(megatron_key):
        return _shard_tp_weight(tensor, dim=1).to(dtype=target.dtype)
    return tensor


def _shard_gate_up(fused_weight: torch.Tensor, target: Optional[torch.Tensor]) -> torch.Tensor:
    if target is None or fused_weight.shape == target.shape:
        return fused_weight.to(dtype=target.dtype) if target is not None else fused_weight
    ffn_size = fused_weight.shape[0] // 2
    return torch.cat(
        [_shard_tp_weight(fused_weight[:ffn_size], dim=0), _shard_tp_weight(fused_weight[ffn_size:], dim=0)], dim=0
    ).to(dtype=target.dtype)


def _map_hf_to_megatron_eagle(
    hf_state: dict[str, torch.Tensor],
    model_state: dict[str, torch.Tensor],
    num_attention_heads: int,
    num_query_groups: int,
    head_dim: int,
) -> dict[str, torch.Tensor]:
    mapped: dict[str, torch.Tensor] = {}

    pending_q: dict[int, torch.Tensor] = {}
    pending_k: dict[int, torch.Tensor] = {}
    pending_v: dict[int, torch.Tensor] = {}
    pending_gate: dict[int, torch.Tensor] = {}
    pending_up: dict[int, torch.Tensor] = {}

    for raw_key, tensor in hf_state.items():
        key = _normalize_key(raw_key)

        if key == "fc.weight":
            megatron_key = "eagle_module.fc.weight"
            mapped[megatron_key] = _shard_for_tp(megatron_key, tensor, model_state)
            continue
        # enorm normalizes the EMBEDDING before layer-0 injection. In Eagle3
        # checkpoints this is the layer's `input_layernorm` (see below); an explicit
        # top-level `enorm` is also accepted.
        if key in ("enorm.weight",):
            mapped["eagle_module.enorm.weight"] = tensor
            continue
        # Bare hidden-norm names (no layer prefix) normalize the HIDDEN state →
        # the decoder layer-0 input_layernorm.
        if key in ("hidden_norm.weight", "hnorm.weight", "pre_fc_norm_hidden.weight"):
            mapped["eagle_module.decoder.layers.0.input_layernorm.weight"] = tensor
            continue
        if key == "norm.weight":
            mapped["eagle_module.decoder.final_layernorm.weight"] = tensor
            continue
        if key in ("lm_head.weight", "eagle_output_layer.weight"):
            megatron_key = "eagle_module.eagle_output_layer.weight"
            mapped[megatron_key] = _shard_for_tp(megatron_key, tensor, model_state)
            continue
        if key == "d2t":
            megatron_key = "eagle_module.d2t"
            if megatron_key in model_state:
                mapped[megatron_key] = tensor
            continue

        if key.startswith("midlayer."):
            layer_idx, sub = 0, key[len("midlayer."):]
        else:
            m = re.match(r"^layers\.(\d+)\.(.+)$", key)
            if not m:
                continue
            layer_idx, sub = int(m.group(1)), m.group(2)

        lp = f"eagle_module.decoder.layers.{layer_idx}"

        # Eagle3 layer norms (per vLLM llama_eagle3 reference):
        #   input_layernorm  -> normalizes the embedding   -> our enorm
        #   hidden_norm      -> normalizes the hidden state -> decoder input_layernorm
        #   post_attention_layernorm -> pre-MLP norm        -> pre_mlp_layernorm
        if sub == "input_layernorm.weight":
            mapped["eagle_module.enorm.weight"] = tensor
        elif sub in ("hidden_norm.weight", "hnorm.weight", "pre_fc_norm_hidden.weight"):
            mapped[f"{lp}.input_layernorm.weight"] = tensor
        elif sub == "post_attention_layernorm.weight":
            mapped[f"{lp}.pre_mlp_layernorm.weight"] = tensor
        elif sub == "self_attn.qkv_proj.weight":
            mk = f"{lp}.self_attention.linear_qkv.weight"
            # A pre-fused HF qkv_proj is flat [all_q | all_k | all_v]; split it back
            # to q/k/v and re-fuse into Megatron's per-group interleaved layout.
            q_dim = num_attention_heads * head_dim
            kv_dim = num_query_groups * head_dim
            q_f, k_f, v_f = tensor.split([q_dim, kv_dim, kv_dim], dim=0)
            mapped[mk] = _fuse_and_shard_qkv(
                q_f, k_f, v_f, num_attention_heads, num_query_groups, head_dim, model_state.get(mk)
            )
        elif sub == "self_attn.q_proj.weight":
            pending_q[layer_idx] = tensor
        elif sub == "self_attn.k_proj.weight":
            pending_k[layer_idx] = tensor
        elif sub == "self_attn.v_proj.weight":
            pending_v[layer_idx] = tensor
        elif sub == "self_attn.o_proj.weight":
            mk = f"{lp}.self_attention.linear_proj.weight"
            mapped[mk] = _shard_for_tp(mk, tensor, model_state)
        elif sub == "mlp.gate_up_proj.weight":
            mk = f"{lp}.mlp.linear_fc1.weight"
            mapped[mk] = _shard_gate_up(tensor, model_state.get(mk))
        elif sub == "mlp.gate_proj.weight":
            pending_gate[layer_idx] = tensor
        elif sub == "mlp.up_proj.weight":
            pending_up[layer_idx] = tensor
        elif sub == "mlp.down_proj.weight":
            mk = f"{lp}.mlp.linear_fc2.weight"
            mapped[mk] = _shard_for_tp(mk, tensor, model_state)

    # Fuse split QKV into Megatron's per-query-group interleaved linear_qkv layout
    # (NOT a flat [all_q|all_k|all_v] concat — see _fuse_qkv_for_megatron).
    for i in set(pending_q) | set(pending_k) | set(pending_v):
        q, k, v = pending_q.get(i), pending_k.get(i), pending_v.get(i)
        if None in (q, k, v):
            logger.warning(f"[eagle] Incomplete QKV for layer {i}")
            continue
        mk = f"eagle_module.decoder.layers.{i}.self_attention.linear_qkv.weight"
        mapped[mk] = _fuse_and_shard_qkv(
            q, k, v, num_attention_heads, num_query_groups, head_dim, model_state.get(mk)
        )

    # Fuse split gate+up
    for i in set(pending_gate) | set(pending_up):
        gate, up = pending_gate.get(i), pending_up.get(i)
        if None in (gate, up):
            logger.warning(f"[eagle] Incomplete gate/up MLP for layer {i}")
            continue
        mk = f"eagle_module.decoder.layers.{i}.mlp.linear_fc1.weight"
        mapped[mk] = _shard_gate_up(torch.cat([gate, up], dim=0), model_state.get(mk))

    return mapped


def map_hf_to_megatron_eagle(
    hf_state: dict[str, torch.Tensor],
    draft_model: nn.Module,
) -> dict[str, torch.Tensor]:
    """Map an HF Eagle3 checkpoint into ``draft_model``'s Megatron state dict.

    Convenience wrapper that derives the QKV head layout (``num_attention_heads``,
    ``num_query_groups``, ``head_dim``) from the model's own config, so callers don't
    have to thread those through. Use this from scripts/tests that hold the model.
    """
    cfg = unwrap_model(draft_model).config
    return _map_hf_to_megatron_eagle(
        hf_state,
        draft_model.state_dict(),
        num_attention_heads=cfg.num_attention_heads,
        num_query_groups=cfg.num_query_groups,
        head_dim=cfg.kv_channels,
    )


# ---------------------------------------------------------------------------
# Public: load model
# ---------------------------------------------------------------------------

def _get_rope_theta(hf_config, default: float = 10000.0) -> float:
    """Read rope_theta across transformers versions.

    transformers < 5 exposes a flat ``hf_config.rope_theta``; transformers >= 5
    nests it under ``hf_config.rope_parameters = {"rope_theta": ..., ...}``.
    Falls back to ``default`` if neither is present.
    """
    val = getattr(hf_config, "rope_theta", None)
    if val is not None:
        return val
    rope_params = getattr(hf_config, "rope_parameters", None)
    if isinstance(rope_params, dict) and rope_params.get("rope_theta") is not None:
        return rope_params["rope_theta"]
    return default


def load_eagle_draft_model(
    eagle_config,
    policy_model: MegatronModule,
    device: Optional[torch.device] = None,
    torch_dtype: torch.dtype = torch.bfloat16,
) -> EagleDraftModel:
    if device is None:
        from verl.utils.device import get_device_name, get_device_id
        device = torch.device(get_device_name(), get_device_id())

    from transformers import AutoConfig
    hf_config = AutoConfig.from_pretrained(eagle_config.model_path, trust_remote_code=True)
    num_aux = len(eagle_config.eagle_aux_hidden_state_layer_ids) if eagle_config.eagle_aux_hidden_state_layer_ids else 3
    config = TransformerConfig(
        num_layers=hf_config.num_hidden_layers,
        hidden_size=hf_config.hidden_size,
        num_attention_heads=hf_config.num_attention_heads,
        num_query_groups=getattr(hf_config, "num_key_value_heads", hf_config.num_attention_heads),
        kv_channels=getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads),
        ffn_hidden_size=getattr(hf_config, "intermediate_size", 4 * hf_config.hidden_size),
        normalization="RMSNorm",
        layernorm_epsilon=getattr(hf_config, "rms_norm_eps", 1e-5),
        activation_func=torch.nn.functional.silu,
        gated_linear_unit=True,
        add_bias_linear=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        gradient_accumulation_fusion=False,
    )
    # seq_length / vocab_size / rotary_base / rope_scaling are not valid
    # TransformerConfig constructor kwargs in mcore 0.16; set as plain attributes.
    config.seq_length = getattr(hf_config, "max_position_embeddings", 4096)
    config.vocab_size = hf_config.vocab_size
    config.rotary_base = _get_rope_theta(hf_config, default=10000.0)
    config.rope_scaling = getattr(hf_config, "rope_scaling", False) or False
    config.rope_scaling_factor = getattr(hf_config, "rope_scaling_factor", 8.0)
    config.eagle_num_aux_hidden_states = num_aux
    config.draft_vocab_size = eagle_config.draft_vocab_size or hf_config.vocab_size
    # Disable activation recomputation for the draft. The draft is 1 layer (recompute
    # saves nothing) AND its layer-0 attention uses a STATEFUL pre-hook that consumes
    # self._embeddings during the forward — recompute re-runs that attention in the
    # backward without re-running EagleModule.forward, so _embeddings is None and it
    # raises "pre-hook called before embeddings set". (Only surfaced once the draft
    # actually received gradients, i.e. after the differentiable-gather fix.)
    config.recompute_granularity = None
    config.recompute_method = None
    config.recompute_num_layers = None
    # On Ascend NPU, MindSpeed's attention path expects use_flash_attn to be set
    # (otherwise it raises on mask generation). Harmless on GPU.
    try:
        from verl.utils.device import is_npu_available
        if is_npu_available:
            config.use_flash_attn = True
    except Exception:
        pass

    model = EagleDraftModel(config).to(dtype=torch_dtype, device=device)
    # NOTE: output-layer trainability is decided AFTER load, once we know whether the
    # checkpoint shipped its own lm_head (see _apply_output_layer_trainability below).

    logger.info(f"[eagle] Loading Eagle3 draft weights from {eagle_config.model_path}")
    hf_state = _load_hf_checkpoint(eagle_config.model_path)
    model_state = model.state_dict()
    mapped = _map_hf_to_megatron_eagle(
        hf_state,
        model_state,
        num_attention_heads=config.num_attention_heads,
        num_query_groups=config.num_query_groups,
        head_dim=config.kv_channels,
    )

    lm_head_key = "eagle_module.eagle_output_layer.weight"
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    has_own_lm_head = lm_head_key not in missing
    if not has_own_lm_head:
        missing = [k for k in missing if k != lm_head_key]
        logger.info("[eagle] No lm_head in checkpoint; will copy from policy.")
    if missing:
        logger.warning(f"[eagle] Missing keys: {missing}")
    if unexpected:
        logger.warning(f"[eagle] Unexpected keys: {unexpected}")

    # Fail fast if a subset draft vocab lacks a real d2t (would distill on wrong columns).
    _validate_draft_d2t(
        int(config.draft_vocab_size),
        int(config.vocab_size),
        getattr(model.eagle_module, "d2t", None),
    )

    # CRITICAL: Eagle3 drafts (e.g. Tengyunw) ship their OWN trained lm_head over
    # the draft vocab; it is NOT tied to the target lm_head (cosine ≈ 0). Overwriting
    # it with the policy lm_head[d2t] resets the draft to garbage (KL ~14.5 vs ~5),
    # which is why the draft loss never drops and acceptance stays 0. Only copy from
    # the policy when the checkpoint did NOT provide a draft lm_head (tied-head case).
    model._eagle_has_own_lm_head = bool(has_own_lm_head)
    _apply_output_layer_trainability(model, has_own_lm_head)
    if has_own_lm_head:
        logger.info("[eagle] Draft has its own lm_head; keeping it and TRAINING it (no policy sync).")
    else:
        copy_policy_lm_head_to_draft(model, policy_model)

    model.train()
    return model


def _check_capture_preconditions(
    sequence_parallel: bool,
    context_parallel_size: int,
    tensor_parallel_size: int,
) -> None:
    """Validate the parallelism layout is compatible with Eagle3 hidden capture.

    The capture hooks record decoder/embedding/output activations and
    ``_unpack_megatron_thd`` reconstructs ``[B, S, *]`` assuming FULL-LENGTH sequences.
    Sequence parallelism (at TP>1) scatters those activations across TP ranks, and
    context parallelism splits the sequence — both silently break the unpack so the
    draft loss is skipped (or mis-sliced) every step. Fail loudly at setup instead.
    """
    problems = []
    if sequence_parallel and tensor_parallel_size > 1:
        problems.append(
            f"sequence_parallel=True at TP={tensor_parallel_size} scatters captured "
            "activations across TP ranks; set actor.megatron.sequence_parallel=False "
            "(and +...override_transformer_config.sequence_parallel=False)."
        )
    if context_parallel_size > 1:
        problems.append(
            f"context_parallel_size={context_parallel_size} splits the sequence; "
            "Eagle3 draft hidden capture requires CP=1."
        )
    if problems:
        raise RuntimeError(
            "[eagle] parallelism incompatible with Eagle3 hidden-state capture: "
            + " ".join(problems)
        )


def assert_capture_preconditions(policy_model: MegatronModule) -> None:
    """Read SP/CP/TP from the policy model + parallel state and validate (see
    :func:`_check_capture_preconditions`)."""
    pcfg = unwrap_model(policy_model).config
    _check_capture_preconditions(
        sequence_parallel=bool(getattr(pcfg, "sequence_parallel", False)),
        context_parallel_size=parallel_state.get_context_parallel_world_size(),
        tensor_parallel_size=parallel_state.get_tensor_model_parallel_world_size(),
    )


def _validate_draft_d2t(
    draft_vocab_size: int,
    vocab_size: int,
    d2t: Optional[torch.Tensor],
) -> None:
    """Require a non-trivial d2t when the draft vocab is a subset of the target vocab.

    The distillation maps teacher logits to the draft vocab via ``target_id = i + d2t[i]``.
    If the draft vocab is smaller than the target's but no real d2t is present, the loss
    silently falls back to the teacher's FIRST ``draft_vocab_size`` columns — the wrong
    token subset for any non-contiguous map — so acceptance never rises and nothing errors.
    Identity (draft_vocab == vocab) needs no d2t.
    """
    if draft_vocab_size == vocab_size:
        return
    if d2t is None or d2t.numel() != draft_vocab_size or int(d2t.abs().sum()) == 0:
        raise RuntimeError(
            f"[eagle] draft_vocab_size={draft_vocab_size} != vocab_size={vocab_size} but the "
            "checkpoint has no non-trivial d2t (draft→target token map). Training would distill "
            "against the wrong teacher-vocab columns. Use a checkpoint that ships a real d2t, or "
            "set draft_vocab_size == vocab_size."
        )


def _apply_output_layer_trainability(model: EagleDraftModel, has_own_lm_head: bool) -> None:
    """Set draft output-layer (lm_head) trainability based on checkpoint provenance.

    * Own lm_head (e.g. Tengyunw): it is NEVER synced from the policy
      (``sync_lm_head`` is a no-op for own-head drafts), so it MUST be trainable —
      otherwise the optimizer skips it (``_build_optimizer`` excludes
      ``requires_grad=False`` params) and the output projection stays frozen at its
      loaded value, the draft can never produce sensible draft-vocab logits, and
      acceptance stays ~0 with no error. (This was the original critical bug: the
      head was frozen unconditionally at load and never unfrozen.)
    * Tied head (no lm_head in checkpoint): it is overwritten from the policy every
      step via ``copy_policy_lm_head_to_draft`` / ``sync_lm_head``, so freeze it (the
      optimizer skips it; it tracks the policy head instead of being trained).
    """
    if has_own_lm_head:
        model.unfreeze_output_layer()
    else:
        model.freeze_output_layer()


# ---------------------------------------------------------------------------
# Public: LM-head sync
# ---------------------------------------------------------------------------

def _get_policy_lm_head(policy_model: MegatronModule) -> torch.Tensor:
    unwrapped = unwrap_model(policy_model)
    if getattr(unwrapped, "share_embeddings_and_output_weights", False):
        return unwrapped.shared_embedding_or_output_weight()
    return unwrapped.output_layer.weight


def _get_draft_output_layer(draft_model: EagleDraftModel) -> nn.Module:
    layer = getattr(getattr(draft_model, "eagle_module", None), "eagle_output_layer", None)
    if layer is None:
        raise ValueError("[eagle] Draft model does not have an eagle_output_layer")
    return layer


def _get_draft_to_target_token_mapping(
    draft_model: EagleDraftModel,
    device: torch.device,
) -> torch.Tensor:
    draft_vocab_size = int(draft_model.config.draft_vocab_size)
    mapping = torch.arange(draft_vocab_size, dtype=torch.long, device=device)
    d2t = getattr(draft_model.eagle_module, "d2t", None)
    if d2t is not None:
        mapping = mapping + d2t.to(device=device, dtype=torch.long)
    return mapping


def copy_policy_lm_head_to_draft(
    draft_model: EagleDraftModel,
    policy_model_chunk: MegatronModule,
) -> None:
    """Copy (a subset of) the policy LM-head shard into the draft output layer."""
    draft_output_layer = _get_draft_output_layer(draft_model)
    policy_lm_head_weight = _get_policy_lm_head(policy_model_chunk).detach()

    # With param_offload, the policy LM-head may sit on CPU. HCCL/NCCL all_gather
    # requires the tensor on the accelerator, so move it to the draft device first
    # (data is preserved by offload; this is a no-op when already on-device).
    if policy_lm_head_weight.device != draft_output_layer.weight.device:
        policy_lm_head_weight = policy_lm_head_weight.to(device=draft_output_layer.weight.device)

    # Fast path: local shard shapes match and no d2t → direct copy, no gather needed
    if (draft_output_layer.weight.shape == policy_lm_head_weight.shape
            and getattr(draft_model.eagle_module, "d2t", None) is None):
        with torch.no_grad():
            draft_output_layer.weight.copy_(
                policy_lm_head_weight.to(
                    device=draft_output_layer.weight.device,
                    dtype=draft_output_layer.weight.dtype,
                )
            )
        return

    # Slow path: gather full policy head, select draft-vocab rows, re-shard
    full_policy_lm_head_weight = _gather_tp_weight(policy_lm_head_weight, dim=0)
    draft_token_mapping = _get_draft_to_target_token_mapping(
        draft_model, device=full_policy_lm_head_weight.device
    )

    if draft_token_mapping.numel() == 0:
        raise RuntimeError("[draft] Draft token mapping is empty.")
    if int(draft_token_mapping.max().item()) >= full_policy_lm_head_weight.shape[0]:
        raise RuntimeError(
            f"[draft] Draft token mapping references index {int(draft_token_mapping.max().item())} "
            f"but policy LM head only has {full_policy_lm_head_weight.shape[0]} rows."
        )

    selected_lm_head_weight = _shard_tp_weight(
        full_policy_lm_head_weight.index_select(0, draft_token_mapping), dim=0
    )

    if draft_output_layer.weight.shape != selected_lm_head_weight.shape:
        raise RuntimeError(
            f"[eagle] lm_head shape mismatch: draft {tuple(draft_output_layer.weight.shape)} "
            f"vs policy_selected {tuple(selected_lm_head_weight.shape)}"
        )
    with torch.no_grad():
        draft_output_layer.weight.copy_(
            selected_lm_head_weight.to(
                device=draft_output_layer.weight.device,
                dtype=draft_output_layer.weight.dtype,
            )
        )


_SYNC_DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


def _sync_export_dtype() -> torch.dtype:
    """Dtype for draft weights shipped to vLLM (env ``EAGLE_SYNC_DTYPE``, default bf16).

    The draft trains/runs in bf16 and vLLM casts to each param's own dtype on load, so
    exporting bf16 is LOSSLESS vs the source while halving the per-step ZMQ transfer
    relative to the previous fp32 export. Set ``EAGLE_SYNC_DTYPE=fp32`` to restore the
    old behavior (diagnostic / bit-exact comparison).
    """
    name = os.environ.get("EAGLE_SYNC_DTYPE", "bf16").strip().lower()
    dtype = _SYNC_DTYPE_MAP.get(name)
    if dtype is None:
        logger.warning(
            "[eagle] unknown EAGLE_SYNC_DTYPE=%r; falling back to bf16. Valid: %s",
            name, sorted(_SYNC_DTYPE_MAP),
        )
        return torch.bfloat16
    return dtype


def export_eagle_weights_to_hf(draft_model: nn.Module) -> list[tuple[str, torch.Tensor]]:
    """Export the Eagle3 draft to the checkpoint (``midlayer.*``) naming that
    vLLM's ``LlamaForCausalLMEagle3.load_weights`` consumes.

    vLLM renames ``midlayer.* -> layers.0.*`` and stacks q/k/v -> qkv_proj and
    gate/up -> gate_up_proj internally, so we emit the *unfused* Tengyunw layout.
    Norm routing is the inverse of the loader (see _map_hf_to_megatron_eagle):
        enorm                          -> midlayer.input_layernorm   (embedding norm)
        decoder.layers.0.input_layernorm -> midlayer.hidden_norm     (hidden norm)
        decoder.layers.0.pre_mlp_layernorm -> midlayer.post_attention_layernorm
    This is what makes the Megatron-trained draft sync correctly into vLLM (whose
    draft is otherwise dummy-initialized under load_format=dummy).
    """
    unwrapped_model = unwrap_model(draft_model)
    source_state = unwrapped_model.state_dict()
    config = unwrapped_model.config

    # Resolve once per export (env EAGLE_SYNC_DTYPE, default bf16 — lossless vs the bf16
    # draft, halves the ZMQ transfer vs the old fp32 export). See _sync_export_dtype.
    sync_dtype = _sync_export_dtype()

    def to_sync(t: torch.Tensor) -> torch.Tensor:
        return t.detach().cpu().to(sync_dtype)

    out: list[tuple[str, torch.Tensor]] = []

    out.append(("fc.weight", to_sync(source_state["eagle_module.fc.weight"])))
    # enorm (embedding norm) -> midlayer.input_layernorm
    if "eagle_module.enorm.weight" in source_state:
        out.append(("midlayer.input_layernorm.weight", to_sync(source_state["eagle_module.enorm.weight"])))

    q_dim = config.num_attention_heads * config.kv_channels
    kv_dim = config.num_query_groups * config.kv_channels

    # Eagle3 has a single decoder layer; export it under the `midlayer.` prefix.
    lp = "eagle_module.decoder.layers.0"
    # decoder input_layernorm (hidden norm) -> midlayer.hidden_norm
    if f"{lp}.input_layernorm.weight" in source_state:
        out.append(("midlayer.hidden_norm.weight", to_sync(source_state[f"{lp}.input_layernorm.weight"])))
    if f"{lp}.pre_mlp_layernorm.weight" in source_state:
        out.append(("midlayer.post_attention_layernorm.weight", to_sync(source_state[f"{lp}.pre_mlp_layernorm.weight"])))

    # De-interleave Megatron's per-group linear_qkv back to per-head q/k/v. vLLM's
    # LlamaForCausalLMEagle3 stacks these into ITS OWN (flat [q|k|v]) qkv_proj on load,
    # so we emit the unfused, head-ordered tensors here.
    q, k, v = _get_tp_qkv_weight(
        source_state[f"{lp}.self_attention.linear_qkv.weight"], q_dim, kv_dim, config.kv_channels
    )
    out.extend([
        ("midlayer.self_attn.q_proj.weight", to_sync(q)),
        ("midlayer.self_attn.k_proj.weight", to_sync(k)),
        ("midlayer.self_attn.v_proj.weight", to_sync(v)),
        ("midlayer.self_attn.o_proj.weight", to_sync(_gather_tp_weight(source_state[f"{lp}.self_attention.linear_proj.weight"], dim=1))),
    ])

    gate, up = _get_tp_gate_up_weight(source_state[f"{lp}.mlp.linear_fc1.weight"], config.ffn_hidden_size)
    out.extend([
        ("midlayer.mlp.gate_proj.weight", to_sync(gate)),
        ("midlayer.mlp.up_proj.weight",   to_sync(up)),
        ("midlayer.mlp.down_proj.weight",  to_sync(_gather_tp_weight(source_state[f"{lp}.mlp.linear_fc2.weight"], dim=1))),
    ])

    if "eagle_module.decoder.final_layernorm.weight" in source_state:
        out.append(("norm.weight", to_sync(source_state["eagle_module.decoder.final_layernorm.weight"])))

    out.append(("lm_head.weight", to_sync(_gather_tp_weight(source_state["eagle_module.eagle_output_layer.weight"], dim=0))))
    # d2t (vLLM renames to draft_id_to_target_id). Critical: under load_format=dummy
    # vLLM's draft d2t is random, so it MUST be synced for the vocab remap to work.
    if "eagle_module.d2t" in source_state:
        out.append(("d2t", source_state["eagle_module.d2t"].cpu()))
    return out


def get_draft_state_dict_for_vllm(
    draft_model: EagleDraftModel,
) -> Iterator[Tuple[str, torch.Tensor]]:
    yield from export_eagle_weights_to_hf(draft_model)
