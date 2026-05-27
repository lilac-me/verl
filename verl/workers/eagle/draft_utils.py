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

import json
import logging
import re
from pathlib import Path
from typing import Iterator, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

from verl.workers.eagle.draft_model import EagleDraftModel, build_eagle_transformer_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TP helpers
# ---------------------------------------------------------------------------

def _tp_info() -> tuple[Optional[dist.ProcessGroup], int, int]:
    try:
        from megatron.core import parallel_state
        if parallel_state.model_parallel_is_initialized():
            return (
                parallel_state.get_tensor_model_parallel_group(),
                parallel_state.get_tensor_model_parallel_world_size(),
                parallel_state.get_tensor_model_parallel_rank(),
            )
    except Exception:
        pass
    return None, 1, 0


def _all_gather(local: torch.Tensor, group, tp_size: int) -> list[torch.Tensor]:
    if tp_size == 1:
        return [local]
    shards = [torch.empty_like(local) for _ in range(tp_size)]
    dist.all_gather(shards, local.contiguous(), group=group)
    return shards


def _shard(tensor: torch.Tensor, tp_size: int, tp_rank: int, dim: int) -> torch.Tensor:
    """Shard tensor along dim for this TP rank."""
    if tp_size == 1:
        return tensor
    return torch.chunk(tensor, tp_size, dim=dim)[tp_rank].contiguous()


def _gather(local: torch.Tensor, tp_size: int, group, dim: int) -> torch.Tensor:
    """All-gather tensor shards along dim."""
    shards = _all_gather(local, group, tp_size)
    return torch.cat(shards, dim=dim).contiguous()


def gather_vocab_parallel_logits(logits: torch.Tensor) -> torch.Tensor:
    """[B, S, vocab/TP] -> [B, S, vocab]."""
    group, tp_size, _ = _tp_info()
    if tp_size == 1:
        return logits
    shards = _all_gather(logits, group, tp_size)
    return torch.cat(shards, dim=-1).contiguous()


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def _load_hf_checkpoint(model_path: str) -> dict[str, torch.Tensor]:
    p = Path(model_path)
    if p.is_file():
        return _load_file(p)
    if p.is_dir():
        for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
            idx = p / name
            if idx.exists():
                weight_map = json.loads(idx.read_text())["weight_map"]
                merged = {}
                for shard in sorted(set(weight_map.values())):
                    merged.update(_load_file(p / shard))
                return merged
        for name in ("model.safetensors", "pytorch_model.bin"):
            if (p / name).exists():
                return _load_file(p / name)
        for shard in sorted(p.glob("model-*.safetensors")):
            pass  # handled by index above
    raise FileNotFoundError(f"[eagle] No checkpoint found at '{model_path}'")


def _load_file(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(str(path))
    obj = torch.load(str(path), map_location="cpu", weights_only=True)
    if isinstance(obj, dict) and "state_dict" in obj:
        obj = obj["state_dict"]
    return {k: v for k, v in obj.items() if isinstance(v, torch.Tensor)}


# ---------------------------------------------------------------------------
# HF -> Megatron key mapping
# ---------------------------------------------------------------------------

_LAYER_RE = re.compile(r"^(?:layers|midlayer)\.?(\d*)\.(.+)$")
# Not TP-sharded: full weight on every rank
_NO_TP_KEYS = {"eagle_module.fc.weight", "eagle_module.enorm.weight"}
# TP-sharded along dim=0 (ColumnParallel)
_COL_PARALLEL_RE = re.compile(
    r"(self_attention\.linear_qkv|mlp\.linear_fc1|eagle_output_layer)\.weight$"
)
# TP-sharded along dim=1 (RowParallel)
_ROW_PARALLEL_RE = re.compile(
    r"(self_attention\.linear_proj|mlp\.linear_fc2)\.weight$"
)


def _normalize_key(key: str) -> str:
    for prefix in ("draft.", "module.", "eagle_module."):
        while key.startswith(prefix):
            key = key[len(prefix):]
    return key


def _shard_for_tp(
    megatron_key: str,
    tensor: torch.Tensor,
    model_state: dict,
    tp_size: int,
    tp_rank: int,
) -> torch.Tensor:
    """Shard tensor to local TP rank by comparing shapes with model."""
    if megatron_key in _NO_TP_KEYS or tp_size == 1:
        target = model_state.get(megatron_key)
        return tensor.to(dtype=target.dtype) if target is not None else tensor

    target = model_state.get(megatron_key)
    if target is None or tensor.shape == target.shape:
        return tensor.to(dtype=target.dtype) if target is not None else tensor

    # Infer shard axis from key pattern
    if _COL_PARALLEL_RE.search(megatron_key):
        dim = 0
    elif _ROW_PARALLEL_RE.search(megatron_key):
        dim = 1
    else:
        return tensor

    return _shard(tensor, tp_size, tp_rank, dim).to(dtype=target.dtype)


def _map_hf_to_megatron(
    hf_state: dict[str, torch.Tensor],
    model_state: dict[str, torch.Tensor],
    tp_size: int,
    tp_rank: int,
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
            mk = "eagle_module.fc.weight"
            mapped[mk] = _shard_for_tp(mk, tensor, model_state, tp_size, tp_rank)
            continue
        if key in ("enorm.weight", "hidden_norm.weight", "hnorm.weight"):
            mk = "eagle_module.enorm.weight"
            mapped[mk] = _shard_for_tp(mk, tensor, model_state, tp_size, tp_rank)
            continue
        if key == "norm.weight":
            mk = "eagle_module.decoder.final_layernorm.weight"
            mapped[mk] = tensor
            continue
        if key in ("lm_head.weight", "eagle_output_layer.weight"):
            mk = "eagle_module.eagle_output_layer.weight"
            mapped[mk] = _shard_for_tp(mk, tensor, model_state, tp_size, tp_rank)
            continue
        if key == "d2t":
            mk = "eagle_module.d2t"
            if mk in model_state:
                mapped[mk] = tensor
            continue

        # Layer weights
        if key.startswith("midlayer."):
            layer_idx, sub = 0, key[len("midlayer."):]
        else:
            m = re.match(r"^layers\.(\d+)\.(.+)$", key)
            if not m:
                continue
            layer_idx, sub = int(m.group(1)), m.group(2)

        lp = f"eagle_module.decoder.layers.{layer_idx}"

        if sub == "input_layernorm.weight":
            mapped[f"{lp}.input_layernorm.weight"] = tensor
        elif sub == "post_attention_layernorm.weight":
            mapped[f"{lp}.pre_mlp_layernorm.weight"] = tensor
        elif sub == "self_attn.qkv_proj.weight":
            mk = f"{lp}.self_attention.linear_qkv.weight"
            mapped[mk] = _shard_qkv(tensor, model_state.get(mk), tp_size, tp_rank)
        elif sub == "self_attn.q_proj.weight":
            pending_q[layer_idx] = tensor
        elif sub == "self_attn.k_proj.weight":
            pending_k[layer_idx] = tensor
        elif sub == "self_attn.v_proj.weight":
            pending_v[layer_idx] = tensor
        elif sub == "self_attn.o_proj.weight":
            mk = f"{lp}.self_attention.linear_proj.weight"
            mapped[mk] = _shard_for_tp(mk, tensor, model_state, tp_size, tp_rank)
        elif sub == "mlp.gate_up_proj.weight":
            mk = f"{lp}.mlp.linear_fc1.weight"
            mapped[mk] = _shard_gate_up(tensor, model_state.get(mk), tp_size, tp_rank)
        elif sub == "mlp.gate_proj.weight":
            pending_gate[layer_idx] = tensor
        elif sub == "mlp.up_proj.weight":
            pending_up[layer_idx] = tensor
        elif sub == "mlp.down_proj.weight":
            mk = f"{lp}.mlp.linear_fc2.weight"
            mapped[mk] = _shard_for_tp(mk, tensor, model_state, tp_size, tp_rank)

    # Fuse split QKV
    for i in set(pending_q) | set(pending_k) | set(pending_v):
        q, k, v = pending_q.get(i), pending_k.get(i), pending_v.get(i)
        if None in (q, k, v):
            logger.warning(f"[eagle] Incomplete QKV for layer {i}")
            continue
        fused = torch.cat([q, k, v], dim=0)
        mk = f"eagle_module.decoder.layers.{i}.self_attention.linear_qkv.weight"
        mapped[mk] = _shard_qkv(fused, model_state.get(mk), tp_size, tp_rank)

    # Fuse split gate+up
    for i in set(pending_gate) | set(pending_up):
        gate, up = pending_gate.get(i), pending_up.get(i)
        if None in (gate, up):
            logger.warning(f"[eagle] Incomplete gate/up MLP for layer {i}")
            continue
        fused = torch.cat([gate, up], dim=0)
        mk = f"eagle_module.decoder.layers.{i}.mlp.linear_fc1.weight"
        mapped[mk] = _shard_gate_up(fused, model_state.get(mk), tp_size, tp_rank)

    return mapped


def _shard_qkv(
    full: torch.Tensor,
    target: Optional[torch.Tensor],
    tp_size: int,
    tp_rank: int,
) -> torch.Tensor:
    """Shard fused [Q, K, V] preserving component boundaries."""
    if target is None or full.shape == target.shape or tp_size == 1:
        return full.to(dtype=target.dtype) if target is not None else full
    # Infer split: full.shape[0] = Q+K+V, target.shape[0] = (Q+K+V)/TP
    # But Q and KV may have different sizes, so we can't just chunk blindly.
    # Infer from shape ratio: each component sharded by same TP
    full_out = full.shape[0]
    local_out = target.shape[0]
    # We don't know Q/KV split here without config; use inferred TP from ratio
    inferred_tp = full_out // local_out
    # Chunk each component separately: Q, K, V in equal thirds if GQA not present
    # For GQA: caller should pass pre-split q/k/v via pending_q/k/v path
    chunk = _shard(full, inferred_tp, tp_rank, dim=0)
    return chunk.to(dtype=target.dtype)


def _shard_gate_up(
    full: torch.Tensor,
    target: Optional[torch.Tensor],
    tp_size: int,
    tp_rank: int,
) -> torch.Tensor:
    """Shard fused [gate, up] preserving component boundaries."""
    if target is None or full.shape == target.shape or tp_size == 1:
        return full.to(dtype=target.dtype) if target is not None else full
    ffn = full.shape[0] // 2
    gate = _shard(full[:ffn], tp_size, tp_rank, dim=0)
    up   = _shard(full[ffn:], tp_size, tp_rank, dim=0)
    return torch.cat([gate, up], dim=0).to(dtype=target.dtype)


# ---------------------------------------------------------------------------
# Public: load model
# ---------------------------------------------------------------------------

def load_eagle_draft_model(
    eagle_config,
    policy_lm_head_weight: Optional[torch.Tensor],
    device: Optional[torch.device] = None,
    torch_dtype: torch.dtype = torch.bfloat16,
) -> EagleDraftModel:
    from transformers import AutoConfig

    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())

    hf_cfg = AutoConfig.from_pretrained(eagle_config.model_path, trust_remote_code=True)
    num_aux = len(eagle_config.aux_layer_indices) if eagle_config.aux_layer_indices else 3
    config = build_eagle_transformer_config(hf_cfg, num_aux_layers=num_aux)

    model = EagleDraftModel(config).to(dtype=torch_dtype, device=device)
    model.freeze_output_layer()

    logger.info(f"[eagle] Loading Eagle3 draft weights from {eagle_config.model_path}")
    hf_state = _load_hf_checkpoint(eagle_config.model_path)
    model_state = model.state_dict()
    _, tp_size, tp_rank = _tp_info()
    mapped = _map_hf_to_megatron(hf_state, model_state, tp_size, tp_rank)

    lm_head_key = "eagle_module.eagle_output_layer.weight"
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    if lm_head_key in missing:
        missing = [k for k in missing if k != lm_head_key]
        logger.info("[eagle] No lm_head in checkpoint; will copy from policy.")
    if missing:
        logger.warning(f"[eagle] Missing keys: {missing}")
    if unexpected:
        logger.warning(f"[eagle] Unexpected keys: {unexpected}")

    if policy_lm_head_weight is not None:
        copy_policy_lm_head(model, policy_lm_head_weight)

    model.train()
    return model


# ---------------------------------------------------------------------------
# Public: sync LM head
# ---------------------------------------------------------------------------

def copy_policy_lm_head(
    draft_model: EagleDraftModel,
    policy_lm_head_local_weight: torch.Tensor,
) -> None:
    """Copy (a subset of) the policy LM-head shard into the draft output layer.

    When draft_vocab_size == vocab_size: direct per-rank copy, no communication.
    When draft_vocab_size < vocab_size: gather the full policy head across TP,
    select the draft-vocab rows (using an optional d2t offset stored on the model),
    then re-shard for the local TP rank.
    """
    target = draft_model.eagle_module.eagle_output_layer.weight
    group, tp_size, tp_rank = _tp_info()

    # Fast path: same vocab size, shapes match → no gather needed.
    d2t = getattr(draft_model.eagle_module, "d2t", None)
    if target.shape == policy_lm_head_local_weight.shape and d2t is None:
        with torch.no_grad():
            target.copy_(policy_lm_head_local_weight.to(device=target.device, dtype=target.dtype))
        return

    # Gather full policy LM head [vocab_size, H] from all TP ranks.
    full_policy = _gather(policy_lm_head_local_weight, tp_size, group, dim=0)

    # Build draft-to-target index mapping.
    # d2t (if present) is a per-token offset: mapping[i] = i + d2t[i].
    draft_vocab_size = draft_model.config.draft_vocab_size
    mapping = torch.arange(draft_vocab_size, dtype=torch.long, device=full_policy.device)
    if d2t is not None:
        mapping = mapping + d2t.to(device=full_policy.device, dtype=torch.long)

    selected = full_policy.index_select(0, mapping)  # [draft_vocab_size, H]

    # Re-shard for this TP rank.
    local_selected = _shard(selected, tp_size, tp_rank, dim=0)

    if target.shape != local_selected.shape:
        raise RuntimeError(
            f"[eagle] lm_head shape mismatch after vocab selection: "
            f"draft {tuple(target.shape)} vs policy_selected {tuple(local_selected.shape)}"
        )
    with torch.no_grad():
        target.copy_(local_selected.to(device=target.device, dtype=target.dtype))


# ---------------------------------------------------------------------------
# Public: export for vLLM
# ---------------------------------------------------------------------------

def export_eagle_weights_to_hf(
    draft_model: EagleDraftModel,
) -> list[tuple[str, torch.Tensor]]:
    """All-gather TP shards and return (HF key, cpu float32 tensor) pairs."""
    state = draft_model.state_dict()
    cfg = draft_model.config
    group, tp_size, _ = _tp_info()

    def cpu32(t):
        return t.detach().cpu().float()

    def gather_col(key):
        return cpu32(_gather(state[key], tp_size, group, dim=0) if tp_size > 1 else state[key])

    def gather_row(key):
        return cpu32(_gather(state[key], tp_size, group, dim=1) if tp_size > 1 else state[key])

    out: list[tuple[str, torch.Tensor]] = []

    # fc and enorm: not TP-sharded
    out.append(("fc.weight", cpu32(state["eagle_module.fc.weight"])))
    if "eagle_module.enorm.weight" in state:
        out.append(("enorm.weight", cpu32(state["eagle_module.enorm.weight"])))

    q_dim = cfg.num_attention_heads * cfg.kv_channels
    kv_dim = cfg.num_query_groups * cfg.kv_channels

    for i in range(cfg.num_layers):
        lp = f"eagle_module.decoder.layers.{i}"
        lk = f"layers.{i}"

        if f"{lp}.input_layernorm.weight" in state:
            out.append((f"{lk}.input_layernorm.weight", cpu32(state[f"{lp}.input_layernorm.weight"])))
        if f"{lp}.pre_mlp_layernorm.weight" in state:
            out.append((f"{lk}.post_attention_layernorm.weight", cpu32(state[f"{lp}.pre_mlp_layernorm.weight"])))

        # QKV: gather col then re-split into q/k/v
        qkv_full = _gather(state[f"{lp}.self_attention.linear_qkv.weight"], tp_size, group, dim=0) if tp_size > 1 else state[f"{lp}.self_attention.linear_qkv.weight"]
        q, k, v = _unshard_qkv(qkv_full, q_dim, kv_dim, tp_size)
        out.extend([
            (f"{lk}.self_attn.q_proj.weight", cpu32(q)),
            (f"{lk}.self_attn.k_proj.weight", cpu32(k)),
            (f"{lk}.self_attn.v_proj.weight", cpu32(v)),
            (f"{lk}.self_attn.o_proj.weight", gather_row(f"{lp}.self_attention.linear_proj.weight")),
        ])

        # gate+up: gather col then re-split
        fc1_full = _gather(state[f"{lp}.mlp.linear_fc1.weight"], tp_size, group, dim=0) if tp_size > 1 else state[f"{lp}.mlp.linear_fc1.weight"]
        gate, up = _unshard_gate_up(fc1_full, cfg.ffn_hidden_size, tp_size)
        out.extend([
            (f"{lk}.mlp.gate_proj.weight", cpu32(gate)),
            (f"{lk}.mlp.up_proj.weight",   cpu32(up)),
            (f"{lk}.mlp.down_proj.weight",  gather_row(f"{lp}.mlp.linear_fc2.weight")),
        ])

    if "eagle_module.decoder.final_layernorm.weight" in state:
        out.append(("norm.weight", cpu32(state["eagle_module.decoder.final_layernorm.weight"])))

    out.append(("lm_head.weight", gather_col("eagle_module.eagle_output_layer.weight")))
    return out


def _unshard_qkv(full, q_dim, kv_dim, tp_size):
    if tp_size == 1:
        return full.split([q_dim, kv_dim, kv_dim], dim=0)
    lq, lkv = q_dim // tp_size, kv_dim // tp_size
    shard_size = lq + 2 * lkv
    qs, ks, vs = [], [], []
    for r in range(tp_size):
        s = full[r * shard_size:(r + 1) * shard_size]
        qs.append(s[:lq]); ks.append(s[lq:lq+lkv]); vs.append(s[lq+lkv:])
    return torch.cat(qs, 0).contiguous(), torch.cat(ks, 0).contiguous(), torch.cat(vs, 0).contiguous()


def _unshard_gate_up(full, ffn_hidden_size, tp_size):
    if tp_size == 1:
        return full.split([ffn_hidden_size, ffn_hidden_size], dim=0)
    lf = ffn_hidden_size // tp_size
    gates, ups = [], []
    for r in range(tp_size):
        s = full[r * 2 * lf:(r + 1) * 2 * lf]
        gates.append(s[:lf]); ups.append(s[lf:])
    return torch.cat(gates, 0).contiguous(), torch.cat(ups, 0).contiguous()


def get_draft_state_dict_for_vllm(
    draft_model: EagleDraftModel,
) -> Iterator[Tuple[str, torch.Tensor]]:
    yield from export_eagle_weights_to_hf(draft_model)
