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


def gather_vocab_parallel_logits(logits: torch.Tensor) -> torch.Tensor:
    """[B, S, vocab/TP] -> [B, S, vocab]."""
    return _gather_tp_weight(logits, dim=-1)


def _get_tp_qkv_weight(
    local_fused_weight: torch.Tensor,
    q_dim: int,
    kv_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shards = _all_gather_tp_shards(local_fused_weight)
    if len(shards) == 1 and local_fused_weight.shape[0] == q_dim + 2 * kv_dim:
        return local_fused_weight.split([q_dim, kv_dim, kv_dim], dim=0)

    tp_world_size = len(shards)
    if q_dim % tp_world_size != 0 or kv_dim % tp_world_size != 0:
        raise RuntimeError("QKV dimensions are not divisible by the tensor-parallel world size.")

    local_q_dim = q_dim // tp_world_size
    local_kv_dim = kv_dim // tp_world_size
    q_shards, k_shards, v_shards = [], [], []
    for shard in shards:
        q_local, k_local, v_local = shard.split([local_q_dim, local_kv_dim, local_kv_dim], dim=0)
        q_shards.append(q_local)
        k_shards.append(k_local)
        v_shards.append(v_local)

    return (
        torch.cat(q_shards, dim=0).contiguous(),
        torch.cat(k_shards, dim=0).contiguous(),
        torch.cat(v_shards, dim=0).contiguous(),
    )


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


def _shard_qkv(fused_weight: torch.Tensor, target: Optional[torch.Tensor]) -> torch.Tensor:
    if target is None or fused_weight.shape == target.shape:
        return fused_weight.to(dtype=target.dtype) if target is not None else fused_weight
    return _shard_tp_weight(fused_weight, dim=0).to(dtype=target.dtype)


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
        if key in ("enorm.weight", "hidden_norm.weight", "hnorm.weight", "pre_fc_norm_hidden.weight"):
            megatron_key = "eagle_module.enorm.weight"
            mapped[megatron_key] = _shard_for_tp(megatron_key, tensor, model_state)
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

        if sub == "input_layernorm.weight":
            mapped[f"{lp}.input_layernorm.weight"] = tensor
        elif sub == "post_attention_layernorm.weight":
            mapped[f"{lp}.pre_mlp_layernorm.weight"] = tensor
        elif sub == "self_attn.qkv_proj.weight":
            mk = f"{lp}.self_attention.linear_qkv.weight"
            mapped[mk] = _shard_qkv(tensor, model_state.get(mk))
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

    # Fuse split QKV — shard each component separately to handle GQA correctly
    for i in set(pending_q) | set(pending_k) | set(pending_v):
        q, k, v = pending_q.get(i), pending_k.get(i), pending_v.get(i)
        if None in (q, k, v):
            logger.warning(f"[eagle] Incomplete QKV for layer {i}")
            continue
        mk = f"eagle_module.decoder.layers.{i}.self_attention.linear_qkv.weight"
        target = model_state.get(mk)
        q_l, k_l, v_l = _shard_tp_weight(q, 0), _shard_tp_weight(k, 0), _shard_tp_weight(v, 0)
        fused = torch.cat([q_l, k_l, v_l], dim=0)
        mapped[mk] = fused.to(dtype=target.dtype) if target is not None else fused

    # Fuse split gate+up
    for i in set(pending_gate) | set(pending_up):
        gate, up = pending_gate.get(i), pending_up.get(i)
        if None in (gate, up):
            logger.warning(f"[eagle] Incomplete gate/up MLP for layer {i}")
            continue
        mk = f"eagle_module.decoder.layers.{i}.mlp.linear_fc1.weight"
        mapped[mk] = _shard_gate_up(torch.cat([gate, up], dim=0), model_state.get(mk))

    return mapped


# ---------------------------------------------------------------------------
# Public: load model
# ---------------------------------------------------------------------------

def load_eagle_draft_model(
    eagle_config,
    policy_model: MegatronModule,
    device: Optional[torch.device] = None,
    torch_dtype: torch.dtype = torch.bfloat16,
) -> EagleDraftModel:
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())

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
        rotary_base=getattr(hf_config, "rope_theta", 10000),
        seq_length=getattr(hf_config, "max_position_embeddings", 4096),
        vocab_size=hf_config.vocab_size,
        gradient_accumulation_fusion=False,
    )
    config.eagle_num_aux_hidden_states = num_aux
    config.draft_vocab_size = eagle_config.draft_vocab_size or hf_config.vocab_size

    model = EagleDraftModel(config).to(dtype=torch_dtype, device=device)
    model.freeze_output_layer()

    logger.info(f"[eagle] Loading Eagle3 draft weights from {eagle_config.model_path}")
    hf_state = _load_hf_checkpoint(eagle_config.model_path)
    model_state = model.state_dict()
    mapped = _map_hf_to_megatron_eagle(hf_state, model_state)

    lm_head_key = "eagle_module.eagle_output_layer.weight"
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    if lm_head_key in missing:
        missing = [k for k in missing if k != lm_head_key]
        logger.info("[eagle] No lm_head in checkpoint; will copy from policy.")
    if missing:
        logger.warning(f"[eagle] Missing keys: {missing}")
    if unexpected:
        logger.warning(f"[eagle] Unexpected keys: {unexpected}")

    copy_policy_lm_head_to_draft(model, policy_model)

    model.train()
    return model


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


def export_eagle_weights_to_hf(draft_model: nn.Module) -> list[tuple[str, torch.Tensor]]:
    """Export the standalone Eagle draft model to HF naming."""
    unwrapped_model = unwrap_model(draft_model)
    source_state = unwrapped_model.state_dict()
    config = unwrapped_model.config

    def cpu32(t: torch.Tensor) -> torch.Tensor:
        return t.detach().cpu().float()

    out: list[tuple[str, torch.Tensor]] = []

    out.append(("fc.weight", cpu32(source_state["eagle_module.fc.weight"])))
    if "eagle_module.enorm.weight" in source_state:
        out.append(("enorm.weight", cpu32(source_state["eagle_module.enorm.weight"])))

    q_dim = config.num_attention_heads * config.kv_channels
    kv_dim = config.num_query_groups * config.kv_channels

    for i in range(config.num_layers):
        lp = f"eagle_module.decoder.layers.{i}"
        lk = f"layers.{i}"

        if f"{lp}.input_layernorm.weight" in source_state:
            out.append((f"{lk}.input_layernorm.weight", cpu32(source_state[f"{lp}.input_layernorm.weight"])))
        if f"{lp}.pre_mlp_layernorm.weight" in source_state:
            out.append((f"{lk}.post_attention_layernorm.weight", cpu32(source_state[f"{lp}.pre_mlp_layernorm.weight"])))

        q, k, v = _get_tp_qkv_weight(source_state[f"{lp}.self_attention.linear_qkv.weight"], q_dim, kv_dim)
        out.extend([
            (f"{lk}.self_attn.q_proj.weight", cpu32(q)),
            (f"{lk}.self_attn.k_proj.weight", cpu32(k)),
            (f"{lk}.self_attn.v_proj.weight", cpu32(v)),
            (f"{lk}.self_attn.o_proj.weight", cpu32(_gather_tp_weight(source_state[f"{lp}.self_attention.linear_proj.weight"], dim=1))),
        ])

        gate, up = _get_tp_gate_up_weight(source_state[f"{lp}.mlp.linear_fc1.weight"], config.ffn_hidden_size)
        out.extend([
            (f"{lk}.mlp.gate_proj.weight", cpu32(gate)),
            (f"{lk}.mlp.up_proj.weight",   cpu32(up)),
            (f"{lk}.mlp.down_proj.weight",  cpu32(_gather_tp_weight(source_state[f"{lp}.mlp.linear_fc2.weight"], dim=1))),
        ])

    if "eagle_module.decoder.final_layernorm.weight" in source_state:
        out.append(("norm.weight", cpu32(source_state["eagle_module.decoder.final_layernorm.weight"])))

    out.append(("lm_head.weight", cpu32(_gather_tp_weight(source_state["eagle_module.eagle_output_layer.weight"], dim=0))))
    if "eagle_module.d2t" in source_state:
        out.append(("d2t", source_state["eagle_module.d2t"].cpu()))
    return out


def get_draft_state_dict_for_vllm(
    draft_model: EagleDraftModel,
) -> Iterator[Tuple[str, torch.Tensor]]:
    yield from export_eagle_weights_to_hf(draft_model)
