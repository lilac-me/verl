# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
"""Self-contained EAGLE3 draft model, mcore-attachable, modelopt-free.

Design notes
------------
* Parameter names deliberately mirror the public HF EAGLE3 checkpoint layout
  (e.g. yuhuili/EAGLE3-*, RedHatAI/*-speculator.eagle3, SpecForge outputs):
      fc.weight
      midlayer.input_layernorm.weight       (normalizes token embeds)
      midlayer.hidden_norm.weight           (normalizes fused hidden)
      midlayer.self_attn.{q,k,v,o}_proj.weight   (q/k/v in_features = 2*h)
      midlayer.mlp.{gate,up,down}_proj.weight
      norm.weight
      lm_head.weight                        ([draft_vocab, h])
      d2t / t2d                             (buffers)
  so HF<->trainer weight mapping is identity (weight_utils.py) and the export
  feeds vLLM's LlamaForCausalLMEagle3.load_weights() unchanged.

* TP strategy (phase 1): all draft parameters are REPLICATED across TP ranks.
  The module is one layer + a small lm_head (draft_vocab is typically 32k),
  so replication costs tens of MB and removes vocab-parallel soft-CE
  complexity. Gradients across TP replicas stay identical because inputs
  (captured states) and the teacher are identical on all TP ranks of the
  last PP stage. A defensive grad all-reduce hook is installed anyway, since
  Megatron DDP will otherwise treat these as TP-sharded params.
  TODO(phase2): vocab-parallel lm_head + vocab-parallel soft CE
  (port NeMo-RL DraftCrossEntropyLossFn) if draft vocab grows.

* Attention (phase 1): plain causal SDPA on [s, b, h] tensors with RoPE.
  Correct for CP=1 only (validate_draft_config enforces this).
  TODO(phase2): build the layer from a mcore TransformerLayer spec with a
  2h-input linear_qkv (the modelopt megatron_eagle approach) to inherit
  CP/FlashAttention/TE support; on Ascend, torch_npu falls back through
  sdpa -> npu_fusion_attention is a further optimization.

Tensor layout: [s, b, h] throughout (mcore convention).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

try:
    from megatron.core import parallel_state
except ImportError:  # unit tests without megatron
    parallel_state = None


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: Tensor) -> Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * x.to(input_dtype)).to(input_dtype)


def _rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """Minimal RoPE matching HF llama convention. [s, b, nh, hd] inputs."""

    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, q: Tensor, k: Tensor, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        # position_ids: [s, b] (or [s] broadcastable)
        freqs = torch.einsum("sb,d->sbd", position_ids.to(torch.float32), self.inv_freq.to(position_ids.device))
        emb = torch.cat((freqs, freqs), dim=-1)  # [s, b, hd]
        cos = emb.cos().unsqueeze(2)  # [s, b, 1, hd]
        sin = emb.sin().unsqueeze(2)
        q = (q * cos) + (_rotate_half(q) * sin)
        k = (k * cos) + (_rotate_half(k) * sin)
        return q.to(q.dtype), k.to(k.dtype)


class EagleAttention(nn.Module):
    """GQA attention whose q/k/v projections consume 2*h (embeds ++ hidden)."""

    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, rope_base: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        in_features = 2 * hidden_size
        self.q_proj = nn.Linear(in_features, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(in_features, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(in_features, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)
        self.rotary = RotaryEmbedding(self.head_dim, base=rope_base)

    def forward(self, x: Tensor, position_ids: Tensor, attention_mask: Optional[Tensor]) -> Tensor:
        # x: [s, b, 2h]
        s, b, _ = x.shape
        q = self.q_proj(x).view(s, b, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(s, b, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(s, b, self.num_kv_heads, self.head_dim)
        q, k = self.rotary(q, k, position_ids)
        # -> [b, nh, s, hd]
        q = q.permute(1, 2, 0, 3)
        k = k.permute(1, 2, 0, 3)
        v = v.permute(1, 2, 0, 3)
        if self.num_kv_heads != self.num_heads:
            rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        # NOTE: padded positions are handled by loss masking, so a pure causal
        # mask is sufficient for training (matches EAGLE3 reference trainers).
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.permute(2, 0, 1, 3).reshape(s, b, self.num_heads * self.head_dim)
        return self.o_proj(out)


class EagleMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class EagleDecoderLayer(nn.Module):
    """EAGLE3 'midlayer': dual-input norm -> 2h attention -> MLP."""

    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int,
                 intermediate_size: int, rms_eps: float, rope_base: float):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_eps)   # on token embeds
        self.hidden_norm = RMSNorm(hidden_size, eps=rms_eps)       # on fused hidden
        self.self_attn = EagleAttention(hidden_size, num_heads, num_kv_heads, rope_base)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_eps)
        self.mlp = EagleMLP(hidden_size, intermediate_size)

    def forward(self, input_embeds: Tensor, hidden_states: Tensor,
                position_ids: Tensor, attention_mask: Optional[Tensor]) -> Tensor:
        # input_embeds, hidden_states: [s, b, h]
        residual = hidden_states
        attn_in = torch.cat([self.input_layernorm(input_embeds), self.hidden_norm(hidden_states)], dim=-1)
        hidden_states = residual + self.self_attn(attn_in, position_ids, attention_mask)
        residual = hidden_states
        hidden_states = residual + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states


class EagleDraftModel(nn.Module):
    """Trainer-owned EAGLE3 draft model.

    forward(hidden_states [s,b,3h], input_embeds [s,b,h]) -> logits [s,b,draft_vocab]
    """

    def __init__(self, hf_config, draft_cfg, target_hidden_size: int, params_dtype=torch.bfloat16):
        super().__init__()
        h = getattr(hf_config, "hidden_size", target_hidden_size)
        if h != target_hidden_size:
            raise ValueError(
                f"EAGLE3 draft hidden_size ({h}) must match target hidden_size ({target_hidden_size}) "
                f"because the drafter consumes target embeddings/hidden states directly."
            )
        self.hidden_size = h
        self.num_aux_hidden_states = 3
        self.draft_vocab_size = (
            draft_cfg.draft_vocab_size
            or getattr(hf_config, "draft_vocab_size", None)
            or getattr(hf_config, "vocab_size", None)
        )
        self.target_vocab_size = getattr(hf_config, "target_vocab_size", None) or getattr(hf_config, "vocab_size")

        # hidden fusion: cat of 3 aux layer hiddens -> h
        self.fc = nn.Linear(self.num_aux_hidden_states * h, h, bias=False)
        self.midlayer = EagleDecoderLayer(
            hidden_size=h,
            num_heads=hf_config.num_attention_heads,
            num_kv_heads=getattr(hf_config, "num_key_value_heads", hf_config.num_attention_heads),
            intermediate_size=hf_config.intermediate_size,
            rms_eps=getattr(hf_config, "rms_norm_eps", 1e-6),
            rope_base=getattr(hf_config, "rope_theta", 10000.0),
        )
        self.norm = RMSNorm(h, eps=getattr(hf_config, "rms_norm_eps", 1e-6))
        self.lm_head = nn.Linear(h, self.draft_vocab_size, bias=False)

        # d2t: offset map, target_id = draft_id + d2t[draft_id]. Identity (zeros)
        # when draft vocab == target vocab.
        self.register_buffer("d2t", torch.zeros(self.draft_vocab_size, dtype=torch.long), persistent=True)

        self.to(dtype=params_dtype)
        # Buffers must stay integer dtype after the cast above.
        self.d2t = self.d2t.to(torch.long)

        # NOTE: gradient synchronization (TP replica averaging + DP all-reduce)
        # is owned by MegatronDraftTrainerPlugin.step(), not by hooks here —
        # the draft model is deliberately NOT attached to the policy chunks so
        # it never enters Megatron DDP grad buffers built at wrap time.


    # ------------------------------------------------------------------ #
    def forward(
        self,
        hidden_states: Tensor,        # [s, b, 3h] captured aux states (already gathered to this rank)
        input_embeds: Tensor,         # [s, b, h] embeds rolled left by 1 (token t+1 at position t)
        position_ids: Optional[Tensor] = None,   # [s, b]
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        s, b, _ = input_embeds.shape
        if position_ids is None:
            position_ids = torch.arange(s, device=input_embeds.device).unsqueeze(-1).expand(s, b)
        fused = self.fc(hidden_states)                       # [s, b, h]
        out = self.midlayer(input_embeds, fused, position_ids, attention_mask)
        out = self.norm(out)
        logits = self.lm_head(out)                           # [s, b, draft_vocab]
        return logits

    # ------------------------------------------------------------------ #
    def sharded_state_dict(self, prefix: str = "", sharded_offsets=(), metadata=None):
        """Megatron dist-ckpt integration: all params replicated across TP/PP,
        sharded only across DP (handled by replica_id).

        TODO(verify): exact replica_id convention for fully-replicated params in
        your Megatron-LM / MindSpeed version; the helper below follows
        make_sharded_tensors_for_checkpoint semantics with tp-rank in
        replica_id so only one TP rank writes.
        """
        from megatron.core.dist_checkpointing.mapping import ShardedTensor

        tp_rank = parallel_state.get_tensor_model_parallel_rank() if parallel_state else 0
        dp_rank = parallel_state.get_data_parallel_rank(with_context_parallel=True) if parallel_state else 0
        sd = {}
        state = self.state_dict(prefix=prefix, keep_vars=True)
        for k, v in state.items():
            sd[k] = ShardedTensor.from_rank_offsets(k, v.data, replica_id=(tp_rank, 0, dp_rank))
        return sd
