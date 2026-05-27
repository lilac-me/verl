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

"""Megatron-native Eagle3 draft model (no modelopt dependency).

Mirrors nemo-rl's EagleModel: MegatronModule wrapping a TransformerBlock,
replacing modelopt's EagleModule.

  1. fc:    nn.Linear(N_aux*H -> H) -- NOT TP-sharded (duplicated).
  2. enorm: RMSNorm(H) -- normalises rolled embeddings.
  3. decoder: Megatron TransformerBlock (gpt_layer_local_spec, no TE).
              Layer-0 linear_qkv replaced: ColumnParallelLinear(2H -> qkv).
              Pre-hook injects cat(enorm(embeds), input_layernorm(h)).
  4. eagle_output_layer: ColumnParallelLinear(H -> vocab, gather_output=False).

State-dict key layout (eagle_module.* prefix):
    eagle_module.fc.weight
    eagle_module.enorm.weight
    eagle_module.decoder.layers.{i}.*
    eagle_module.decoder.final_layernorm.weight
    eagle_module.eagle_output_layer.weight
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from megatron.core import tensor_parallel
from megatron.core.models.common.embeddings import RotaryEmbedding
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.transformer import MegatronModule, TransformerConfig
from megatron.core.transformer.transformer_block import TransformerBlock


def build_eagle_transformer_config(hf_config, num_aux_layers: int = 3) -> TransformerConfig:
    """Build a Megatron TransformerConfig from an HF Eagle3 config."""
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
    # Custom attributes consumed by EagleModule.__init__
    config.eagle_num_aux_hidden_states = num_aux_layers
    config.draft_vocab_size = getattr(hf_config, "draft_vocab_size", hf_config.vocab_size)
    return config


class EagleModule(MegatronModule):
    def __init__(self, config: TransformerConfig):
        super().__init__(config=config)
        num_aux_hidden_states = getattr(config, "eagle_num_aux_hidden_states", 3)

        self.fc = nn.Linear(num_aux_hidden_states * config.hidden_size, config.hidden_size, bias=False)

        # enorm: normalise rolled embeddings before layer-0 injection
        self.enorm = nn.RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)

        self.rotary_pos_emb = RotaryEmbedding(
            kv_channels=config.kv_channels,
            rotary_percent=1.0,
            rotary_interleaved=False,
            seq_len_interpolation_factor=None,
            rotary_base=getattr(config, "rotary_base", 10000),
            rope_scaling=getattr(config, "rope_scaling", False),
            rope_scaling_factor=getattr(config, "rope_scaling_factor", 8.0),
            use_cpu_initialization=getattr(config, "use_cpu_initialization", not torch.cuda.is_available()),
        )
        
        self.decoder = TransformerBlock(
            config=config,
            spec=get_gpt_layer_local_spec(normalization="RMSNorm"),
        )

        self._embeddings = None
        # Optional draft-to-target token mapping: mapping[i] = i + d2t[i].
        # Populated from checkpoint when draft_vocab_size < vocab_size.
        self.register_buffer("d2t", None, persistent=True)

        last_layer = self.decoder.layers[-1]
        last_layer.register_forward_hook(self._eagle3_layer_forward_hook)

        layer = self.decoder.layers[0]
        self_attention = layer.self_attention
        self_attention.register_forward_pre_hook(self._eagle3_attention_forward_pre_hook)
        self_attention.linear_qkv = tensor_parallel.ColumnParallelLinear(
            self_attention.config.hidden_size * 2,
            self_attention.query_projection_size + 2 * self_attention.kv_projection_size,
            config=self_attention.config,
            init_method=self_attention.config.init_method,
            gather_output=False,
            bias=self_attention.config.add_bias_linear or self_attention.config.add_qkv_bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="qkv",
        )

        if self.config.draft_vocab_size != self.config.vocab_size:
            # Need an extra lm_head for eagle module since vocab size is reduced.
            if self.config.draft_vocab_size > self.config.vocab_size:
                raise ValueError(
                    "EAGLE module's vocab size should be <= base model vocab size!"
                )
        self.eagle_output_layer = tensor_parallel.ColumnParallelLinear(
            self.config.hidden_size,
            self.config.draft_vocab_size,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            gather_output=False,
            skip_weight_param_allocation=False,
        )

    def _eagle3_layer_forward_hook(self, _module, input, output):
        hidden_states = (
            output.clone().detach()
            if isinstance(output, torch.Tensor)
            else output[0].clone().detach()
        )
        self._next_hidden_states_input = hidden_states

    def _eagle3_attention_forward_pre_hook(self, _module, input_layernorm_output):
        if self._embeddings is None:
            raise ValueError("EagleModule attention pre-hook called before embeddings set")
        embeddings = self._embeddings
        self._embeddings = None
        # shape: [S, B, 2 * H]
        return torch.cat([embeddings, input_layernorm_output[0]], dim=-1)

    def _freeze_output_layer(self):
        for p in self.eagle_output_layer.parameters():
            p.requires_grad_(False)

    def _unfreeze_output_layer(self):
        for p in self.eagle_output_layer.parameters():
            p.requires_grad_(True)

    def forward(
        self,
        embeddings: torch.Tensor,      # [S, B, N_aux*H]
        hidden_states: torch.Tensor,   # [S, B, N_aux*H]
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._embeddings = self.enorm(embeddings)
        self._next_hidden_states_input = None
        rotary_pos_emb = self.rotary_pos_emb(hidden_states.shape[0])
        hidden_states = self.decoder(
            hidden_states,
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
        )
        if self._next_hidden_states_input is None:
            next_hidden_states_input = hidden_states
        else:
            next_hidden_states_input = self._next_hidden_states_input
            self._next_hidden_states_input = None

        return hidden_states, next_hidden_states_input


class EagleDraftModel(MegatronModule):
    """Eagle3 draft model. Forward: [B, S, *] batch-first. Returns [B, S, vocab/TP]."""

    def __init__(self, config: TransformerConfig):
        super().__init__(config=config)
        self.eagle_module = EagleModule(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.eagle_module.fc(hidden_states)

        hidden_states, _ = self.eagle_module(
            embeddings=inputs_embeds,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )
        logits, _ = self.eagle_module.eagle_output_layer(hidden_states)
        logits = logits.transpose(0, 1).contiguous()  # [S, B, vocab/TP] -> [B, S, vocab/TP]
        return logits

    def freeze_output_layer(self):
        self.eagle_module._freeze_output_layer()

    def unfreeze_output_layer(self):
        self.eagle_module._unfreeze_output_layer()
