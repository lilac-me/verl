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

"""Captures intermediate hidden states from the policy model for Eagle3 draft training.

Registers persistent forward hooks on selected decoder layers and the embedding
layer.  After the policy forward pass completes, ``get_captured_states()``
assembles the captured tensors into:

* ``hidden_states``:  concatenation of aux-layer outputs  [batch, seq, N_aux * hidden]
* ``inputs_embeds``:  embedding-layer output              [batch, seq, hidden]

Teacher logits for distillation are taken directly from ``model_output["logits"]``
in ``EagleLossWrapper`` — no separate LM-head hook is needed.

Supported backends
------------------
* **HuggingFace / FSDP**: ``model.model.embed_tokens``, ``model.model.layers``
* **Megatron-Core (mcore), PP=1**: ``model.embedding.word_embeddings``,
  ``model.decoder.layers``

Pipeline parallelism (PP > 1) is NOT supported: each pipeline stage holds only
a subset of layers, so all required activations cannot be collected on a single
rank without explicit inter-stage communication.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from contextlib import contextmanager

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Eagle3 auxiliary-layer index heuristic (mirrors nemo-rl)
# ---------------------------------------------------------------------------

def get_eagle3_aux_layer_indices(num_layers: int) -> Tuple[int, ...]:
    """Return default aux-layer indices for Eagle3.

    Uses the nemo-rl heuristic: (1, num_layers // 2 - 1, num_layers - 4),
    deduplicated and sorted.
    """
    candidates = (
        1,
        max(0, num_layers // 2 - 1),
        max(1, num_layers - 1),
    )
    return tuple(sorted(set(candidates)))


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class CapturedStates:
    """Tensors captured from the policy model during a single forward pass."""

    # [batch, seq, N_aux * hidden_size]
    hidden_states: Optional[torch.Tensor] = None
    # [batch, seq, hidden_size]
    inputs_embeds: Optional[torch.Tensor] = None

# ---------------------------------------------------------------------------
# Main capture class
# ---------------------------------------------------------------------------

class HiddenStateCapture:
    """Registers persistent forward hooks on the policy model to collect activations.

    The hooks fire on every policy forward pass; captured data is cleared by the
    caller after each use to prevent stale state from leaking across steps.

    Args:
        model:             The policy nn.Module (or list[GPTModel] for Megatron).
        aux_layer_indices: Indices of decoder layers to capture.  None → auto.
        capture_logits:    Whether to hook the LM head for teacher logits.
    """

    def __init__(
        self,
        model: nn.Module,
        aux_layer_indices: Optional[Tuple[int, ...]] = None,
    ):
        self._model = self._unwrap_model(model)
        self._layers = self._find_layers(self._model)
        self._embed = self._find_embed(self._model)

        num_layers = len(self._layers)
        self._aux_indices = (
            aux_layer_indices
            if aux_layer_indices is not None
            else get_eagle3_aux_layer_indices(num_layers)
        )

        self._captured: Dict[str, torch.Tensor] = {}
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []

    # ------------------------------------------------------------------
    # Module discovery
    # ------------------------------------------------------------------

    @staticmethod
    def _unwrap_model(model, model_instances=None):
        """Unwrap model to return the final model instance"""
        if model_instances is None:
            from megatron.core.distributed import DistributedDataParallel as DDP
            from megatron.core.transformer.module import Float16Module
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            module_instances = (DDP, FSDP, Float16Module)

        return_list = True
        if not isinstance(model, list):
            model = [model]
            return_list = False
        unwrapped_model = []
        for model_module in model:
            while isinstance(model_module, module_instances):
                model_module = model_module.module
            unwrapped_model.append(model_module)
        if not return_list:
            return unwrapped_model[0]
        return unwrapped_model

    @staticmethod
    def _find_layers(inner: nn.Module) -> nn.ModuleList:
        """
        Locate the decoder layer ModuleList.

        HF path:   model.model.layers  (Qwen2/3, LLaMA, Mistral, …)
        Mcore path: model.decoder.layers (GPTModel with TransformerBlock)
        """
        # HF: model.model.layers
        candidate = getattr(getattr(inner, "model", None), "layers", None)
        if isinstance(candidate, nn.ModuleList) and len(candidate) > 0:
            return candidate

        # Mcore: model.decoder.layers
        candidate = getattr(getattr(inner, "decoder", None), "layers", None)
        if isinstance(candidate, nn.ModuleList) and len(candidate) > 0:
            return candidate

        raise AttributeError(
            "Cannot locate decoder layers. "
            "Expected model.model.layers (HF) or model.decoder.layers (Mcore)."
        )

    @staticmethod
    def _find_embed(inner: nn.Module) -> nn.Module:
        """
        Locate the token embedding module.

        HF path:   model.model.embed_tokens  (Qwen2/3, LLaMA, Mistral, …)
        Mcore path: model.embedding.word_embeddings (GPTModel)
        """
        # HF: model.model.embed_tokens
        embed = getattr(getattr(inner, "model", None), "embed_tokens", None)
        if embed is not None:
            return embed

        # Mcore: model.embedding.word_embeddings
        embed = getattr(getattr(inner, "embedding", None), "word_embeddings", None)
        if embed is not None:
            return embed

        raise AttributeError(
            "Cannot locate embedding layer. "
            "Expected model.model.embed_tokens (HF) or "
            "model.embedding.word_embeddings (Mcore)."
        )
    
    @staticmethod
    def _find_lm_head(inner: nn.Module) -> nn.Module:
        """
        Locate the LM head

        Check "lm_head", "output_layer"
        """
        for attr in ("lm_head", "output_layer"):
            candidate = getattr(inner, attr, None)
            if candidate is not None and isinstance(candidate, nn.Module):
                return candidate
        return None

    # ------------------------------------------------------------------
    # Hook construction
    # ------------------------------------------------------------------

    def _make_layer_hook(self, layer_idx: int):
        def hook(_module, _args, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            if hidden_states is None:
                return
            # Accept [batch, seq, hidden] (bshd) and [total_tokens, hidden] (thd)
            if hidden_states.dim() in (2, 3):
                self._captured[f"layer_{layer_idx}"] = hidden_states.detach().clone()
        return hook

    def _make_embed_hook(self):
        def hook(_module, _args, output):
            self._captured["embeds"] = output.detach().clone()
        return hook

    # ------------------------------------------------------------------
    # Hook lifecycle
    # ------------------------------------------------------------------

    def register_hooks(self) -> None:
        """
        Register persistent hooks on the policy model.

        Called once during initialization; hooks remain active for the
        lifetime of the manager.
        """
        self.clear_hooks()
        self._captured.clear()
        self._hooks.append(self._embed.register_forward_hook(self._make_embed_hook()))

        for idx in self._aux_indices:
            if idx < len(self._layers):
                self._hooks.append(
                    self._layers[idx].register_forward_hook(self._make_layer_hook(idx))
                )

    def clear_hooks(self) -> None:
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    # ------------------------------------------------------------------
    # State assembly
    # ------------------------------------------------------------------

    @contextmanager
    def capture_context(self):
        try:
            self.register_hooks()
            yield self
        finally:
            self.clear_hooks()

    def _assemble_captured_states(self) -> CapturedStates:
        """Assemble captured tensors into a CapturedStates container.

        Must be called after the policy forward pass and before _captured is
        cleared.
        """
        embeds = self._captured.get("embeds")

        hidden_chunks: List[torch.Tensor] = []
        for idx in sorted(self._aux_indices):
            tensor = self._captured.get(f"layer_{idx}")
            if tensor is not None:
                hidden_chunks.append(tensor)

        if not hidden_chunks:
            return CapturedStates(
                hidden_states=None,
                inputs_embeds=embeds,
            )

        return CapturedStates(
            hidden_states=torch.cat(hidden_chunks, dim=-1),  # [batch, seq, N*hidden]
            inputs_embeds=embeds,
        )
    
    def get_captured_states(self) -> CapturedStates:
        return self._assemble_captured_states()
    

def get_capture_context(model, aux_layer_indices):
    capture = HiddenStateCapture(model=model, aux_layer_indices=aux_layer_indices)
    return capture.capture_context(), capture
