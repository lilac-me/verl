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

"""Captures intermediate hidden states from the policy model for Eagle draft training.

Registers temporary forward hooks on selected decoder layers and the embedding
layer.  After the policy forward pass, ``get_captured_states()`` assembles the
captured tensors into the format expected by the Eagle draft model:

* ``hidden_states``:  concatenation of aux-layer outputs  [batch, seq, N * hidden]
* ``inputs_embeds``:  embedding-layer output              [batch, seq, hidden]

The caller must call ``roll_inputs_embeds`` on the returned embeds to apply the
time-step shift required by Eagle3 before passing to the draft model.

Supports FSDP and DDP (single-node and multi-node DP).  Pipeline parallelism is
NOT supported — see note in EagleDraftConfig.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import ContextManager, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


def get_eagle3_aux_layer_indices(num_layers: int) -> Tuple[int, ...]:
    """Return the default auxiliary layer indices for Eagle3.

    Mirrors the heuristic used in nemo-rl:
      (1, num_layers // 2 - 1, num_layers - 4)
    with deduplication and sorting.
    """
    candidates = (
        1,
        max(0, num_layers // 2 - 1),
        max(1, num_layers - 4),
    )
    return tuple(sorted(set(candidates)))


@dataclass
class CapturedStates:
    """Hidden states captured from the policy model during a forward pass."""

    # [batch, seq, N_aux * hidden_size]  — concatenation along last dim
    hidden_states: Optional[torch.Tensor] = None
    # [batch, seq, hidden_size]
    inputs_embeds: Optional[torch.Tensor] = None
    # [batch, seq, vocab_size]  — LM head output (teacher logits for draft distillation)
    lm_head_logits: Optional[torch.Tensor] = None


def roll_inputs_embeds(embeds: torch.Tensor) -> torch.Tensor:
    """Left-shift input embeddings by one token for Eagle3 time-step alignment.

    Eagle3 trains the draft at position t to predict the policy distribution
    for position t+1.  Shifting the embeddings left by one aligns the draft
    input at position t with the teacher output at position t+1.
    """
    # embeds: [batch, seq, hidden] — roll along the sequence dim (dim=1)
    return torch.roll(embeds, shifts=-1, dims=1)


class HiddenStateCapture:
    """Register forward hooks on a HF-style policy model and collect activations.

    Works with any ``nn.Module`` that exposes:
      * An embedding block accessible at ``model.model.embed_tokens``
        (HuggingFace convention) or ``model.embed_tokens``.
      * Decoder layers accessible at ``model.model.layers``
        or ``model.layers``.

    FSDP / DDP compatibility:
      Hooks are registered on the *unwrapped* inner module so they fire
      during the forward pass regardless of the distributed wrapper.
    """

    def __init__(
        self,
        model: nn.Module,
        aux_layer_indices: Optional[Tuple[int, ...]] = None,
        capture_logits: bool = True,
    ):
        self._raw_model = model
        self._inner = self._unwrap(model)
        self._layers = self._find_layers(self._inner)
        self._embed = self._find_embed(self._inner)
        self._lm_head = self._find_lm_head(self._inner) if capture_logits else None

        num_layers = len(self._layers)
        self._aux_indices: Tuple[int, ...] = (
            aux_layer_indices
            if aux_layer_indices is not None
            else get_eagle3_aux_layer_indices(num_layers)
        )

        self._captured: Dict[str, torch.Tensor] = {}
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []

    # ------------------------------------------------------------------
    # Module discovery helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _unwrap(model: nn.Module) -> nn.Module:
        """Strip FSDP / DDP / compiled wrappers to reach the inner module."""
        from torch.nn.parallel import DistributedDataParallel

        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            while isinstance(model, FSDP):
                model = model.module
        except ImportError:
            pass

        while isinstance(model, DistributedDataParallel):
            model = model.module

        # torch.compile wraps in OptimizedModule
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod

        return model

    @staticmethod
    def _find_layers(inner: nn.Module) -> nn.ModuleList:
        """Return the transformer decoder layer list."""
        for attr in ("model", ""):
            obj = getattr(inner, attr, inner) if attr else inner
            for layers_attr in ("layers", "decoder", "h", "blocks"):
                candidate = getattr(obj, layers_attr, None)
                if isinstance(candidate, nn.ModuleList) and len(candidate) > 0:
                    return candidate
        raise AttributeError(
            "Cannot locate decoder layers in model. "
            "Expected model.model.layers or model.layers (HF convention)."
        )

    @staticmethod
    def _find_lm_head(inner: nn.Module) -> Optional[nn.Module]:
        """Return the LM head module (linear projection to vocab), or None if not found."""
        for attr in ("lm_head", "embed_out", "output", "head"):
            candidate = getattr(inner, attr, None)
            if candidate is not None and isinstance(candidate, nn.Module):
                return candidate
        return None

    @staticmethod
    def _find_embed(inner: nn.Module) -> nn.Module:
        """Return the token embedding module."""
        for path in (
            ("model", "embed_tokens"),
            ("embed_tokens",),
            ("model", "wte"),
            ("wte",),
            ("model", "embedding"),
            ("embedding",),
        ):
            obj = inner
            found = True
            for part in path:
                obj = getattr(obj, part, None)
                if obj is None:
                    found = False
                    break
            if found and obj is not None:
                return obj
        raise AttributeError(
            "Cannot locate embedding layer in model. "
            "Expected model.model.embed_tokens (HF convention)."
        )

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def _make_layer_hook(self, layer_idx: int):
        def hook(_module, _args, output):
            # HF decoder layers return a tuple; first element is hidden states
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden is not None:
                # Store as [batch, seq, hidden] — HF convention
                self._captured[f"layer_{layer_idx}"] = hidden.detach()

        return hook

    def _make_embed_hook(self):
        def hook(_module, _args, output):
            embeds = output[0] if isinstance(output, tuple) else output
            if embeds is not None:
                self._captured["embeds"] = embeds.detach()

        return hook

    def _make_lm_head_hook(self):
        def hook(_module, _args, output):
            # LM head: [batch, seq, vocab] or nested tensor; keep in float32 for soft CE
            logits = output[0] if isinstance(output, tuple) else output
            if logits is not None and isinstance(logits, torch.Tensor) and logits.is_floating_point():
                self._captured["lm_head_logits"] = logits.detach().float()

        return hook

    def _register_hooks(self) -> None:
        self._hooks.clear()
        self._captured.clear()

        self._hooks.append(self._embed.register_forward_hook(self._make_embed_hook()))

        for idx in self._aux_indices:
            if idx < len(self._layers):
                self._hooks.append(
                    self._layers[idx].register_forward_hook(self._make_layer_hook(idx))
                )

        if self._lm_head is not None:
            self._hooks.append(self._lm_head.register_forward_hook(self._make_lm_head_hook()))

    def _clear_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @contextmanager
    def capture_context(self):
        """Context manager: hooks are active inside the ``with`` block."""
        try:
            self._register_hooks()
            yield self
        finally:
            self._clear_hooks()

    def get_captured_states(self) -> CapturedStates:
        """Assemble captured tensors into CapturedStates.

        Must be called after the policy forward pass completes (still inside
        ``capture_context``).
        """
        embeds = self._captured.get("embeds")

        hidden_chunks: List[torch.Tensor] = []
        for idx in sorted(self._aux_indices):
            tensor = self._captured.get(f"layer_{idx}")
            if tensor is not None:
                hidden_chunks.append(tensor)

        lm_head_logits = self._captured.get("lm_head_logits")

        if not hidden_chunks:
            return CapturedStates(hidden_states=None, inputs_embeds=embeds, lm_head_logits=lm_head_logits)

        return CapturedStates(
            hidden_states=torch.cat(hidden_chunks, dim=-1),  # [batch, seq, N*hidden]
            inputs_embeds=embeds,
            lm_head_logits=lm_head_logits,
        )


def get_capture_context(
    model: nn.Module,
    enabled: bool = False,
    aux_layer_indices: Optional[Tuple[int, ...]] = None,
) -> Tuple[ContextManager, Optional[HiddenStateCapture]]:
    """Return a (context_manager, capture) pair.

    If ``enabled`` is False a no-op context manager is returned and ``capture``
    is None, incurring zero overhead.
    """
    if not enabled:
        return nullcontext(), None

    capture = HiddenStateCapture(model=model, aux_layer_indices=aux_layer_indices)
    return capture.capture_context(), capture
