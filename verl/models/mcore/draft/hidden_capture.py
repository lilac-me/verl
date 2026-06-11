# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
"""Capture policy input embeddings and auxiliary hidden states for EAGLE3.

Ported from NeMo-RL PR #2078 hidden_capture.py with verl adjustments:
  * supports a LIST of model chunks (verl Megatron engine VPP layout)
  * batched isend/irecv for the PP gather (HCCL-friendlier on Ascend than
    sequential send/recv; still guarded — see PP WARNING below)
  * detach is mandatory by default: the draft loss must not perturb the
    policy backbone. `policy-grad invariance` unit test relies on this.

PP WARNING: with pipeline parallelism > 1 the gather issues extra P2P traffic
interleaved with the 1F1B schedule's own P2P. NeMo-RL ships the same pattern,
but ordering across microbatches with interleaved (VPP) schedules is the
riskiest part of this feature. Validate PP>1 with NCCL/HCCL blocking-wait
enabled before trusting it at scale; until then prefer PP=1 or non-interleaved
schedules. The capture raises if VPP>1 and PP>1 simultaneously.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import Tensor, nn

from megatron.core import parallel_state
from megatron.core.utils import unwrap_model


def get_eagle3_aux_hidden_state_layers(num_layers: int) -> tuple[int, ...]:
    """Default aux layers; matches vLLM's eagle3 convention (early/mid/late)."""
    candidate = (1, max(0, num_layers // 2 - 1), max(1, num_layers - 4))
    return tuple(sorted(set(candidate)))


_DTYPE_TO_CODE = {torch.float16: 0, torch.bfloat16: 1, torch.float32: 2}
_CODE_TO_DTYPE = {v: k for k, v in _DTYPE_TO_CODE.items()}


@dataclass
class CapturedStates:
    hidden_states: Optional[Tensor] = None   # [s, b, 3h] on the draft-owner rank
    inputs_embeds: Optional[Tensor] = None   # [s, b, h]  on the draft-owner rank


class HiddenStateCapture:
    """Registers forward hooks on policy decoder layers + embedding.

    Usage (inside MegatronEngine.forward_step):
        with capture.capture_context():
            output = forward_fn(model, ...)
        states = capture.get_captured_states()   # collective across PP group
    """

    def __init__(
        self,
        model_chunks: List[nn.Module],
        num_layers: int,
        aux_layer_indices: Optional[Tuple[int, ...]] = None,
        detach: bool = True,
    ):
        self.model_chunks = [unwrap_model(m) for m in model_chunks]
        self.num_layers = num_layers
        self.aux_layer_indices = (
            tuple(aux_layer_indices) if aux_layer_indices is not None
            else get_eagle3_aux_hidden_state_layers(num_layers)
        )
        self.detach = detach

        self.pp_size = parallel_state.get_pipeline_model_parallel_world_size()
        self.pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        self.is_first_stage = parallel_state.is_pipeline_first_stage(ignore_virtual=True)
        self.is_last_stage = parallel_state.is_pipeline_last_stage(ignore_virtual=True)

        vpp = parallel_state.get_virtual_pipeline_model_parallel_world_size()
        if self.pp_size > 1 and vpp is not None and vpp > 1:
            raise NotImplementedError(
                "EAGLE3 hidden capture: interleaved VPP with PP>1 not validated (see PP WARNING)."
            )

        # global layer idx -> (chunk_idx, local layer module)
        self._local_aux: Dict[int, nn.Module] = {}
        for chunk in self.model_chunks:
            decoder = getattr(chunk, "decoder", None)
            if decoder is None:
                continue
            for layer in decoder.layers:
                gidx = int(layer.layer_number) - 1
                if gidx in self.aux_layer_indices:
                    self._local_aux[gidx] = layer

        self._layer_owner = self._compute_layer_owner_map()
        self._captured: Dict[str, Tensor] = {}
        self._hooks: list = []

    # ------------------------------------------------------------------ #
    def _compute_layer_owner_map(self) -> Dict[int, int]:
        if self.pp_size == 1 or not dist.is_initialized():
            return {i: 0 for i in range(self.num_layers)}
        pp_group = parallel_state.get_pipeline_model_parallel_group()
        local_mask = torch.zeros(self.num_layers, dtype=torch.int64, device=torch.cuda.current_device())
        for chunk in self.model_chunks:
            decoder = getattr(chunk, "decoder", None)
            if decoder is None:
                continue
            for layer in decoder.layers:
                gidx = int(layer.layer_number) - 1
                if 0 <= gidx < self.num_layers:
                    local_mask[gidx] = 1
        gathered = [torch.zeros_like(local_mask) for _ in range(self.pp_size)]
        dist.all_gather(gathered, local_mask, group=pp_group)
        owner = {}
        for gidx in range(self.num_layers):
            for rank, mask in enumerate(gathered):
                if int(mask[gidx].item()) == 1:
                    owner[gidx] = rank
                    break
        return owner

    # ------------------------------------------------------------------ #
    def _make_layer_hook(self, gidx: int):
        def hook(_m, _args, output):
            hs = output[0] if isinstance(output, tuple) else output
            if hs is None:
                return
            self._captured[f"layer_{gidx}"] = hs.detach().clone() if self.detach else hs

        return hook

    def _make_embedding_hook(self):
        def hook(_m, _args, output):
            self._captured["embeds"] = output.detach().clone() if self.detach else output

        return hook

    def register_hooks(self) -> None:
        self.clear_hooks()
        self._captured.clear()
        if self.is_first_stage:
            for chunk in self.model_chunks:
                emb = getattr(chunk, "embedding", None)
                if emb is not None:
                    self._hooks.append(emb.register_forward_hook(self._make_embedding_hook()))
                    break
        for gidx, layer in self._local_aux.items():
            self._hooks.append(layer.register_forward_hook(self._make_layer_hook(gidx)))

    def clear_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    @contextmanager
    def capture_context(self):
        try:
            self.register_hooks()
            yield self
        finally:
            self.clear_hooks()

    # ------------------------------------------------------------------ #
    # PP gather: every PP rank must call get_captured_states() exactly once
    # per microbatch forward (collective).
    # ------------------------------------------------------------------ #
    def get_captured_states(self) -> CapturedStates:
        if self.pp_size == 1:
            return self._assemble_local()

        pp_group = parallel_state.get_pipeline_model_parallel_group()
        pp_ranks = dist.get_process_group_ranks(pp_group)
        last_global_rank = pp_ranks[-1]
        my_global_rank = dist.get_rank()
        is_owner = my_global_rank == last_global_rank

        p2p_ops = []
        recv_bufs: Dict[str, Tensor] = {}

        def schedule(key: str, src_pp_rank: int, local_tensor: Optional[Tensor]):
            # exchange shape/dtype metadata first (small blocking exchange),
            # then batch the payload isend/irecv.
            src_global = pp_ranks[src_pp_rank]
            if src_pp_rank == self.pp_size - 1:
                return  # already on owner
            if my_global_rank == src_global:
                meta = torch.tensor(
                    [_DTYPE_TO_CODE[local_tensor.dtype], *local_tensor.shape],
                    dtype=torch.int64, device=local_tensor.device,
                )
                dist.send(meta, dst=last_global_rank, group=pp_group)
                p2p_ops.append(dist.P2POp(dist.isend, local_tensor.contiguous(), last_global_rank, group=pp_group))
            elif is_owner:
                meta = torch.empty(4, dtype=torch.int64, device=torch.cuda.current_device())
                dist.recv(meta, src=src_global, group=pp_group)
                dtype = _CODE_TO_DTYPE[int(meta[0].item())]
                shape = tuple(int(x) for x in meta[1:].tolist())
                buf = torch.empty(shape, dtype=dtype, device=torch.cuda.current_device())
                recv_bufs[key] = buf
                p2p_ops.append(dist.P2POp(dist.irecv, buf, src_global, group=pp_group))

        # deterministic order across ranks: embeds first, then sorted layers
        schedule("embeds", 0, self._captured.get("embeds"))
        for gidx in sorted(self.aux_layer_indices):
            schedule(f"layer_{gidx}", self._layer_owner[gidx], self._captured.get(f"layer_{gidx}"))

        if p2p_ops:
            reqs = dist.batch_isend_irecv(p2p_ops)
            for r in reqs:
                r.wait()

        if not is_owner:
            return CapturedStates()
        for k, v in recv_bufs.items():
            self._captured[k] = v
        return self._assemble_local()

    def _assemble_local(self) -> CapturedStates:
        embeds = self._captured.get("embeds")
        chunks = [self._captured[f"layer_{g}"] for g in sorted(self.aux_layer_indices)
                  if f"layer_{g}" in self._captured]
        hidden = torch.cat(chunks, dim=-1) if chunks else None  # [s, b, 3h]
        return CapturedStates(hidden_states=hidden, inputs_embeds=embeds)
