# Eagle3 Draft Model Loss 机制

本文档说明在 verl FSDP 训练后端中，Eagle3 草稿模型（draft model）的损失函数是如何计算、如何反传、以及梯度如何正确同步到所有数据并行节点的。

---

## 1. 训练目标

Eagle3 草稿模型的目标是**实时追踪主 policy 的 token 概率分布**。RL 训练过程中 policy 持续更新，如果草稿模型的分布与 policy 偏差越来越大，推理阶段的接受率（acceptance rate）就会下降。

解决方案：把草稿模型的蒸馏训练嵌入到 RL 训练循环里，每步都用当前 policy 的输出作为"教师"来更新草稿模型。

---

## 2. 教师 logits 的来源

RL 训练时，policy 做一次 forward 就会产出 LM head 的完整 logits。但 verl 的 FSDP engine 只把 `log_probs` 放进 `model_output`，原始 logits 会被丢弃。

为此，在 LM head 模块上挂了一个 **forward hook**，在 FSDP forward 过程中实时捕获并保存 logits：

```python
# verl/workers/eagle/hidden_capture.py
def _make_lm_head_hook(self):
    def hook(_module, _args, output):
        logits = output[0] if isinstance(output, tuple) else output
        if logits is not None and logits.is_floating_point():
            self._captured["lm_head_logits"] = logits.detach().float()
    return hook
```

同时还会捕获若干中间层的 hidden states 和 embedding 层输出，这些是草稿模型的输入。

> **注意**：教师是**当前训练步的 policy**，而非冻结的 reference policy。草稿模型每步都在追 policy 的最新分布，这是维持 acceptance rate 上升的核心机制。

---

## 3. 损失函数

### 3.1 公式

使用**前向 KL 散度**（等价于以教师概率为权重的 soft cross-entropy）：

```
L_draft = -Σ_t mask[t] · Σ_v softmax(teacher_logits[t, v]) · log_softmax(draft_logits[t, v])
          / Σ_t mask[t]
```

其中 `mask[t]` 为 response token 掩码（prompt 位置 = 0，padding = 0，response = 1）。

梯度**只通过 `draft_logits`** 流向草稿模型参数——`teacher_logits` 在进入 loss 计算前已经 `detach()`，不会影响 policy 的梯度。

### 3.2 实现

```python
# verl/workers/eagle/losses.py
teacher_probs     = F.softmax(teacher_logits.detach(), dim=-1)
student_log_probs = F.log_softmax(draft_logits, dim=-1)
per_token_loss    = -(teacher_probs * student_log_probs).sum(dim=-1)  # [batch, seq]

num_valid = response_mask.bool().sum().clamp(min=1)
loss = (per_token_loss * response_mask).sum() / num_valid
```

---

## 4. 时间步对齐（Eagle3 Time-Step Alignment）

Eagle3 的结构要求：**位置 t 的草稿输出预测位置 t+1 的 token 分布**。这通过两步实现：

### 4.1 输入 embedding 左移

把 `inputs_embeds` 向左滚动一位，使位置 t 处看到的是 token t+1 的 embedding（即下一个 token 的嵌入向量，模拟推理时的 teacher forcing）：

```python
rolled_embeds = torch.roll(inputs_embeds, shifts=-1, dims=1)
# rolled_embeds[t] = inputs_embeds[t+1]
# rolled_embeds[T-1] = inputs_embeds[0]  ← 最后一位发生了 wrap-around，需排除
```

### 4.2 logits 与 mask 左移

`draft_model(hidden_states, rolled_embeds)` 返回的 `draft_logits[t]` 已经是对 t+1 的预测，与 `teacher_logits[t]`（policy 对 t+1 的预测）直接对应。

`roll_for_eagle_alignment` 再把 draft/teacher logits 和 mask 统一左移一位，然后**将最后一个位置的 mask 置零**，排除 `rolled_embeds[T-1]` 的 wrap-around 位置对损失的污染：

```python
# verl/workers/eagle/losses.py
def roll_for_eagle_alignment(draft_logits, teacher_logits, response_mask):
    draft_logits   = torch.roll(draft_logits,   shifts=-1, dims=1)
    teacher_logits = torch.roll(teacher_logits, shifts=-1, dims=1)
    response_mask  = torch.roll(response_mask,  shifts=-1, dims=1).clone()
    response_mask[:, -1] = 0   # 排除 wrap-around 位置
    return draft_logits, teacher_logits, response_mask
```

---

## 5. 梯度反传与双路 Optimizer

### 5.1 计算图结构

`EagleLossWrapper` 返回的总损失为：

```
L_total = L_policy + λ × L_draft
```

FSDP engine 对 `L_total` 调用 `backward()`，计算图分两路：

```
L_policy ──→ policy 参数梯度   （FSDP 自动跨 rank all-reduce）
L_draft  ──→ draft 参数梯度    （draft model 不在 FSDP 里，需手动同步）
              ↑
         hidden_states 已经 detach，梯度不会回流到 policy
```

### 5.2 两个独立的 Optimizer

| | Policy | Draft |
|---|---|---|
| 参数管理 | FSDP | 单独的 `nn.Module` |
| Optimizer | FSDP 内置 AdamW | 独立 AdamW（可配置 lr/wd） |
| 步进时机 | FSDP engine 内部 | `train_batch()` 结束后调用 `optimizer_step()` |

```yaml
# 配置示例
actor_rollout_ref:
  model:
    eagle_draft:
      enabled: true
      loss_weight: 0.1
      optimizer:
        lr: 1.0e-4
        weight_decay: 0.0
```

---

## 6. 多卡梯度同步

### 6.1 问题

draft model 没有 FSDP/DDP wrapper，`backward()` 后每个 DP rank 只积累了自己 micro-batch 对应的本地梯度。如果直接调用 `optimizer.step()`，各 rank 的 draft model 参数会逐渐分叉；后续 `state_dict_for_vllm()` 只取 rank 0 的权重同步到 vLLM，等价于只用了 1/N 的数据做更新。

### 6.2 修复

在 `optimizer_step()` 中，步进前对所有参数梯度做显式 `all_reduce`，行为与 DDP 的 grad hook 等价：

```python
# verl/workers/eagle/manager.py
def optimizer_step(self) -> None:
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        world_size = dist.get_world_size()
        for p in self.draft_model.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad.div_(world_size)

    torch.nn.utils.clip_grad_norm_(
        [p for p in self.draft_model.parameters() if p.requires_grad],
        max_norm=1.0,
    )
    self.optimizer.step()
    self.optimizer.zero_grad()
```

---

## 7. 完整的单步数据流

```
┌─ FSDP Forward ────────────────────────────────────────────────────────────┐
│  policy model forward                                                      │
│    → forward hooks 触发，捕获：                                            │
│        hidden_states: layer 1, mid-1, last-4  [1, total_nnz, hidden]     │
│        inputs_embeds:                          [1, total_nnz, hidden]     │
│        lm_head_logits (detach + float32):      [1, total_nnz, vocab]     │
└────────────────────────────────────────────────────────────────────────────┘
                            │
                   EagleLossWrapper.__call__()
                            │
┌─ Loss 计算 ───────────────────────────────────────────────────────────────┐
│  ① base_loss_fn()  →  L_policy                                            │
│  ② unpack packed tensors（use_remove_padding 适配）                        │
│  ③ roll(inputs_embeds, -1)  →  rolled_embeds                              │
│  ④ draft_model(hidden_states, rolled_embeds)  →  draft_logits             │
│  ⑤ roll_for_eagle_alignment(draft_logits, teacher_logits, mask)           │
│  ⑥ L_draft = soft_CE(draft_logits.float(), teacher_logits, mask)         │
│  ⑦ return L_total = L_policy + λ × L_draft                               │
└────────────────────────────────────────────────────────────────────────────┘
                            │
                   L_total.backward()
                            │
          ┌─────────────────┴──────────────────┐
          ▼                                     ▼
  policy 参数梯度                        draft 参数梯度
  （FSDP 自动 all-reduce）               （手动 all-reduce）
          │                                     │
  FSDP optimizer.step()          eagle_manager.optimizer_step()
          │                                     │
  policy 参数更新 ✓                     draft 参数更新（所有 rank 一致）✓
                            │
                   update_weights()
                            │
          ┌─────────────────┴──────────────────┐
          ▼                                     ▼
  policy weights → vLLM main model     draft weights → vLLM Eagle3 proposer
```

---

## 相关文件

| 文件 | 职责 |
|------|------|
| `verl/workers/eagle/losses.py` | 损失函数：`eagle_draft_loss`, `roll_for_eagle_alignment` |
| `verl/workers/eagle/hidden_capture.py` | Forward hooks：捕获 hidden states 和 LM head logits |
| `verl/workers/eagle/manager.py` | `EagleDraftManager`：optimizer 管理；`EagleLossWrapper`：包装 policy loss |
| `verl/workers/eagle/draft_model.py` | `EagleDraftModelWrapper`：封装 HF Eagle3 模型的 forward |
| `verl/workers/eagle/config.py` | `EagleDraftConfig`：loss_weight、optimizer lr/wd 等配置 |
