# Eagle3 Online Draft Model Training in verl

## 背景

[Eagle3](https://arxiv.org/abs/2503.05030) 是一种投机解码（Speculative Decoding）方法：一个小的 **draft 模型** 快速提议后续 token，大的 **policy 模型** 并行验证，被接受的 token 无需额外推理成本，从而显著加速 rollout 生成。

**在线训练（Online Draft Training）** 指在 RL 训练循环中同步训练 draft 模型，使其始终与不断更新的 policy 对齐。若不在线训练，draft 模型会随着 policy 更新而逐渐过时，导致接受率下降、加速效果劣化。

本文档描述如何在 verl 中实现该功能，设计参考 [nemo-rl PR #2078](https://github.com/NVIDIA-NeMo/RL/pull/2078)。

---

## 与 nemo-rl 实现的对比

| 维度 | nemo-rl (原始) | verl (本实现) |
|---|---|---|
| 训练后端 | Megatron Core | FSDP (主) |
| Draft 模型加载 | modelopt `EagleModule` | HuggingFace `trust_remote_code` |
| 隐状态捕获 | Megatron hook + PP 跨 rank 通信 | PyTorch forward hook，FSDP 兼容 |
| vLLM 权重同步 | 专用 IPC refit | 复用 `BucketedWeightSender` |
| 配置系统 | TypedDict + YAML | dataclass + OmegaConf |
| Pipeline 并行 | 不支持 | 不支持（同） |

---

## 整体架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                    verl Eagle3 Online Draft Training                 │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  ROLLOUT PHASE  (vLLM + Eagle3 Speculative Decoding)         │   │
│  │                                                              │   │
│  │   Prompts ──► vLLM Engine ──► Responses + Log Probs         │   │
│  │                  │                                           │   │
│  │        ┌─────────┴──────────┐                               │   │
│  │   Policy Weights        Draft Weights  ← synced each step   │   │
│  │   (base model)          (Eagle3 head)                       │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                         │                                            │
│                         ▼  Advantages (GRPO / PPO)                  │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  TRAINING PHASE  (FSDP)                                      │   │
│  │                                                              │   │
│  │   Policy Model (FSDP)           Eagle Draft Model            │   │
│  │   ┌─────────────────┐           ┌──────────────────────┐    │   │
│  │   │ embed ──hook────┼──────────►│ rolled_embeds         │    │   │
│  │   │ layer[1] ──hook─┼──────────►│ concat(hidden[0],     │    │   │
│  │   │ layer[N/2]─hook─┼──────────►│        hidden[1],     │    │   │
│  │   │ layer[N-4]─hook─┼──────────►│        hidden[2])     │    │   │
│  │   │ lm_head ───hook─┼──────────►│ → draft logits        │    │   │
│  │   │  (teacher logits)│           └──────────────────────┘    │   │
│  │   └─────────────────┘                    │                   │   │
│  │            │                             │                   │   │
│  │            └──────────────┬──────────────┘                   │   │
│  │                           ▼                                  │   │
│  │       L_policy = GRPO(policy_logits, advantages)             │   │
│  │       L_draft  = SoftCE(draft_logits, lm_head_logits.detach) │   │
│  │       L_total  = L_policy + λ × L_draft                      │   │
│  │                           │                                  │   │
│  │              backward() + optimizer.step() × 2               │   │
│  │              (policy FSDP optimizer + draft AdamW)           │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                         │                                            │
│              After each step: sync policy + draft → vLLM            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Eagle3 时步对齐（Time-Step Alignment）

Eagle3 训练的关键细节：draft 模型在位置 `t` 处预测 policy 在位置 `t+1` 的分布。

实现方式：将输入嵌入向量向左滚动 1 步，同步将 teacher logits 也左滚 1 步，最后一位用 mask 置零。

```
原始序列:   [t0, t1, t2, t3, t4]
滚动后输入: [t1, t2, t3, t4, t0]   ← left roll by 1
滚动后目标: [t1, t2, t3, t4,  0]   ← last pos masked
```

相关代码：[`verl/workers/eagle/losses.py`](../verl/workers/eagle/losses.py) 中的 `roll_for_eagle_alignment()`。

---

## 蒸馏损失

对每个有效 response token，最小化 draft 分布与 policy 分布的前向 KL：

```
L_draft = -Σ_t mask[t] · Σ_v softmax(teacher_logits[t,v]) · log_softmax(draft_logits[t,v])
```

总损失：

```
L_total = L_policy + λ × L_draft
```

其中 `λ = eagle_draft.loss_weight`（默认 0.1）。

---

## 文件清单

### 新增文件

```
verl/workers/eagle/
├── __init__.py           # 模块导出
├── config.py             # EagleDraftConfig 配置 dataclass
├── hidden_capture.py     # HiddenStateCapture：forward hook 捕获隐状态 + LM head logits
├── losses.py             # eagle_draft_loss()：软交叉熵蒸馏损失 + 时步对齐
├── draft_model.py        # EagleDraftModelWrapper：HF Eagle3 模型封装 + FSDP 初始化
└── manager.py            # EagleDraftManager + EagleLossWrapper：训练生命周期管理

examples/eagle_grpo/
├── grpo_eagle3_qwen3_1.7b.yaml   # 示例训练配置
└── run_eagle_grpo.sh             # 示例训练脚本
```

### 修改文件

```
verl/workers/config/model.py
  + EagleDraftConfig dataclass
  + HFModelConfig.eagle_draft 字段

verl/workers/utils/losses.py
  + eagle_ppo_loss()：ppo_loss 增量封装，含 draft loss 项

verl/workers/engine_workers.py
  + TrainingWorker.init_eagle_draft()       ← 初始化 draft 模型、注册 hooks、替换 loss_fn
  + TrainingWorker.get_eagle_draft_state_dict()  ← 获取 draft 权重供 vLLM 同步
  + TrainingWorker.train_batch()            ← 在 policy optimizer step 后调用 draft optimizer step
  + ActorRolloutRefWorker.__init__()        ← 根据配置自动启用 Eagle

verl/workers/rollout/vllm_rollout/vllm_rollout.py
  + ServerAdapter.update_draft_weights()   ← 通过 ZMQ/IPC 同步 draft 权重到 vLLM

verl/workers/rollout/vllm_rollout/vllm_async_server.py
  + update_draft_weights_from_ipc()        ← server 侧接收 draft 权重并加载到推理引擎
```

---

## 核心组件详解

### 1. `EagleDraftConfig`

```python
@dataclass
class EagleDraftConfig:
    enabled: bool = False
    model_path: str = None            # HF repo 或本地路径
    loss_weight: float = 0.1          # λ
    aux_layer_indices: list[int] = None  # 默认自动选取 (layer 1, mid-1, last-4)
    optimizer: EagleDraftOptimizerConfig = ...
```

### 2. `HiddenStateCapture`

在 policy 模型的前向传播中通过 PyTorch `register_forward_hook` 捕获：
- **嵌入层输出**：`[batch, seq, hidden]`
- **3 个辅助层输出**（layer 1、中间层、倒数第 4 层）：拼接为 `[batch, seq, 3×hidden]`
- **LM head 输出**：`[batch, seq, vocab]`（teacher logits，float32）

Hooks 持久注册（Manager 初始化后全程有效），每次 forward 后由 `EagleLossWrapper` 清空缓存。

### 3. `EagleLossWrapper`

替换 `TrainingWorker` 中的 `loss_fn`，在每次 `forward_step` 调用时：

```
policy_loss, metrics = base_loss_fn(model_output, data)
captured = capture.get_captured_states()
draft_logits = draft_model(captured.hidden_states, rolled(captured.inputs_embeds))
draft_loss = SoftCE(draft_logits, captured.lm_head_logits)
return policy_loss + λ * draft_loss, metrics
```

### 4. `EagleDraftManager`

- 管理 draft 模型及其独立 AdamW optimizer
- `optimizer_step()`：在 `engine.train_batch()` 完成后被 `TrainingWorker.train_batch()` 调用
- `state_dict_for_vllm()`：导出 draft 模型权重（HF 格式）供 vLLM 同步

---

## 配置方法

在 YAML 配置中添加以下内容即可启用：

```yaml
actor_rollout_ref:
  model:
    path: Qwen/Qwen3-1.7B

    eagle_draft:
      enabled: true
      model_path: AngelSlim/Qwen3-1.7B_eagle3
      loss_weight: 0.1
      aux_layer_indices: null     # 自动选取
      optimizer:
        lr: 1.0e-4
        weight_decay: 0.0

  rollout:
    name: vllm
    vllm_kwargs:
      speculative_config:
        method: eagle3
        model: AngelSlim/Qwen3-1.7B_eagle3
        num_speculative_tokens: 3
        draft_tensor_parallel_size: 1
```

---

## 限制与后续工作

| 项目 | 状态 | 说明 |
|---|---|---|
| FSDP 后端 | ✅ 完整支持 | 主实现目标 |
| Megatron 后端 | ⬜ 未支持 | 可参考 nemo-rl 实现迁移 |
| Pipeline 并行 | ❌ 不支持 | 与 nemo-rl 当前限制相同 |
| vLLM draft 同步 | 🔶 部分完成 | server 侧 `update_draft_weights_from_ipc` 需对接具体 vLLM 版本 |
| Draft LR Scheduler | ⬜ 未支持 | draft optimizer 暂用固定 LR |
| Sequence Packing | ❌ 不支持 | 与 nemo-rl 当前限制相同 |
| Draft 模型 Checkpoint | ⬜ 未支持 | `EagleDraftManager.save_pretrained()` 已提供接口 |

---

## 参考资料

- [Eagle3 论文](https://arxiv.org/abs/2503.05030)
- [nemo-rl PR #2078](https://github.com/NVIDIA-NeMo/RL/pull/2078) — 原始 Megatron 实现
- [nemo-rl Eagle3 使用指南](../../../nemo-rl/docs/guides/eagle3-speculative-decoding.md)
- [vLLM Eagle3 speculative decoding 文档](https://docs.vllm.ai/en/latest/features/spec_decode.html)
