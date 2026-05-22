# Eagle3 Token 生成流程与时间步移位

以 `"can you do my homework"` 为例，完整展示 Eagle3 投机解码（推理）和训练时移位对齐的全过程。

---

## Part 1 — 推理阶段：草稿模型如何"吐 token"

### 1.1 整体机制

Eagle3 的核心思路是用一个**小而快的草稿模型**先提议若干候选 token，再让**主模型一次验证全部**，从而把原本 N 步的主模型 forward 压缩成 1 步。

```
┌────────────────────────────────────────────────────────────────────────────┐
│               Eagle3 投机解码 — 一个完整的 Loop                             │
│                                                                            │
│  ┌─────────────────────────────────────────────────────────────────┐      │
│  │ ① 主模型 forward（上一轮已完成）                                  │      │
│  │                                                                  │      │
│  │  [can] [you]  ──►  Main Model  ──►  h_can, h_you                │      │
│  │                       (FSDP)           (aux hidden states)       │      │
│  └─────────────────────────────────────────────────────────────────┘      │
│                              │                                             │
│                     h_can, h_you 传给草稿模型                              │
│                              │                                             │
│  ┌─────────────────────────────────────────────────────────────────┐      │
│  │ ② 草稿模型自回归生成 K=3 个候选 token（快！）                      │      │
│  │                                                                  │      │
│  │  Draft step 1:  (h_you + e_you)  → d₁ = "do"                   │      │
│  │  Draft step 2:  (h_you + e_do)   → d₂ = "my"                   │      │
│  │  Draft step 3:  (h_you + e_my)   → d₃ = "homework"             │      │
│  │                                                                  │      │
│  └─────────────────────────────────────────────────────────────────┘      │
│                              │                                             │
│                         候选序列提交验证                                   │
│                              │                                             │
│  ┌─────────────────────────────────────────────────────────────────┐      │
│  │ ③ 主模型一次 forward 验证全部候选（3 个 token，只需 1 次！）         │      │
│  │                                                                  │      │
│  │  [can][you][do][my][homework]  ──►  Main Model                  │      │
│  │                                         │                        │      │
│  │  pos you→do:      p_main("do")=0.88    → 接受 d₁ ✓              │      │
│  │  pos do→my:       p_main("my")=0.73    → 接受 d₂ ✓              │      │
│  │  pos my→homework: p_main("homework")=0.91 → 接受 d₃ ✓           │      │
│  │  pos homework→?:  p_main("for")=0.65   → 额外输出一个 bonus token  │      │
│  │                                                                  │      │
│  └─────────────────────────────────────────────────────────────────┘      │
│                                                                            │
│  结果：生成了 "do my homework for"（4 个 token），主模型只跑了 1 次 forward  │
└────────────────────────────────────────────────────────────────────────────┘
```

---

### 1.2 Token 级别详细流程

**初始状态**：已生成 `[can][you]`，等待生成后续 token。

```
已有序列:  [can]   [you]
位置:        1       2
主模型上一轮输出的 aux 隐层:  h₁(can), h₂(you)
```

---

#### 草稿模型第 1 步 — 预测 position 3

```
                        草稿模型第 1 步
 ┌──────────────────────────────────────────────────────┐
 │                                                      │
 │  已有 token:  [can]     [you]                        │
 │  位置:          1         2                          │
 │                                                      │
 │  Draft 输入:                                         │
 │    hidden_states:  [h₁(can), h₂(you)]               │
 │    inputs_embeds:  [e₁(can), e₂(you)]               │
 │                               ↑                     │
 │                      最后一位 = e₂(you)               │
 │                                                      │
 │  Draft 在 pos 2 的输出:                               │
 │    logits → top: "do"(0.72), "help"(0.11), ...      │
 │    采样得到  d₁ = "do"                               │
 │                                                      │
 └──────────────────────────────────────────────────────┘
                            │
                        d₁ = "do"
```

---

#### 草稿模型第 2 步 — 预测 position 4

```
                        草稿模型第 2 步
 ┌──────────────────────────────────────────────────────┐
 │                                                      │
 │  序列延伸:  [can]  [you]  [do]                       │
 │  位置:        1      2      3                        │
 │                                                      │
 │  Draft 输入（追加一个位置）:                           │
 │    hidden_states:  [h₁, h₂, h₂*]                   │
 │                              ↑                      │
 │                   复用 h₂（主模型还没跑 pos 3）         │
 │    inputs_embeds:  [e₁, e₂, e₃(do)]                │
 │                             ↑                       │
 │                       d₁ 的 embedding                │
 │                                                      │
 │  Draft 在 pos 3 的输出:                               │
 │    logits → top: "my"(0.61), "the"(0.15), ...       │
 │    采样得到  d₂ = "my"                               │
 │                                                      │
 └──────────────────────────────────────────────────────┘
                            │
                        d₂ = "my"
```

---

#### 草稿模型第 3 步 — 预测 position 5

```
                        草稿模型第 3 步
 ┌──────────────────────────────────────────────────────┐
 │                                                      │
 │  序列延伸:  [can]  [you]  [do]   [my]                │
 │  位置:        1      2      3      4                 │
 │                                                      │
 │  Draft 输入（再追加一位）:                             │
 │    hidden_states:  [h₁, h₂, h₂*, h₂*]              │
 │    inputs_embeds:  [e₁, e₂, e₃,  e₄(my)]           │
 │                                   ↑                 │
 │                             d₂ 的 embedding          │
 │                                                      │
 │  Draft 在 pos 4 的输出:                               │
 │    logits → top: "homework"(0.88), "work"(0.06), …  │
 │    采样得到  d₃ = "homework"                         │
 │                                                      │
 └──────────────────────────────────────────────────────┘
                            │
                        d₃ = "homework"
```

---

#### 主模型一次性验证

```
主模型验证（单次 forward，输入长度 = 2 + 3 = 5）

 输入:  [can]  [you]  [do]   [my]   [homework]
 位置:    1      2      3      4         5
         ↓      ↓      ↓      ↓          ↓
 ┌───────────────────────────────────────────────┐
 │              Main Model (FSDP)                │
 └───────────────────────────────────────────────┘
         ↓      ↓      ↓      ↓          ↓
 logits: L₁    L₂     L₃     L₄         L₅

 验证结果:
 ┌──────┬──────────────┬──────────────────┬───────────┐
 │ 位置  │ 草稿提议      │ 主模型概率        │ 结果      │
 ├──────┼──────────────┼──────────────────┼───────────┤
 │  2→3 │ d₁ = "do"   │ p("do"|L₂)=0.88 │  接受 ✓   │
 │  3→4 │ d₂ = "my"   │ p("my"|L₃)=0.73 │  接受 ✓   │
 │  4→5 │ d₃="homework"│ p("hmk"|L₄)=0.91│  接受 ✓   │
 │  5→6 │ 无草稿        │ p("for"|L₅)=0.65│ bonus ✓   │
 └──────┴──────────────┴──────────────────┴───────────┘

 最终输出: "do my homework for"  ← 4 个 token，主模型只 forward 了 1 次
```

---

### 1.3 接受/拒绝机制（Speculative Sampling）

对每个草稿 token dᵢ，用概率比做随机接受检验：

```
u ~ Uniform(0, 1)

if u < min(1,  p_main(dᵢ)  )  →  接受 dᵢ
                ──────────
                p_draft(dᵢ)

else                           →  拒绝 dᵢ，从修正分布重采样:
                                   p_corrected = normalize(max(0, p_main - p_draft))
                                   后续 draft token 全部丢弃
```

> 当 `p_main ≥ p_draft`（主模型比草稿更"确定"），必然接受。  
> 当 `p_main < p_draft`（草稿过于自信），以比例概率拒绝，防止分布偏移。

---

## Part 2 — 训练阶段：时间步移位（Time-Step Alignment）

### 2.1 为什么需要移位

推理时，草稿模型在每一步会看到**刚刚采样出的 token 的 embedding** 来预测下一个 token。训练时用 teacher forcing 模拟这一行为——把 embedding 序列整体向左移 1 位，使位置 t 看到 token t+1 的 embedding。

```
原始序列:  [can]  [you]  [do]  [my]  [home] [work] ...
位置:        1      2      3     4      5      6
            e₁     e₂     e₃    e₄     e₅     e₆

左移 1 位后（rolled_embeds）:
位置:        1      2      3     4      5      6
            e₂     e₃     e₄    e₅     e₆    e₁(wrap!)
                                               ↑
                                       最后一位 wrap-around
                                       → 训练时 mask[:,-1]=0 排除
```

---

### 2.2 完整的 Token 级别对应关系

```
Token:    [can]   [you]   [do]    [my]   [homework]  [for]
Position:   1       2       3       4        5          6
            │       │       │       │         │          │
            ▼       ▼       ▼       ▼         ▼          ▼
Policy   ┌────────────────────────────────────────────────┐
(FSDP)   │  h₁      h₂      h₃      h₄        h₅        h₆  │  aux hidden states
         └─────────────────────────────────────────────────┘
            │       │       │       │         │          │
            ▼       ▼       ▼       ▼         ▼          ▼
LM Head  T₁→you  T₂→do  T₃→my  T₄→hmk   T₅→for    T₆→me
         (teacher logits, detach 后作为蒸馏目标)
            ┆       ┆       ┆       ┆         ┆          ┆
            ┆       ┆       ┆       ┆         ┆          ┆
Rolled   e₂(you) e₃(do) e₄(my) e₅(hmk)  e₆(for)  e₇(me)
embeds:    ↑       ↑       ↑       ↑         ↑          ↑
           └───────┴───────┴───────┴─────────┴──────────┘
                      left-shift by 1

Draft   (h₁,e₂)→ (h₂,e₃)→ (h₃,e₄)→ (h₄,e₅)→ (h₅,e₆)→ excluded
input:    │         │         │         │         │      (wrap)
          ▼         ▼         ▼         ▼         ▼
Draft   D₁→you   D₂→do    D₃→my    D₄→hmk   D₅→for
output:

Loss:   CE(D₁,T₁) CE(D₂,T₂) CE(D₃,T₃) CE(D₄,T₄) CE(D₅,T₅)
        ← 仅对 response mask=1 的位置计算 →
```

**关键对齐逻辑**：
- 位置 3（`do`）的草稿输入是 `h₃(do) + e₄(my)`  
- 草稿输出 D₃ 预测 `my`，与教师 T₃ 对齐  
- 这与推理时"看到上一个已生成 token 的 embedding 来预测下一个"完全一致

---

### 2.3 Loss 计算的 Roll 对齐

`roll_for_eagle_alignment` 对 draft/teacher logits 和 mask **再次左移 1 位**，然后清零最后位置（排除 wrap-around）：

```
                      roll_for_eagle_alignment 前后对比

Before roll:
  pos:        1      2      3      4      5      6
  Draft:      D₁     D₂     D₃     D₄     D₅    D_wrap
  Teacher:    T₁     T₂     T₃     T₄     T₅    T_wrap
  Mask:       0      0      1      1      1      0
              ↑      ↑      ↑
           prompt  prompt  response starts here

After roll (shifts=-1) + mask[:,-1]=0:
  pos:        1      2      3      4      5      6
  Draft:      D₂     D₃     D₄     D₅    D_wrap  [0]
  Teacher:    T₂     T₃     T₄     T₅    T_wrap  [0]
  Mask:       0      1      1      1      0       0
                     ↑                   ↑
              response starts         wrap-around 被清零

Loss = CE(D₃,T₃) + CE(D₄,T₄) + CE(D₅,T₅)  ← 只有 response 位置参与
```

---

### 2.4 训练 vs 推理的对应关系

```
┌─────────────────┬──────────────────────────────┬──────────────────────────────────┐
│                 │  训练（Teacher Forcing）        │  推理（Autoregressive）            │
├─────────────────┼──────────────────────────────┼──────────────────────────────────┤
│ 位置 t 的输入    │ h_t（policy 隐层）             │ h_t（主模型上一次 forward 的隐层）    │
│                 │ + e_{t+1}（真实 token 嵌入）   │ + e_{d_t}（上一步草稿 token 嵌入）  │
├─────────────────┼──────────────────────────────┼──────────────────────────────────┤
│ 位置 t 的目标    │ T_t（policy LM head logits）   │ 从 draft_logits[t] 采样           │
├─────────────────┼──────────────────────────────┼──────────────────────────────────┤
│ 梯度            │ soft-CE loss，只更新草稿模型    │ 无梯度（inference only）             │
├─────────────────┼──────────────────────────────┼──────────────────────────────────┤
│ 最后位置（T-1）  │ mask=0，excluded（wrap）       │ 无此问题（逐步扩展序列）              │
└─────────────────┴──────────────────────────────┴──────────────────────────────────┘
```

---

## Part 3 — 推理与训练的闭环

```
                    RL 训练 + Eagle3 的完整闭环

  ┌────────────────────────────────────────────────────────────┐
  │                      每个训练 Step                          │
  │                                                            │
  │  1. 草稿模型（旧权重）在 vLLM 里做投机解码 → rollout          │
  │     "can" → draft: "you"(0.81) "do"(0.76) "my"(0.69)      │
  │     主模型验证 → acceptance rate ≈ 75%                     │
  │                         │                                  │
  │  2. FSDP 训练（policy + draft 联合更新）                    │
  │     hooks 捕获 h_t, e_t, teacher_logits                    │
  │     L_total = L_policy + 0.1 × L_draft                    │
  │     backward → policy grads (FSDP all-reduce)              │
  │               → draft grads (手动 all-reduce)              │
  │                         │                                  │
  │  3. 权重同步到 vLLM                                        │
  │     policy weights  → vLLM main model                      │
  │     draft weights   → vLLM Eagle3 proposer                 │
  │                         │                                  │
  │  4. 下一步 rollout 用最新 draft → acceptance rate ↑         │
  │     "can" → draft: "you"(0.89) "do"(0.83) "my"(0.78)      │
  │                                                            │
  └────────────────────────────────────────────────────────────┘
```

---

## 相关文件

| 文件 | 对应上述哪个环节 |
|------|------------------|
| `verl/workers/eagle/hidden_capture.py` | 捕获 h_t, e_t, teacher_logits |
| `verl/workers/eagle/losses.py` | `roll_for_eagle_alignment` + `eagle_draft_loss` |
| `verl/workers/eagle/manager.py` | `EagleLossWrapper`（联合 loss）+ `optimizer_step`（梯度同步） |
| `verl/workers/rollout/vllm_rollout/utils.py` | `update_draft_weights_from_ipc`（投机解码用最新权重） |
