# MedicalGPT 学习日志

> 写给明天的自己（或明天打开这个项目的 Claude）：读这个文件，你就可以接着上次的进度继续辅导我。

## 学习目标

通过 MedicalGPT 项目，学习大模型**后训练（Post-Training）**的完整流水线代码。

## 学习方法（四步法）

| 步骤 | 名称 | 谁主导 | 规则 |
|------|------|--------|------|
| 1 | 全局扫盲与直觉构建 | AI | 不讲公式和代码，只用比喻建立直觉 |
| 2 | 物理隔离与逻辑推演 | 我 | 关屏幕，纸笔推导，流程图+伪代码 |
| 3 | 积木式编码 | 我 + AI | 手敲代码，15分钟死磕再求助 |
| 4 | 费曼输出 | 我 → AI | 向AI讲解，让AI反问逼问到底 |

两条红线：
1. 禁止整段 Copy-Paste
2. 不明白原理的代码不留在项目中

---

## 当前进度：Step 1 → Step 2 过渡

### 已完成：直觉建立

**MedicalGPT 的核心概念（已理解）：**

这是一个标准的大模型后训练流水线，类比"培养医生"：

| 阶段 | 类比 | 核心动作 |
|------|------|---------|
| PT（增量预训练） | 读进修班 | 喂海量医学文本，适应领域 |
| SFT（有监督微调） | 跟老医生出门诊 | 学"问答对"，对齐指令意图 |
| DPO | 看两个作业选好的 | chosen vs rejected 对比学习 |
| RLHF（RM+RLOO） | 训练打分助教再练 | RM评分 → 强化学习优化 |
| GRPO | 按规则书自己练 | 不用RM，手写规则函数打分 |

**SFT 核心数据流（已理解）：**

```
jsonl → 模板格式化 → tokenize → label mask (query=-100) → Trainer → loss
```

**三样核心产物的文件映射（已给出，见下方）：**
- 数据格式
- 损失函数
- 奖励定义

---

## 关键文件清单

```
项目根目录: /root/autodl-tmp/MedicalGPT_experiments/

training/
├── supervised_finetuning.py   ← SFT 训练（最核心，先读这个）
├── dpo_training.py            ← DPO 训练（调用 trl.DPOTrainer）
├── reward_modeling.py         ← RM 训练（logsigmoid loss）
├── grpo_training.py           ← GRPO 训练（规则奖励函数）
├── orpo_training.py           ← ORPO 训练
├── ppo_training.py            ← PPO/RLOO 训练
├── template.py                ← 对话模板（qwen3_5 等）
└── tool_utils.py              ← Agent/Function Call 工具

data/
├── sft/
│   ├── medical_sft_1K_format.jsonl      ← SFT 训练数据样例
│   └── sharegpt_zh_1K_format.jsonl
├── reward/
│   └── dpo_zh_500.jsonl                 ← DPO 偏好数据样例
├── grpo/
│   └── sample.jsonl                     ← GRPO 训练数据样例
└── experiments/
    ├── preference_medical_safety/train.jsonl  ← 医疗安全偏好数据（250对）
    └── eval/medical_eval_50.jsonl            ← 固定评测集

scripts/
├── run_sft.sh                   ← SFT 启动脚本
├── run_dpo.sh                   ← DPO 启动脚本
├── run_rm.sh                    ← RM 启动脚本
├── run_ppo.sh                   ← PPO/RLOO 启动脚本
└── run_grpo.sh                  ← GRPO 启动脚本

scripts/experiments/
├── run_sft_training.sh          ← SFT 消融实验 (A/B/C 三组)
└── run_dpo_medical.sh           ← DPO 医疗安全实验

PLAN.md                         ← 完整实验方案（必读）
CLAUDE.md                       ← 环境配置和快速启动
```

---

## 三样核心产物 — 精确文件映射

### 1. 数据格式

| 训练方法 | 样例数据 | 解析代码 |
|---------|---------|---------|
| SFT | `data/sft/medical_sft_1K_format.jsonl` | `supervised_finetuning.py` 的 `get_dialog()` (L443) |
| DPO | `data/reward/dpo_zh_500.jsonl` | `dpo_training.py` (trl 库解析) |
| GRPO | `data/grpo/sample.jsonl` | `grpo_training.py` 的 `grpo_train()` (L167) |

三种格式：
- **SFT**: `{"conversations": [{"from":"human","value":"..."}, {"from":"gpt","value":"..."}]}`
- **DPO**: `{"conversations": [...], "chosen": "...", "rejected": "..."}`
- **GRPO**: `{"question": "...", "answer": "..."}`

### 2. 损失函数

| 方法 | 文件 | 关键行 | 核心 |
|------|------|--------|------|
| SFT | `supervised_finetuning.py` | L567 | label=-100 掩码 + 标准 CrossEntropy |
| DPO | `dpo_training.py` | L595 | `DPOTrainer(...)` → loss 在 trl 库内 |
| RM | `reward_modeling.py` | L237 | `-logsigmoid(r_chosen - r_rejected).mean()` |

### 3. 奖励定义

| 方法 | 文件 | 关键行 | 谁打分 |
|------|------|--------|--------|
| RM | `reward_modeling.py` | L225-240 | 神经网络模型 |
| GRPO | `grpo_training.py` | L71-127 | 手写规则函数 `accuracy_reward` + `format_reward` |

---

## 推荐阅读顺序

```
第 1 站: training/supervised_finetuning.py  L437-L571 (preprocess_function)
          理解"数据 → token → label mask"的完整流水线

第 2 站: training/reward_modeling.py  L225-L240 (compute_loss)
          理解 logsigmoid 这个干净到极致的 RM loss

第 3 站: training/grpo_training.py  L71-L127 (accuracy_reward + format_reward)
          理解"用规则代替模型"的奖励定义

第 4 站: training/dpo_training.py  L554-L603 (DPOConfig + DPOTrainer)
          理解 DPO 是"数据格式 + 调用 trl 库"
```

---

## 当前卡点 & 下一步

### 当前状态

- Step 1（直觉）已完成：理解了 PT/SFT/DPO/RLHF/GRPO 的本质区别
- 已读完 `supervised_finetuning.py` 的结构（中文注释 + 数据流讲解）
- 已拿到"数据格式 + 损失函数 + 奖励定义"的精确文件映射
- **还没进入 Step 2（纸笔推导）**

### 明天的任务（优先级从高到低）

1. **关掉屏幕，纸笔推导** `preprocess_function` 的数据变形过程。画出一个 jsonl 行如何变成 input_ids 和 labels 列表。卡住了回看 `supervised_finetuning.py:437-571`

2. **纸笔推导 RM loss**：`loss = -logsigmoid(r_chosen - r_rejected).mean()`。画出 logsigmoid 的函数图像，标注"chosen 远好于 rejected"和"rejected 比 chosen 好"两种情况下的 loss 值

3. **读第 2 站代码**：`reward_modeling.py` L225-L240，对着纸上的 RM loss 公式看代码实现

4. **读第 3 站代码**：`grpo_training.py` L71-L127，理解两个奖励函数的输入输出

---

## 项目环境速查

```bash
conda activate medicalgpt
cd /root/autodl-tmp/MedicalGPT_experiments
```

- Base model: `/root/autodl-tmp/models/Qwen3.5-2B`
- GPUs: 2 × RTX 4090 (24GB)
- API: 阿里百炼 `.env` 已配置

---

## 实验进度速查

| 阶段 | 状态 | 关键结果 |
|------|------|---------|
| SFT 消融 (A/B/C) | ✅ 完成 | best=SFT-C |
| 医疗偏好数据 | ✅ 完成 | 250对 |
| DPO-Medical | ✅ 完成 | safety 0.64→0.83 |
| RM + RLOO | ⏳ 待做 | |
| GRPO Safety | ⏳ 待做 | |
| LLM Judge 评测 | ⏳ 待做 | |

---

*最后更新: 2026-05-12*
