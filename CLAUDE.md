# MedicalGPT 实验项目 — 快速启动与进度

## 环境

```bash
conda activate medicalgpt
cd /root/autodl-tmp/MedicalGPT_experiments
```

- **2B 基座**: `/root/autodl-tmp/models/Qwen3.5-2B`
- **7B 基座**: `/root/autodl-tmp/models/Qwen2.5-7B-Instruct` (QLoRA 4bit)
- GPUs: 2 × RTX 4090 (24GB)
- API: 阿里百炼 `.env` 已配置 (qwen-plus)
- 所有脚本从项目根目录执行

## 实验进度

### 2B 流水线 (Qwen3.5-2B, LoRA)

| 阶段 | 状态 | 核心结果 |
|------|------|---------|
| SFT 消融 (A/B/C) | ✅ 完成 | best=SFT-C, safety=0.627 |
| 医疗偏好数据 | ✅ 完成 | 250对, Qwen-Plus API 双prompt |
| DPO-Medical | ✅ 完成 | safety 0.64→0.83, 急症 0.13→0.47 |
| RM 训练 | ✅ 完成 | MAE=0.65, pairwise accuracy=95.6% |
| RLOO | ✅ 完成 | reward 稳定在 +0.15, cp-100/200/300 |
| GRPO Safety | ✅ 完成 | v1→v2→v3 三版迭代 |
| LLM Judge 评测 | ✅ 完成 | DeepSeek V4 Flash, 5维度 |

### 7B 流水线 (Qwen2.5-7B-Instruct, QLoRA 4bit)

| 阶段 | 状态 | 核心结果 |
|------|------|---------|
| SFT-A/B/C | ✅ 完成 | best=SFT-C (perplexity=7.82) |
| DPO-Medical | ✅ 完成 | LLM Judge overall=6.58 (三者最优) |
| RM 训练 | ✅ 完成 | MAE=1.77 (差，数据太少+基座选错) |
| RLOO | ❌ 失败 | RM 给所有输出打负分，100步无改善 |
| GRPO Safety | ✅ 完成 | 规则评分 0.831 (三者最优)，用药满分 |
| LLM Judge 评测 | ✅ 完成 | DeepSeek V4 Flash, 5维度 |

### 核心结论

- **DPO 赢在回答质量** (LLM Judge: 6.58)，偏好数据利用效率最高
- **GRPO 赢在安全合规** (规则评分: 0.831)，奖励-评测同构但可能 reward hacking
- **RLOO 7B 失败**：RM 基座用了 merged SFT 而非 pretrained，MAE 1.77 未验证就进 RL
- **就医类别是所有模型的最短板** (safety 0.33-0.43)

## 关键文件

```
.env                                                    # API key (百炼)
PLAN.md                                                 # 原始实验方案（含执行状态）
LEARNING_LOG.md                                         # 学习日志 + 实验教训
outputs/experiments/
  models/
    sft_C/                      sft_7b_C_merged/        # best SFT (2B adapter / 7B merged)
    dpo_medical/                dpo_7b_medical/          # DPO
    rm_medical/                 rm_7b_medical/           # Reward Model
    rloo_medical/               rloo_7b_medical/         # RLOO (7B failed)
    grpo_medical_safety/        grpo_7b_medical_safety/  # GRPO
  predictions/                  predictions_7b/          # 推理结果
  scores/                       scores_7b/               # 评分结果
  reports/
    phase1_sft_ablation.md                              # 2B SFT 消融报告
    phase3_dpo_medical.md                               # 2B DPO 报告
    7b_full_report.md                                   # 7B 完整实验报告
    7b_alignment_analysis.md                            # 7B 对齐机理分析
data/experiments/
  preference_medical_safety/train.jsonl                 # 偏好数据 (250对)
  grpo_medical_safety/train.jsonl                       # GRPO 训练数据
  eval/medical_eval_50.jsonl                            # 固定评测集 (50条)
tools/experiments/
  score_reward_model.py                                 # RM 评估（训完必须先跑！）
  batch_inference.py                                    # 批量推理
  score_medical_outputs.py                              # 规则评分
  llm_judge.py                                          # LLM Judge 评分
```

## 训练命令

### 2B 实验

```bash
# SFT
bash scripts/experiments/run_sft_training.sh --group C

# DPO
bash scripts/experiments/run_dpo_medical.sh

# RM
bash scripts/experiments/run_rm_medical.sh

# RM 评估（必须跑！）
python tools/experiments/score_reward_model.py \
    --model_path outputs/experiments/models/rm_medical \
    --base_model /root/autodl-tmp/models/Qwen3.5-2B \
    --peft_path outputs/experiments/models/sft_C \
    --data data/experiments/preference_medical_safety/train.jsonl

# RLOO
bash scripts/experiments/run_rloo_medical.sh

# GRPO
bash scripts/experiments/run_grpo_medical_safety.sh

# 评测
bash scripts/experiments/run_medical_eval.sh
```

### 7B 实验

```bash
# SFT
bash scripts/experiments/run_sft_training_7b.sh --group C

# DPO
bash scripts/experiments/run_dpo_training_7b.sh

# RM
bash scripts/experiments/run_rm_medical_7b.sh

# RLOO (失败，勿直接跑)
bash scripts/experiments/run_rloo_medical_7b.sh

# GRPO
bash scripts/experiments/run_grpo_medical_safety_7b.sh

# 评测
bash scripts/experiments/run_7b_eval.sh
```

## 教训

1. **先 conda activate medicalgpt**，否则 import 报错
2. eval 集不能用训练同源数据，必须 holdout
3. DPO 偏好数据不要用自己模型生成（自循环），用强模型 API 双 prompt
4. batch_inference.py 用 `--lora_models` 多模型依次推理，自动 merge+unload
5. 安全关键词匹配会误报 "切勿自行用药" 为 danger，需人工抽查
6. Python 后台任务加 `-u` 参数禁用输出缓冲
7. **RM 基座要用 pretrained 模型，不能直接用 merged SFT**（否则表示偏差→MAE爆炸→RLOO必死）
8. **RM 训完必须先跑 `score_reward_model.py` 验证**，MAE > 1.0 不进 RL
9. **7B QLoRA 需保守超参**：batch_size=1, grad_accum=16, lr=2e-5
10. RLOO 的 num_generations 至少 8（7B 用了 4，噪声太大）
