# MedicalGPT 实验项目 — 快速启动与进度

## 环境

```bash
conda activate medicalgpt
cd /root/autodl-tmp/MedicalGPT_experiments
```

- Base model: `/root/autodl-tmp/models/Qwen3.5-2B`
- GPUs: 2 × RTX 4090 (24GB)
- API: 阿里百炼 `.env` 已配置 (qwen-plus)
- 所有脚本从项目根目录执行

## 实验进度

| 阶段 | 状态 | 核心结果 |
|------|------|---------|
| SFT 消融 (A/B/C) | ✅ 完成 | best=SFT-C, safety=0.627, 急症safety=0.133 |
| 医疗偏好数据 | ✅ 完成 | 250对, Qwen-Plus API 双prompt生成 |
| DPO-Medical | ✅ 完成 | safety 0.64→0.83, 急症 0.13→0.47, 用药 0.07→0.67 |
| RM + RLOO | 🔧 脚本就绪 | RM 训练 + RLOO 优化，脚本已写好待运行 |
| GRPO Safety | 🔧 脚本就绪 | 规则奖励 + 安全 reward，脚本已写好待运行 |
| LLM Judge 评测 | ⏳ 待做 | API 细粒度评分 |

## 关键文件

```
.env                                    # API key (百炼)
PLAN.md                                 # 完整实验方案
outputs/experiments/models/sft_C/       # best SFT (LoRA adapter)
outputs/experiments/models/dpo_medical/ # DPO model
outputs/experiments/reports/
  phase1_sft_ablation.md                # SFT 消融报告
  phase3_dpo_medical.md                 # DPO 实验报告
data/experiments/eval/medical_eval_50.jsonl        # 固定评测集
data/experiments/preference_medical_safety/train.jsonl  # 偏好数据
data/experiments/grpo_medical_safety/train.jsonl   # GRPO 训练数据
tools/experiments/score_reward_model.py            # RM 评估脚本
```

## 训练脚本

```bash
# SFT (run_sft_training.sh 支持 --group A/B/C)
bash scripts/experiments/run_sft_training.sh --group C

# DPO
bash scripts/experiments/run_dpo_medical.sh

# RM (先跑，产出 reward model)
bash scripts/experiments/run_rm_medical.sh

# RM 评估
python tools/experiments/score_reward_model.py \
    --model_path outputs/experiments/models/rm_medical \
    --base_model /root/autodl-tmp/models/Qwen3.5-2B \
    --data data/experiments/preference_medical_safety/train.jsonl

# RLOO (需要先跑 RM)
bash scripts/experiments/run_rloo_medical.sh

# 评测三步走
python tools/experiments/build_eval_set.py ...
python tools/experiments/batch_inference.py --lora_models <path1> <path2> ...
python tools/experiments/score_medical_outputs.py --predictions <paths> ...
```

## 教训

1. **先 conda activate medicalgpt**，否则 import 报错
2. eval 集不能用训练同源数据，必须 holdout（`medical_sft_1K_format.jsonl` 是独立来源）
3. build_eval_set.py 的去重只取首轮 human turn，避免多轮对话膨胀去重集
4. DPO 偏好数据不要用自己模型生成（自循环），用强模型 API 双 prompt
5. batch_inference.py 用 `--lora_models` 多模型依次推理，自动 merge+unload
6. 安全关键词匹配会误报 "切勿自行用药" 为 danger，需人工抽查
7. Python 后台任务加 `-u` 参数禁用输出缓冲，否则看不到进度
