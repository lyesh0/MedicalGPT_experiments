# 大模型后训练全流程：原理、算法与 MedicalGPT 代码导读

本文面向已经具备 Python、PyTorch、Transformer 基础的学生，目标是讲清楚大模型后训练的完整链路：后训练是什么，为什么预训练之后还要做这些步骤，主流算法如何从监督学习、奖励建模、强化学习和偏好优化逐步演进，以及这些思想在 MedicalGPT 项目中分别落在哪些代码文件里。

阅读本文时建议同时打开这些文件：

- `training/supervised_finetuning.py`
- `training/reward_modeling.py`
- `training/ppo_training.py`
- `training/dpo_training.py`
- `training/orpo_training.py`
- `training/grpo_training.py`
- `training/template.py`
- `training/tool_utils.py`
- `scripts/run_sft.sh`
- `scripts/run_rm.sh`
- `scripts/run_ppo.sh`
- `scripts/run_dpo.sh`
- `scripts/run_orpo.sh`
- `scripts/run_grpo.sh`
- `data/sft/*.jsonl`
- `data/reward/*.jsonl`
- `data/grpo/sample.jsonl`

## 1. 什么是后训练

大模型训练通常可以粗略分成两个阶段：

1. 预训练：在大规模无标注文本上做 next-token prediction，让模型学习语言、知识、模式和一部分推理能力。
2. 后训练：在预训练模型基础上，用指令数据、偏好数据、奖励模型或可验证奖励继续训练，让模型更符合用户任务、对话格式、安全边界、工具调用规范和特定领域需求。

预训练的核心目标通常是最大化语料概率：

$$
\max_{\theta}\sum_{t}\log \pi_{\theta}(x_t \mid x_{<t})
$$

也就是给定前文预测下一个 token。这个目标很强，但它不直接等价于“听懂用户指令”“拒绝危险请求”“优先输出有用答案”“严格按工具调用格式返回 JSON”“医学问题中表达不确定性并提示就医”。

后训练要解决的是另一个问题：

给定用户上下文 $x$，在多个可能回答 $y$ 中，使模型更偏向人类、规则或奖励函数认为更好的 $y$。

如果形式化为策略优化：

$$
\max_{\pi}\ \mathbb{E}_{x \sim \mathcal{D},\ y \sim \pi(\cdot \mid x)}
\left[r(x,y)\right]
- \beta\,\mathrm{KL}\left(\pi(\cdot \mid x)\ \Vert\ \pi_{\mathrm{ref}}(\cdot \mid x)\right)
$$

这里：

- $\pi$ 是要训练的当前模型。
- $\pi_{\mathrm{ref}}$ 是参考模型，通常是 SFT 模型或冻结的旧模型。
- $r(x,y)$ 是奖励，可以来自人类偏好、奖励模型、规则验证器或自动评测。
- $\mathrm{KL}$ 项限制模型不要为了追求奖励而偏离原模型太远。
- $\beta$ 控制“对齐偏好”和“保持原能力”的权衡。

这就是后训练的核心矛盾：让模型更符合目标行为，同时不破坏预训练和 SFT 已经学到的能力。

## 2. 为什么必须有后训练

### 2.1 预训练只学会“像文本”，不保证“像助手”

预训练模型学习的是互联网文本分布。它可能会补全百科、小说、论坛、代码、错误答案、偏见文本，也可能模仿不合适的表达。它没有被直接优化为“遵循用户意图”。

典型问题：

- 用户问问题时，base model 可能继续补全文档，而不是回答。
- 输出可能没有稳定的角色边界。
- 医疗场景可能过度自信。
- 工具调用任务可能不会输出可解析的函数调用。
- 多轮对话中可能忘记上下文格式。

### 2.2 SFT 把模型变成“会模仿示范的助手”

SFT 使用人工或模型生成的高质量示范回答训练模型，本质仍是监督学习：

$$
\mathcal{L}_{\mathrm{SFT}}(\theta)
= -\sum_{t}\log \pi_{\theta}(y_t \mid x, y_{<t})
$$

它教模型“遇到这样的指令，应该像示范答案这样回答”。这是后训练的第一步，也是最稳定、最容易调试的一步。

在 MedicalGPT 中，SFT 由 `training/supervised_finetuning.py` 实现，对应脚本是 `scripts/run_sft.sh`，数据来自 `data/sft/*.jsonl`。

### 2.3 偏好优化解决“多个答案谁更好”

SFT 只告诉模型一个标准答案，不能很好表达“回答 A 比回答 B 更好”。偏好数据更适合表达人类评价：

```json
{
  "conversations": [{"from": "human", "value": "问题"}],
  "chosen": "更好的回答",
  "rejected": "较差的回答"
}
```

这类数据允许我们训练模型拉开 `chosen` 和 `rejected` 的概率或奖励差距。

MedicalGPT 中偏好数据在 `data/reward/*.jsonl`，对应算法包括：

- RM：`training/reward_modeling.py`
- RLOO/PPO 类 RLHF：`training/ppo_training.py`
- DPO：`training/dpo_training.py`
- ORPO：`training/orpo_training.py`

### 2.4 可验证奖励解决“结果是否正确”

数学题、代码题、格式题、工具调用题有时不需要人工逐条打分，可以用规则或验证器给奖励。比如答案是否等于标准答案，是否满足 `<think>...</think><answer>...</answer>` 格式。

MedicalGPT 的 `training/grpo_training.py` 就展示了这类思路：

- `accuracy_reward`：解析答案并验证正确性。
- `format_reward`：检查输出是否符合指定格式。
- `GRPOTrainer`：用组内相对奖励做优化。

## 3. 后训练全流程总览

一个完整的后训练流程通常长这样：

```text
Base Model
  |
  | 1. 指令数据 / 对话数据
  v
SFT Model
  |
  | 2A. 偏好数据 -> Reward Model -> RL 优化
  | 2B. 偏好数据 -> DPO / ORPO / SimPO 等直接偏好优化
  | 2C. 可验证任务 -> GRPO / RLOO / rejection sampling
  v
Aligned Model
  |
  | 3. 评测、安全检查、医疗边界检查、工具调用解析检查
  v
Deployable Model
```

在工程上，它会拆成下面几个数据和模型工件：

| 阶段 | 输入 | 输出 | MedicalGPT 入口 |
| --- | --- | --- | --- |
| SFT | base model + `conversations` | SFT adapter 或 SFT model | `supervised_finetuning.py` |
| RM | SFT/base model + `chosen/rejected` | reward model | `reward_modeling.py` |
| RLHF/RLOO | SFT model + reward model + prompts | aligned policy | `ppo_training.py` |
| DPO | SFT model + `chosen/rejected` | aligned policy | `dpo_training.py` |
| ORPO | base/SFT model + `chosen/rejected` | aligned policy | `orpo_training.py` |
| GRPO | policy model + question/answer + reward functions | reasoning/aligned policy | `grpo_training.py` |
| 推理 | aligned model + tokenizer + optional LoRA | answer | `demo/inference.py` |

## 4. 第一步：SFT，有监督指令微调

### 4.1 SFT 要解决什么

SFT 的任务是把 base model 调整成能稳定遵循指令的 assistant。它不要求模型自己探索，也不要求比较两个回答，只要求模仿高质量示范。

数学目标：

给定样本 $(x,y)$，最大化 $y$ 在 $x$ 条件下的概率：

$$
\mathcal{L}_{\mathrm{SFT}}(\theta)
= -\sum_{t \in \mathcal{T}_{\mathrm{assistant}}}
\log \pi_{\theta}(y_t \mid x, y_{<t})
$$

关键点是：通常只让 assistant 回答部分参与 loss，用户 prompt 部分不参与 loss。否则模型会被训练去复述用户输入。

### 4.2 MedicalGPT 中的数据格式

普通 SFT 样例：

```json
{"conversations":[
  {"from":"human","value":"治疗阳痿吃什么药呢？"},
  {"from":"gpt","value":"建议到正规医院就诊..."}
]}
```

多轮对话是 `human` 和 `gpt` 交替出现。

工具调用样例会多出：

- `tools`：可用工具 schema。
- `function_call`：模型应该输出的函数调用。
- `observation`：工具执行结果。

### 4.3 MedicalGPT 中的 SFT 代码链路

核心文件：`training/supervised_finetuning.py`

阅读顺序：

1. `ModelArguments`：模型、tokenizer、量化、FlashAttention、RoPE 等参数。
2. `DataArguments`：训练集、验证集、样本截断、数据预处理 workers。
3. `ScriptArguments`：LoRA、QLoRA、模板、工具调用格式。
4. `main()`：训练全流程。
5. `preprocess_function()`：最关键的数据到 token 的转换。

数据流：

```text
data/sft/*.jsonl
  -> load_dataset("json")
  -> preprocess_function
  -> prompt_template 或 tokenizer.apply_chat_template
  -> tokenizer
  -> input_ids / attention_mask / labels
  -> Trainer
  -> LoRA adapter 或全参模型
```

SFT 中你最应该追的变量是：

- `input_ids_list`
- `attention_mask_list`
- `targets_list`
- `IGNORE_INDEX`
- `train_on_inputs`

`IGNORE_INDEX` 通常用于 mask 掉不参与 loss 的 token：

```text
labels = [-100, -100, ..., assistant_token_1, assistant_token_2, ...]
```

PyTorch 的交叉熵会忽略 label 为 `-100` 的位置。

### 4.4 SFT 的工程取舍

| 方案 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- |
| 全参 SFT | 表达能力强，可更新所有参数 | 显存和存储成本高，灾难性遗忘风险更高 | 大算力、关键模型训练 |
| LoRA SFT | 显存低，训练快，易保存和切换 | 容量受 rank 限制 | 教学、小团队、领域适配 |
| QLoRA SFT | 进一步降低显存 | 量化和训练稳定性更敏感 | 单卡低显存训练 |
| 只训练输出层/部分模块 | 成本低 | 能力受限明显 | 小实验或受限硬件 |

MedicalGPT 默认更偏向 LoRA/QLoRA 路线，相关参数在 `scripts/run_sft.sh`：

```bash
--use_peft True
--target_modules all
--lora_rank 8
--lora_alpha 16
--lora_dropout 0.05
```

## 5. 第二步：奖励模型 RM

### 5.1 RM 要解决什么

奖励模型把“人类更喜欢哪个回答”变成可计算的标量函数：

$$
r_{\phi}(x,y) \in \mathbb{R}
$$

给定同一个 prompt 的两个回答 $y_w$ 和 $y_l$，其中 $w$ 表示 chosen/winner，$l$ 表示 rejected/loser，奖励模型应该满足：

$$
r_{\phi}(x,y_w) > r_{\phi}(x,y_l)
$$

常见的 pairwise loss 来自 Bradley-Terry 模型：

$$
P(y_w \succ y_l \mid x)
= \sigma\left(r_{\phi}(x,y_w)-r_{\phi}(x,y_l)\right)
$$

$$
\mathcal{L}_{\mathrm{RM}}(\phi)
= -\log \sigma\left(r_{\phi}(x,y_w)-r_{\phi}(x,y_l)\right)
$$

### 5.2 MedicalGPT 中的 RM 实现

核心文件：`training/reward_modeling.py`

关键代码结构：

- `AutoModelForSequenceClassification`：把语言模型改造成输出一个标量分数的模型。
- `RewardDataCollatorWithPadding`：同时 padding chosen 和 rejected。
- `RewardTrainer.compute_loss()`：实现 pairwise log loss。

核心 loss 在代码中对应：

```python
loss = -torch.nn.functional.logsigmoid(
    rewards_chosen - rewards_rejected
).mean()
```

这行代码就是 RM 的数学核心。它不关心具体答案文本的 token-level 交叉熵，而关心 chosen 的整体 reward 是否高于 rejected。

### 5.3 RM 的技术风险

RM 是 RLHF 的关键瓶颈：

- 奖励模型可能学到标注偏差。
- 奖励模型只是在训练分布上拟合偏好，分布外不可靠。
- policy 可能利用 RM 的漏洞，出现 reward hacking。
- RM 标量很难完整表达真实性、安全性、帮助性、格式正确性等多个目标。

因此 RL 阶段通常还要加 KL 约束，防止 policy 为了刷高 reward 偏离太远。

## 6. 第三步：RLHF / PPO / RLOO

### 6.1 RLHF 的标准框架

InstructGPT 式 RLHF 通常是三步：

1. 收集示范数据，做 SFT。
2. 收集偏好比较数据，训练 reward model。
3. 用 PPO 等 RL 算法优化 policy，同时用 KL 限制 policy 偏离 SFT model。

目标可以写成：

$$
\max_{\theta}\ 
\mathbb{E}_{y \sim \pi_{\theta}(\cdot \mid x)}
\left[
r_{\phi}(x,y)
- \beta\,\mathrm{KL}\left(
\pi_{\theta}(\cdot \mid x)\ \Vert\ \pi_{\mathrm{ref}}(\cdot \mid x)
\right)
\right]
$$

这里 $\pi_{\mathrm{ref}}$ 通常是冻结的 SFT 模型。

### 6.2 PPO 的核心思想

PPO 是一种 policy gradient 方法，它不直接做无限制的大步更新，而是限制新旧策略比率：

$$
\rho_t(\theta)
= \frac{\pi_{\theta}(a_t \mid s_t)}
{\pi_{\mathrm{old}}(a_t \mid s_t)}
$$

$$
\mathcal{L}_{\mathrm{CLIP}}(\theta)
= \mathbb{E}_t\left[
\min\left(
\rho_t(\theta)A_t,\ 
\mathrm{clip}\left(\rho_t(\theta), 1-\epsilon, 1+\epsilon\right)A_t
\right)
\right]
$$

在语言模型里：

- state 可以理解为 prompt + 已生成前缀。
- action 是下一个 token。
- reward 来自 reward model 或规则。
- advantage 衡量这个 token/序列比平均水平好多少。

PPO 的优点是能在线采样，让模型探索自己当前会生成的答案；缺点是工程复杂、显存占用高、稳定性敏感。

### 6.3 MedicalGPT 中的 RLOO/PPO 类实现

本项目的 `training/ppo_training.py` 文件描述是 RLOO，即 REINFORCE Leave-One-Out，作为 PPO 替代方案使用 TRL 的 `RLOOTrainer`。

主要对象：

- `sft_model_path`：待优化的 SFT policy。
- `reward_model_path`：奖励模型。
- `AutoModelForSequenceClassification`：加载 reward model。
- `AutoModelForCausalLM`：加载 policy model。
- `RLOOTrainer`：执行 RL 优化。

代码数据流：

```text
prompt dataset
  -> preprocess_function 只抽取 prompt
  -> policy 生成多个 completion
  -> reward_model 打分
  -> RLOO 估计相对优势
  -> 更新 policy
```

RLOO 的直觉是：对同一个 prompt 采样多个回答，用“某个回答的奖励 - 其他回答平均奖励”作为优势估计，降低 REINFORCE 的方差。

### 6.4 RLHF 的工程成本

RLHF 相比 SFT/DPO 更重，因为训练时可能同时需要：

- policy model
- reference model
- reward model
- value model 或 baseline
- rollout 生成过程
- KL 计算
- 分布式训练和显存管理

这也是后来 DPO、ORPO、SimPO、KTO 等方法流行的重要原因：它们试图用更接近监督学习的方式完成偏好对齐，减少在线 RL 的复杂度。

## 7. DPO：直接偏好优化

### 7.1 DPO 为什么出现

DPO 的动机是：既然 RLHF 最终也是希望 policy 偏向 chosen、远离 rejected，能不能不显式训练 reward model、不跑在线 RL，而直接在偏好数据上优化 policy？

DPO 从带 $\mathrm{KL}$ 约束的 RLHF 目标出发，推导出一个闭式关系：最优 policy 和 reward 之间可以互相表示。于是可以把 reward model 消掉，直接得到一个分类式 loss。

### 7.2 DPO 的核心公式

给定偏好样本 $(x, y_w, y_l)$：

$$
\mathcal{L}_{\mathrm{DPO}}(\theta)
= -\log \sigma\left(
\beta\left[
\log \pi_{\theta}(y_w \mid x)
- \log \pi_{\mathrm{ref}}(y_w \mid x)
- \log \pi_{\theta}(y_l \mid x)
+ \log \pi_{\mathrm{ref}}(y_l \mid x)
\right]
\right)
$$

直觉：

- 如果当前模型相对参考模型更偏向 chosen，loss 下降。
- 如果当前模型相对参考模型更偏向 rejected，loss 上升。
- $\pi_{\mathrm{ref}}$ 负责提供锚点，避免模型无约束漂移。

### 7.3 MedicalGPT 中的 DPO 实现

核心文件：`training/dpo_training.py`

关键代码路径：

1. 加载 tokenizer。
2. 加载 `data/reward/*.jsonl`。
3. `return_prompt_and_responses()` 把原始样本转成 `prompt/chosen/rejected`。
4. 加载 `AutoModelForCausalLM`。
5. 配置 LoRA。
6. 使用 TRL 的 `DPOTrainer`。

核心数据转换结果是：

```python
{
    "prompt": prompts,
    "chosen": chosen_list,
    "rejected": rejected_list,
}
```

这正是 `DPOTrainer` 需要的格式。

### 7.4 DPO 和 RM+PPO 的对比

| 维度 | RM + PPO/RLOO | DPO |
| --- | --- | --- |
| 是否训练 reward model | 是 | 否 |
| 是否在线采样 | 通常是 | 否 |
| 训练稳定性 | 更敏感 | 通常更稳定 |
| 显存成本 | 高 | 中等 |
| 数据需求 | 偏好数据 + prompt | 偏好数据 |
| 探索能力 | 较强 | 受离线数据限制 |
| 工程复杂度 | 高 | 低 |

DPO 的核心局限是：它主要学习离线偏好数据中的比较关系。如果偏好数据覆盖不足，模型很难通过在线探索发现更好的答案。

## 8. ORPO：把 SFT 和偏好优化合并

### 8.1 ORPO 的动机

ORPO 试图进一步简化流程：不使用 reference model，把 SFT 和偏好惩罚放在一个目标里。

它的思想是：

- 对 chosen 做普通 SFT，让模型学会好回答。
- 同时用 odds ratio 惩罚 rejected，让模型降低坏回答风格的相对概率。

### 8.2 ORPO 的目标直觉

可以把 ORPO 简化理解为：

$$
\mathcal{L}_{\mathrm{ORPO}}
= \mathcal{L}_{\mathrm{SFT}}(y_w)
+ \lambda\,\mathcal{L}_{\mathrm{OR}}(y_l, y_w)
$$

其中 odds ratio 项关注 chosen 和 rejected 的相对可取性，而不是引入单独的 reference model。

### 8.3 MedicalGPT 中的 ORPO 实现

核心文件：`training/orpo_training.py`

这个文件结构和 DPO 很像：

- 读取 `conversations/chosen/rejected`。
- 构建 `prompt/chosen/rejected`。
- 加载 causal LM。
- 配置 LoRA。
- 训练偏好模型。

MedicalGPT 当前文件中仍复用了 TRL 的 `DPOConfig`、`DPOTrainer` 风格接口，并额外定义了 `orpo_beta` 参数。阅读时要注意区分“文件名和参数表达的 ORPO 意图”与“底层 trainer 接口复用”的工程实现方式。

### 8.4 ORPO 的适用场景

ORPO 更适合：

- 想减少 reference model 显存成本。
- 偏好数据质量较高。
- 希望把 SFT 和偏好对齐合并成单阶段。

风险：

- 没有 reference model 约束时，训练稳定性依赖 loss 设计和数据质量。
- 如果 rejected 样本构造不合理，模型可能学到错误的负偏好。

## 9. GRPO：面向可验证奖励和推理任务

### 9.1 GRPO 的基本动机

GRPO 常见于数学、代码、推理这类可验证任务。它不一定需要人工偏好对，而是对同一个问题生成多个答案，用规则或验证器给奖励，再做组内相对优化。

直觉：

同一个 prompt 采样 $K$ 个回答：

$$
y_1, y_2, \ldots, y_K
$$

分别计算奖励：

$$
r_1, r_2, \ldots, r_K
$$

用组内归一化奖励作为 advantage：

$$
A_i = \frac{r_i - \mathrm{mean}(r_1,\ldots,r_K)}
{\mathrm{std}(r_1,\ldots,r_K)}
$$

这样模型不需要单独训练 value model，也能得到相对优势信号。

### 9.2 MedicalGPT 中的 GRPO 实现

核心文件：`training/grpo_training.py`

关键函数：

- `accuracy_reward(completions, answer, **kwargs)`
- `format_reward(completions, **kwargs)`
- `grpo_train(...)`

数据格式来自 `data/grpo/sample.jsonl`：

```json
{"question": "肛门病变可能是什么疾病的症状?", "answer": "食管克罗恩病"}
```

代码会把样本转成：

```python
{
    "prompt": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": x["question"]}
    ],
    "answer": x["answer"]
}
```

`SYSTEM_PROMPT` 要求模型输出：

```text
<think> reasoning process here </think><answer> answer here </answer>
```

于是奖励函数分成两类：

- 格式奖励：是否满足 `<think>` 和 `<answer>` 标签。
- 正确性奖励：解析答案后是否和标准答案等价。

### 9.3 GRPO 的优势和风险

优势：

- 适合数学、代码、结构化答案、工具调用等可验证任务。
- 不一定需要人工偏好标注。
- 可以直接优化推理格式和最终答案。

风险：

- 奖励函数写得不好会导致 reward hacking。
- 可验证任务不等于所有任务，开放式医疗问答很难只靠规则奖励。
- 格式奖励过强时，模型可能形式正确但内容空洞。

## 10. Agent 工具调用后训练

MedicalGPT v2.6 支持 Agent 工具调用微调，关键文件是：

- `training/tool_utils.py`
- `data/sft/glaive_toolcall_zh_demo.jsonl`
- `data/reward/toolcall_dpo_zh_demo.jsonl`

工具调用后训练要让模型学会三件事：

1. 根据工具 schema 判断是否需要调用工具。
2. 输出符合指定模型格式的 function call。
3. 读取 observation 后生成自然语言总结。

不同模型族的工具调用格式不同：

| 格式 | 代码类 | 特点 |
| --- | --- | --- |
| default | `DefaultToolUtils` | `Action` / `Action Input` |
| GLM4 | `GLM4ToolUtils` | GLM 风格工具提示 |
| Llama3 | `Llama3ToolUtils` | JSON function call |
| Qwen | `QwenToolUtils` | `<tools>`、`<tool_call>`、`<tool_response>` |
| Mistral | 相关 formatter | Mistral 工具调用标签 |

工具调用 SFT 的关键不只是回答质量，而是可解析性。模型输出如果多一个逗号、少一个标签，都可能导致执行器无法解析。

因此工具调用后训练至少要评估：

- 是否在该调用工具时调用工具。
- function name 是否正确。
- arguments 是否是合法 JSON。
- 参数字段是否满足 schema。
- observation 后是否能正确总结。
- 不该调用工具时是否避免乱调工具。

## 11. 主流后训练算法谱系

截至 2026 年 4 月，工程上常见的后训练算法可以按监督信号分成几类。

### 11.1 示范学习类

| 算法 | 信号 | 核心目标 | 代表场景 |
| --- | --- | --- | --- |
| SFT | 标准答案 | 最大化示范答案概率 | 指令微调、领域适配 |
| Behavior Cloning | 专家轨迹 | 模仿专家行为 | Agent 轨迹学习 |
| Rejection Sampling Fine-tuning | 采样后筛选的高分答案 | 对高质量样本继续 SFT | 用 RM/规则筛出好样本 |
| RAFT | reward-ranked samples | 生成、排序、再微调 | 比 PPO 更稳定的迭代式对齐 |

### 11.2 奖励模型 + RL 类

| 算法 | 信号 | 特点 |
| --- | --- | --- |
| RM + PPO | pairwise preference + online rollout | 经典 RLHF，能力强但复杂 |
| RM + RLOO | reward + 多样本 baseline | 去掉 value model 或降低方差 |
| GRPO | 组内相对奖励 | 常用于可验证推理任务 |

### 11.3 直接偏好优化类

| 算法 | 是否需要 reference model | 数据 | 核心差异 |
| --- | --- | --- | --- |
| DPO | 需要 | chosen/rejected | 从 KL-RLHF 推导出的分类 loss |
| IPO | 通常需要 | chosen/rejected | 修改偏好目标，缓解过拟合偏好强度 |
| ORPO | 不需要 | chosen/rejected | SFT + odds ratio 偏好项 |
| SimPO | 不需要 | chosen/rejected | 用平均 log probability 作为隐式 reward |
| KTO | 通常不需要 pairwise | desirable/undesirable | 用前景理论式人类效用建模 |

这些算法的共同点是：尽量避免在线 RL 的复杂工程成本，把偏好学习变成更接近监督学习的训练过程。

## 12. 该如何选择后训练算法

### 12.1 如果你只有问答示范数据

选择 SFT。

原因：

- 数据格式简单。
- loss 稳定。
- 易调试。
- 最适合教学和领域模型入门。

在本项目中使用：

```bash
bash scripts/run_sft.sh
```

### 12.2 如果你有 chosen/rejected 偏好数据

优先选择 DPO。

原因：

- 不需要单独训练 reward model。
- 不需要在线 rollout。
- TRL 支持成熟。
- 工程成本低于 PPO。

在本项目中使用：

```bash
bash scripts/run_dpo.sh
```

### 12.3 如果你希望完整理解 RLHF

选择 RM + RLOO/PPO 路线。

原因：

- 能理解 reward model、policy、reference model、KL 约束之间的关系。
- 更接近经典 InstructGPT 技术路线。

在本项目中依次看：

```bash
bash scripts/run_rm.sh
bash scripts/run_ppo.sh
```

### 12.4 如果任务答案可自动验证

选择 GRPO 或类似 rule-based RL。

适合：

- 数学题。
- 代码题。
- 格式严格的结构化输出。
- 工具调用可执行验证。

在本项目中看：

```bash
bash scripts/run_grpo.sh
```

### 12.5 如果显存紧张

优先组合：

```text
LoRA / QLoRA + SFT
LoRA / QLoRA + DPO
```

尽量先避开完整 PPO/RLHF，因为它通常需要更多模型副本和 rollout 过程。

## 13. MedicalGPT 后训练代码阅读路线

建议按下面路线读：

### 13.1 先读 SFT

文件：

- `training/supervised_finetuning.py`
- `scripts/run_sft.sh`
- `data/sft/*.jsonl`

目标：

- 搞懂 `conversations` 如何变成 token。
- 搞懂 assistant token 的 label 如何保留，prompt token 如何 mask。
- 搞懂 LoRA 如何挂载。

### 13.2 再读 RM

文件：

- `training/reward_modeling.py`
- `scripts/run_rm.sh`
- `data/reward/dpo_zh_500.jsonl`

目标：

- 搞懂 `chosen/rejected` 如何分别 tokenize。
- 搞懂 pairwise log loss。
- 搞懂 reward model 为什么是 sequence classification。

### 13.3 再读 DPO

文件：

- `training/dpo_training.py`
- `scripts/run_dpo.sh`

目标：

- 搞懂 `prompt/chosen/rejected` 格式。
- 搞懂 DPO 为什么不需要显式 reward model。
- 搞懂 `DPOTrainer` 的输入格式。

### 13.4 最后读 RL 和 GRPO

文件：

- `training/ppo_training.py`
- `training/grpo_training.py`

目标：

- 搞懂 prompt-only 数据如何用于 rollout。
- 搞懂 reward function 如何替代人工偏好。
- 搞懂为什么 RL 类训练比 DPO 更难调。

## 14. 后训练中的评估问题

后训练不能只看 loss。不同阶段要看不同指标。

### 14.1 SFT 评估

关注：

- eval loss 是否下降。
- 回答是否遵循指令。
- 是否发生过拟合。
- 多轮对话格式是否稳定。
- 医疗回答是否过度自信。

### 14.2 RM 评估

关注：

- chosen reward 是否高于 rejected reward。
- pairwise accuracy。
- reward 分布是否塌缩。
- 是否对长度、格式、套话产生偏置。

### 14.3 DPO/ORPO 评估

关注：

- chosen logprob 是否相对提升。
- rejected 是否被压低。
- 输出是否变短、模板化或过度拒答。
- 通用能力是否退化。

### 14.4 RL/GRPO 评估

关注：

- reward 是否上升。
- KL 是否失控。
- 格式通过率。
- 答案正确率。
- 是否出现 reward hacking。

医疗项目还必须增加：

- 安全拒答能力。
- 不确定性表达。
- 引导就医和咨询专业医生。
- 避免编造诊断和用药剂量。
- 数据隐私和合规检查。

## 15. 数学公式与代码对应表

| 数学对象 | 含义 | MedicalGPT 代码位置 |
| --- | --- | --- |
| $\pi_{\theta}(y \mid x)$ | 当前语言模型 | `AutoModelForCausalLM` |
| $\pi_{\mathrm{ref}}(y \mid x)$ | 参考模型 | `DPOTrainer` 内部或 `ref_model` |
| $r_{\phi}(x,y)$ | 奖励模型分数 | `AutoModelForSequenceClassification` |
| $\mathcal{L}_{\mathrm{SFT}}$ | token-level 交叉熵 | `supervised_finetuning.py` + `Trainer` |
| $-\log \sigma(r_w-r_l)$ | RM pairwise loss | `RewardTrainer.compute_loss()` |
| DPO loss | 直接偏好优化 | `DPOTrainer` |
| ORPO odds ratio | reference-free 偏好项 | `orpo_training.py` |
| reward function | 规则奖励/验证奖励 | `accuracy_reward`、`format_reward` |
| LoRA 参数 | 低秩适配器 | `LoraConfig` |

## 16. 学生实验建议

### 实验 1：跑通 SFT

目标：理解后训练最小闭环。

步骤：

1. 读取 `data/sft/medical_sft_1K_format.jsonl`。
2. 修改或复制 `scripts/run_sft.sh`，设置小样本。
3. 跑 100 条以内样本。
4. 用 `demo/inference.py` 测试训练前后差异。

建议参数：

```bash
--max_train_samples 100
--max_eval_samples 10
--num_train_epochs 1
--model_max_length 512
--per_device_train_batch_size 1
```

### 实验 2：观察 RM 的 chosen/rejected 分数

目标：理解奖励模型。

步骤：

1. 用 `data/reward/dpo_zh_500.jsonl`。
2. 跑小样本 RM。
3. 打印同一 prompt 下 chosen 和 rejected 的 reward。
4. 观察 chosen 是否整体更高。

### 实验 3：比较 SFT 和 DPO 输出

目标：理解偏好优化的效果。

步骤：

1. 先训练一个小样本 SFT。
2. 再用同一 base/SFT 模型跑 DPO。
3. 固定测试 prompt，比较两个模型回答。
4. 重点观察：回答是否更符合偏好，是否变得更保守或模板化。

### 实验 4：改写 GRPO 奖励函数

目标：理解 rule-based reward。

步骤：

1. 阅读 `accuracy_reward` 和 `format_reward`。
2. 增加一个简单奖励，例如回答长度在合理范围内加分。
3. 观察 reward 变化。

注意：奖励函数越容易被投机，模型越容易学到奇怪行为。奖励设计必须谨慎。

## 17. 常见错误和排查方式

### 17.1 数据格式错误

表现：

- `KeyError: conversations`
- `KeyError: chosen`
- 训练样本数为 0。

排查：

- SFT 数据必须有 `conversations`。
- DPO/RM 数据必须有 `conversations/chosen/rejected`。
- jsonl 每行必须是完整 JSON object。

### 17.2 prompt 模板不匹配

表现：

- 模型回答格式混乱。
- 推理和训练表现差异大。
- stop token 不生效。

排查：

- 训练和推理使用同一 tokenizer 或同一 `template_name`。
- 检查 `training/template.py`。
- 检查 `tokenizer.apply_chat_template` 生成的文本。

### 17.3 labels mask 错误

表现：

- 模型复述用户问题。
- loss 下降但回答质量差。

排查：

- 检查 prompt token 是否被设为 `IGNORE_INDEX`。
- 检查 `train_on_inputs` 是否符合预期。

### 17.4 DPO 训练后模型退化

表现：

- 输出变短。
- 泛化能力下降。
- 过度拒答。

排查：

- 降低学习率。
- 检查偏好数据质量。
- 减少训练步数。
- 调整 $\beta$。
- 混入少量高质量 SFT 数据或做 staged training。

### 17.5 RL reward hacking

表现：

- reward 上升但人工看质量下降。
- 模型输出固定模板骗过格式奖励。

排查：

- 增加人工抽检。
- 拆分 reward：格式、正确性、安全性分别统计。
- 限制 KL。
- 修改奖励函数，避免单一可钻空子的指标。

## 18. 推荐阅读资料

这些资料可以帮助你把 MedicalGPT 代码和主流后训练算法联系起来：

- InstructGPT / RLHF: https://arxiv.org/abs/2203.02155
- PPO: https://arxiv.org/abs/1707.06347
- DPO: https://arxiv.org/abs/2305.18290
- ORPO: https://arxiv.org/abs/2403.07691
- KTO: https://arxiv.org/abs/2402.01306
- SimPO: https://arxiv.org/abs/2405.14734
- RAFT: https://arxiv.org/abs/2304.06767
- TRL 文档: https://huggingface.co/docs/trl
- PEFT 文档: https://huggingface.co/docs/peft
- Transformers 文档: https://huggingface.co/docs/transformers

## 19. 一句话总结

后训练不是一个单独算法，而是一组把 base model 变成可用 assistant 的工程和数学方法。SFT 负责教模型“怎么答”，RM/PPO 负责用奖励继续优化“答得好不好”，DPO/ORPO 等方法把偏好学习压缩成更稳定的离线训练，GRPO 则把可验证任务中的规则奖励引入推理能力训练。

在 MedicalGPT 中，最适合学生的学习顺序是：

```text
SFT -> DPO -> RM -> RLOO/PPO -> ORPO -> GRPO -> Agent tool call
```

只要你能把 `conversations/chosen/rejected` 这几类数据如何变成 loss 讲清楚，就已经抓住了后训练的主干。
