# MedicalGPT 数据处理全流程：阶段、格式与样例

这份文档专门讲 MedicalGPT 项目里的数据：每个训练阶段用什么数据、数据格式是什么、代码如何读取和预处理、学生应该如何检查自己的数据。读完后，你应该能回答三个问题：

1. `data/` 目录下每类文件分别服务哪个训练阶段。
2. `train_file_dir` 和 `validation_file_dir` 传入后，代码如何把文件变成模型输入。
3. 自己准备数据时，应该整理成什么字段，怎么验证格式。

建议配合这些文件阅读：

- `docs/datasets.md`
- `training/pretraining.py`
- `training/supervised_finetuning.py`
- `training/reward_modeling.py`
- `training/dpo_training.py`
- `training/orpo_training.py`
- `training/ppo_training.py`
- `training/grpo_training.py`
- `training/tool_utils.py`
- `tools/convert_dataset.py`
- `tools/validate_jsonl.py`

## 1. 总览：不同阶段用什么数据

MedicalGPT 的数据不是一种格式通吃。不同训练目标需要不同监督信号：

| 阶段 | 目标 | 数据目录 | 主要格式 | 训练入口 |
| --- | --- | --- | --- | --- |
| PT 增量预训练 | 继续学习领域语料分布 | `data/pretrain/` | `.txt` 每行文本，或 `.jsonl` 的 `text` 字段 | `training/pretraining.py` |
| SFT 有监督微调 | 学会按指令回答 | `data/sft/` | ShareGPT 风格 `conversations` | `training/supervised_finetuning.py` |
| Agent SFT | 学会工具调用 | `data/sft/` | `conversations` + 可选 `tools`，角色含 `function_call`、`observation` | `training/supervised_finetuning.py` |
| RM 奖励模型 | 学会给好坏回答打分 | `data/reward/` | `conversations` + `chosen` + `rejected` | `training/reward_modeling.py` |
| DPO | 直接学习偏好 | `data/reward/` | `conversations` + `chosen` + `rejected` | `training/dpo_training.py` |
| ORPO | 合并 SFT 和偏好优化 | `data/reward/` | `conversations` + `chosen` + `rejected` | `training/orpo_training.py` |
| RLOO/PPO | 根据 reward model 做 RL 优化 | `data/sft/` | `conversations`，代码主要抽取 prompt | `training/ppo_training.py` |
| GRPO | 用规则/验证器奖励训练 | `data/grpo/` | `question` + `answer` | `training/grpo_training.py` |
| RAG 演示 | 检索增强问答知识库 | `data/rag/` | 当前样例为每行 JSON：`问` + `答` | `demo/chatpdf.py` |

最重要的是两类后训练格式：

- SFT：`conversations`
- 偏好优化/RM/DPO/ORPO：`conversations` + `chosen` + `rejected`

## 2. 项目如何读取数据

绝大多数训练脚本都遵循同一套入口参数：

```bash
--train_file_dir ./data/xxx
--validation_file_dir ./data/xxx
--max_train_samples 1000
--max_eval_samples 10
--preprocessing_num_workers 4
```

代码内部通常这样处理：

```text
train_file_dir / validation_file_dir
  -> glob 搜索目录下文件
  -> datasets.load_dataset(...)
  -> map(preprocess_function)
  -> tokenizer 或 prompt builder
  -> input_ids / attention_mask / labels
  -> Trainer
```

注意：

- 本项目大部分训练文件都要求从项目根目录运行脚本。
- `train_file_dir` 通常传目录，不是单个文件。
- SFT/RM/DPO/ORPO 会递归读取目录下的 `.jsonl`。
- PT 会读取 `.txt` 或 `.jsonl`，但同一个训练目录下不能混用 `.txt` 和 `.jsonl`。
- 如果没有单独验证集，部分脚本会用 `validation_split_percentage` 从训练集切一部分做验证。

## 3. PT 数据：增量预训练语料

### 3.1 数据目标

PT 不是问答训练，而是继续做 causal language modeling。它让模型继续适应医疗或领域文本分布。

模型看到的是连续文本，目标是预测下一个 token：

$$
\mathcal{L}_{\mathrm{PT}}
= -\sum_t \log p_\theta(x_t \mid x_{<t})
$$

### 3.2 文件位置

样例目录：

- `data/pretrain/fever.txt`
- `data/pretrain/en_article_tail500.txt`
- `data/pretrain/tianlongbabu.txt`

启动脚本：

- `scripts/run_pt.sh`

代码入口：

- `training/pretraining.py`

### 3.3 支持格式

格式 A：`.txt`，每行一个文档或段落。

```text
第一章论
传染病是指由病原微生物，如朊粒、病毒、衣原体、立克次体...
```

格式 B：`.jsonl`，每行一个 JSON object，通常包含 `text` 字段。

```json
{"text": "传染病是指由病原微生物感染人体后产生的有传染性疾病。"}
{"text": "感染性疾病包括传染病和非传染性感染性疾病。"}
```

### 3.4 代码如何处理

`training/pretraining.py` 会：

1. 搜索 `train_file_dir` 下的 `.txt` 和 `.jsonl`。
2. 判断扩展名，使用 `load_dataset("text")` 或 `load_dataset("json")`。
3. 如果 `--packing True`，先 tokenize，再把多个文本拼接后按 `block_size` 切块。
4. 设置 `labels = input_ids.copy()`。

关键函数：

- `tokenize_function`
- `tokenize_wo_pad_function`
- `group_text_function`

packing 的数据流：

```text
doc1 -> token ids
doc2 -> token ids
doc3 -> token ids
拼接为 [doc1][EOS][doc2][EOS][doc3][EOS]
按 block_size 切成固定长度 chunk
labels = input_ids
```

学生要注意：

- PT 数据不需要 `conversations`。
- PT 数据不要混入问答 JSON，除非你明确把它整理成 `text` 字段。
- `block_size` 控制每个训练样本长度。
- `packing=True` 会提高训练效率，但样本边界被拼接了。

## 4. SFT 数据：普通指令微调

### 4.1 数据目标

SFT 训练模型根据用户问题输出示范答案。它使用 token-level 交叉熵，但通常只让 assistant 的回答部分参与 loss。

$$
\mathcal{L}_{\mathrm{SFT}}
= -\sum_{t \in \mathcal{T}_{\mathrm{assistant}}}
\log p_\theta(y_t \mid x, y_{<t})
$$

### 4.2 文件位置

样例目录：

- `data/sft/medical_sft_1K_format.jsonl`
- `data/sft/sharegpt_zh_1K_format.jsonl`

启动脚本：

- `scripts/run_sft.sh`

代码入口：

- `training/supervised_finetuning.py`

### 4.3 单轮问答格式

每一行是一个 JSON object：

```json
{
  "conversations": [
    {
      "from": "human",
      "value": "治疗阳痿吃什么药呢？"
    },
    {
      "from": "gpt",
      "value": "建议到正规医院就诊，由专业医生评估病因后再决定治疗方案。"
    }
  ]
}
```

字段解释：

- `conversations`：一个对话列表。
- `from`：消息角色。
- `value`：消息内容。
- `human`：用户。
- `gpt`：助手答案，也就是模型要学习的输出。

### 4.4 多轮对话格式

```json
{
  "conversations": [
    {"from": "human", "value": "什么是高血压？"},
    {"from": "gpt", "value": "高血压是指动脉血压持续升高的一类情况。"},
    {"from": "human", "value": "日常生活应该注意什么？"},
    {"from": "gpt", "value": "建议低盐饮食、规律运动、监测血压，并遵医嘱治疗。"}
  ]
}
```

多轮格式要求角色基本交替。代码会把一组 `human/gpt` 消息整理成历史轮次，再结合 prompt template 或 tokenizer 的 chat template 拼成训练文本。

### 4.5 代码如何处理

`training/supervised_finetuning.py` 中的核心是 `preprocess_function`。

它大致做这些事：

1. 从 `examples["conversations"]` 取出每条对话。
2. 处理 `system`、`human`、`gpt` 等角色。
3. 如果设置了 `template_name`，使用 `training/template.py` 中的模板。
4. 否则使用 `tokenizer.apply_chat_template`。
5. tokenize 成 `input_ids` 和 `attention_mask`。
6. 构造 `labels`，通常把非 assistant 部分置为 `IGNORE_INDEX`。

概念流程：

```text
conversations
  -> history_messages: [[user1, assistant1], [user2, assistant2]]
  -> prompt string
  -> tokenizer
  -> input_ids
  -> labels: 用户 token 为 -100，助手 token 保留真实 id
```

学生阅读时要重点追踪：

- `roles = ["human", "gpt"]`
- `get_dialog`
- `prompt_template.get_dialog`
- `tokenizer.apply_chat_template`
- `IGNORE_INDEX`
- `train_on_inputs`

## 5. Agent SFT 数据：工具调用微调

### 5.1 数据目标

Agent SFT 不只训练自然语言回答，还训练模型输出函数调用，并在工具返回 observation 后继续回答。

它要学会三件事：

1. 判断是否需要调用工具。
2. 输出符合格式的 function call。
3. 读取 observation 后给用户总结结果。

### 5.2 文件位置

样例文件：

- `data/sft/glaive_toolcall_zh_demo.jsonl`

代码入口：

- `training/supervised_finetuning.py`
- `training/tool_utils.py`

### 5.3 工具调用 SFT 格式

```json
{
  "conversations": [
    {
      "from": "human",
      "value": "我需要为John Doe生成一张发票。他购买了2个苹果，每个1美元，以及3根香蕉，每根0.5美元。"
    },
    {
      "from": "function_call",
      "value": "{\"name\":\"generate_invoice\",\"arguments\":{\"customer_name\":\"John Doe\",\"items\":[{\"name\":\"apple\",\"quantity\":2,\"price\":1},{\"name\":\"banana\",\"quantity\":3,\"price\":0.5}]}}"
    },
    {
      "from": "observation",
      "value": "{\"invoice_id\":\"INV12345\",\"total\":3.5,\"status\":\"generated\"}"
    },
    {
      "from": "gpt",
      "value": "发票已成功生成，发票编号为 INV12345，总金额为 3.5 美元。"
    }
  ],
  "tools": "[{\"name\":\"generate_invoice\",\"description\":\"生成发票\",\"parameters\":{\"type\":\"object\",\"properties\":{\"customer_name\":{\"type\":\"string\"}},\"required\":[\"customer_name\"]}}]"
}
```

注意这里有两个容易出错的点：

- `tools` 在当前样例里是 JSON 字符串，不是 JSON 数组对象。
- `function_call.value` 也是 JSON 字符串，里面包含 `name` 和 `arguments`。

### 5.4 代码如何处理

`training/supervised_finetuning.py` 会识别：

- `tools`
- `function_call`
- `observation`

如果设置了 `--tool_format default/qwen/glm4/mistral/llama3`，会调用 `training/tool_utils.py` 中的 formatter：

- `DefaultToolUtils`
- `GLM4ToolUtils`
- `Llama3ToolUtils`
- `QwenToolUtils`

处理逻辑：

```text
tools schema
  -> tool_formatter
  -> 拼入 system prompt

function_call
  -> function_formatter
  -> 变成模型要学习的 assistant 输出

observation
  -> 按模型格式包装
  -> 作为下一轮 user/observation 输入
```

学生要注意：工具调用任务的核心不是“答案看起来对”，而是“输出能不能被程序解析”。训练和评估时要检查 function name、JSON 参数、必填字段和 observation 总结。

## 6. RM / DPO / ORPO 数据：偏好对

### 6.1 数据目标

偏好数据不是给一个标准答案，而是告诉模型：

```text
在同一个 prompt 下，chosen 比 rejected 更好。
```

这个格式同时服务：

- Reward Modeling
- DPO
- ORPO

### 6.2 文件位置

样例文件：

- `data/reward/dpo_zh_500.jsonl`
- `data/reward/toolcall_dpo_zh_demo.jsonl`

启动脚本：

- `scripts/run_rm.sh`
- `scripts/run_dpo.sh`
- `scripts/run_orpo.sh`

代码入口：

- `training/reward_modeling.py`
- `training/dpo_training.py`
- `training/orpo_training.py`

### 6.3 普通偏好格式

```json
{
  "conversations": [
    {
      "from": "human",
      "value": "20个关于新鲜果汁菜单的口号，适用于一家名为 Dishes 的餐厅"
    }
  ],
  "chosen": "这里是 20 个突出新鲜果汁菜单的口号：...",
  "rejected": "1. 与菜肴一起品尝新鲜！..."
}
```

字段解释：

- `conversations`：用户上下文，可能是单轮，也可能是多轮。
- `chosen`：更好的回答。
- `rejected`：较差的回答。

### 6.4 工具调用偏好格式

```json
{
  "conversations": [
    {"from": "human", "value": "北京今天天气怎么样？"}
  ],
  "tools": "[{\"name\":\"get_weather\",\"description\":\"获取指定城市的天气信息\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}}]",
  "chosen": "Action: get_weather\nAction Input: {\"city\":\"北京\"}\n",
  "rejected": "北京今天天气晴朗，气温大约25度。"
}
```

这个样例表达的偏好是：用户问实时天气时，模型应该调用工具，而不是凭空编造天气。

### 6.5 RM 如何处理偏好数据

`training/reward_modeling.py` 的目标是训练一个标量奖励模型。

数据处理逻辑：

```text
conversations + chosen
  -> chosen_prompt
  -> tokenizer
  -> input_ids_chosen / attention_mask_chosen

conversations + rejected
  -> rejected_prompt
  -> tokenizer
  -> input_ids_rejected / attention_mask_rejected
```

核心函数：

- `_parse_conversations_to_messages`
- `preprocess_reward_function`
- `RewardDataCollatorWithPadding`

训练时，`RewardTrainer.compute_loss()` 会让 chosen 的 reward 高于 rejected：

$$
\mathcal{L}_{\mathrm{RM}}
= -\log \sigma(r_\phi(x,y_w)-r_\phi(x,y_l))
$$

### 6.6 DPO / ORPO 如何处理偏好数据

`training/dpo_training.py` 和 `training/orpo_training.py` 会把数据转换成：

```python
{
    "prompt": "...",
    "chosen": "...",
    "rejected": "..."
}
```

核心函数：

- DPO：`return_prompt_and_responses`
- ORPO：`return_prompt_and_responses`

处理逻辑：

```text
conversations
  -> prompt
chosen
  -> 正样本回答
rejected
  -> 负样本回答
DPOTrainer / ORPO 风格训练器
```

和 RM 的区别：

- RM 会把 `prompt + chosen`、`prompt + rejected` 变成两个 sequence classification 输入。
- DPO/ORPO 保留 `prompt/chosen/rejected` 三段，让 trainer 自己计算回答的 log probability 和偏好 loss。

## 7. RLOO/PPO 数据：从 SFT 数据中抽 prompt

### 7.1 数据目标

RLOO/PPO 阶段不直接学习原始答案，而是让 policy 根据 prompt 自己生成 completion，然后用 reward model 打分。

因此它更关心 prompt，而不是 SFT 里的目标回答。

### 7.2 文件位置

启动脚本：

- `scripts/run_ppo.sh`

代码入口：

- `training/ppo_training.py`

默认脚本使用：

```bash
--train_file_dir ./data/sft
--validation_file_dir ./data/sft
```

### 7.3 支持格式

它复用 SFT 的 `conversations` 格式：

```json
{
  "conversations": [
    {"from": "human", "value": "什么是高血压？"},
    {"from": "gpt", "value": "高血压是指动脉血压持续升高。"}
  ]
}
```

### 7.4 代码如何处理

`training/ppo_training.py` 中的 `preprocess_function` 会生成：

```python
{"prompt": [...]}
```

概念上：

```text
conversations
  -> 只抽出 user prompt
  -> policy 生成 completion
  -> reward model 给 completion 打分
  -> RLOOTrainer 更新 policy
```

学生要注意：PPO/RLOO 阶段不是简单复读 SFT 答案。它更像“给模型题目，让模型作答，再根据奖励更新”。

## 8. GRPO 数据：question / answer

### 8.1 数据目标

GRPO 更适合有可验证答案的任务，例如数学题、代码题、结构化医学问答。它需要一个问题和一个可验证的标准答案。

### 8.2 文件位置

样例文件：

- `data/grpo/sample.jsonl`

启动脚本：

- `scripts/run_grpo.sh`

代码入口：

- `training/grpo_training.py`

### 8.3 数据格式

```json
{"question": "1+2=?", "answer": "3"}
```

医学问答样例：

```json
{
  "question": "膺窗穴的定位是什么?",
  "answer": "第3肋间隙，距前正中线4寸。"
}
```

### 8.4 代码如何处理

`training/grpo_training.py` 会把每条数据转成：

```python
{
    "prompt": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": x["question"]}
    ],
    "answer": x["answer"]
}
```

然后用奖励函数打分：

- `format_reward`：检查输出是否满足 `<think>...</think><answer>...</answer>`。
- `accuracy_reward`：检查答案是否和 `answer` 等价或匹配。

简化流程：

```text
question
  -> prompt
  -> model 生成多个 completion
  -> format_reward + accuracy_reward
  -> GRPOTrainer 按组内相对奖励更新模型
```

## 9. RAG 数据：知识库文件

### 9.1 数据目标

RAG 数据不是训练数据，而是检索增强问答的外部知识。模型回答前先检索相关文本，再把检索结果作为上下文提供给模型。

### 9.2 文件位置

样例文件：

- `data/rag/medical_corpus.txt`

代码入口：

- `demo/chatpdf.py`

### 9.3 当前样例格式

当前样例每行是一个 JSON object：

```json
{"问": "肛门病变可能是什么疾病的症状?", "答": "食管克罗恩病"}
```

用于 RAG 时，关键不是训练 loss，而是能否被检索系统切分、向量化、召回。准备 RAG 数据时应关注：

- 文档是否干净。
- 每条知识是否足够完整。
- 是否有重复、冲突或过期医学信息。
- 是否能追溯来源。

## 10. 数据转换工具

### 10.1 `tools/convert_dataset.py`

这个工具把常见格式转换成本项目使用的 ShareGPT jsonl。

支持：

- Alpaca 格式：`instruction/input/output`
- QA 格式：`input/output`
- JSON array 转 JSONL
- 已有 ShareGPT-like 格式清理字段

Alpaca 转 ShareGPT：

```bash
python tools/convert_dataset.py \
  --in_file alpaca_data.json \
  --out_file out.jsonl \
  --data_type alpaca
```

输入：

```json
{
  "instruction": "解释什么是高血压",
  "input": "",
  "output": "高血压是指动脉血压持续升高..."
}
```

输出：

```json
{"conversations":[{"from":"human","value":"解释什么是高血压"},{"from":"gpt","value":"高血压是指动脉血压持续升高..."}]}
```

QA 转 ShareGPT：

```bash
python tools/convert_dataset.py \
  --in_file qa_data.jsonl \
  --out_file out.jsonl \
  --data_type qa \
  --file_type jsonl
```

输入：

```json
{"input": "糖尿病患者饮食注意什么？", "output": "建议控制总热量，规律进餐，遵医嘱管理血糖。"}
```

输出：

```json
{"conversations":[{"from":"human","value":"糖尿病患者饮食注意什么？"},{"from":"gpt","value":"建议控制总热量，规律进餐，遵医嘱管理血糖。"}]}
```

JSON array 转 JSONL：

```bash
python tools/convert_dataset.py \
  --in_file data.json \
  --out_file data.jsonl \
  --data_type json2jsonl
```

### 10.2 `tools/validate_jsonl.py`

这个工具检查 SFT jsonl 是否有基本字段：

```bash
python tools/validate_jsonl.py \
  --file_path data/sft/sharegpt_zh_1K_format.jsonl
```

它会检查：

- 每行是否是合法 JSON。
- 是否包含 `conversations`。
- `conversations` 是否是列表。
- 每条消息是否有 `from` 和 `value`。
- `from` 是否属于 `system/human/gpt`。

注意：当前 `validate_jsonl.py` 更适合普通 SFT 数据。Agent 数据里有 `function_call`、`observation`，会被它判为非法角色；这不一定代表训练脚本不能处理，只是验证工具较简单。

## 11. 自己准备数据时的检查清单

### 11.1 通用检查

- 每行必须是一个完整 JSON object，不能跨行。
- 文件编码使用 UTF-8。
- 不要有空行。
- 字段名必须和代码期望一致。
- 大文本里的换行要作为 JSON 字符串中的 `\n`。
- 训练集和验证集最好分开，至少要能用 `validation_split_percentage` 切分。

### 11.2 SFT 检查

- 必须有 `conversations`。
- `conversations` 必须是 list。
- 普通 SFT 的角色应主要是 `human/gpt`，可选 `system`。
- 最后一轮最好有 `gpt` 回答，否则这条样本可能无法形成有效监督。
- 回答中不要包含明显错误医学建议。

### 11.3 偏好数据检查

- 必须有 `conversations/chosen/rejected`。
- `chosen` 和 `rejected` 应该是同一个 prompt 下的两个回答。
- `chosen` 不一定要更长，但应该更正确、更安全或更符合格式。
- `rejected` 不要过于离谱，否则模型只学会区分容易样本。
- 偏好标准要一致，例如安全性、事实性、工具调用正确性不要混乱。

### 11.4 Agent 数据检查

- `tools` 必须能被 `json.loads` 解析。
- `function_call.value` 必须能被 `json.loads` 解析。
- function call 中最好包含 `name` 和 `arguments`。
- `arguments` 字段要符合工具 schema。
- `observation` 应该是工具执行结果，不要写成模型自然语言回答。

### 11.5 GRPO 数据检查

- 必须有 `question` 和 `answer`。
- `answer` 应尽量可验证。
- 如果用数学验证器，答案格式要容易解析。
- 如果是医学问答，注意很多答案不是唯一字符串，简单 exact match 可能不够。

## 12. 最小练习路线

建议学生按这个顺序练：

1. 打开 `data/sft/medical_sft_1K_format.jsonl`，手写 3 条同格式医学问答。
2. 用 `tools/validate_jsonl.py` 检查普通 SFT 数据。
3. 打开 `scripts/run_sft.sh`，确认 `--train_file_dir ./data/sft`。
4. 阅读 `training/supervised_finetuning.py` 中 `preprocess_function`。
5. 打开 `data/reward/dpo_zh_500.jsonl`，手写 3 条 `chosen/rejected` 偏好数据。
6. 阅读 `training/dpo_training.py` 中 `return_prompt_and_responses`。
7. 打开 `data/grpo/sample.jsonl`，手写 5 条 `question/answer` 可验证数据。
8. 阅读 `training/grpo_training.py` 中 `accuracy_reward` 和 `format_reward`。

## 13. 一句话总结

MedicalGPT 的数据处理主线可以概括为：

```text
PT 用 text 学领域语言分布；
SFT 用 conversations 学会回答；
Agent SFT 用 tools/function_call/observation 学会调用工具；
RM/DPO/ORPO 用 conversations + chosen/rejected 学会偏好；
RLOO/PPO 从 conversations 抽 prompt，再用 reward model 打分；
GRPO 用 question/answer 和规则奖励训练可验证推理；
RAG 用外部知识库做检索，不直接参与训练 loss。
```

对学生来说，最重要的不是背字段，而是理解“训练目标决定数据格式”：预测下一个 token 需要文本，模仿助手需要对话，学习偏好需要成对好坏答案，规则强化学习需要可验证答案。
