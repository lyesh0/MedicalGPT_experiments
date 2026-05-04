# MedicalGPT 项目阅读与学习指导

这份文档面向有 Python 基础、刚开始接触大模型训练项目的学生。它的目标不是替代 `README.md`，而是告诉你应该从哪里看起、每一步重点看什么、读完以后应该能回答哪些问题，以及如何用小样本跑通一个训练到推理的最小闭环。

MedicalGPT 是一个医学领域大模型训练与推理项目，覆盖增量预训练、有监督微调、奖励模型、强化学习、直接偏好优化、ORPO、GRPO、Agent 工具调用训练、推理部署和模型后处理工具。初学时不要一上来追求读完所有代码，应该先抓住一条主线：数据如何进入训练脚本，如何被拼成 prompt，模型如何训练，训练结果如何被推理脚本加载。

## 1. 先建立项目地图

建议按下面顺序阅读：

1. `README.md`
2. `data/`
3. `scripts/`
4. `training/`
5. `demo/`
6. `tools/`
7. `docs/`

各目录的学习重点如下：

| 目录或文件 | 主要作用 | 学生应该关注什么 |
| --- | --- | --- |
| `README.md` | 项目总览、安装方式、训练管线、示例命令 | 先理解项目能做什么，不急着记参数 |
| `data/` | 小样本训练数据 | 看清 PT、SFT、DPO/Reward、GRPO 数据格式 |
| `scripts/` | 一键训练脚本和 DeepSpeed 配置 | 从 shell 命令反推训练入口和关键参数 |
| `training/` | 核心训练代码 | 学习数据加载、prompt 构造、LoRA、Trainer、保存逻辑 |
| `demo/` | 推理、Gradio、FastAPI、RAG 示例 | 学习训练后模型如何被加载和使用 |
| `tools/` | 数据转换、模型合并、量化、词表处理 | 理解训练前后常见工程工具 |
| `docs/` | 数据集、训练参数、FAQ 等补充说明 | 遇到参数或数据格式疑问时查阅 |

第一遍阅读时，不要陷入所有模型和所有训练方法的细节。先围绕 SFT 有监督微调读，因为 SFT 同时连接了数据格式、tokenizer、prompt 模板、LoRA、Hugging Face Trainer 和推理流程，是理解项目的最佳入口。

## 2. 核心训练主线

项目支持的训练阶段可以理解为下面几类：

| 阶段 | 含义 | 入口脚本 | 启动脚本 |
| --- | --- | --- | --- |
| PT | Continue PreTraining，增量预训练 | `training/pretraining.py` | `scripts/run_pt.sh` |
| SFT | Supervised Fine-tuning，有监督微调 | `training/supervised_finetuning.py` | `scripts/run_sft.sh` |
| RM | Reward Model，奖励模型 | `training/reward_modeling.py` | `scripts/run_rm.sh` |
| PPO | RLHF 强化学习训练 | `training/ppo_training.py` | `scripts/run_ppo.sh` |
| DPO | Direct Preference Optimization，直接偏好优化 | `training/dpo_training.py` | `scripts/run_dpo.sh` |
| ORPO | 偏好优化方法 | `training/orpo_training.py` | `scripts/run_orpo.sh` |
| GRPO | 强化学习/偏好优化方法 | `training/grpo_training.py` | `scripts/run_grpo.sh` |

推荐的学习路线是：

1. 先读 SFT：理解普通指令微调如何工作。
2. 再读 DPO：理解 `chosen` 和 `rejected` 偏好数据如何进入训练。
3. 再看 RM 和 PPO：理解传统 RLHF 管线。
4. 最后看 ORPO、GRPO 和 Agent tool call：理解项目扩展出来的新训练能力。

## 3. 初学者优先阅读路径

### 第一步：读 `README.md`

阅读目标：

- 知道项目支持哪些训练阶段。
- 知道训练脚本和 shell 脚本之间的对应关系。
- 知道项目默认使用 Hugging Face Transformers、PEFT/LoRA、TRL、DeepSpeed 等生态。
- 知道医疗模型输出不能直接作为临床诊断或治疗建议。

读完后你应该能回答：

- PT、SFT、DPO、RM、PPO 分别解决什么问题？
- 如果我要跑 SFT，应该看哪个 Python 文件和哪个 shell 文件？
- 如果显存不够，项目提供了哪些方向，例如 LoRA、QLoRA、DeepSpeed？

### 第二步：读 `data/` 里的样例数据

优先打开这些文件：

- `data/sft/medical_sft_1K_format.jsonl`
- `data/sft/sharegpt_zh_1K_format.jsonl`
- `data/sft/glaive_toolcall_zh_demo.jsonl`
- `data/reward/dpo_zh_500.jsonl`
- `data/reward/toolcall_dpo_zh_demo.jsonl`
- `data/grpo/sample.jsonl`
- `data/pretrain/*.txt`

重点观察字段：

- SFT 数据使用 `conversations`，里面通常有 `human` 和 `gpt`。
- 多轮对话就是 `human`、`gpt` 多次交替。
- DPO/Reward 数据包含 `conversations`、`chosen`、`rejected`。
- Agent 工具调用数据可能包含 `tools`、`function_call`、`observation`。
- PT 数据可以是普通文本，每行一个样本。

读完后你应该能回答：

- SFT 和 DPO 数据格式有什么不同？
- 多轮对话在 jsonl 中如何表示？
- 工具调用样本比普通对话样本多了哪些字段？

### 第三步：读 `scripts/run_sft.sh`

这个文件是理解训练参数的入口。建议把它当作一份参数清单来看，而不是立刻运行。

重点看这些参数：

- `--model_name_or_path`：基础模型路径或 Hugging Face 模型名。
- `--train_file_dir`、`--validation_file_dir`：训练和验证数据目录。
- `--use_peft`：是否使用 PEFT/LoRA。
- `--max_train_samples`、`--max_eval_samples`：小样本调试用。
- `--model_max_length`：训练上下文长度。
- `--per_device_train_batch_size`、`--gradient_accumulation_steps`：实际 batch 相关。
- `--output_dir`：训练结果保存目录。
- `--target_modules`、`--lora_rank`、`--lora_alpha`、`--lora_dropout`：LoRA 参数。
- `--tool_format`：Agent 工具调用格式。

读完后你应该能回答：

- shell 脚本最终调用了哪个 Python 训练入口？
- 数据目录从哪个参数传进去？
- 小样本训练由哪些参数控制？
- LoRA 的输出会保存到哪里？

### 第四步：读 `training/supervised_finetuning.py`

这是初学者最应该精读的核心文件。建议按函数和代码块顺序阅读：

1. `ModelArguments`、`DataArguments`、`ScriptArguments`
2. `main()`
3. tokenizer 加载与 special token 处理
4. 数据集加载逻辑
5. `preprocess_function`
6. 模型加载逻辑
7. LoRA/QLoRA 配置逻辑
8. `Trainer` 创建、训练、评估、保存

读源码时重点追踪一条数据流：

```text
jsonl 文件
  -> datasets.load_dataset
  -> preprocess_function
  -> tokenizer
  -> input_ids / attention_mask / labels
  -> Trainer
  -> model.forward
  -> loss
  -> output_dir
```

读完后你应该能回答：

- 命令行参数是如何被 `HfArgumentParser` 解析的？
- `train_file_dir` 下的 jsonl 文件是如何被加载的？
- `conversations` 如何变成模型输入？
- 哪些 token 会参与 loss，哪些 token 会被忽略？
- LoRA 是在哪里创建并挂到模型上的？
- tokenizer 和模型权重最后保存在哪里？

### 第五步：读 `training/template.py`

这个文件定义对话模板。不同模型往往需要不同的 prompt 格式，例如 Vicuna、Alpaca、Baichuan、ChatGLM、Qwen 等。

重点看：

- `Conversation` 数据结构。
- `get_prompt()` 和 `get_dialog()`。
- `register_conv_template()`。
- 不同模板的 `system_prompt`、`prompt`、`sep`。

读完后你应该能回答：

- 为什么同一条对话数据给不同模型训练时，prompt 格式可能不同？
- `template_name` 参数会影响什么？
- `sep` 和 `stop_str` 在训练/推理中有什么作用？

### 第六步：读 `training/tool_utils.py`

这个文件处理 Agent 工具调用格式。普通 SFT 只训练问答，工具调用训练还要让模型学会输出函数调用。

重点看：

- `FunctionCall`
- `ToolUtils`
- `DefaultToolUtils`
- `GLM4ToolUtils`
- `Llama3ToolUtils`
- `QwenToolUtils`
- `get_tool_utils`

读完后你应该能回答：

- `tools` 字段如何被格式化进 prompt？
- `function_call` 如何变成模型要学习的 assistant 输出？
- 不同模型的工具调用格式为什么不一样？

### 第七步：读 `demo/inference.py` 和 `demo/gradio_demo.py`

训练代码读完后，需要看模型如何被加载回来。

重点看：

- `AutoTokenizer.from_pretrained`
- `AutoModelForCausalLM.from_pretrained`
- `PeftModel.from_pretrained`
- `model.generate`
- `tokenizer.apply_chat_template`
- 流式输出和批量输出的差异

读完后你应该能回答：

- 只加载 base model 和加载 base model + LoRA 有什么区别？
- 推理时 prompt 是如何构造的？
- `max_new_tokens`、`temperature`、`repetition_penalty` 会影响什么？

## 4. 七天学习任务清单

### 第 1 天：理解项目目录和 README

任务：

- 阅读 `README.md` 的 Introduction、Features、Project Structure、Training Pipeline。
- 画出你自己的项目结构图。
- 记录所有训练阶段的 Python 文件和 shell 文件对应关系。

验收：

- 能用自己的话解释 MedicalGPT 是训练项目，不只是聊天 demo。
- 能说出为什么 SFT 是初学者最好的入口。

### 第 2 天：理解数据格式和 SFT 脚本

任务：

- 阅读 `data/sft/*.jsonl` 前几行。
- 阅读 `data/reward/*.jsonl` 前几行。
- 阅读 `scripts/run_sft.sh`。
- 开始阅读 `training/supervised_finetuning.py` 的参数定义。

验收：

- 能区分 SFT 数据和 DPO 数据。
- 能解释 `max_train_samples` 和 `max_eval_samples` 为什么适合调试。

### 第 3 天：跑小样本 SFT

任务：

- 在有 GPU 和依赖环境的情况下，优先使用小模型和小样本参数。
- 保留 `--max_train_samples 1000` 或更小的值。
- 保留 `--max_eval_samples 10` 或更小的值。
- 观察 `output_dir` 下生成了哪些文件。

建议：

```bash
bash scripts/run_sft.sh
```

如果显存不足，不要直接改成全量训练。先降低模型规模、batch size、上下文长度，或改用 4bit/QLoRA 方案。

验收：

- 能说明训练日志里 loss、eval loss、保存 checkpoint 的含义。
- 能说明 LoRA adapter 和完整模型权重的区别。

### 第 4 天：跑推理

任务：

- 阅读并运行 `demo/inference.py`。
- 如果训练出了 LoRA adapter，用 `--lora_model` 加载。
- 如果只有 base model，则不传 `--lora_model`。

示例：

```bash
python demo/inference.py \
  --base_model Qwen/Qwen3.5-2B \
  --lora_model outputs-sft-qwen-v1 \
  --interactive
```

验收：

- 能解释 base model、tokenizer、LoRA adapter 三者的关系。
- 能说出推理脚本和训练脚本都依赖哪些 Hugging Face API。

### 第 5 天：理解 DPO 和偏好数据

任务：

- 阅读 `data/reward/dpo_zh_500.jsonl`。
- 阅读 `scripts/run_dpo.sh`。
- 阅读 `training/dpo_training.py` 的参数定义、数据处理和 `DPOTrainer` 创建逻辑。

验收：

- 能解释 `chosen` 和 `rejected` 的含义。
- 能解释 DPO 为什么不需要先单独训练 reward model。
- 能说出 DPO 通常接在 SFT 之后。

### 第 6 天：阅读工具和部署相关代码

任务：

- 阅读 `tools/merge_peft_adapter.py`，理解 LoRA 合并。
- 阅读 `tools/validate_jsonl.py`，理解数据检查。
- 阅读 `demo/gradio_demo.py` 和 `demo/fastapi_server_demo.py`。

验收：

- 能说明为什么训练后可能需要合并 LoRA。
- 能说明 Gradio demo 和 FastAPI demo 面向的使用场景不同。

### 第 7 天：整理笔记并改一个小功能

建议选择一个小任务：

- 给 `scripts/run_sft.sh` 增加一份自己的注释版副本。
- 写一个脚本检查 SFT jsonl 是否包含空回答。
- 给 `demo/inference.py` 增加一个你自己的测试问题文件。
- 整理一页学习笔记，画出 SFT 的数据流。

验收：

- 能独立讲清楚 SFT 从数据到模型保存的全过程。
- 能定位一个训练参数最终在哪段代码里被使用。

## 5. 读源码时必须回答的问题

阅读 `training/supervised_finetuning.py` 时，建议带着下面的问题逐段查找答案：

| 问题 | 你应该去哪里找 |
| --- | --- |
| 参数从哪里进入？ | `HfArgumentParser` 和三个 dataclass |
| 数据从哪里进入？ | `load_dataset` 和 `train_file_dir` |
| jsonl 的 `conversations` 如何处理？ | `preprocess_function` |
| prompt 如何拼接？ | `training/template.py` 或 tokenizer 的 chat template |
| 哪些 token 参与 loss？ | `labels`、`IGNORE_INDEX`、`train_on_inputs` 相关逻辑 |
| LoRA 如何挂载？ | `LoraConfig`、`get_peft_model` |
| QLoRA 如何启用？ | `load_in_4bit`、`qlora`、`BitsAndBytesConfig` |
| 训练由谁执行？ | `Trainer.train()` |
| 模型如何保存？ | `save_model`、`SavePeftModelTrainer`、`save_model_zero3` |
| 推理如何加载 LoRA？ | `demo/inference.py` 中的 `PeftModel.from_pretrained` |

不要只看函数名，要用“参数 -> 数据 -> prompt -> token -> label -> loss -> 保存”的链路把代码串起来。

## 6. 最小闭环实践路线

初学者不要从全量医疗数据开始。建议用下面的最小闭环：

1. 看 `data/sft/medical_sft_1K_format.jsonl` 的格式。
2. 看 `scripts/run_sft.sh` 如何指定数据和模型。
3. 用小样本参数跑 SFT。
4. 查看 `outputs-sft-qwen-v1/` 是否生成 adapter、tokenizer、日志等文件。
5. 用 `demo/inference.py` 加载 base model 和 LoRA adapter。
6. 手动问几个医疗相关和通用问题，比较训练前后的输出差异。

推荐小样本参数：

```bash
--max_train_samples 100
--max_eval_samples 10
--num_train_epochs 1
--model_max_length 512
--per_device_train_batch_size 1
```

这些参数适合验证流程是否能跑通，不代表最终训练质量。真正训练时需要更大的数据、更稳定的超参数、更严格的评估和更充足的硬件。

## 7. 常见学习误区

### 一开始就读所有训练方法

不建议。PT、SFT、RM、PPO、DPO、ORPO、GRPO 都读一遍很容易混乱。先精读 SFT，再扩展到 DPO 和 RLHF。

### 一开始就跑全量医疗数据

不建议。全量训练会消耗大量时间和显存，也不利于调试。先用 `max_train_samples` 和 `max_eval_samples` 跑小样本，确认代码链路和数据格式没有问题。

### 只看 shell 脚本，不看 Python 代码

shell 脚本告诉你怎么启动，Python 代码告诉你为什么这样启动。学习项目时必须从 shell 参数追到 Python 变量，再追到模型和数据处理逻辑。

### 只看模型，不看数据

大模型训练项目里，数据格式和 prompt 模板非常关键。很多训练错误不是模型问题，而是数据字段、模板、label mask 或 tokenizer 设置不正确。

### 忽略医疗场景风险

本项目用于医疗领域模型训练研究和工程实践。模型输出不能直接作为临床诊断、治疗方案或用药依据。涉及真实医疗场景时，必须由有资质的专业人员审核，并遵守数据隐私、伦理和合规要求。

## 8. 进阶阅读路线

完成 SFT 最小闭环后，可以按兴趣继续深入：

| 方向 | 阅读文件 | 学习目标 |
| --- | --- | --- |
| 偏好优化 | `training/dpo_training.py`、`scripts/run_dpo.sh` | 理解 chosen/rejected 如何训练模型偏好 |
| RLHF | `training/reward_modeling.py`、`training/ppo_training.py` | 理解奖励模型和 PPO 训练流程 |
| ORPO/GRPO | `training/orpo_training.py`、`training/grpo_training.py` | 理解更新的偏好优化和强化学习方法 |
| Agent 工具调用 | `training/tool_utils.py`、`data/sft/glaive_toolcall_zh_demo.jsonl` | 理解工具 schema、函数调用和 observation |
| RAG 应用 | `demo/chatpdf.py`、`data/rag/medical_corpus.txt` | 理解基于知识库文件的问答 |
| 模型后处理 | `tools/merge_peft_adapter.py`、`tools/model_quant.py` | 理解 LoRA 合并和量化 |
| 词表扩充 | `tools/build_domain_tokenizer.py`、`docs/extend_vocab.md` | 理解领域词表扩展流程 |

## 9. 建议的学习产出

学习这个项目时，建议最终产出三份材料：

1. 一张 SFT 数据流图：从 jsonl 到 loss 到 output_dir。
2. 一份参数说明表：整理 `run_sft.sh` 中每个关键参数的作用。
3. 一个小实验记录：记录模型、数据量、训练参数、loss 曲线、推理样例和观察结论。

如果能完成这三份材料，说明你已经不是只会运行脚本，而是开始理解这个项目的工程结构和训练逻辑。

## 10. 最后提醒

这个项目的正确阅读方式是“先主线，后分支；先小样本，后全量；先理解数据，再调整模型”。学生第一次阅读时，只需要把 SFT 这条线读透，就已经能理解大多数大模型微调项目的基本结构。

当你能独立回答下面这句话，就可以进入更深入的训练方法：

> 一条 `conversations` 样本如何经过模板拼接和 tokenizer，变成模型训练时的 `input_ids`、`attention_mask` 和 `labels`，并最终通过 LoRA 更新少量参数保存到 `output_dir`。
