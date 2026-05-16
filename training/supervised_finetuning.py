# -*- coding: utf-8 -*-
# Copyright 2023 XuMing(xuming624@qq.com) and The HuggingFace Inc. team. All rights reserved.

"""
SFT (Supervised Fine-Tuning) 监督微调 —— 整个项目的核心训练脚本。

============================================================================
训练流程总览：
============================================================================

  原始数据 (jsonl)                  训练好的模型
      │                                ▲
      ▼                                │
  ┌──────────────┐    ┌───────────┐    │
  │ 1.加载数据集  │ → │ 2.预处理   │    │
  │   (jsonl)    │   │ (tokenize) │    │
  └──────────────┘    └─────┬─────┘    │
                            │          │
                     ┌──────▼──────┐    │
                     │ 3.加载模型   │    │
                     │ (Qwen3.5-2B)│    │
                     └──────┬──────┘    │
                            │          │
                     ┌──────▼──────┐    │
                     │ 4.LoRA 配置 │    │
                     │ (低秩适配)   │    │
                     └──────┬──────┘    │
                            │          │
                     ┌──────▼──────┐    │
                     │ 5.Trainer   │ ──┘
                     │   训练循环   │
                     └─────────────┘

============================================================================
关键技术点：
  - LoRA: 只训练少量低秩矩阵，冻结原模型权重，大幅降低显存
  - DataCollatorForSeq2Seq: 动态填充到批次内最长序列，负位置填充
  - IGNORE_INDEX: labels 中 query 部分标记为 -100，loss 计算时跳过
  - FlashAttention-2: 加速注意力计算，显存效率更高
  - QLoRA: 4bit 量化 + LoRA，极限降低显存
============================================================================
"""

import math
import os
import sys
import json
from dataclasses import dataclass, field
from glob import glob
from types import MethodType
from typing import Literal, Optional, Tuple

import torch
from datasets import load_dataset
from loguru import logger
from peft import LoraConfig, TaskType, get_peft_model, PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    Seq2SeqTrainingArguments,
    set_seed,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
)
from transformers.trainer import TRAINING_ARGS_NAME
from transformers.trainer_pt_utils import LabelSmoother
from transformers.utils.versions import require_version

from transformers.integrations import is_deepspeed_zero3_enabled
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.tool_utils import get_tool_utils, FunctionCall
from training.template import get_conv_template
try:
    import flash_attn  # noqa: F401

    is_flash_attn_2_available = True
except ImportError:
    is_flash_attn_2_available = False



# ============================================================================
# 参数配置（三大 dataclass）：使用 HfArgumentParser 从命令行解析
#   ModelArguments  — 模型相关（路径、量化、精度）
#   DataArguments   — 数据相关（jsonl 路径、采样数）
#   ScriptArguments — 训练策略（LoRA 参数、模板名、上下文长度）
#   Seq2SeqTrainingArguments — HuggingFace 内置，训练超参（lr、epochs、batch_size等）
# ============================================================================

@dataclass
class ModelArguments:
    """
    模型加载相关参数。
    关键字段：
      model_name_or_path: 预训练模型路径，本项目用 /root/autodl-tmp/models/Qwen3.5-2B
      load_in_4bit/8bit: 量化加载，4bit≈省75%显存
      torch_dtype: 模型权重的数据类型，默认 float16
      flash_attn: 开启 FlashAttention-2 加速（RTX 4090 支持）
      rope_scaling: 扩展上下文长度的 RoPE 缩放策略
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "预训练模型路径，如 /root/autodl-tmp/models/Qwen3.5-2B"},
    )
    load_in_8bit: bool = field(default=False, metadata={"help": "8bit 量化加载模型"})
    load_in_4bit: bool = field(default=False, metadata={"help": "4bit 量化加载模型"})
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "分词器路径，默认与 model_name_or_path 相同"},
    )
    cache_dir: Optional[str] = field(default=None, metadata={"help": "模型缓存目录"})
    model_revision: Optional[str] = field(default="main", metadata={"help": "模型版本"})
    hf_hub_token: Optional[str] = field(default=None, metadata={"help": "HuggingFace Hub 认证 token"})
    use_fast_tokenizer: bool = field(default=False, metadata={"help": "是否使用 fast tokenizer"})
    torch_dtype: Optional[str] = field(
        default="float16",
        metadata={"help": "模型数据类型: auto/bfloat16/float16/float32", "choices": ["auto", "bfloat16", "float16", "float32"]},
    )
    device_map: Optional[str] = field(default="auto", metadata={"help": "设备映射策略，auto 自动分配到多 GPU"})
    trust_remote_code: bool = field(default=True, metadata={"help": "信任远程仓库中的自定义代码"})
    rope_scaling: Optional[Literal["linear", "dynamic"]] = field(default=None, metadata={"help": "RoPE 位置编码缩放"})
    flash_attn: Optional[bool] = field(default=False, metadata={"help": "启用 FlashAttention-2"})
    shift_attn: Optional[bool] = field(default=False, metadata={"help": "启用 LongLoRA 的稀疏注意力"})
    neft_alpha: Optional[float] = field(default=0, metadata={"help": "NEFTune 噪声正则化强度，如 5"})

    def __post_init__(self):
        if self.model_name_or_path is None:
            raise ValueError("You must specify a valid model_name_or_path to run training.")


@dataclass
class DataArguments:
    """
    数据加载相关参数。
    支持两种数据来源：
      1. HuggingFace Datasets Hub（dataset_name）
      2. 本地 jsonl 文件（train_file_dir / validation_file_dir）← 本项目用这个
    """

    dataset_name: Optional[str] = field(default=None, metadata={"help": "HuggingFace 数据集名称"})
    dataset_config_name: Optional[str] = field(default=None, metadata={"help": "数据集配置名"})
    train_file_dir: Optional[str] = field(default=None, metadata={"help": "训练数据 jsonl 文件夹路径"})
    validation_file_dir: Optional[str] = field(default=None, metadata={"help": "验证数据 jsonl 文件夹路径"})
    max_train_samples: Optional[int] = field(default=None, metadata={"help": "最多用多少训练样本"})
    max_eval_samples: Optional[int] = field(default=None, metadata={"help": "最多用多少验证样本"})
    ignore_pad_token_for_loss: bool = field(default=True, metadata={"help": "loss 计算时忽略 pad token"})
    overwrite_cache: bool = field(default=False, metadata={"help": "是否覆盖缓存的 tokenized 数据"})
    validation_split_percentage: Optional[int] = field(default=1, metadata={"help": "从训练集切出多少%做验证集"})
    preprocessing_num_workers: Optional[int] = field(default=None, metadata={"help": "预处理并行进程数"})

    def __post_init__(self):
        if self.max_train_samples is not None and 0 < self.max_train_samples <= 1000:
            logger.warning("You may set max_train_samples = -1 to run all samples in production.")


@dataclass
class ScriptArguments:
    """
    训练策略相关参数。
    关键字段：
      use_peft: 是否用 LoRA（本项目默认开启）
      lora_rank: LoRA 秩 r，决定低秩矩阵的大小。r 越大 → 可训参数越多 → 表达能力更强
      lora_alpha: LoRA 缩放因子，实际学习率 = lr * (alpha / r)
      target_modules: 对哪些层加 LoRA，"all" = 所有线性层
      template_name: 对话模板名，如 "qwen3_5"，对应 template.py 中的注册名
      train_on_inputs: 是否连用户输入一起算 loss（默认 False，只算回复部分）
      model_max_length: 最大上下文长度，超过的序列会截断
    """

    use_peft: bool = field(default=True, metadata={"help": "是否使用 LoRA 参数高效微调"})
    train_on_inputs: bool = field(default=False, metadata={"help": "是否在用户输入上也计算 loss（通常关闭）"})
    target_modules: Optional[str] = field(default="all", metadata={"help": "LoRA 目标模块，all=所有线性层"})
    lora_rank: Optional[int] = field(default=8, metadata={"help": "LoRA 秩 r"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "LoRA dropout 率"})
    lora_alpha: Optional[float] = field(default=32.0, metadata={"help": "LoRA 缩放因子"})
    modules_to_save: Optional[str] = field(default=None, metadata={"help": "额外需要完整保存的模块"})
    peft_path: Optional[str] = field(default=None, metadata={"help": "已有 LoRA 权重路径（用于继续训练）"})
    qlora: bool = field(default=False, metadata={"help": "是否使用 QLoRA（4bit 量化 + LoRA）"})
    model_max_length: int = field(default=512, metadata={"help": "最大上下文长度"})
    template_name: Optional[str] = field(default=None, metadata={"help": "对话模板名，如 qwen3_5"})
    tool_format: Optional[str] = field(default=None, metadata={"help": "工具调用格式（Agent 训练用）"})

    def __post_init__(self):
        if self.model_max_length < 60:
            raise ValueError("You must specify a valid model_max_length >= 60 to run training")


class SavePeftModelTrainer(Trainer):
    """
    继承 HuggingFace Trainer，覆盖 save_model 方法，
    使其正确保存 LoRA adapter 权重而非完整模型。
    LoRA 只保存 adapter_config.json + adapter_model.safetensors
    """

    def save_model(self, output_dir=None, _internal_call=False):
        """保存 LoRA adapter 权重"""
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        self.model.save_pretrained(output_dir)


def save_model(model, tokenizer, args):
    """Save the model and the tokenizer."""
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Take care of distributed/parallel training
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def save_model_zero3(model, tokenizer, args, trainer):
    """Save the model for deepspeed zero3.
    refer https://github.com/lm-sys/FastChat/blob/main/fastchat/train/train_lora.py#L209
    """
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    state_dict_zero3 = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(args.output_dir, state_dict=state_dict_zero3)
    tokenizer.save_pretrained(output_dir)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def find_all_linear_names(peft_model, int4=False, int8=False):
    """Find all linear layer names in the model. reference from qlora paper."""
    cls = torch.nn.Linear
    if int4 or int8:
        import bitsandbytes as bnb
        if int4:
            cls = bnb.nn.Linear4bit
        elif int8:
            cls = bnb.nn.Linear8bitLt
    lora_module_names = set()
    for name, module in peft_model.named_modules():
        if isinstance(module, cls):
            # last layer is not add to lora_module_names
            if 'lm_head' in name:
                continue
            if 'output_layer' in name:
                continue
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return sorted(lora_module_names)


def check_and_optimize_memory():
    """检查并优化GPU内存使用"""
    if not torch.cuda.is_available():
        return

    logger.info("🔍 检查GPU内存状态...")

    # 清理缓存
    torch.cuda.empty_cache()

    # 检查每个GPU的内存状态
    num_gpus = torch.cuda.device_count()
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        total_memory = props.total_memory / 1024 ** 3
        allocated = torch.cuda.memory_allocated(i) / 1024 ** 3
        cached = torch.cuda.memory_reserved(i) / 1024 ** 3
        free = total_memory - allocated - cached

        logger.info(f"GPU {i} ({props.name}):")
        logger.info(f"  总内存: {total_memory:.1f}GB")
        logger.info(f"  已分配: {allocated:.1f}GB")
        logger.info(f"  已缓存: {cached:.1f}GB")
        logger.info(f"  可用: {free:.1f}GB")

    # 设置内存优化选项
    if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        torch.backends.cuda.enable_flash_sdp(True)
        logger.info("✅ 启用Flash Attention优化")

    # 启用内存高效的注意力机制
    if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        logger.info("✅ 启用内存高效注意力机制")


def main():
    """
    SFT 训练主函数。
    执行顺序：解析参数 → 加载 tokenizer → 加载数据 → 预处理 → 加载模型 → LoRA → Trainer → 训练 → 评估
    """
    parser = HfArgumentParser((ModelArguments, DataArguments, Seq2SeqTrainingArguments, ScriptArguments))

    # 使用 parse_args_into_dataclasses 时忽略未知参数
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # 如果我们传递了一个 JSON 文件，让我们用它来配置参数
        model_args, data_args, training_args, script_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        # 否则解析命令行参数，忽略未知参数
        model_args, data_args, training_args, script_args = parser.parse_args_into_dataclasses(look_for_args_file=False)

    # 确保 DeepSpeed 配置正确加载
    if training_args.deepspeed is not None:
        training_args.distributed_state.deepspeed_plugin = None

    # The Trainer will handle distributed training setup
    is_main_process = training_args.local_rank in [-1, 0]

    # Only log on main process
    if is_main_process:
        logger.info(f"Model args: {model_args}")
        logger.info(f"Data args: {data_args}")
        logger.info(f"Training args: {training_args}")
        logger.info(f"Script args: {script_args}")
        logger.info(
            f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
            + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
        )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # ========================================================================
    # 加载 Tokenizer，确保有 eos_token, bos_token, pad_token
    # ========================================================================
    # Load tokenizer
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "trust_remote_code": model_args.trust_remote_code,
    }
    tokenizer_name_or_path = model_args.tokenizer_name_or_path
    if not tokenizer_name_or_path:
        tokenizer_name_or_path = model_args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, **tokenizer_kwargs)
    prompt_template = None
    if script_args.template_name:
        prompt_template = get_conv_template(script_args.template_name)
    if tokenizer.eos_token_id is None:
        if prompt_template:
            tokenizer.eos_token = prompt_template.stop_str
        else:
            tokenizer.eos_token = "</s>"
        tokenizer.add_special_tokens({"eos_token": tokenizer.eos_token})
        logger.info(f"Add eos_token: {tokenizer.eos_token}, eos_token_id: {tokenizer.eos_token_id}")
    if tokenizer.bos_token_id is None:
        tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})
        tokenizer.bos_token_id = tokenizer.eos_token_id
        logger.info(f"Add bos_token: {tokenizer.bos_token}, bos_token_id: {tokenizer.bos_token_id}")
    if tokenizer.pad_token_id is None:
        if tokenizer.unk_token_id is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Add pad_token: {tokenizer.pad_token}, pad_token_id: {tokenizer.pad_token_id}")
    logger.debug(f"Tokenizer: {tokenizer}")

    IGNORE_INDEX = LabelSmoother.ignore_index if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id

    # Get datasets
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
        )
        if "validation" not in raw_datasets.keys():
            shuffled_train_dataset = raw_datasets["train"].shuffle(seed=42)
            # Split the shuffled train dataset into training and validation sets
            split = shuffled_train_dataset.train_test_split(
                test_size=data_args.validation_split_percentage / 100,
                seed=42
            )
            # Assign the split datasets back to raw_datasets
            raw_datasets["train"] = split["train"]
            raw_datasets["validation"] = split["test"]
    else:
        # 从本地 jsonl 文件加载数据（本项目使用此方式）
        data_files = {}
        if data_args.train_file_dir is not None and os.path.exists(data_args.train_file_dir):
            train_data_files = glob(f'{data_args.train_file_dir}/**/*.jsonl', recursive=True)
            logger.info(f"train files: {train_data_files}")
            data_files["train"] = train_data_files
        if data_args.validation_file_dir is not None and os.path.exists(data_args.validation_file_dir):
            eval_data_files = glob(f'{data_args.validation_file_dir}/**/*.jsonl', recursive=True)
            logger.info(f"eval files: {eval_data_files}")
            data_files["validation"] = eval_data_files
        raw_datasets = load_dataset(
            'json',
            data_files=data_files,
            cache_dir=model_args.cache_dir,
        )
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            shuffled_train_dataset = raw_datasets["train"].shuffle(seed=42)
            split = shuffled_train_dataset.train_test_split(
                test_size=float(data_args.validation_split_percentage / 100),
                seed=42
            )
            raw_datasets["train"] = split["train"]
            raw_datasets["validation"] = split["test"]
    logger.info(f"Raw datasets: {raw_datasets}")

    # Preprocessing the datasets
    max_length = script_args.model_max_length

    # ========================================================================
    # 核心函数：preprocess_function
    # 将原始 jsonl 数据（对话格式）转换为模型可用的 input_ids 和 labels
    #
    # 数据流:
    #   jsonl 一行:
    #     {"conversations": [{"from":"human","value":"..."}, {"from":"gpt","value":"..."}]}
    #       ↓ get_dialog()  →  ["<|im_start|>user\n...<|im_end|>\n...", "回复"]
    #       ↓ tokenize     →  input_ids = [1, 2, 3, ...]
    #       ↓ label masking →  labels = [-100, -100, ..., 4, 5, 6]  (query 部分标记为 -100)
    #
    # labels 中 -100 是 IGNORE_INDEX，loss 计算时会跳过，这样模型只学习生成回复部分
    # ========================================================================
    def preprocess_function(examples):
        input_ids_list = []
        attention_mask_list = []
        targets_list = []
        roles = ["human", "gpt"]

        def get_dialog(examples):
            system_prompts = examples.get("system_prompt", "")
            for i, source in enumerate(examples['conversations']):
                system_prompt = ""
                tools_text = ""
                if "tools" in examples and examples["tools"][i]:
                    tools_json = examples["tools"][i]
                    if isinstance(tools_json, str):
                        tools_parsed = json.loads(tools_json)
                        if tools_parsed and script_args.tool_format:
                            tu = get_tool_utils(script_args.tool_format)
                            tools_text = tu.tool_formatter(tools_parsed)
                
                messages = []
                for sentence in source:
                    role = sentence.get("from", "")
                    value = sentence.get("value", "")
                    
                    if role == "system":
                        system_prompt = value
                        continue
                    
                    if role in ["human", "user", "observation"]:
                        if role == "observation":
                            if script_args.tool_format == "qwen":
                                value = f"<tool_response>\n{value}\n</tool_response>"
                            elif script_args.tool_format == "glm4":
                                value = f"<|observation|>\n{value}"
                            elif script_args.tool_format == "mistral":
                                value = f"[TOOL_RESULTS] {{\"content\": {value}}}[/TOOL_RESULTS]"
                            else:
                                value = f"Observation: {value}"
                        messages.append({"role": "user", "content": value})
                    elif role in ["gpt", "assistant", "function_call"]:
                        if role == "function_call":
                            fc_dict = json.loads(value)
                            if "name" in fc_dict and "arguments" in fc_dict:
                                if script_args.tool_format:
                                    tu = get_tool_utils(script_args.tool_format)
                                    value = tu.function_formatter([FunctionCall(fc_dict["name"], json.dumps(fc_dict["arguments"], ensure_ascii=False))])
                                else:
                                    value = f"Action: {fc_dict['name']}\nAction Input: {json.dumps(fc_dict['arguments'], ensure_ascii=False)}"
                        messages.append({"role": "assistant", "content": value})

                if tools_text:
                    system_prompt = system_prompt + ("\n\n" if system_prompt else "") + tools_text

                history_messages = []
                temp_history = []
                for msg in messages:
                    if not temp_history and msg["role"] == "user":
                        temp_history.append(msg["content"])
                    elif len(temp_history) == 1 and msg["role"] == "assistant":
                        temp_history.append(msg["content"])
                        history_messages.append(temp_history)
                        temp_history = []
                    elif msg["role"] == "user" and len(temp_history) == 1:
                        temp_history[0] += "\n" + msg["content"]
                    elif msg["role"] == "assistant" and len(temp_history) == 0:
                        pass
                    elif msg["role"] == "assistant" and len(temp_history) == 2:
                        history_messages[-1][1] += "\n" + msg["content"]
                        
                if not history_messages:
                    continue

                if not system_prompt:
                    system_prompt = system_prompts[i] if system_prompts else ""
                # 如果有指定模板名（如 qwen3_5），用模板的 get_dialog 格式化
                # 否则用 tokenizer 内置的 chat_template
                if prompt_template:
                    yield prompt_template.get_dialog(history_messages, system_prompt=system_prompt)
                else:
                    convs = []
                    accumulated = []
                    if system_prompt:
                        accumulated.append({"role": "system", "content": system_prompt})
                    prev_text = ""
                    for uq, br in history_messages:
                        accumulated.append({"role": "user", "content": uq})
                        cur_text = tokenizer.apply_chat_template(
                            accumulated, tokenize=False, add_generation_prompt=True
                        )
                        convs.append(cur_text[len(prev_text):])  # 截出本轮新增的 query 部分
                        convs.append(br)                          # response 原样保留
                        accumulated.append({"role": "assistant", "content": br})
                        prev_text = tokenizer.apply_chat_template(
                            accumulated, tokenize=False, add_generation_prompt=False
                        )
                    yield convs

        for dialog in get_dialog(examples):
            input_ids, labels = [], []

            for i in range(len(dialog) // 2):
                # source_ids = tokenize 后的 query（用户输入 + 模板格式）
                # target_ids = tokenize 后的 response（助手回复）
                source_ids = tokenizer.encode(text=dialog[2 * i], add_special_tokens=(i == 0))
                target_ids = tokenizer.encode(text=dialog[2 * i + 1], add_special_tokens=False)

                # 按长度比例分配截断额度，防止长 query 吃掉全部 context
                total_len = len(source_ids) + len(target_ids)
                max_source_len = int(max_length * (len(source_ids) / total_len))
                max_target_len = int(max_length * (len(target_ids) / total_len))

                if len(source_ids) > max_source_len:
                    source_ids = source_ids[:max_source_len]
                if len(target_ids) > max_target_len - 1:  # 给 eos token 留位置
                    target_ids = target_ids[:max_target_len - 1]
                if len(source_ids) > 0 and source_ids[0] == tokenizer.eos_token_id:
                    source_ids = source_ids[1:]
                if len(target_ids) > 0 and target_ids[-1] == tokenizer.eos_token_id:
                    target_ids = target_ids[:-1]
                if len(input_ids) + len(source_ids) + len(target_ids) + 1 > max_length:
                    break  # 超出 context 长度，丢弃后续轮次

                # 拼接：source + target + eos
                input_ids += source_ids + target_ids + [tokenizer.eos_token_id]
                if script_args.train_on_inputs:
                    # 所有 token 都参与 loss 计算（罕见）
                    labels += source_ids + target_ids + [tokenizer.eos_token_id]
                else:
                    # 关键：query 部分的 labels 设为 IGNORE_INDEX(-100)
                    # 这样 loss 只在 target(reply) + eos 上计算
                    labels += [IGNORE_INDEX] * len(source_ids) + target_ids + [tokenizer.eos_token_id]

            input_ids_list.append(input_ids)
            attention_mask_list.append([1] * len(input_ids))
            targets_list.append(labels)

        return dict(
            input_ids=input_ids_list,
            attention_mask=attention_mask_list,
            labels=targets_list,
        )

    def filter_empty_labels(example):
        """Remove empty labels dataset."""
        return not all(label == IGNORE_INDEX for label in example["labels"])

    train_dataset = None
    max_train_samples = 0
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets['train'].shuffle(seed=42)
        max_train_samples = len(train_dataset)
        if data_args.max_train_samples is not None and data_args.max_train_samples > 0:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

        if is_main_process:
            logger.debug(f"Example train_dataset[0]: {train_dataset[0]}")

        with training_args.main_process_first(desc="Train dataset tokenization"):
            tokenized_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=train_dataset.column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset" if is_main_process else None,
            )
            train_dataset = tokenized_dataset.filter(
                filter_empty_labels,
                num_proc=data_args.preprocessing_num_workers
            )

            if is_main_process:
                logger.debug(f"Num train_samples: {len(train_dataset)}")
                logger.debug("Tokenized training example:")
                logger.debug(f"Decode input_ids[0]:\n{tokenizer.decode(train_dataset[0]['input_ids'])}")
                replaced_labels = [label if label != IGNORE_INDEX else tokenizer.pad_token_id
                                   for label in list(train_dataset[0]['labels'])]
                logger.debug(f"Decode labels[0]:\n{tokenizer.decode(replaced_labels)}")

    eval_dataset = None
    max_eval_samples = 0
    if training_args.do_eval:
        with training_args.main_process_first(desc="Eval dataset tokenization"):
            if "validation" not in raw_datasets:
                raise ValueError("--do_eval requires a validation dataset")
            eval_dataset = raw_datasets["validation"]
            max_eval_samples = len(eval_dataset)
            if data_args.max_eval_samples is not None and data_args.max_eval_samples > 0:
                max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
                eval_dataset = eval_dataset.select(range(max_eval_samples))
            eval_size = len(eval_dataset)
            logger.debug(f"Num eval_samples: {eval_size}")
            if eval_size > 500:
                logger.warning(f"Num eval_samples is large: {eval_size}, "
                               f"training slow, consider reduce it by `--max_eval_samples=50`")
            logger.debug(f"Example eval_dataset[0]: {eval_dataset[0]}")
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=eval_dataset.column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )
            eval_dataset = eval_dataset.filter(filter_empty_labels, num_proc=data_args.preprocessing_num_workers)
            logger.debug(f"Num eval_samples: {len(eval_dataset)}")
            logger.debug("Tokenized eval example:")
            logger.debug(tokenizer.decode(eval_dataset[0]['input_ids']))

    # ========================================================================
    # 加载模型
    # 流程：AutoConfig → 量化配置 → AutoModelForCausalLM.from_pretrained
    # 之后：LoRA 配置 → get_peft_model（冻结原权重，添加可训练低秩矩阵）
    # ========================================================================
    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        ddp = world_size != 1
        if ddp:
            model_args.device_map = None
        if model_args.device_map in ["None", "none", ""]:
            model_args.device_map = None
        if script_args.qlora and (len(training_args.fsdp) > 0 or is_deepspeed_zero3_enabled()):
            logger.warning("FSDP and DeepSpeed ZeRO-3 are both currently incompatible with QLoRA.")

        config_kwargs = {
            "trust_remote_code": model_args.trust_remote_code,
            "cache_dir": model_args.cache_dir,
            "revision": model_args.model_revision,
            "token": model_args.hf_hub_token,
        }
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)

        # Set RoPE scaling
        if model_args.rope_scaling is not None:
            if hasattr(config, "rope_scaling"):
                if model_args.rope_scaling == "dynamic":
                    logger.warning(
                        "Dynamic NTK may not work well with fine-tuning. "
                        "See: https://github.com/huggingface/transformers/pull/24653"
                    )
                current_max_length = getattr(config, "max_position_embeddings", None)
                if current_max_length and script_args.model_max_length > current_max_length:
                    scaling_factor = float(math.ceil(script_args.model_max_length / current_max_length))
                else:
                    logger.warning(f"The model_max_length({script_args.model_max_length}) is smaller than max "
                                   f"length({current_max_length}). Consider increase model_max_length.")
                    scaling_factor = 1.0

                setattr(config, "rope_scaling", {"type": model_args.rope_scaling, "factor": scaling_factor})
                logger.info("Using {} scaling strategy and setting scaling factor to {}".format(
                    model_args.rope_scaling, scaling_factor
                ))
            else:
                logger.warning("Current model does not support RoPE scaling.")

        # Set FlashAttention-2
        if model_args.flash_attn:
            if is_flash_attn_2_available:
                config_kwargs["use_flash_attention_2"] = True
                logger.info("Using FlashAttention-2 for faster training and inference.")
            else:
                logger.warning("FlashAttention-2 is not installed.")
        elif model_args.shift_attn and getattr(config, "model_type", None) == "llama":
            logger.warning("Using `--flash_attn` for faster training in large context length, enable if your GPU"
                           " is RTX3090, RTX4090, A100 or H100.")

        # Set shifted sparse attention (S^2-Attn)
        if model_args.shift_attn:
            if getattr(config, "model_type", None) == "llama":
                setattr(config, "group_size_ratio", 0.25)
                logger.info("Using shifted sparse attention with group_size_ratio=1/4.")
            else:
                logger.warning("Current model does not support shifted sparse attention.")

        load_in_4bit = model_args.load_in_4bit
        load_in_8bit = model_args.load_in_8bit
        quantization_config = None
        if load_in_4bit and load_in_8bit:
            raise ValueError("Error, load_in_4bit and load_in_8bit cannot be set at the same time")
        elif load_in_8bit or load_in_4bit:
            logger.info(f"Quantizing model, load_in_4bit: {load_in_4bit}, load_in_8bit: {load_in_8bit}")
            if is_deepspeed_zero3_enabled():
                raise ValueError("DeepSpeed ZeRO-3 is incompatible with quantization.")
            if load_in_8bit:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            elif load_in_4bit:
                if script_args.qlora:
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch_dtype,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                else:
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch_dtype,
                    )

        model_kwargs = {
            "config": config,
            "torch_dtype": torch_dtype,
            "trust_remote_code": model_args.trust_remote_code,
            "quantization_config": quantization_config,
            "low_cpu_mem_usage": True,  # 减少CPU内存使用
            "device_map": model_args.device_map,
        }

        # 设置device_map
        num_gpus = torch.cuda.device_count()
        if model_args.device_map == 'auto':
            if num_gpus > 1 and not ddp:
                # 大模型多GPU：使用auto进行张量并行
                model_kwargs["device_map"] = "auto"
                # 设置最大内存使用
                max_memory = {}
                for i in range(num_gpus):
                    # 为每个GPU预留一些内存给梯度和优化器
                    gpu_props = torch.cuda.get_device_properties(i)
                    total_mem = gpu_props.total_memory
                    # 预留20%内存给训练时的梯度、优化器状态等
                    usable_mem = int(total_mem * 0.8)
                    max_memory[i] = f"{usable_mem // (1024 ** 3)}GiB"

                model_kwargs["max_memory"] = max_memory

        logger.info(f"🔧 大模型训练配置:")
        logger.info(f"  model_kwargs: {model_kwargs}")

        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            **model_kwargs
        )

        logger.info("✅ 模型加载完成")

        # 显示模型分布信息
        logger.info("📊 模型分布情况:")
        if hasattr(model, 'hf_device_map') and model.hf_device_map:
            logger.info("🔧 使用HuggingFace设备映射:")
            for module_name, device in model.hf_device_map.items():
                logger.info(f"  {module_name}: {device}")

            # 统计每个GPU上的模块数量
            device_count = {}
            for device in model.hf_device_map.values():
                device_str = str(device)
                device_count[device_str] = device_count.get(device_str, 0) + 1

            logger.info("📈 设备使用统计:")
            for device, count in device_count.items():
                logger.info(f"  {device}: {count} 个模块")
        else:
            # 检查模型参数的设备分布
            device_params = {}
            total_params = 0
            for name, param in model.named_parameters():
                device = str(param.device)
                if device not in device_params:
                    device_params[device] = {'count': 0, 'size': 0}
                device_params[device]['count'] += 1
                device_params[device]['size'] += param.numel()
                total_params += param.numel()

            logger.info("📈 参数设备分布:")
            for device, info in device_params.items():
                param_size_gb = info['size'] * 4 / 1024 ** 3  # 假设float32
                percentage = info['size'] / total_params * 100
                logger.info(f"  {device}: {info['count']} 个参数组, {param_size_gb:.2f}GB ({percentage:.1f}%)")

        # 显示GPU内存使用情况
        if torch.cuda.is_available():
            logger.info("💾 GPU内存使用情况:")
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024 ** 3
                cached = torch.cuda.memory_reserved(i) / 1024 ** 3
                total = torch.cuda.get_device_properties(i).total_memory / 1024 ** 3
                logger.info(f"  GPU {i}: 已分配={allocated:.1f}GB, 缓存={cached:.1f}GB, 总计={total:.1f}GB")

        # Fix ChatGLM2 and ChatGLM3 and internlm2 LM head
        if getattr(config, "model_type", None) == "chatglm" or getattr(config, "model_type", None) == "internlm2":
            setattr(model, "lm_head", model.transformer.output_layer)
            setattr(model, "_keys_to_ignore_on_save", ["lm_head.weight"])

        # Set NEFTune trick for fine-tuning
        if model_args.neft_alpha > 0:
            input_embed = model.get_input_embeddings()
            if isinstance(input_embed, torch.nn.Embedding):
                def noisy_forward(self: torch.nn.Embedding, x: torch.Tensor) -> torch.Tensor:
                    embeddings = input_embed.__class__.forward(self, x)
                    dims = self.num_embeddings * self.embedding_dim
                    mag_norm = model_args.neft_alpha / (dims ** 0.5)
                    embeddings += torch.zeros_like(embeddings).uniform_(-mag_norm, mag_norm)
                    return embeddings

                input_embed.forward = MethodType(noisy_forward, input_embed)
                logger.info("Using noisy embedding with alpha={:.2f}".format(model_args.neft_alpha))
            else:
                logger.warning("Input embeddings are not normal nn.Embedding, cannot transform into noisy embedding.")

        # Patch Mixtral MOE model
        if getattr(config, "model_type", None) == "mixtral" and is_deepspeed_zero3_enabled():
            require_version("deepspeed>=0.13.0", "To fix: pip install deepspeed>=0.13.0")
            from deepspeed.utils import set_z3_leaf_modules  # type: ignore
            from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock  # type: ignore

            set_z3_leaf_modules(model, [MixtralSparseMoeBlock])

        # Patch DeepSeek-V3 MoE module
        if getattr(config, "model_type", None) == "deepseek_v3" and is_deepspeed_zero3_enabled():
            require_version("deepspeed>=0.13.0", "To fix: pip install deepspeed>=0.13.0")
            for layer in model.model.layers:
                if 'DeepseekV3MoE' in str(type(layer.mlp)):
                    layer.mlp._z3_leaf = True

        # Patch Qwen3 MoE module
        if getattr(config, "model_type", None) == "qwen3_moe" and is_deepspeed_zero3_enabled():
            require_version("deepspeed>=0.13.0", "To fix: pip install deepspeed>=0.13.0")
            from deepspeed.utils import set_z3_leaf_modules
            from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock
            set_z3_leaf_modules(model, [Qwen3MoeSparseMoeBlock])

        # Patch Qwen3.5 MoE module
        if getattr(config, "model_type", None) == "qwen3_5_moe" and is_deepspeed_zero3_enabled():
            require_version("deepspeed>=0.13.0", "To fix: pip install deepspeed>=0.13.0")
            from deepspeed.utils import set_z3_leaf_modules
            from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeSparseMoeBlock
            set_z3_leaf_modules(model, [Qwen3_5MoeSparseMoeBlock])
    else:
        raise ValueError(f"Error, model_name_or_path is None, SFT must be loaded from a pre-trained model")

    if script_args.use_peft:
        logger.info("Fine-tuning method: LoRA(PEFT)")

        # lm_head（输出层）用 fp32 精度，避免数值不稳定
        output_layer = getattr(model, "lm_head")
        if isinstance(output_layer, torch.nn.Linear) and output_layer.weight.dtype != torch.float32:
            def fp32_forward_post_hook(module: torch.nn.Module, args: Tuple[torch.Tensor], output: torch.Tensor):
                return output.to(torch.float32)

            output_layer.register_forward_hook(fp32_forward_post_hook)

        # 如果提供了已有 LoRA 路径（peft_path），从该路径加载继续训练
        # 否则创建全新的 LoRA 配置
        if script_args.peft_path is not None:
            logger.info(f"Peft from pre-trained model: {script_args.peft_path}")
            model = PeftModel.from_pretrained(model, script_args.peft_path, is_trainable=True)
        else:
            logger.info("Init new peft model")
            if load_in_8bit or load_in_4bit:
                # 量化模型需要特殊预处理，使其兼容 LoRA 训练
                model = prepare_model_for_kbit_training(model, training_args.gradient_checkpointing)
            target_modules = script_args.target_modules.split(',') if script_args.target_modules else None
            if target_modules and 'all' in target_modules:
                # "all" = 自动找到模型中的所有线性层作为 LoRA 目标
                target_modules = find_all_linear_names(model, int4=load_in_4bit, int8=load_in_8bit)
            modules_to_save = script_args.modules_to_save
            if modules_to_save is not None:
                modules_to_save = modules_to_save.split(',')
            logger.info(f"Peft target_modules: {target_modules}")
            logger.info(f"Peft lora_rank: {script_args.lora_rank}")
            # LoRA 配置：
            #   r=8:     低秩矩阵秩（越大可训参数越多）
            #   alpha=32: 缩放因子，实际学习率 ≈ lr * alpha / r
            #   dropout=0.05: LoRA 层的 dropout
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,      # 因果语言模型任务
                target_modules=target_modules,      # 哪些层添加 LoRA
                inference_mode=False,               # 训练模式
                r=script_args.lora_rank,           # LoRA 秩
                lora_alpha=script_args.lora_alpha, # 缩放因子
                lora_dropout=script_args.lora_dropout,
                modules_to_save=modules_to_save)
            model = get_peft_model(model, peft_config)  # 包装模型：冻结原权重 + 插入 LoRA 层
        for param in filter(lambda p: p.requires_grad, model.parameters()):
            param.data = param.data.to(torch.float32)
        model.print_trainable_parameters()
    else:
        logger.info("Fine-tuning method: Full parameters training")
        model = model.float()
        print_trainable_parameters(model)

    # ========================================================================
    # 初始化 Trainer
    # ========================================================================
    # Gradient Checkpointing: 用时间换空间，不存所有中间激活，反向时重算
    if training_args.gradient_checkpointing and getattr(model, "supports_gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False  # gradient checkpointing 时不能用 KV cache
        logger.info("Gradient checkpointing enabled.")
    else:
        model.config.use_cache = True
        logger.info("Gradient checkpointing disabled.")
    model.enable_input_require_grads()
    if not ddp and torch.cuda.device_count() > 1:
        # 多 GPU 但不使用 DDP 时，启用模型并行
        model.is_parallelizable = True
        model.model_parallel = True

    # DataCollatorForSeq2Seq: 自动将 batch 内序列填充到相同长度
    # label_pad_token_id = IGNORE_INDEX(-100): 填充位置的 label 也是 -100，不参与 loss
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=IGNORE_INDEX,
        pad_to_multiple_of=4 if tokenizer.padding_side == "right" else None,
    )
    trainer = SavePeftModelTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    # ========================================================================
    # 开始训练
    # ========================================================================
    if training_args.do_train:
        if trainer.is_world_process_zero():
            logger.info("*** Train ***")
            sample = next(iter(trainer.get_train_dataloader()))
            logger.debug(f"Train dataloader example: {sample}")
            logger.debug(f"input_ids:\n{list(sample['input_ids'])[:3]}, \nlabels:\n{list(sample['labels'])[:3]}")
            logger.debug(f"Decode input_ids[0]:\n{tokenizer.decode(sample['input_ids'][0])}")
            replaced_labels = [label if label != IGNORE_INDEX else tokenizer.pad_token_id for label in
                               sample['labels'][0]]
            logger.debug(f"Decode labels[0]:\n{tokenizer.decode(replaced_labels)}")
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        metrics = train_result.metrics
        metrics["train_samples"] = max_train_samples
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        model.config.use_cache = True  # enable cache after training
        tokenizer.padding_side = "left"  # restore padding side
        tokenizer.init_kwargs["padding_side"] = "left"

        if trainer.is_world_process_zero():
            logger.debug(f"Training metrics: {metrics}")
            logger.info(f"Saving model checkpoint to {training_args.output_dir}")
            if is_deepspeed_zero3_enabled():
                # DeepSpeed ZeRO-3 需要特殊保存方式：先合并分散的权重再保存
                save_model_zero3(model, tokenizer, training_args, trainer)
            else:
                save_model(model, tokenizer, training_args)

    # ========================================================================
    # 验证评估
    # ========================================================================
    if training_args.do_eval:
        if trainer.is_world_process_zero():
            logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")

        metrics["eval_samples"] = max_eval_samples
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        if trainer.is_world_process_zero():
            logger.debug(f"Eval metrics: {metrics}")


if __name__ == "__main__":
    main()
