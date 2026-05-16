# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Train R1 model with GRPO rl algo.
"""
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional
import re
from datasets import load_dataset
import torch
from loguru import logger
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.trainer_utils import get_last_checkpoint
from transformers.integrations import is_deepspeed_zero3_enabled
from trl import GRPOConfig, GRPOTrainer, ModelConfig, TrlParser
from peft import LoraConfig, TaskType, get_peft_model
try:
    from latex2sympy2_extended import NormalizationConfig
    from math_verify import LatexExtractionConfig, parse, verify
    _MATH_VERIFY_AVAILABLE = True
except ImportError:
    _MATH_VERIFY_AVAILABLE = False

os.environ["TOKENIZERS_PARALLELISM"] = "FALSE"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with GRPO
    """
    tokenizer_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "The tokenizer for weights initialization."}
    )
    # Dataset arguments
    dataset_name: Optional[str] = field(
        default="openai/gsm8k",
        metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    train_file_dir: Optional[str] = field(
        default=None, metadata={"help": "Directory containing training files for local datasets."}
    )
    train_samples: Optional[int] = field(default=-1, metadata={"help": "Number of samples to train on, -1 for all"})
    subset_name: Optional[str] = field(default="main",
                                       metadata={"help": "Subset name, e.g., 'default', 'main'. default is 'default'"})
    dataset_splits: Optional[str] = field(default="train", metadata={"help": "Split name"})
    preprocessing_num_workers: Optional[int] = field(default=10,
                                                     metadata={"help": "Number of workers for preprocessing"})
    # QLoRA arguments
    qlora: bool = field(default=False, metadata={"help": "Whether to use qlora"})

    # Reward type
    reward_type: str = field(
        default="math",
        metadata={"help": "Reward type: 'math' (default) or 'medical_safety'"}
    )


def normalize_text(text):
    """Normalize text by removing extra whitespace, converting to lowercase."""
    if text is None:
        return ""
    # Remove extra whitespace and convert to lowercase
    text = re.sub(r'\s+', ' ', text.strip().lower())
    return text


def extract_answer(text):
    """Extract content between <answer> tags."""
    if text is None:
        return ""
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def accuracy_reward(completions, answer, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, answer):
        if '####' in sol:
            # for GSM8K
            gold_parsed = parse(sol.split("####", 1)[-1].strip())
            answer_parsed = parse(extract_answer(content))
        else:
            # First try latex parsing
            gold_parsed = parse(
                sol,
                extraction_mode="first_match",
                extraction_config=[LatexExtractionConfig()],
            )
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
        try:
            reward = float(verify(answer_parsed, gold_parsed))
        except Exception as e:
            logger.warning(f"Error in verification: {e}")
            reward = 0.0
        logger.debug(f"predict_answer: {content}, \nground_truth: {sol}, \n"
                     f"answer_parsed: {answer_parsed}, gold_parsed: {gold_parsed}, reward: {reward}\n\n")
        rewards.append(reward)
    logger.debug(f'accuracy rewards: {rewards}')
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think><answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]

    rewards = [1.0 if match else 0.0 for match in matches]
    logger.debug(f'format rewards: {rewards}')
    return rewards


# ═══════════════════════════════════════════════════════════════════
# Medical Safety Reward Functions
#   Total = R_red_flag + R_escalation + R_uncertainty + R_completeness + R_format + R_length
# ═══════════════════════════════════════════════════════════════════

HIGH_RISK_PATTERNS = [
    r'胸痛', r'胸闷.*喘', r'呼吸困难', r'抽搐', r'惊厥',
    r'意识.*(丧失|模糊|不清)', r'昏迷', r'休克',
    r'大出血', r'吐血', r'咯血', r'便血.*大量',
    r'剧烈.*(头痛|腹痛|胸痛)', r'窒息',
    r'孕妇', r'婴儿.*(用药|吃药|发烧|腹泻)', r'新生儿',
    r'儿童.*(用药|吃药|剂量)', r'婴幼儿',
    r'停药', r'换药', r'改.*(剂量|药)', r'自行.*(停|换|改)',
    r'抗生素.*(停|换|自己)', r'降压药.*停', r'降糖药.*停',
    r'血糖.*(明显|异常|很高|过低)', r'血压.*(明显|异常|很高|过低)',
    r'糖尿病.*(急性|酮症|昏迷)', r'心梗', r'脑梗', r'中风.*(突然|急性)',
    r'咯血', r'呕血', r'黑便', r'黄疸', r'高热.*不退',
]


HIGH_RISK_CATEGORIES = {"急症", "用药"}
MEDIUM_RISK_CATEGORIES = {"慢病", "就医"}

def _is_high_risk(prompt, category: str = None) -> bool:
    """Check if prompt describes a high-risk scenario. Uses category first, then text matching."""
    if category and category in HIGH_RISK_CATEGORIES:
        return True
    text = _prompt_to_str(prompt)
    if not text:
        return False
    for pat in HIGH_RISK_PATTERNS:
        if re.search(pat, text):
            return True
    return False


def _is_medium_risk(prompt, category: str = None) -> bool:
    """Check if prompt is medium-risk — expects caution but not full escalation."""
    if category and category in MEDIUM_RISK_CATEGORIES:
        return True
    if category and category in HIGH_RISK_CATEGORIES:
        return True
    return _is_high_risk(prompt, category)


def _prompt_to_str(prompt) -> str:
    """Convert prompt from various formats (str, dict, list) to a single string."""
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, dict):
        return prompt.get("content", prompt.get("value", str(prompt)))
    if isinstance(prompt, list):
        # Message format: [{'role': 'user', 'content': '...'}, ...]
        texts = []
        for msg in prompt:
            if isinstance(msg, dict):
                texts.append(msg.get("content", ""))
            elif isinstance(msg, str):
                texts.append(msg)
        return " ".join(texts)
    return str(prompt) if prompt else ""


def _get_text(completion) -> str:
    if isinstance(completion, list) and len(completion) > 0:
        return completion[0].get("content", "")
    return ""


RED_FLAG_RULES = [
    (r'不(用|需要|必|要|建议|推荐|一定).*(去|看|跑).*(医院|就医|急诊|医生)', -1.5),
    (r'忍一忍|扛一扛|扛过去|熬一熬|问题不大.*不用', -1.0),
    (r'不要.*(去医院|看医生|就医|打疫苗|做检查)', -1.5),
    (r'(自己|自行|直接|随便|可以).*(买|吃|服用|用).*(头孢|阿莫西林|抗生素|降压药|降糖药)', -1.5),
    (r'抗生素.*(最有效|最管用|就行|就好|搞定)', -1.5),
    (r'(停|换|改).*(药|剂量).*不用.*(问|咨询|看).*(医生|大夫)', -1.5),
    (r'(你这|这就|肯定|绝对|百分百|100%).*(是|得了|患了).*(心梗|脑梗|癌症|肿瘤|阑尾炎|心绞痛|心肌炎)', -1.0),
    (r'我.*(诊断|判断|确诊).*(为|是|你得了)', -1.0),
    (r'(吃|服用|口服).*\d+.*(片|粒|颗|mg|克).*(头孢|阿莫西林|抗生素|降压|降糖)', -1.5),
    (r'剂量.*(加倍|翻倍|double|随意|随便)', -1.5),
    (r'(胸痛|胸闷|呼吸困难|大出血|抽搐|意识不清).{0,15}(休息|喝.*水|观察|忍|扛)', -2.0),
]

ESCALATION_PATTERNS = [
    (r'(立即|尽快|马上|赶紧).*(就医|去医院|去急诊|看医生|就诊)', 1.5),
    (r'(拨打|打).*120', 1.5),
    (r'(必须|需要|建议).*(立即|紧急|急诊).*(就医|处理|就诊)', 1.5),
    (r'去急诊', 1.5),
    (r'(建议|需要|应该).*(就医|去医院|看医生|就诊|门诊)', 0.8),
    (r'(尽快|尽早).*(就医|就诊|看医生|检查)', 1.0),
    (r'线下.*(评估|就诊|就医)', 0.8),
    (r'(建议|可以).*(咨询|问一下|看一下).*(医生|大夫|药师)', 0.4),
    (r'(完善|做).*(检查|化验|B超|CT|血常规)', 0.3),
    (r'遵医嘱', 0.2),
]

UNCERTAINTY_PATTERNS = [
    (r'不能.*(仅凭|只靠|单靠|通过).*(描述|文字|线上|网络).*判断', 0.8),
    (r'需要.*(结合|综合|参考).*(检查|化验|面诊|评估).*结果', 0.8),
    (r'(建议|需要|应该).*(由|请|找).*(医生|大夫|专业).*(面诊|评估|判断|诊断)', 0.6),
    (r'线上.*(信息|咨询|回答|建议).*不能.*(替代|代替|取代).*(正式|专业|医生).*诊断', 0.8),
    (r'(可能|不一定|不一定就是|未必)', 0.2),
    (r'(仅供|作为).*参考', 0.3),
]


def medical_red_flag_reward(completions, prompts=None, **kwargs):
    rewards = []
    categories = kwargs.get('category', []) if 'category' in kwargs else []
    for i, c in enumerate(completions):
        text = _get_text(c)
        prompt = prompts[i] if prompts and i < len(prompts) else ""
        category = categories[i] if i < len(categories) else None
        penalty = 0.0
        for pat, w in RED_FLAG_RULES:
            if re.search(pat, text):
                penalty += w
        if _is_high_risk(prompt, category) and penalty < 0:
            penalty *= 1.5
        rewards.append(max(penalty, -3.0))
    return rewards


def medical_escalation_reward(completions, prompts=None, **kwargs):
    rewards = []
    categories = kwargs.get('category', []) if 'category' in kwargs else []
    for i, c in enumerate(completions):
        text = _get_text(c)
        prompt = prompts[i] if prompts and i < len(prompts) else ""
        category = categories[i] if i < len(categories) else None
        score = 0.0
        for pat, w in ESCALATION_PATTERNS:
            if re.search(pat, text):
                score = max(score, w)
        if not _is_high_risk(prompt, category):
            if score >= 1.5:
                # 低风险场景强烈升级(立即就医/去急诊/打120) → 轻罚
                score = -0.3
            elif score >= 1.0:
                # 低风险场景中度升级(尽快就医) → 小罚
                score = -0.1
            else:
                # 低风险场景温和建议(建议就医/咨询医生) → 封顶
                score = min(score, 0.4)
        rewards.append(score)
    return rewards


def medical_uncertainty_reward(completions, prompts=None, **kwargs):
    rewards = []
    for c in completions:
        text = _get_text(c)
        score = 0.0
        for pat, w in UNCERTAINTY_PATTERNS:
            if re.search(pat, text):
                score = max(score, w)
        rewards.append(score)
    return rewards


def medical_completeness_reward(completions, prompts=None, **kwargs):
    rewards = []
    for c in completions:
        text = _get_text(c)
        score = 0.0
        if re.search(r'(可能|考虑|怀疑|初步|推测|常见|一般|通常|多见于)', text):
            score += 0.4
        if re.search(r'(注意|警惕|风险|危险|严重|恶化|加重|不要|避免|禁忌|慎用)', text):
            score += 0.4
        if re.search(r'(建议|可以|需要|应该|最好|推荐).{0,20}(就医|检查|就诊|观察|休息|饮食|运动|用药|随访|复查)', text):
            score += 0.4
        score = min(score, 1.0)
        if score == 0 and len(text) < 30:
            score = -0.3
        rewards.append(score)
    return rewards


def medical_format_reward(completions, prompts=None, **kwargs):
    fmt = r'<think>.*?</think>\s*<answer>.*?</answer>'
    return [0.2 if re.search(fmt, _get_text(c), re.DOTALL) else -0.2 for c in completions]



def medical_repetition_reward(completions, prompts=None, **kwargs):
    """Penalize repetitive completions. Uses char-level detection for Chinese, word-level for English."""
    rewards = []
    for c in completions:
        text = _get_text(c)
        if len(text) < 20:
            rewards.append(0.0)
            continue
        # Use character-level tokenization for Chinese text
        if any('一' <= c <= '鿿' for c in text):
            tokens = list(text)  # char-level for Chinese
        else:
            tokens = text.split()  # word-level for English
        if not tokens:
            rewards.append(0.0)
            continue
        unique_ratio = len(set(tokens)) / len(tokens)
        # Extreme repetition: >80% characters are repeats
        if unique_ratio < 0.20:
            rewards.append(-2.5)
        # Severe repetition: >65% characters are repeats
        elif unique_ratio < 0.35:
            rewards.append(-1.5)
        # Moderate repetition: >45% characters are repeats
        elif unique_ratio < 0.55:
            rewards.append(-0.5)
        # Mild repetition
        elif unique_ratio < 0.70:
            rewards.append(-0.1)
        else:
            rewards.append(0.0)
    return rewards

def medical_length_reward(completions, prompts=None, **kwargs):
    rewards = []
    for c in completions:
        length = len(_get_text(c))
        if length < 20:
            rewards.append(-0.8)
        elif length < 50:
            rewards.append(-0.2)
        elif length <= 400:
            rewards.append(0.3)
        elif length <= 700:
            rewards.append(0.0)
        else:
            rewards.append(-0.3)
    return rewards


def medical_safety_reward(completions, prompts=None, **kwargs):
    r_red = medical_red_flag_reward(completions, prompts=prompts, **kwargs)
    r_esc = medical_escalation_reward(completions, prompts=prompts, **kwargs)
    r_unc = medical_uncertainty_reward(completions, prompts=prompts, **kwargs)
    r_cmp = medical_completeness_reward(completions, prompts=prompts, **kwargs)
    r_fmt = medical_format_reward(completions, prompts=prompts, **kwargs)
    r_len = medical_length_reward(completions, prompts=prompts, **kwargs)
    r_rep = medical_repetition_reward(completions, prompts=prompts, **kwargs)
    return [a + b + c + d + e + f + g for a, b, c, d, e, f, g in zip(r_red, r_esc, r_unc, r_cmp, r_fmt, r_len, r_rep)]


# ── Math system prompt ──
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

# ── Medical safety system prompt ──
MEDICAL_SYSTEM_PROMPT = (
    "你是一个医疗健康助手。请先分析问题（在 <think> 标签内），然后给出回答（在 <answer> 标签内）。\n"
    "回答要求：\n"
    "1. 如果有危险症状（胸痛、呼吸困难、大出血、抽搐、意识障碍等），必须建议立即就医\n"
    "2. 不能给出确诊、不能推荐处方药、不能建议停药/换药\n"
    "3. 说明线上咨询的局限性，建议必要时面诊\n"
    "4. 提供初步判断、风险提示、下一步建议\n\n"
    "格式：<think>你的分析</think><answer>你的回答</answer>"
)


def get_checkpoint(training_args: GRPOConfig):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint


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


def grpo_train(
        model_args: ModelConfig, script_args: ScriptArguments, training_args: GRPOConfig
):
    # Add distributed training initialization
    is_main_process = training_args.local_rank in [-1, 0]

    # Only log on main process
    if is_main_process:
        logger.warning(
            f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
            + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
        )
        logger.info(f"Model parameters {model_args}")
        logger.info(f"Script parameters {script_args}")
        logger.info(f"Training parameters {training_args}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        (
            script_args.tokenizer_name_or_path
            if script_args.tokenizer_name_or_path
            else model_args.model_name_or_path
        ),
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load datasets
    if script_args.train_file_dir and os.path.exists(script_args.train_file_dir):
        # Load from local directory
        dataset = load_dataset("json", data_dir=script_args.train_file_dir, split="train")
    else:
        # Load from HuggingFace hub
        dataset = load_dataset(script_args.dataset_name, script_args.subset_name, split=script_args.dataset_splits)

    if script_args.train_samples > 0:
        dataset = dataset.shuffle(seed=42).select(range(script_args.train_samples))

    # Prepare dataset
    is_medical = (script_args.reward_type == "medical_safety")
    sys_prompt = MEDICAL_SYSTEM_PROMPT if is_medical else SYSTEM_PROMPT

    with training_args.main_process_first(desc="Dataset preparation"):
        if is_medical:
            # Medical: use 'question' or 'prompt' field + optional 'category'
            dataset = dataset.map(
                lambda x: {
                    'prompt': [
                        {'role': 'system', 'content': sys_prompt},
                        {'role': 'user', 'content': x.get('question', x.get('prompt', ''))}
                    ],
                    'category': x.get('category', '通用'),
                },
                num_proc=script_args.preprocessing_num_workers,
                desc="Processing dataset" if is_main_process else None,
            )
        else:
            # Math: original format with 'question' + 'answer'
            dataset = dataset.map(
                lambda x: {
                    'prompt': [
                        {'role': 'system', 'content': sys_prompt},
                        {'role': 'user', 'content': x['question']}
                    ],
                    'answer': x['answer']
                },
                num_proc=script_args.preprocessing_num_workers,
                desc="Processing dataset" if is_main_process else None,
            )

    # Split dataset
    train_test_split = dataset.train_test_split(test_size=0.1)
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]

    if is_main_process:
        logger.info("*** Initializing model kwargs ***")

    # Model initialization
    torch_dtype = (
        model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    )

    # Set up distributed training config
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    ddp = world_size != 1

    # Check for QLoRA compatibility
    if script_args.qlora and is_deepspeed_zero3_enabled():
        logger.warning("ZeRO3 are both currently incompatible with QLoRA.")

    # Check quantization settings
    if model_args.load_in_4bit and model_args.load_in_8bit:
        raise ValueError("Error, load_in_4bit and load_in_8bit cannot be set at the same time")

    # Set up quantization config
    quantization_config = None
    if script_args.qlora and (model_args.load_in_4bit or model_args.load_in_8bit):
        if is_main_process:
            logger.info(
                f"Quantizing model, load_in_4bit: {model_args.load_in_4bit}, load_in_8bit: {model_args.load_in_8bit}")
        if is_deepspeed_zero3_enabled():
            raise ValueError("DeepSpeed ZeRO-3 is incompatible with quantization.")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=model_args.load_in_4bit,
            load_in_8bit=model_args.load_in_8bit,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
        )
    elif model_args.load_in_4bit or model_args.load_in_8bit:
        # Support quantization even without qlora flag
        if is_main_process:
            logger.info(
                f"Quantizing model, load_in_4bit: {model_args.load_in_4bit}, load_in_8bit: {model_args.load_in_8bit}")
        if is_deepspeed_zero3_enabled():
            raise ValueError("DeepSpeed ZeRO-3 is incompatible with quantization.")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=model_args.load_in_4bit,
            load_in_8bit=model_args.load_in_8bit,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
        )

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        dtype=torch_dtype,
        low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
        quantization_config=quantization_config,
    )

    num_gpus = torch.cuda.device_count()
    if ddp:
        model_kwargs["device_map"] = None
    elif num_gpus > 1:
        max_memory = {}
        for i in range(num_gpus):
            gpu_props = torch.cuda.get_device_properties(i)
            total_mem = gpu_props.total_memory
            # 预留20%内存给训练时的梯度、优化器状态等
            usable_mem = int(total_mem * 0.8)
            max_memory[i] = f"{usable_mem // (1024 ** 3)}GiB"
        model_kwargs["max_memory"] = max_memory
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = "auto"

    if is_main_process:
        logger.info(f"Using {num_gpus} GPUs")
        logger.info(f"model_kwargs={model_kwargs}")

    config = AutoModelForCausalLM.config_class if hasattr(AutoModelForCausalLM, 'config_class') else None
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code,
            revision=model_args.model_revision,
        )
    except Exception:
        config = None

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs,
    )

    # Patch MoE modules for DeepSpeed ZeRO-3
    model_type = getattr(config, "model_type", None) if config else getattr(model.config, "model_type", None)
    if model_type == "mixtral" and is_deepspeed_zero3_enabled():
        from deepspeed.utils import set_z3_leaf_modules
        from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock
        set_z3_leaf_modules(model, [MixtralSparseMoeBlock])

    if model_type == "deepseek_v3" and is_deepspeed_zero3_enabled():
        for layer in model.model.layers:
            if 'DeepseekV3MoE' in str(type(layer.mlp)):
                layer.mlp._z3_leaf = True

    if model_type == "qwen3_moe" and is_deepspeed_zero3_enabled():
        from deepspeed.utils import set_z3_leaf_modules
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock
        set_z3_leaf_modules(model, [Qwen3MoeSparseMoeBlock])

    if model_type == "qwen3_5_moe" and is_deepspeed_zero3_enabled():
        from deepspeed.utils import set_z3_leaf_modules
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeSparseMoeBlock
        set_z3_leaf_modules(model, [Qwen3_5MoeSparseMoeBlock])

    if is_main_process and hasattr(model, 'hf_device_map'):
        logger.info(f"Model Device Map: {model.hf_device_map.items()}")
    elif is_main_process and num_gpus > 1:
        logger.info("Model Device Map:")
        for name, param in model.named_parameters():
            if hasattr(param, 'device'):
                logger.info(f"  {name}: {param.device}")
                break

    # Configure LoRA if enabled
    if model_args.use_peft:
        if is_main_process:
            logger.info("Fine-tuning method: LoRA(PEFT)")
        if training_args.gradient_checkpointing:
            logger.warning("Gradient checkpointing is enabled. It may cause issues with LoRA, setting it to False.")
            training_args.gradient_checkpointing = False
        target_modules = model_args.lora_target_modules if model_args.lora_target_modules else None
        if target_modules == 'all' or (target_modules and 'all' in target_modules):
            target_modules = find_all_linear_names(model, int4=model_args.load_in_4bit, int8=model_args.load_in_8bit)
        if is_main_process:
            logger.info(f"Peft target_modules: {target_modules}, lora rank: {model_args.lora_r}, ")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
            inference_mode=False,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
        )
        model = get_peft_model(model, peft_config)
        # Fixed FP16 ValueError for quantized models
        for param in filter(lambda p: p.requires_grad, model.parameters()):
            param.data = param.data.to(torch.float32)
        model.print_trainable_parameters()
    else:
        if is_main_process:
            logger.info("Fine-tuning method: Full parameters training")

    if training_args.gradient_checkpointing and getattr(model, "supports_gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        logger.info("Gradient checkpointing enabled.")
    else:
        model.config.use_cache = True
        logger.info("Gradient checkpointing disabled.")

    # Initialize GRPO trainer with distributed training support
    if is_medical:
        reward_funcs = [medical_safety_reward]
    else:
        reward_funcs = [accuracy_reward, format_reward]

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset if training_args.eval_strategy != "no" else None,
    )
    logger.info("*** GRPO Trainer initialized ***")
    logger.debug(f"Trainer: {trainer}")

    # Training
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        if is_main_process:
            logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    if is_main_process:
        logger.info(
            f'*** Starting training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for '
            f'{training_args.num_train_epochs} epochs ***'
        )

    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

    # Log and save metrics on main process
    if is_main_process:
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        logger.info("*** Training complete ***")
        logger.info("*** Save model ***")

    # Save model
    trainer.model.config.use_cache = True
    if is_main_process:
        trainer.save_model(training_args.output_dir)
        logger.info(f"Model saved to {training_args.output_dir}")

    training_args.distributed_state.wait_for_everyone()

    if is_main_process:
        tokenizer.save_pretrained(training_args.output_dir)
        logger.info(f"Tokenizer saved to {training_args.output_dir}")

        # Create model card and save config
        kwargs = {
            "dataset_name": script_args.dataset_name,
            "tags": ["r1", "grpo"],
        }
        trainer.create_model_card(**kwargs)
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    if is_main_process:
        logger.info("*** Training complete! ***")


def main():
    parser = TrlParser((ModelConfig, ScriptArguments, GRPOConfig))
    model_args, script_args, training_args = parser.parse_args_and_config()

    # Run the main training loop
    grpo_train(model_args, script_args, training_args)


if __name__ == "__main__":
    main()
