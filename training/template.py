# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 对话模板管理模块

核心思路：
  不同基座模型（Qwen、LLaMA、ChatGLM...）使用不同的对话格式（ChatML、Alpaca...），
  需要一个统一的模板系统来将「用户消息 + 助手回复」拼成模型认识的 prompt 字符串。

  这个模块定义了：
  1. Conversation 数据类 —— 持有单个模板的所有信息
  2. conv_templates 全局字典 —— 注册所有已知模板
  3. register_conv_template() —— 注册新模板
  4. get_conv_template() —— 按名称取模板
 
  使用流程：
    template = get_conv_template("qwen")
    dialog_parts = template.get_dialog(messages=[["问题1", "回答1"], ["问题2", "回答2"]])
    # 得到 ["<|im_start|>user\n问题1<|im_end|>\n<|im_start|>assistant\n", "回答1", ...]
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Sequence

__all__ = ['Conversation', 'register_conv_template', 'get_conv_template']


@dataclass
class Conversation:
    """
    一个提示模板的完整定义。

    核心字段说明：
      name          : 模板名称，如 "qwen", "llama3"
      system_prompt : 系统提示词（角色设定）
      prompt        : 单轮用户问题模板，用 {query} 占位
      sep           : 分隔符（通常是 eos_token）
      stop_str      : 停止生成的标记

    两个核心方法：
      get_prompt()  → 返回纯字符串，用于推理时拼接 prompt
      get_dialog()  → 返回字符串列表 [query1, response1, query2, response2, ...]
                       训练时需要分别拿到 query 和 response 来计算 loss
    """
    """A class that manages prompt templates and keeps all conversation history."""

    name: str                                          # 模板名称
    system_prompt: str                                # 系统提示词
    messages: Optional[List[Sequence[str]]]           # 历史对话，格式: [[问题, 回答], ...]
    roles: Optional[Sequence[str]]                    # 对话角色名，如 ("USER", "ASSISTANT")
    prompt: str                                       # 单轮用户提问模板，{query} 会被替换为实际输入
    sep: str                                          # 轮次分隔符，如 "</s>"、"\n"
    stop_str: Optional[str] = "</s>"                  # 停止标记，推理时遇到此标记停止生成

    def get_prompt(
            self,
            messages: Optional[List[Sequence[str]]] = None,
            system_prompt: Optional[str] = ""
    ) -> str:
        """
        返回拼接后的完整 prompt 字符串（不含回复内容）。
        用于推理阶段：把 prompt 喂给模型，让模型接着生成。
        """
        return "".join(self._format_example(messages, system_prompt))

    def get_dialog(
            self,
            messages: Optional[List[Sequence[str]]] = None,
            system_prompt: Optional[str] = ""
    ) -> List[str]:
        """
        返回一个字符串列表，偶数位是格式化后的 query，奇数位是对应的 response。
        用于训练阶段：需要分别拿到每个 query 和 response 来做 tokenize 和 loss 计算。

        示例返回：
          ["<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n",  # query
           "你好！有什么可以帮你的？"]                                      # response
        """
        return self._format_example(messages, system_prompt)

    def _format_example(
            self,
            messages: Optional[List[Sequence[str]]] = None,
            system_prompt: Optional[str] = ""
    ) -> List[str]:
        """
        将多轮对话格式化为 [query1, response1, query2, response2, ...] 列表。

        格式化逻辑：
          - 第 0 轮：system_prompt + "USER: {query} ASSISTANT:"  → query1，然后 response1
          - 第 1+ 轮：sep + "USER: {query} ASSISTANT:"            → query2，然后 response2

        关键：每轮的 query 都以分隔符结尾（如 "</s>"），这样模型知道一轮对话的边界。
        """
        system_prompt = system_prompt or self.system_prompt
        system_prompt = system_prompt + self.sep if system_prompt else ""  # 非空 system_prompt 后面加分隔符
        messages = messages or self.messages
        convs = []
        if not messages:
            messages = []
        for turn_idx, [user_query, bot_resp] in enumerate(messages):
            if turn_idx == 0:
                convs.append(system_prompt + self.prompt.format(query=user_query))
                convs.append(bot_resp)
            else:
                convs.append(self.sep + self.prompt.format(query=user_query))
                convs.append(bot_resp)
        return convs

    def append_message(self, query: str, answer: str):
        """Append a new message."""
        self.messages.append([query, answer])


# ============================================================================
# 全局模板注册表 —— conv_templates 字典存所有已注册的模板
# 用 register_conv_template() 注册新模板，用 get_conv_template(name) 按名称取模板
# ============================================================================
conv_templates: Dict[str, Conversation] = {}


def register_conv_template(template: Conversation):
    """将模板注册到全局字典。在文件末尾调用，一次性注册所有已知模板。"""
    conv_templates[template.name] = template


# ============================================================================
# 以下是所有预定义的对话模板
# 每个模板对应一种基座模型的 prompt 格式
# 关键区别在于 prompt 字段中用 {query} 占位符和 sep 分隔符的组合方式
# ============================================================================

"""Vicuna v1.1 template
Supports: https://huggingface.co/lmsys/vicuna-7b-delta-v1.1
Format: USER: {query} ASSISTANT: {response}</s>
"""
register_conv_template(
    Conversation(
        name="vicuna",
        system_prompt="A chat between a curious user and an artificial intelligence assistant. "
                      "The assistant gives helpful, detailed, and polite answers to the user's questions.",
        messages=[],
        roles=("USER", "ASSISTANT"),
        prompt="USER: {query} ASSISTANT:",
        sep="</s>",
    )
)

"""Base model template, for few shot — 无格式，直接拼接原始文本"""
register_conv_template(
    Conversation(
        name="base",
        system_prompt="",
        messages=[],
        roles=("USER", "ASSISTANT"),
        prompt="{query}",
        sep="</s>",
    )
)

"""Alpaca template"""
register_conv_template(
    Conversation(
        name="alpaca",
        system_prompt="Below is an instruction that describes a task. "
                      "Write a response that appropriately completes the request.",
        messages=[],
        roles=("### Instruction", "### Response"),
        prompt="### Instruction:\n{query}\n\n### Response:\n",
        sep="\n\n",
    )
)

"""Baichuan template
source: https://huggingface.co/baichuan-inc/Baichuan-13B-Chat/blob/main/generation_utils.py#L31
Support: https://huggingface.co/baichuan-inc/Baichuan-13B-Chat
"""
register_conv_template(
    Conversation(
        name="baichuan",
        system_prompt="",
        messages=[],
        roles=("<reserved_102>", "<reserved_103>"),
        prompt="<reserved_102>{query}<reserved_103>",
        sep="</s>",
    )
)

"""Baichuan2 template
Support: https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat
         https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat
"""
register_conv_template(
    Conversation(
        name="baichuan2",
        system_prompt="",
        messages=[],
        roles=("<reserved_106>", "<reserved_107>"),
        prompt="<reserved_106>{query}<reserved_107>",
        sep="</s>",
    )
)

"""ziya template"""
register_conv_template(
    Conversation(
        name="ziya",
        system_prompt="",
        messages=[],
        roles=("<human>", "<bot>"),
        prompt="<human>:{query}\n<bot>:",
        sep="\n",
    )
)

"""Linly template"""
register_conv_template(
    Conversation(
        name="linly",
        system_prompt="",
        messages=[],
        roles=("User", "Bot"),
        prompt="User: {query}\nBot: ",
        sep="\n",
    )
)

"""ChatGLM1 template
Support: https://huggingface.co/THUDM/chatglm-6b
source: https://huggingface.co/THUDM/chatglm-6b/blob/main/modeling_chatglm.py#L1307
"""
register_conv_template(
    Conversation(
        name="chatglm",
        system_prompt="",
        messages=[],
        roles=("问", "答"),
        prompt="问：{query}\n答：",
        sep="\n",
    )
)

"""ChatGLM2 template
Support: https://huggingface.co/THUDM/chatglm2-6b
source: https://huggingface.co/THUDM/chatglm2-6b/blob/main/modeling_chatglm.py#L1007
"""
register_conv_template(
    Conversation(
        name="chatglm2",
        system_prompt="",
        messages=[],
        roles=("问", "答"),
        prompt="问：{query}\n\n答：",
        sep="\n\n",
    )
)

"""ChatGLM3 template
Support: https://huggingface.co/THUDM/chatglm3-6b
source: https://huggingface.co/THUDM/chatglm3-6b/blob/main/tokenization_chatglm.py#L179
"""
register_conv_template(
    Conversation(
        name="chatglm3",
        system_prompt="",
        messages=[],
        roles=("<|user|>", "<|assistant|>"),
        prompt="<|user|>\n{query}<|assistant|>",
        sep="\n",
        stop_str="<|user|>",
    )
)

"""Phoenix template"""
register_conv_template(
    Conversation(
        name="phoenix",
        system_prompt="A chat between a curious human and an artificial intelligence assistant. "
                      "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n",
        messages=[],
        roles=("Human", "Assistant"),
        prompt="Human: <s>{query}</s>Assistant: ",
        sep="</s>",
    )
)

"""belle template
Supports: https://huggingface.co/BelleGroup/BELLE-LLaMA-EXT-13B
"""
register_conv_template(
    Conversation(
        name="belle",
        system_prompt="",
        messages=[],
        roles=("Human", "Belle"),
        prompt="Human: {query}\n\nBelle: ",
        sep="\n\n",
    )
)

"""aquila template
Supports: https://huggingface.co/qhduan/aquilachat-7b
          https://huggingface.co/BAAI/AquilaChat2-34B
"""
register_conv_template(
    Conversation(
        name="aquila",
        system_prompt="A chat between a curious human and an artificial intelligence assistant. "
                      "The assistant gives helpful, detailed, and polite answers to the human's questions.",
        messages=[],
        roles=("Human", "Assistant"),
        prompt="Human: {query}###Assistant:",
        sep="###",
    )
)

"""intern template
Supports: https://huggingface.co/internlm/internlm-chat-7b
          https://huggingface.co/internlm/internlm-chat-20b
"""
register_conv_template(
    Conversation(
        name="intern",
        system_prompt="",
        messages=[],
        roles=("<|User|>", "<|Bot|>"),
        prompt="<|User|>:{query}<eoh>\n<|Bot|>:",
        sep="<eoa>\n",
        stop_str="<eoa>",
    )
)

"""intern2 template
Supports: https://huggingface.co/internlm/internlm2-1_8b
"""
register_conv_template(
    Conversation(
        name="intern2",
        system_prompt="<|im_start|>system\nYou are an AI assistant whose name is InternLM (书生·浦语).\n<|im_end|>\n",
        messages=[],
        roles=("user", "assistant"),
        prompt="<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n",
        sep="<|im_end|>\n",
        stop_str="<|im_end|>",
    )
)

"""StarChat template
Supports: https://huggingface.co/HuggingFaceH4/starchat-alpha
          https://huggingface.co/HuggingFaceH4/starchat-beta
"""
register_conv_template(
    Conversation(
        name="starchat",
        system_prompt="<system>\n",
        messages=[],
        roles=("<|user|>", "<|assistant|>"),
        prompt="<|user|>\n{query}<|end|>\n<|assistant|>\n",
        sep="<|end|>\n",
        stop_str="<|end|>",
    )
)

"""llama2 template
Supports: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
          https://huggingface.co/meta-llama/Llama-2-13b-chat-hf
          https://huggingface.co/meta-llama/Llama-2-70b-chat-hf
reference: https://github.com/facebookresearch/llama/blob/cfc3fc8c1968d390eb830e65c63865e980873a06/llama/generation.py#L212
"""
register_conv_template(
    Conversation(
        name="llama2",
        system_prompt=(
            "<<SYS>>\nYou are a helpful, respectful and honest assistant. "
            "Always answer as helpfully as possible, while being safe. "
            "Your answers should not include any harmful, unethical, racist, sexist, "
            "toxic, dangerous, or illegal content. "
            "Please ensure that your responses are socially unbiased and positive in nature.\n\n"
            "If a question does not make any sense, or is not factually coherent, "
            "explain why instead of answering something not correct. "
            "If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n"
        ),
        messages=[],
        roles=("[INST]", "[/INST]"),
        prompt="[INST] {query} [/INST]",
        sep="</s>",
    )
)

"""llama3 template
source: https://huggingface.co/meta-llama
Supports: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
chat template:
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{{ system_prompt }}<|eot_id|><|start_header_id|>user<|end_header_id|>
{{ user_msg_1 }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{{ model_answer_1 }}<|eot_id|>
"""
register_conv_template(
    Conversation(
        name="llama3",
        system_prompt=(
            "<|start_header_id|>system<|end_header_id|>\n\n"
            "You are a helpful, excellent and smart assistant."
        ),
        messages=[],
        roles=("user", "assistant"),
        prompt=(
            "<|start_header_id|>user<|end_header_id|>\n\n{query}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        ),
        sep="<|eot_id|>",
        stop_str="<|eot_id|>",
    )
)

"""llama2-zh template
source: https://github.com/ymcui/Chinese-LLaMA-Alpaca-2
Supports: https://huggingface.co/ziqingyang/chinese-alpaca-2-7b
"""
register_conv_template(
    Conversation(
        name="llama2-zh",
        system_prompt="[INST] <<SYS>>\nYou are a helpful assistant. 你是一个乐于助人的助手。\n<</SYS>>\n\n [/INST]",
        messages=[],
        roles=("[INST]", "[/INST]"),
        prompt="[INST] {query} [/INST]",
        sep="</s>",
    )
)

"""mistral template
Supports: https://huggingface.co/mistralai/Mistral-7B-v0.1
          https://huggingface.co/HuggingFaceH4/zephyr-7b-beta
source: https://docs.mistral.ai/llm/mistral-instruct-v0.1
"""
register_conv_template(
    Conversation(
        name="mistral",
        system_prompt="",
        messages=[],
        roles=("[INST]", "[/INST]"),
        prompt="[INST] {query} [/INST]",
        sep="</s>",
    )
)

"""XVERSE template
Supports: https://huggingface.co/xverse/XVERSE-13B-Chat
"""
register_conv_template(
    Conversation(
        name="xverse",
        system_prompt="",
        messages=[],
        roles=("Human", "Assistant"),
        prompt="Human: {query}\n\nAssistant: ",
        sep="</s>",
    )
)

"""chatml template
chatml: https://xbot123.com/645a461b922f176d7cfdbc2d/
"""
register_conv_template(
    Conversation(
        name="chatml",
        system_prompt="You are a helpful assistant.",
        messages=[],
        roles=("user", "assistant"),
        prompt="<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n",
        sep="<|im_end|>\n",
        stop_str="<|im_end|>",
    )
)

"""deepseek template
Supports: https://huggingface.co/deepseek-ai/deepseek-llm-7b-chat
          https://huggingface.co/deepseek-ai/deepseek-moe-16b-chat
"""
register_conv_template(
    Conversation(
        name="deepseek",
        system_prompt="",
        messages=[],
        roles=("User", "Assistant"),
        prompt="User: {query}\n\nAssistant:",
        sep="</s>",
    )
)


"""deepseek3 template
Supports: https://huggingface.co/deepseek-ai/DeepSeek-V3
"""
register_conv_template(
    Conversation(
        name="deepseek3",
        system_prompt="",
        messages=[],
        roles=("<｜User｜>", "<｜Assistant｜>"),
        prompt="<｜User｜>{query}<｜Assistant｜>",
        sep="</s>",
    )
)

"""deepseekcoder template
Supports: https://huggingface.co/deepseek-ai/deepseek-coder-33b-instruct
"""
register_conv_template(
    Conversation(
        name="deepseekcoder",
        system_prompt=(
            "You are an AI programming assistant, utilizing the Deepseek Coder model, "
            "developed by Deepseek Company, and you only answer questions related to computer science. "
            "For politically sensitive questions, security and privacy issues, "
            "and other non-computer science questions, you will refuse to answer\n"
        ),
        messages=[],
        roles=("### Instruction", "### Response"),
        prompt="### Instruction:\n{{content}}\n### Response:\n",
        sep="\n",
        stop_str="<|EOT|>",
    )
)

"""Yi template
source: https://github.com/01-ai/Yi
Supports: https://huggingface.co/01-ai/Yi-34B-Chat
          https://huggingface.co/01-ai/Yi-6B-Chat
"""
register_conv_template(
    Conversation(
        name="yi",
        system_prompt="",
        messages=[],
        roles=("user", "assistant"),
        prompt="<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n",
        sep="\n",
        stop_str="<|im_end|>",
    )
)

"""Orion template
source: https://github.com/OrionStarAI/Orion
Supports: https://huggingface.co/OrionStarAI/Orion-14B-Chat
"""
register_conv_template(
    Conversation(
        name="orion",
        system_prompt="",
        messages=[],
        roles=("Human", "Assistant"),
        prompt="Human: {query}\n\nAssistant: </s>",
        sep="</s>",
    )
)

"""Cohere template
source: https://huggingface.co/CohereForAI/c4ai-command-r-plus
Supports: https://huggingface.co/CohereForAI/c4ai-command-r-plus-4bit
          https://huggingface.co/CohereForAI/c4ai-command-r-plus
"""
register_conv_template(
    Conversation(
        name="cohere",
        system_prompt="<BOS_TOKEN>",
        messages=[],
        roles=("User", "Assistant"),
        prompt=(
            "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{query}<|END_OF_TURN_TOKEN|>"
            "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
        ),
        sep="</s>",
    )
)

"""Qwen template (ChatML format)
通用 Qwen 系列模板，使用 ChatML 格式。

ChatML 格式示例：
  <|im_start|>system
  You are a helpful assistant.<|im_end|>
  <|im_start|>user
  {query}<|im_end|>
  <|im_start|>assistant
  {response}<|im_end|>

本项目使用 qwen3_5 模板（Qwen3.5 也是 ChatML 格式，字段一致）。
"""
register_conv_template(
    Conversation(
        name="qwen",
        system_prompt="<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n",
        messages=[],
        roles=("user", "assistant"),
        prompt="<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n",
        sep="\n",
        stop_str="<|im_end|>",
    )
)

"""Qwen3 template (ChatML format, same as Qwen2.5)
Supports: https://huggingface.co/Qwen/Qwen3-8B
          https://huggingface.co/Qwen/Qwen3-4B
          https://huggingface.co/Qwen/Qwen3-32B
For thinking models, the model will auto-generate <think>...</think> blocks.
Use qwen3_nothink for non-thinking mode.
"""
register_conv_template(
    Conversation(
        name="qwen3",
        system_prompt="<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n",
        messages=[],
        roles=("user", "assistant"),
        prompt="<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n",
        sep="\n",
        stop_str="<|im_end|>",
    )
)

"""Qwen3 no-think template
Use this template to disable thinking mode for Qwen3 models.
Add '/no_think' or set enable_thinking=False to disable thinking.
"""
register_conv_template(
    Conversation(
        name="qwen3_nothink",
        system_prompt="<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n",
        messages=[],
        roles=("user", "assistant"),
        prompt="<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n",
        sep="\n",
        stop_str="<|im_end|>",
    )
)

"""Qwen3.5 template (ChatML format with thinking support)
本项目实际使用的模板。Qwen3.5 使用与 Qwen 系列相同的 ChatML 格式。
基座模型 Qwen3.5-2B 位于 /root/autodl-tmp/models/Qwen3.5-2B
"""
register_conv_template(
    Conversation(
        name="qwen3_5",
        system_prompt="<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n",
        messages=[],
        roles=("user", "assistant"),
        prompt="<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n",
        sep="\n",
        stop_str="<|im_end|>",
    )
)


def get_conv_template(name: str) -> Conversation:
    """按名称获取已注册的对话模板。例如 get_conv_template("qwen3_5")"""
    return conv_templates[name]
