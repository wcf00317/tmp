# 文件: GDesigner/llm/qianwen_chat.py

import os
from typing import List, Union, Optional, Dict
from tenacity import retry, wait_random_exponential, stop_after_attempt
from dotenv import load_dotenv
from openai import AsyncOpenAI

from GDesigner.llm.format import Message
from GDesigner.llm.llm import LLM
# 注意：我们遵循最终的方案，不在该文件中导入和使用 LLMRegistry

# --- 配置加载 ---
# 从您指定的 'qwen.env' 文件加载配置
load_dotenv(dotenv_path='qwen.env')

DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY')
DASHSCOPE_BASE_URL = os.getenv('DASHSCOPE_BASE_URL')

if not DASHSCOPE_API_KEY:
    raise ValueError("请在 qwen.env 文件中设置您的 DASHSCOPE_API_KEY")
if not DASHSCOPE_BASE_URL:
    DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    print(f"Warning: DASHSCOPE_BASE_URL not found in qwen.env, using default: {DASHSCOPE_BASE_URL}")

client = AsyncOpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL,timeout=60.0, max_retries=10 )

# --- API 调用函数 ---
@retry(wait=wait_random_exponential(max=60), stop=stop_after_attempt(10))
async def achat(
        model: str,
        msg: List[Dict],
        max_tokens: int,
        temperature: float,
        num_comps: int,
):
    """
    使用 openai SDK 异步调用千问模型。
    """
    try:
        print(">>> 开始调用 Qwen API ...")
        completion = await client.chat.completions.create(
            model=model,
            messages=msg,
            max_tokens=max_tokens,
            temperature=temperature,
            n=num_comps,
        )
        print(f"--- QWEN API RESPONSE ---\n{completion}\n-------------------------")
    except Exception as e:
        print(f"!!! Qwen API 调用失败: {repr(e)}")
        raise
    if num_comps == 1:
        return completion.choices[0].message.content
    else:
        return [choice.message.content for choice in completion.choices]

# --- LLM 实现类 (干净版本) ---
class QianwenChat(LLM):
    """
    通义千问大语言模型
    """
    def __init__(self, model_name: str):
        self.model_name = model_name

    async def agen(
            self,
            messages: List[Message],
            max_tokens: Optional[int] = None,
            temperature: Optional[float] = None,
            num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS

        msg_dicts = []
        if messages:  # 确保 messages 列表不为空
            # --- ↓↓↓ 最终的、唯一的正确修复逻辑 ↓↓↓ ---
            # 检查列表中的第一个元素，以确定如何处理整个列表
            if isinstance(messages[0], dict):
                # 如果是字典列表，直接使用（因为 achat 就需要字典列表）
                msg_dicts = messages
            elif isinstance(messages[0], Message):
                # 如果是 Message 对象列表，则转换为字典列表
                msg_dicts = [{"role": m.role, "content": m.content} for m in messages]
            # --- ↑↑↑ 修复逻辑结束 ↑↑↑ ---


        return await achat(
            self.model_name,
            msg_dicts,
            max_tokens,
            temperature,
            num_comps,
        )

    def gen(
            self,
            messages: List[Message],
            max_tokens: Optional[int] = None,
            temperature: Optional[float] = None,
            num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        # 同步方法可以留空
        pass