from typing import Optional
from class_registry import ClassRegistry

from GDesigner.llm.llm import LLM
# 确保两个模型实现都已导入
#from GDesigner.llm.gpt_chat import GPTChat
#from GDesigner.llm.qianwen_chat import QianwenChat


class LLMRegistry:
    registry = ClassRegistry()

    @classmethod
    def register(cls, *args, **kwargs):
        return cls.registry.register(*args, **kwargs)

    @classmethod
    def keys(cls):
        return cls.registry.keys()

    @classmethod
    def get(cls, model_name: Optional[str] = None) -> LLM:
        # 1. 默认模型为通义千问
        if model_name is None or model_name == "":
            model_name = "qwen-turbo"

        # 2. 根据模型名称关键字，精确匹配对应的处理类
        if model_name == 'mock':
            model = cls.registry.get(model_name)
        elif 'qwen' in model_name:
            model = cls.registry.get('QianwenChat', model_name)
        elif 'gpt' in model_name:
            model = cls.registry.get('GPTChat', model_name)
        else:
            # 3. 对于任何无法识别的模型名称，直接抛出异常
            raise ValueError(
                f"Unsupported model name: '{model_name}'. "
                f"Please use a model name containing 'qwen' or 'gpt', "
                f"or add support for the new model in GDesigner/llm/llm_registry.py"
            )

        return model
from GDesigner.llm.gpt_chat import GPTChat
from GDesigner.llm.qianwen_chat import QianwenChat

# 2. 手动进行注册
LLMRegistry.register('GPTChat')(GPTChat)
LLMRegistry.register('QianwenChat')(QianwenChat)