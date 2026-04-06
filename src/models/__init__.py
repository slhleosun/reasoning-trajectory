"""Model adapters and interfaces"""

from .base import ModelAdapter, ModelOutput
from .huggingface import HuggingFaceAdapter
from .openai_adapter import OpenAIAdapter
from .factory import get_model_adapter

__all__ = [
    "ModelAdapter",
    "ModelOutput",
    "HuggingFaceAdapter",
    "OpenAIAdapter",
    "get_model_adapter",
]
