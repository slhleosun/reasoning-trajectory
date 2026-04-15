"""Model adapter factory"""

from typing import Any, Dict, Optional

from .base import ModelAdapter
from .huggingface import HuggingFaceAdapter
from ..config import get_model_config


def get_model_adapter(
    model_name: str,
    config: Optional[Dict[str, Any]] = None,
) -> ModelAdapter:
    """Factory function to get appropriate model adapter

    Args:
        model_name: Name of the model from configuration
        config: Optional model-specific configuration overrides

    Returns:
        ModelAdapter instance

    Raises:
        ValueError: If model type is not supported
    """
    model_config = get_model_config(model_name)
    model_type = model_config.model_type.lower()

    if model_type == "huggingface":
        return HuggingFaceAdapter(model_name, config)
    else:
        raise ValueError(
            f"Unsupported model type: {model_type}. "
            f"Supported types: huggingface"
        )
