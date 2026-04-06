"""Base model adapter interface"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union


@dataclass
class ModelOutput:
    """Model output container"""
    text: str
    logprobs: Optional[List[float]] = None
    tokens: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    raw_output: Optional[Any] = None

    def __str__(self) -> str:
        return self.text


class ModelAdapter(ABC):
    """Abstract base class for model adapters"""

    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None):
        """Initialize model adapter

        Args:
            model_name: Name of the model from configuration
            config: Optional model-specific configuration
        """
        self.model_name = model_name
        self.config = config or {}
        self._model = None
        self._tokenizer = None

    @abstractmethod
    def load(self):
        """Load the model and tokenizer"""
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs,
    ) -> ModelOutput:
        """Generate text from prompt

        Args:
            prompt: Input text prompt
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            **kwargs: Additional model-specific parameters

        Returns:
            ModelOutput object containing generated text and metadata
        """
        pass

    @abstractmethod
    def generate_batch(
        self,
        prompts: List[str],
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs,
    ) -> List[ModelOutput]:
        """Generate text from multiple prompts

        Args:
            prompts: List of input text prompts
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            **kwargs: Additional model-specific parameters

        Returns:
            List of ModelOutput objects
        """
        pass

    def encode(self, text: str) -> List[int]:
        """Encode text to token ids

        Args:
            text: Input text

        Returns:
            List of token ids
        """
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call load() first.")
        return self._tokenizer.encode(text)

    def decode(self, token_ids: List[int]) -> str:
        """Decode token ids to text

        Args:
            token_ids: List of token ids

        Returns:
            Decoded text
        """
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call load() first.")
        return self._tokenizer.decode(token_ids)

    def count_tokens(self, text: str) -> int:
        """Count number of tokens in text

        Args:
            text: Input text

        Returns:
            Number of tokens
        """
        return len(self.encode(text))

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._model is not None

    def unload(self):
        """Unload model to free memory"""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

    def __enter__(self):
        """Context manager entry"""
        if not self.is_loaded:
            self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.unload()
        return False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_name='{self.model_name}', loaded={self.is_loaded})"
