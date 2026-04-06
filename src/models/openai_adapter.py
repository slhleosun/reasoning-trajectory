"""OpenAI model adapter (including CloudGPT)"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import sys
from pathlib import Path

from .base import ModelAdapter, ModelOutput
from ..config import get_model_config


class OpenAIAdapter(ModelAdapter):
    """Adapter for OpenAI models (including CloudGPT OpenAI)"""

    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None):
        """Initialize OpenAI adapter

        Args:
            model_name: Name of the model from configuration
            config: Optional model-specific configuration overrides
        """
        super().__init__(model_name, config)
        self.model_config = get_model_config(model_name)

        # Merge configurations
        self.generation_config = {**self.model_config.config, **(config or {})}

        # Get model ID (e.g., "gpt-4o-20241120")
        self.model_id = self.model_config.model_id
        if not self.model_id:
            raise ValueError(
                f"OpenAI model {model_name} requires 'model_id' in configuration"
            )

    def load(self):
        """Load the OpenAI client"""
        # Check if we should use CloudGPT or standard OpenAI
        project_root = Path(__file__).parent.parent.parent
        cloudgpt_module = project_root / "cloudgpt_oaai.py"

        if cloudgpt_module.exists():
            # Use CloudGPT
            print(f"Loading CloudGPT OpenAI client for model: {self.model_id}")
            try:
                # Add parent directory to path to import cloudgpt_oaai
                sys.path.insert(0, str(project_root))
                import cloudgpt_oaai

                self._client = cloudgpt_oaai.get_openai_client()
                print(f"CloudGPT client loaded successfully")
            except Exception as e:
                raise RuntimeError(f"Failed to load CloudGPT client: {e}")
            finally:
                # Clean up path
                if str(project_root) in sys.path:
                    sys.path.remove(str(project_root))
        else:
            # Use standard OpenAI
            print(f"Loading standard OpenAI client for model: {self.model_id}")
            try:
                from openai import OpenAI

                api_key = self.generation_config.get("api_key")
                if not api_key:
                    raise ValueError(
                        "OpenAI API key required. Set in config or OPENAI_API_KEY env var"
                    )

                self._client = OpenAI(api_key=api_key)
                print(f"OpenAI client loaded successfully")
            except ImportError:
                raise ImportError(
                    "openai library required for OpenAI models. "
                    "Install with: pip install openai"
                )

        self._model = self._client  # Set model to client for is_loaded check

        # For OpenAI, we'll use tiktoken for tokenization if available
        try:
            import tiktoken

            self._tokenizer = tiktoken.encoding_for_model(self.model_id)
        except Exception as e:
            print(f"Warning: Could not load tokenizer: {e}")
            self._tokenizer = None

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
            max_length: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            **kwargs: Additional generation parameters

        Returns:
            ModelOutput object
        """
        if not self.is_loaded:
            raise RuntimeError("Client not loaded. Call load() first.")

        # Prepare API parameters
        api_params = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_length or self.generation_config.get("max_tokens", 4096),
            "temperature": temperature or self.generation_config.get("temperature", 0.7),
            "top_p": top_p or self.generation_config.get("top_p", 0.9),
            **kwargs,
        }

        # Request logprobs if available
        if "logprobs" not in api_params:
            api_params["logprobs"] = True

        try:
            # Make API call
            response = self._client.chat.completions.create(**api_params)

            # Extract response
            choice = response.choices[0]
            generated_text = choice.message.content

            # Extract logprobs and tokens if available
            tokens = None
            logprobs = None
            if hasattr(choice, "logprobs") and choice.logprobs:
                try:
                    tokens = [t.token for t in choice.logprobs.content]
                    logprobs = [t.logprob for t in choice.logprobs.content]
                except Exception as e:
                    print(f"Warning: Could not extract logprobs: {e}")

            return ModelOutput(
                text=generated_text,
                tokens=tokens,
                logprobs=logprobs,
                metadata={
                    "model_name": self.model_name,
                    "model_id": self.model_id,
                    "finish_reason": choice.finish_reason,
                    "usage": response.usage.dict() if hasattr(response, "usage") else None,
                },
                raw_output=response,
            )

        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {e}")

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
            max_length: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            **kwargs: Additional generation parameters

        Returns:
            List of ModelOutput objects
        """
        # OpenAI API doesn't support true batching, so we'll call sequentially
        results = []
        for i, prompt in enumerate(prompts):
            print(f"Generating {i+1}/{len(prompts)}...")
            output = self.generate(
                prompt=prompt,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                **kwargs,
            )
            # Add batch index to metadata
            output.metadata["batch_index"] = i
            results.append(output)

        return results

    def encode(self, text: str) -> List[int]:
        """Encode text to token ids

        Args:
            text: Input text

        Returns:
            List of token ids
        """
        if self._tokenizer is None:
            # Fallback: rough estimation (4 chars per token)
            return list(range(len(text) // 4))

        return self._tokenizer.encode(text)

    def decode(self, token_ids: List[int]) -> str:
        """Decode token ids to text

        Args:
            token_ids: List of token ids

        Returns:
            Decoded text
        """
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not available for this model")

        return self._tokenizer.decode(token_ids)
