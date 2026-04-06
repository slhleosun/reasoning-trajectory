"""HuggingFace model adapter"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from pathlib import Path
import sys

from .base import ModelAdapter, ModelOutput
from ..config import get_model_config
from .greedy_generate_twopass import greedy_generate_with_artifacts_twopass
from .batch_greedy_generate_twopass import batch_greedy_generate_with_artifacts_twopass
from .generation_output import CompleteGenerationOutput


class HuggingFaceAdapter(ModelAdapter):
    """Adapter for HuggingFace transformers models"""

    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None):
        """Initialize HuggingFace adapter

        Args:
            model_name: Name of the model from configuration
            config: Optional model-specific configuration overrides
        """
        super().__init__(model_name, config)
        self.model_config = get_model_config(model_name)

        # Merge configurations (config overrides model_config)
        self.generation_config = {**self.model_config.config, **(config or {})}

    def load(self):
        """Load the HuggingFace model and tokenizer"""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers and torch are required for HuggingFace models. "
                "Install with: pip install transformers torch"
            )

        print(f"Loading model: {self.model_name}")

        # Determine model path (local or HuggingFace Hub)
        model_path = self.model_config.path
        if not Path(model_path).exists():
            # Try HuggingFace Hub ID
            if self.model_config.huggingface_id:
                model_path = self.model_config.huggingface_id
            else:
                raise ValueError(
                    f"Model path not found: {model_path} and no HuggingFace ID specified"
                )

        # Load tokenizer
        tokenizer_path = self.model_config.tokenizer_path or model_path
        print(f"Loading tokenizer from: {tokenizer_path}")

        tokenizer_kwargs = {}
        if self.model_config.cache_dir is not None:
            tokenizer_kwargs["cache_dir"] = self.model_config.cache_dir

        self._tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            **tokenizer_kwargs,
        )

        # Set padding token if not set
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Set left padding for batch generation (decoder-only models)
        self._tokenizer.padding_side = "left"

        # Load model with configuration
        print(f"Loading model from: {model_path}")

        # Parse torch dtype
        torch_dtype = self.generation_config.get("torch_dtype", "float16")
        if torch_dtype == "float16":
            dtype = torch.float16
        elif torch_dtype == "bfloat16":
            dtype = torch.bfloat16
        elif torch_dtype == "float32":
            dtype = torch.float32
        else:
            dtype = torch.float16

        load_kwargs = {
            "torch_dtype": dtype,
            "device_map": self.generation_config.get("device_map", "auto"),
        }

        if self.model_config.cache_dir is not None:
            load_kwargs["cache_dir"] = self.model_config.cache_dir

        # Add quantization options if specified
        if self.generation_config.get("load_in_8bit", False):
            load_kwargs["load_in_8bit"] = True
        elif self.generation_config.get("load_in_4bit", False):
            load_kwargs["load_in_4bit"] = True

        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **load_kwargs,
        ).eval()  # Set to evaluation mode for inference

        # Verify model is on GPU if CUDA is available
        print(f"Model loaded successfully: {self.model_name}")
        print(f"Model device: {self._model.device}")
        print(f"Model dtype: {self._model.dtype}")

        # CRITICAL: Check if model ended up on CPU when CUDA is available
        if torch.cuda.is_available() and self._model.device.type == "cpu":
            print(f"WARNING: Model loaded on CPU despite CUDA being available!")
            print(f"  device_map setting: {self.generation_config.get('device_map', 'auto')}")
            print(f"  Attempting to move model to cuda:0...")
            try:
                self._model = self._model.to("cuda:0")
                print(f"  Model successfully moved to: {self._model.device}")
            except Exception as e:
                print(f"  ERROR: Failed to move model to GPU: {e}")
                raise

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
            **kwargs: Additional generation parameters

        Returns:
            ModelOutput object
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        try:
            import torch
        except ImportError:
            raise ImportError("torch is required")

        # Prepare generation config for deterministic decoding
        do_sample = self.generation_config.get("do_sample", False)

        gen_config = {
            "max_length": max_length or self.generation_config.get("max_length", 4096),
            "do_sample": do_sample,
            "use_cache": self.generation_config.get("use_cache", True),
            "pad_token_id": self._tokenizer.eos_token_id,  # Suppress warning
            **kwargs,
        }

        # Only add temperature/top_p if sampling is explicitly enabled
        if do_sample:
            temperature_val = temperature if temperature is not None else self.generation_config.get("temperature", 0.7)
            top_p_val = top_p if top_p is not None else self.generation_config.get("top_p", 0.9)
            gen_config["temperature"] = temperature_val
            gen_config["top_p"] = top_p_val

        # Tokenize input
        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                **gen_config,
                return_dict_in_generate=True,
                output_scores=True,
            )

        # Decode output
        generated_ids = outputs.sequences[0]
        generated_text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Remove prompt from output
        generated_text = generated_text[len(prompt):].strip()

        # Extract tokens and logprobs if available
        tokens = None
        logprobs = None
        if hasattr(outputs, "scores") and outputs.scores:
            try:
                # Get top token at each step
                tokens = [
                    self._tokenizer.decode([generated_ids[i]])
                    for i in range(len(generated_ids))
                ]
                # Get logprobs (simplified)
                logprobs = [
                    torch.log_softmax(score, dim=-1).max().item()
                    for score in outputs.scores
                ]
            except Exception as e:
                print(f"Warning: Could not extract tokens/logprobs: {e}")

        return ModelOutput(
            text=generated_text,
            tokens=tokens,
            logprobs=logprobs,
            metadata={
                "model_name": self.model_name,
                "prompt_length": len(inputs["input_ids"][0]),
                "output_length": len(generated_ids),
            },
            raw_output=outputs,
        )

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
            **kwargs: Additional generation parameters

        Returns:
            List of ModelOutput objects
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        try:
            import torch
        except ImportError:
            raise ImportError("torch is required")

        # Prepare generation config for deterministic decoding
        do_sample = self.generation_config.get("do_sample", False)

        gen_config = {
            "max_length": max_length or self.generation_config.get("max_length", 4096),
            "do_sample": do_sample,
            "use_cache": self.generation_config.get("use_cache", True),
            "pad_token_id": self._tokenizer.eos_token_id,
            **kwargs,
        }

        # Only add temperature/top_p if sampling is explicitly enabled
        if do_sample:
            temperature_val = temperature if temperature is not None else self.generation_config.get("temperature", 0.7)
            top_p_val = top_p if top_p is not None else self.generation_config.get("top_p", 0.9)
            gen_config["temperature"] = temperature_val
            gen_config["top_p"] = top_p_val

        # Get max padding length from config (for prompt truncation/padding)
        max_padding_length = self.generation_config.get("max_padding_length", None)

        # Tokenize inputs with left padding
        # padding=True pads to max length in batch
        # max_length limits individual prompt length (truncates if longer)
        tokenizer_kwargs = {
            "return_tensors": "pt",
            "padding": True,
            "padding_side": "left",
            "truncation": True,
        }
        if max_padding_length is not None:
            tokenizer_kwargs["max_length"] = max_padding_length

        inputs = self._tokenizer(prompts, **tokenizer_kwargs)
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        # Get max prompt length for logging
        max_prompt_length = inputs["input_ids"].shape[1]

        # Generate
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                **gen_config,
                return_dict_in_generate=True,
            )

        # Decode outputs
        results = []
        for i, (prompt, generated_ids) in enumerate(zip(prompts, outputs.sequences)):
            generated_text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
            # Remove prompt from output
            generated_text = generated_text[len(prompt):].strip()

            results.append(
                ModelOutput(
                    text=generated_text,
                    metadata={
                        "model_name": self.model_name,
                        "batch_index": i,
                        "prompt_length": max_prompt_length,  # Max prompt length in batch
                        "output_length": len(generated_ids),
                    },
                    raw_output=generated_ids,
                )
            )

        return results

    def generate_with_complete_artifacts(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        gold_answer_token_id: Optional[int] = None,
    ) -> CompleteGenerationOutput:
        """Generate with complete per-timestep artifact capture

        Args:
            prompt: Input prompt string
            max_new_tokens: Maximum tokens to generate
            gold_answer_token_id: Optional first token ID of gold answer

        Returns:
            CompleteGenerationOutput with all artifacts
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Tokenize
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Call two-pass greedy generation
        output = greedy_generate_with_artifacts_twopass(
            model=self._model,
            tokenizer=self._tokenizer,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            gold_answer_token_id=gold_answer_token_id,
            capture_hidden_states=True,
            pad_token_id=self._tokenizer.pad_token_id,
        )

        return output

    def generate_batch_with_complete_artifacts(
        self,
        prompts: List[str],
        max_new_tokens: int = 512,
        gold_answer_token_ids: Optional[List[Optional[int]]] = None,
        max_input_length: int = 250,
    ) -> List[CompleteGenerationOutput]:
        """Generate batch with complete per-timestep artifact capture

        Args:
            prompts: List of input prompt strings
            max_new_tokens: Maximum tokens to generate
            gold_answer_token_ids: Optional list of gold answer first token IDs
            max_input_length: Maximum input length (tokens) with padding (default: 250, MATH: 824)

        Returns:
            List of CompleteGenerationOutput with all artifacts
        """
        import os
        print(f"[ADAPTER PID={os.getpid()}] generate_batch_with_complete_artifacts called", flush=True)
        print(f"[ADAPTER PID={os.getpid()}]   Batch size: {len(prompts)}", flush=True)
        print(f"[ADAPTER PID={os.getpid()}]   Max new tokens: {max_new_tokens}", flush=True)
        print(f"[ADAPTER PID={os.getpid()}]   Max input length: {max_input_length}", flush=True)
        sys.stdout.flush()

        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        try:
            # Tokenize batch with left padding to fixed length
            print(f"[ADAPTER PID={os.getpid()}] Tokenizing {len(prompts)} prompts with left padding (max_length={max_input_length})...", flush=True)
            sys.stdout.flush()

            inputs = self._tokenizer(
                prompts,
                return_tensors="pt",
                padding="max_length",  # Pad to max_length
                max_length=max_input_length,  # Configurable padding length
                truncation=True,
            )
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            print(f"[ADAPTER PID={os.getpid()}] Tokenization complete. Input shape: {input_ids.shape} (padded to {max_input_length})", flush=True)
            sys.stdout.flush()

            # Call two-pass batched greedy generation
            print(f"[ADAPTER PID={os.getpid()}] Calling batch_greedy_generate_with_artifacts_twopass...", flush=True)
            sys.stdout.flush()

            outputs = batch_greedy_generate_with_artifacts_twopass(
                model=self._model,
                tokenizer=self._tokenizer,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                gold_answer_token_ids=gold_answer_token_ids,
                capture_hidden_states=True,
                pad_token_id=self._tokenizer.pad_token_id,
            )

            print(f"[ADAPTER PID={os.getpid()}] batch_greedy_generate_with_artifacts_twopass returned {len(outputs)} outputs", flush=True)
            sys.stdout.flush()

            return outputs

        except Exception as e:
            print(f"\n{'='*80}", flush=True)
            print(f"❌❌❌ ERROR IN ADAPTER generate_batch_with_complete_artifacts ❌❌❌", flush=True)
            print(f"PID={os.getpid()}", flush=True)
            print(f"Error type: {type(e).__name__}", flush=True)
            print(f"Error message: {e}", flush=True)
            print(f"{'='*80}\n", flush=True)
            sys.stdout.flush()
            sys.stderr.flush()
            raise
