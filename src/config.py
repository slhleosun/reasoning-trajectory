"""Configuration management for datasets and models"""

from __future__ import annotations
import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass


@dataclass
class DatasetConfig:
    """Dataset configuration"""
    name: str
    description: str
    path: str
    huggingface_id: str
    split: Dict[str, str]
    cache_dir: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DatasetConfig:
        """Create DatasetConfig from dictionary"""
        return cls(
            name=data["name"],
            description=data["description"],
            path=data["path"],
            huggingface_id=data["huggingface_id"],
            split=data["split"],
            cache_dir=data["cache_dir"],
        )


@dataclass
class ModelConfig:
    """Model configuration"""
    name: str
    description: str
    model_type: str
    path: Optional[str] = None
    huggingface_id: Optional[str] = None
    model_id: Optional[str] = None
    cache_dir: Optional[str] = None
    tokenizer_path: Optional[str] = None
    config: Dict[str, Any] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ModelConfig:
        """Create ModelConfig from dictionary"""
        return cls(
            name=data["name"],
            description=data["description"],
            model_type=data["model_type"],
            path=data.get("path"),
            huggingface_id=data.get("huggingface_id"),
            model_id=data.get("model_id"),
            cache_dir=data.get("cache_dir"),
            tokenizer_path=data.get("tokenizer_path"),
            config=data.get("config", {}),
        )


class Config:
    """Main configuration class"""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration

        Args:
            config_path: Path to configuration YAML file. If None, uses default path.
        """
        if config_path is None:
            # Default to config/paths.yaml in the project root
            project_root = Path(__file__).parent.parent
            config_path = project_root / "config" / "paths.yaml"

        self.config_path = Path(config_path)
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    def get_dataset_config(self, dataset_name: str) -> DatasetConfig:
        """Get dataset configuration by name

        Args:
            dataset_name: Name of the dataset (e.g., 'gsm8k')

        Returns:
            DatasetConfig object

        Raises:
            KeyError: If dataset not found in configuration
        """
        if dataset_name not in self._config["datasets"]:
            available = list(self._config["datasets"].keys())
            raise KeyError(
                f"Dataset '{dataset_name}' not found in configuration. "
                f"Available datasets: {available}"
            )

        return DatasetConfig.from_dict(self._config["datasets"][dataset_name])

    def get_model_config(self, model_name: str) -> ModelConfig:
        """Get model configuration by name

        Args:
            model_name: Name of the model (e.g., 'llama-3.1-8b')

        Returns:
            ModelConfig object

        Raises:
            KeyError: If model not found in configuration
        """
        if model_name not in self._config["models"]:
            available = list(self._config["models"].keys())
            raise KeyError(
                f"Model '{model_name}' not found in configuration. "
                f"Available models: {available}"
            )

        return ModelConfig.from_dict(self._config["models"][model_name])

    def get_output_dir(self, output_type: str) -> Path:
        """Get output directory path

        Args:
            output_type: Type of output (e.g., 'results', 'trajectories', 'logs')

        Returns:
            Path to output directory
        """
        project_root = Path(__file__).parent.parent
        output_path = project_root / self._config["output"][output_type]
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path

    def get_settings(self) -> Dict[str, Any]:
        """Get general settings"""
        return self._config["settings"]

    def list_datasets(self) -> list[str]:
        """List all available datasets"""
        return list(self._config["datasets"].keys())

    def list_models(self) -> list[str]:
        """List all available models"""
        return list(self._config["models"].keys())


# Global configuration instance
_global_config: Optional[Config] = None


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration (singleton pattern)

    Args:
        config_path: Path to configuration file. If None, uses default.

    Returns:
        Config instance
    """
    global _global_config
    if _global_config is None or config_path is not None:
        _global_config = Config(config_path)
    return _global_config


def get_dataset_config(dataset_name: str) -> DatasetConfig:
    """Convenience function to get dataset configuration

    Args:
        dataset_name: Name of the dataset

    Returns:
        DatasetConfig object
    """
    config = load_config()
    return config.get_dataset_config(dataset_name)


def get_model_config(model_name: str) -> ModelConfig:
    """Convenience function to get model configuration

    Args:
        model_name: Name of the model

    Returns:
        ModelConfig object
    """
    config = load_config()
    return config.get_model_config(model_name)
