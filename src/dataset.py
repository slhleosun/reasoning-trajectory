"""Dataset loading and preparation utilities"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
import json

try:
    from datasets import load_dataset, Dataset, DatasetDict
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    Dataset = None
    DatasetDict = None

from .config import load_config, DatasetConfig


@dataclass
class DataSample:
    """Single data sample"""
    id: str
    question: str
    answer: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "question": self.question,
            "answer": self.answer,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DataSample:
        """Create from dictionary"""
        return cls(
            id=data.get("id", ""),
            question=data["question"],
            answer=data["answer"],
            metadata=data.get("metadata", {}),
        )


class DatasetLoader:
    """Dataset loader with support for multiple formats"""

    def __init__(self, dataset_name: str, config_path: Optional[str] = None):
        """Initialize dataset loader

        Args:
            dataset_name: Name of dataset (e.g., 'gsm8k')
            config_path: Optional path to configuration file
        """
        self.dataset_name = dataset_name
        self.config = load_config(config_path)
        self.dataset_config = self.config.get_dataset_config(dataset_name)
        self._dataset = None

    def load(self, split: str = "train", force_reload: bool = False) -> List[DataSample]:
        """Load dataset

        Args:
            split: Dataset split to load ('train', 'test', etc.)
            force_reload: Force reload from source

        Returns:
            List of DataSample objects
        """
        if not force_reload and self._dataset is not None:
            return self._dataset

        # Check if dataset is already downloaded locally
        dataset_path = Path(self.dataset_config.path)
        local_file = dataset_path / f"{split}.jsonl"

        if local_file.exists() and not force_reload:
            print(f"Loading {self.dataset_name} from local file: {local_file}")
            samples = self._load_from_jsonl(local_file)
        elif HF_AVAILABLE:
            print(f"Loading {self.dataset_name} from HuggingFace: {self.dataset_config.huggingface_id}")
            samples = self._load_from_huggingface(split)
            # Cache locally
            self._save_to_jsonl(samples, local_file)
        else:
            raise RuntimeError(
                "Dataset not found locally and HuggingFace datasets library is not available. "
                "Install with: pip install datasets"
            )

        self._dataset = samples
        return samples

    def _load_from_huggingface(self, split: str) -> List[DataSample]:
        """Load dataset from HuggingFace"""
        if not HF_AVAILABLE:
            raise ImportError("datasets library not available")

        # Get the actual split name from config
        actual_split = self.dataset_config.split.get(split, split)

        # Parse dataset ID and config name
        # Format can be: "username/dataset" or "username/dataset:config_name"
        dataset_id = self.dataset_config.huggingface_id
        config_name = None

        if ":" in dataset_id:
            dataset_id, config_name = dataset_id.split(":", 1)

        # For GSM8K specifically, default to 'main' config if not specified
        if "gsm8k" in dataset_id.lower() and config_name is None:
            config_name = "main"

        # Load dataset with optional cache_dir
        load_args = [dataset_id]
        if config_name is not None:
            load_args.append(config_name)

        load_kwargs = {
            "split": actual_split,
        }

        if self.dataset_config.cache_dir is not None:
            cache_dir = Path(self.dataset_config.cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            load_kwargs["cache_dir"] = str(cache_dir)

        dataset = load_dataset(
            *load_args,
            **load_kwargs,
        )

        # Convert to our format
        samples = []
        for idx, item in enumerate(dataset):
            samples.append(self._convert_to_sample(item, idx))

        return samples

    def _convert_to_sample(self, item: Dict[str, Any], idx: int) -> DataSample:
        """Convert dataset item to DataSample

        This method should be customized for each dataset format
        """
        # GSM8K format
        if self.dataset_name == "gsm8k":
            return DataSample(
                id=f"gsm8k_{idx}",
                question=item.get("question", ""),
                answer=item.get("answer", ""),
                metadata={"original": item},
            )

        # MATH-500 format
        if self.dataset_name == "math-500":
            return DataSample(
                id=item.get("unique_id", f"math500_{idx}").replace("/", "_").replace(".json", ""),
                question=item.get("problem", ""),
                answer=item.get("answer", ""),
                metadata={
                    "solution": item.get("solution", ""),
                    "subject": item.get("subject", ""),
                    "level": item.get("level", ""),
                    "original": item,
                },
            )

        # Generic format
        return DataSample(
            id=item.get("id", f"{self.dataset_name}_{idx}"),
            question=item.get("question", item.get("input", "")),
            answer=item.get("answer", item.get("output", "")),
            metadata={"original": item},
        )

    def _load_from_jsonl(self, file_path: Path) -> List[DataSample]:
        """Load dataset from JSONL file"""
        samples = []
        with open(file_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if line.strip():
                    data = json.loads(line)
                    # Check if it's already in DataSample format (has 'question' and 'answer' keys)
                    if "question" in data and "answer" in data:
                        samples.append(DataSample.from_dict(data))
                    else:
                        # Convert from raw format (e.g., MATH-500 with 'problem' key)
                        samples.append(self._convert_to_sample(data, idx))
        return samples

    def _save_to_jsonl(self, samples: List[DataSample], file_path: Path):
        """Save dataset to JSONL file"""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample.to_dict()) + "\n")
        print(f"Cached dataset to {file_path}")

    def get_sample(self, idx: int, split: str = "train") -> DataSample:
        """Get a single sample by index

        Args:
            idx: Index of sample
            split: Dataset split

        Returns:
            DataSample object
        """
        samples = self.load(split)
        return samples[idx]

    def __len__(self) -> int:
        """Get dataset size"""
        if self._dataset is None:
            return 0
        return len(self._dataset)

    def __getitem__(self, idx: int) -> DataSample:
        """Get sample by index"""
        if self._dataset is None:
            raise RuntimeError("Dataset not loaded. Call load() first.")
        return self._dataset[idx]


def prepare_dataset(
    dataset_name: str,
    split: str = "train",
    config_path: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> List[DataSample]:
    """Prepare dataset for use

    Args:
        dataset_name: Name of dataset
        split: Dataset split to load
        config_path: Optional configuration file path
        max_samples: Maximum number of samples to load (for debugging)

    Returns:
        List of DataSample objects
    """
    loader = DatasetLoader(dataset_name, config_path)
    samples = loader.load(split)

    if max_samples is not None:
        samples = samples[:max_samples]

    print(f"Loaded {len(samples)} samples from {dataset_name} ({split} split)")
    return samples


def create_dummy_gsm8k_dataset(output_dir: Optional[Path] = None):
    """Create a dummy GSM8K dataset for testing

    Args:
        output_dir: Output directory. If None, uses config path.
    """
    if output_dir is None:
        config = load_config()
        dataset_config = config.get_dataset_config("gsm8k")
        output_dir = Path(dataset_config.path)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy samples
    dummy_samples = [
        DataSample(
            id="gsm8k_0",
            question="Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
            answer="Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs every day.\nShe makes 9 * 2 = $<<9*2=18>>18 every day at the farmers' market.\n#### 18",
            metadata={"source": "dummy", "difficulty": "easy"},
        ),
        DataSample(
            id="gsm8k_1",
            question="A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
            answer="It takes 2/2=<<2/2=1>>1 bolt of white fiber\nSo the total amount of fabric is 2+1=<<2+1=3>>3 bolts of fiber\n#### 3",
            metadata={"source": "dummy", "difficulty": "easy"},
        ),
        DataSample(
            id="gsm8k_2",
            question="Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
            answer="The cost of the house and repairs came out to 80,000+50,000=$<<80000+50000=130000>>130,000\nHe increased the value of the house by 80,000*1.5=<<80000*1.5=120000>>120,000\nSo the new value of the house is 120,000+80,000=$<<120000+80000=200000>>200,000\nSo he made a profit of 200,000-130,000=$<<200000-130000=70000>>70,000\n#### 70000",
            metadata={"source": "dummy", "difficulty": "medium"},
        ),
    ]

    # Save train and test splits
    for split in ["train", "test"]:
        output_file = output_dir / f"{split}.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for sample in dummy_samples:
                f.write(json.dumps(sample.to_dict()) + "\n")

        print(f"Created dummy {split} dataset at {output_file}")


if __name__ == "__main__":
    # Create dummy dataset if run directly
    print("Creating dummy GSM8K dataset...")
    create_dummy_gsm8k_dataset()
