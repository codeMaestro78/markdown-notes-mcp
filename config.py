"""
Advanced Configuration management for markdown-notes-mcp project.

This module provides centralized configuration for all components of the system,
with support for environment variables, multiple models, dynamic chunking,
custom preprocessing pipelines, and environment-specific settings.
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, List, Any, Callable
from dataclasses import dataclass, field
from enum import Enum


class ModelProvider(Enum):
    """Supported embedding model providers."""
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    COHERE = "cohere"


class ChunkingStrategy(Enum):
    """Available chunking strategies."""
    FIXED_SIZE = "fixed_size"
    SENTENCE_BASED = "sentence_based"
    HEADING_BASED = "heading_based"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


class Environment(Enum):
    """Deployment environments."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


@dataclass
class ModelConfig:
    """Configuration for embedding models."""
    provider: ModelProvider
    model_name: str
    dimensions: int
    max_seq_length: int = 512
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    model_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkingConfig:
    """Configuration for text chunking strategies."""
    strategy: ChunkingStrategy
    chunk_size: int = 150
    overlap: int = 30
    min_chunk_size: int = 50
    max_chunk_size: int = 500
    preserve_headings: bool = True
    semantic_threshold: float = 0.7
    custom_splitters: List[str] = field(default_factory=lambda: ["\n## ", "\n### ", "\n#### "])
    sentence_splitter: str = "[.!?]+\\s+"


@dataclass
class PreprocessingConfig:
    """Configuration for text preprocessing pipelines."""
    enabled: bool = True
    remove_frontmatter: bool = True
    normalize_unicode: bool = True
    remove_code_blocks: bool = False
    remove_links: bool = True
    remove_tables: bool = False
    custom_filters: List[Callable] = field(default_factory=list)
    language: str = "en"


@dataclass
class SearchConfig:
    """Configuration for search behavior."""
    default_top_k: int = 5
    max_top_k: int = 20
    lexical_weight: float = 0.3
    semantic_weight: float = 0.7
    rerank_results: bool = True
    diversity_threshold: float = 0.8
    boost_recent: bool = False
    recent_boost_days: int = 30


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization."""
    use_faiss: bool = True
    faiss_index_type: str = "IndexFlatIP"
    cache_embeddings: bool = True
    preload_models: bool = False
    max_workers: int = 4
    memory_limit_mb: int = 2048
    enable_gpu: bool = False


class AdvancedConfig:
    """
    Advanced configuration class with multiple models, dynamic chunking,
    custom preprocessing, and environment-specific settings.
    """

    def __init__(self):
        # Environment detection
        self.environment = self._detect_environment()

        # Load base configuration
        self._load_base_config()

        # Load environment-specific configuration
        self._load_environment_config()

        # Initialize advanced components
        self.models = self._initialize_models()
        self.chunking = self._initialize_chunking()
        self.preprocessing = self._initialize_preprocessing()
        self.search = self._initialize_search()
        self.performance = self._initialize_performance()

    def _detect_environment(self) -> Environment:
        """Detect the current deployment environment."""
        env_var = os.getenv("MCP_ENVIRONMENT", "").lower()
        env_mapping = {
            "dev": Environment.DEVELOPMENT,
            "development": Environment.DEVELOPMENT,
            "staging": Environment.STAGING,
            "prod": Environment.PRODUCTION,
            "production": Environment.PRODUCTION,
            "test": Environment.TESTING,
            "testing": Environment.TESTING,
        }
        return env_mapping.get(env_var, Environment.DEVELOPMENT)

    def _load_base_config(self):
        """Load basic configuration settings."""
        # Model settings
        self.model_name = os.getenv("MCP_MODEL_NAME", "all-MiniLM-L6-v2")

        # Chunking settings
        self.chunk_size = int(os.getenv("MCP_CHUNK_SIZE", "150"))
        self.overlap = int(os.getenv("MCP_OVERLAP", "30"))
        self.batch_size = int(os.getenv("MCP_BATCH_SIZE", "64"))

        # File paths
        self.index_file = os.getenv("MCP_INDEX_FILE", "notes_index.npz")
        self.meta_file = os.getenv("MCP_META_FILE", "notes_meta.json")
        self.notes_root = Path(os.getenv("MCP_NOTES_ROOT", "./notes"))
        self.config_dir = Path(os.getenv("MCP_CONFIG_DIR", "./config"))

        # Server settings
        self.host = os.getenv("MCP_HOST", "127.0.0.1")
        self.port = int(os.getenv("MCP_PORT", "8181"))
        self.admin_token = os.getenv("MCP_ADMIN_TOKEN", "Devarshi")

        # Advanced settings
        self.max_file_size = int(os.getenv("MCP_MAX_FILE_SIZE", "10485760"))  # 10MB
        self.request_timeout = int(os.getenv("MCP_REQUEST_TIMEOUT", "30"))

    def _load_environment_config(self):
        """Load environment-specific configuration."""
        config_file = self.config_dir / f"{self.environment.value}.json"
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    env_config = json.load(f)
                # Override base config with environment-specific values
                for key, value in env_config.items():
                    if hasattr(self, key):
                        setattr(self, key, value)
            except Exception as e:
                print(f"Warning: Could not load environment config {config_file}: {e}")

    def _initialize_models(self) -> Dict[str, ModelConfig]:
        """Initialize multiple model configurations."""
        models = {}

        # Default Sentence Transformers model
        models["default"] = ModelConfig(
            provider=ModelProvider.SENTENCE_TRANSFORMERS,
            model_name=self.model_name,
            dimensions=384,
            max_seq_length=512
        )

        # High-quality model for production
        models["high_quality"] = ModelConfig(
            provider=ModelProvider.SENTENCE_TRANSFORMERS,
            model_name="all-mpnet-base-v2",
            dimensions=768,
            max_seq_length=512
        )

        # Fast model for development
        models["fast"] = ModelConfig(
            provider=ModelProvider.SENTENCE_TRANSFORMERS,
            model_name="all-MiniLM-L6-v2",
            dimensions=384,
            max_seq_length=256
        )

        # Multilingual model
        models["multilingual"] = ModelConfig(
            provider=ModelProvider.SENTENCE_TRANSFORMERS,
            model_name="paraphrase-multilingual-MiniLM-L12-v2",
            dimensions=384,
            max_seq_length=512
        )

        # Load custom models from config
        custom_models_file = self.config_dir / "custom_models.json"
        if custom_models_file.exists():
            try:
                with open(custom_models_file, 'r', encoding='utf-8') as f:
                    custom_models = json.load(f)
                for name, model_data in custom_models.items():
                    models[name] = ModelConfig(**model_data)
            except Exception as e:
                print(f"Warning: Could not load custom models: {e}")

        return models

    def _initialize_chunking(self) -> Dict[str, ChunkingConfig]:
        """Initialize chunking strategy configurations."""
        chunking = {}

        # Fixed size chunking (default)
        chunking["fixed"] = ChunkingConfig(
            strategy=ChunkingStrategy.FIXED_SIZE,
            chunk_size=self.chunk_size,
            overlap=self.overlap
        )

        # Sentence-based chunking
        chunking["sentence"] = ChunkingConfig(
            strategy=ChunkingStrategy.SENTENCE_BASED,
            chunk_size=200,
            overlap=50,
            sentence_splitter=r"[.!?]+\s+"
        )

        # Heading-based chunking
        chunking["heading"] = ChunkingConfig(
            strategy=ChunkingStrategy.HEADING_BASED,
            preserve_headings=True,
            custom_splitters=["\n# ", "\n## ", "\n### ", "\n#### ", "\n##### "]
        )

        # Hybrid chunking
        chunking["hybrid"] = ChunkingConfig(
            strategy=ChunkingStrategy.HYBRID,
            chunk_size=150,
            overlap=30,
            preserve_headings=True,
            semantic_threshold=0.7
        )

        return chunking

    def _initialize_preprocessing(self) -> PreprocessingConfig:
        """Initialize preprocessing pipeline configuration."""
        return PreprocessingConfig(
            enabled=True,
            remove_frontmatter=True,
            normalize_unicode=True,
            remove_code_blocks=self.environment == Environment.PRODUCTION,
            remove_links=True,
            remove_tables=False,
            language="en"
        )

    def _initialize_search(self) -> SearchConfig:
        """Initialize search configuration."""
        return SearchConfig(
            default_top_k=5,
            max_top_k=20,
            lexical_weight=0.3,
            semantic_weight=0.7,
            rerank_results=True,
            diversity_threshold=0.8,
            boost_recent=self.environment == Environment.PRODUCTION,
            recent_boost_days=30
        )

    def _initialize_performance(self) -> PerformanceConfig:
        """Initialize performance configuration."""
        return PerformanceConfig(
            use_faiss=True,
            faiss_index_type="IndexFlatIP",
            cache_embeddings=True,
            preload_models=self.environment == Environment.PRODUCTION,
            max_workers=4 if self.environment == Environment.PRODUCTION else 2,
            memory_limit_mb=4096 if self.environment == Environment.PRODUCTION else 2048,
            enable_gpu=os.getenv("MCP_ENABLE_GPU", "false").lower() == "true"
        )

    def get_model_config(self, model_name: str = "default") -> ModelConfig:
        """Get configuration for a specific model."""
        return self.models.get(model_name, self.models["default"])

    def get_chunking_config(self, strategy: str = "fixed") -> ChunkingConfig:
        """Get configuration for a specific chunking strategy."""
        return self.chunking.get(strategy, self.chunking["fixed"])

    def validate(self) -> bool:
        """Validate all configuration values."""
        try:
            # Validate chunk settings
            if self.chunk_size <= 0 or self.overlap < 0:
                raise ValueError("Invalid chunk settings")

            if self.overlap >= self.chunk_size:
                raise ValueError("Overlap must be less than chunk size")

            # Validate batch size
            if self.batch_size <= 0:
                raise ValueError("Batch size must be positive")

            # Validate port
            if not (1 <= self.port <= 65535):
                raise ValueError("Port must be between 1 and 65535")

            # Validate model configurations
            for name, model in self.models.items():
                if not model.model_name:
                    raise ValueError(f"Model {name} has no model_name")

            return True

        except Exception as e:
            print(f"Configuration validation failed: {e}")
            return False

    def to_dict(self) -> dict:
        """Convert configuration to dictionary for serialization."""
        return {
            "environment": self.environment.value,
            "model_name": self.model_name,
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "batch_size": self.batch_size,
            "index_file": self.index_file,
            "meta_file": self.meta_file,
            "notes_root": str(self.notes_root),
            "config_dir": str(self.config_dir),
            "host": self.host,
            "port": self.port,
            "max_file_size": self.max_file_size,
            "request_timeout": self.request_timeout,
            "models": {name: {
                "provider": model.provider.value,
                "model_name": model.model_name,
                "dimensions": model.dimensions,
                "max_seq_length": model.max_seq_length
            } for name, model in self.models.items()},
            "chunking_strategies": list(self.chunking.keys()),
            "preprocessing": {
                "enabled": self.preprocessing.enabled,
                "language": self.preprocessing.language
            },
            "search": {
                "default_top_k": self.search.default_top_k,
                "lexical_weight": self.search.lexical_weight
            },
            "performance": {
                "use_faiss": self.performance.use_faiss,
                "max_workers": self.performance.max_workers,
                "enable_gpu": self.performance.enable_gpu
            }
        }

    def save_config(self, filepath: Optional[Path] = None):
        """Save current configuration to file."""
        if filepath is None:
            filepath = self.config_dir / "current_config.json"

        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    def load_config(self, filepath: Path):
        """Load configuration from file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            config_data = json.load(f)

        # Update current configuration
        for key, value in config_data.items():
            if hasattr(self, key):
                setattr(self, key, value)


# Global configuration instance
config = AdvancedConfig()

# Validate configuration on import
if not config.validate():
    raise RuntimeError("Invalid configuration. Please check your environment variables and config files.")

class Config:
    def __init__(self):
        self.index_file = "notes_index.npz"
        self.meta_file = "notes_meta.json"
        self.notes_root = Path("./notes")
        self.model_name = "all-MiniLM-L6-v2"

config = Config()
