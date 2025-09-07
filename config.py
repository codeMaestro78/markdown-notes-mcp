"""
Configuration management for markdown-notes-mcp project.

This module provides centralized configuration for all components of the system,
with support for environment variables and default values.
"""

import os
from pathlib import Path
from typing import Optional


class Config:
    """
    Centralized configuration class for the markdown-notes-mcp project.

    All settings can be overridden using environment variables.
    """

    def __init__(self):
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

        # Server settings
        self.host = os.getenv("MCP_HOST", "127.0.0.1")
        self.port = int(os.getenv("MCP_PORT", "8181"))
        self.admin_token = os.getenv("MCP_ADMIN_TOKEN", "Devarshi")

        # Advanced settings
        self.max_file_size = int(os.getenv("MCP_MAX_FILE_SIZE", "10485760"))  # 10MB
        self.request_timeout = int(os.getenv("MCP_REQUEST_TIMEOUT", "30"))

    def validate(self) -> bool:
        """
        Validate configuration values.

        Returns:
            bool: True if configuration is valid
        """
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

            return True

        except Exception as e:
            print(f"Configuration validation failed: {e}")
            return False

    def to_dict(self) -> dict:
        """
        Convert configuration to dictionary for serialization.

        Returns:
            dict: Configuration as dictionary
        """
        return {
            "model_name": self.model_name,
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "batch_size": self.batch_size,
            "index_file": self.index_file,
            "meta_file": self.meta_file,
            "notes_root": str(self.notes_root),
            "host": self.host,
            "port": self.port,
            "max_file_size": self.max_file_size,
            "request_timeout": self.request_timeout
        }


# Global configuration instance
config = Config()

# Validate configuration on import
if not config.validate():
    raise RuntimeError("Invalid configuration. Please check your environment variables.")
