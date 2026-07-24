import os
import yaml
from typing import Literal
from pathlib import Path
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    source: str
    processed: str


class DoclingConfig(BaseModel):
    use_flash_attention: bool = Field(default=False)
    device: Literal["cuda", "cpu"] = Field(default="cuda")


class EmbeddingConfig(BaseModel):
    model: str
    max_tokens: int


class LLMConfig(BaseModel):
    model: str


class VLMConfig(BaseModel):
    model: str
    scale: float
    picture_area_threshold: float


class AppConfig(BaseModel):
    data: DataConfig
    docling: DoclingConfig
    embedding: EmbeddingConfig
    llm: LLMConfig
    vlm: VLMConfig


class MainConfig:
    """Main configuration class to hold all settings"""

    def __init__(self):
        self.PACKAGE_ROOT = self._get_file_path().parent
        self.ROOT = self.PACKAGE_ROOT.parent.parent
        self.CONFIG_FILE_PATH = self.ROOT / "configs" / "main.yaml"
        self.ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
        self.raw_config = self._fetch_config_from_yaml(file_path=self.CONFIG_FILE_PATH)
        self.app = AppConfig(**self.raw_config)

    def _get_file_path(self) -> Path:
        return Path(__file__).resolve()

    def _fetch_config_from_yaml(self, file_path: Path) -> dict:
        """Parse YAML containing the package configuration"""
        with open(file_path, "r") as yaml_file:
            parsed_config = yaml.safe_load(yaml_file)
            return parsed_config
