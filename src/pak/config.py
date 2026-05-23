"""Environment variables, paths, and constant definitions."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PAKSettings(BaseSettings):
    """Global settings for the PAK project."""

    model_config = SettingsConfigDict(
        env_prefix="PAK_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    project_root: Path = Field(default_factory=Path.cwd)
    data_root: Path = Path("data")

    pdf_dpi: int = 300
    pdf_render_format: str = "png"

    # Same stack as Nemotron-Personas-Korea (Apache-2.0 open model + OpenAI-compatible inference).
    # PAK v0.1 starts with the Ollama native API + qwen3:8b (driver compatibility).
    # When using an OpenAI-compatible backend such as vLLM/NIM, change llm_backend to "openai_compatible".
    llm_budget_usd: float = 0.0  # 0 means only self-hosting is allowed
    # vLLM attempt: Qwen3.5-9B bf16 (18GB) + KV cache conflict caused OOM on a 24GB 4090.
    # Conclusion: stick with Ollama qwen3:8b Q4 (~5GB) — verified on Pilot 100, 12.8s per item.
    llm_default_model: str = "qwen3:8b"
    llm_judge_model: str = "qwen3:8b"
    llm_backend: str = "ollama_native"
    llm_base_url: str = "http://localhost:11434"
    llm_api_key: str = "EMPTY"
    llm_max_concurrent: int = 4

    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    nvidia_api_key: str | None = Field(default=None, alias="NVIDIA_API_KEY")
    openrouter_api_key: str | None = Field(default=None, alias="OPENROUTER_API_KEY")
    hf_token: str | None = Field(default=None, alias="HF_TOKEN")
    zenodo_token: str | None = Field(default=None, alias="ZENODO_TOKEN")

    @property
    def source_dir(self) -> Path:
        return self.data_root / "source"

    @property
    def pages_dir(self) -> Path:
        return self.data_root / "pages"

    @property
    def extracted_dir(self) -> Path:
        return self.data_root / "extracted"

    @property
    def grounding_dir(self) -> Path:
        return self.data_root / "grounding"

    @property
    def prompts_dir(self) -> Path:
        return self.data_root / "prompts"

    @property
    def synthetic_dir(self) -> Path:
        return self.data_root / "synthetic"

    @property
    def outputs_dir(self) -> Path:
        return self.project_root / "outputs"


@lru_cache(maxsize=1)
def get_settings() -> PAKSettings:
    return PAKSettings()


settings = get_settings()
