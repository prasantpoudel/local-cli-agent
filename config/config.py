from pathlib import Path
from typing import Optional, Any
import os

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    name: str = "mistralai/devstral-2512:free"
    temperature: float = Field(default=1, ge=0.0, le=2.0)
    context_window: int = 250_000


class Config(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    cwd: Path = Field(default_factory=Path.cwd())

    max_tunrs: int = 100
    max_tool_output_tokens: int = 50_000

    developer_instructions: Optional[str] = None
    uers_instruction: Optional[str] = None

    debug: bool = False

    @property
    def api_key(self) -> Optional[str]:
        return os.environ.get("API_KEY")

    @property
    def base_url(self) -> Optional[str]:
        return os.environ.get("BASE_URL")

    @property
    def model_name(self) -> str:
        return self.model.name

    @model_name.setter
    def model_name(self, value: str) -> None:
        self.model.name = value

    @property
    def temperature(self) -> float:
        return self.model.temperature

    @temperature.setter
    def temperature(self, value: str) -> None:
        self.model.temperature = value

    def validate(self) -> list[str]:
        errors: list[str] = []

        if not self.api_key:
            errors.append("No API key found. Set API_KEY environment variable")

        if not self.base_url:
            errors.append("No base url available.")

        if not self.cwd.exists():
            errors.append(f"Working directory does not exist: {self.cwd}")

        return errors

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
