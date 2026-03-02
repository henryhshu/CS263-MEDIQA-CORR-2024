"""
Concrete LLM Provider Implementations.

Currently includes:
    - GeminiProvider: Google Gemini models
    - OpenAIProvider: OpenAI GPT models (for teammate compatibility)
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from pipeline.base import LLMProvider

logger = logging.getLogger(__name__)


class GeminiProvider(LLMProvider):
    """
    LLM provider for Google Gemini models.

    Uses the google-generativeai SDK. Requires GOOGLE_API_KEY
    in environment or a .env file.

    Usage:
        provider = GeminiProvider(model_name="gemini-2.5-flash")
        response = provider.generate("Check this text for errors", system_prompt="...")
    """

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        env_path: Optional[str] = None,
    ):
        import google.generativeai as genai

        # Load API key
        if env_path:
            load_dotenv(env_path)
        else:
            # Try common locations
            for candidate in [
                Path.cwd() / ".env",
                Path(__file__).parent.parent / ".env",
            ]:
                if candidate.exists():
                    load_dotenv(str(candidate))
                    break

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found. Set it in your .env file or environment."
            )

        genai.configure(api_key=api_key)
        self._model_name = model_name
        self._model = genai.GenerativeModel(model_name)
        logger.info(f"GeminiProvider initialized: {model_name}")

    def generate(self, user_prompt: str, system_prompt: str = "") -> str:
        """Generate a response using Gemini."""
        full_prompt = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
        response = self._model.generate_content(full_prompt)
        return response.text.strip() if response.text else ""

    @property
    def name(self) -> str:
        return f"gemini-{self._model_name}"


class OpenAIProvider(LLMProvider):
    """
    LLM provider for OpenAI GPT models.

    Uses the openai SDK. Requires OPENAI_API_KEY in environment.

    Usage:
        provider = OpenAIProvider(model_name="gpt-4o")
        response = provider.generate("Check this text for errors", system_prompt="...")
    """

    def __init__(
        self,
        model_name: str = "gpt-4o",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 600,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package not installed. Install with: pip install openai"
            )

        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            # Try loading from file
            key_path = Path.home() / "env" / "openai_secret_key.txt"
            if key_path.exists():
                key = key_path.read_text(encoding="utf-8").strip()

        if not key:
            raise ValueError(
                "OPENAI_API_KEY not found. Set it in environment or pass api_key=."
            )

        self._client = OpenAI(api_key=key)
        self._model_name = model_name
        self._temperature = temperature
        self._max_tokens = max_tokens
        logger.info(f"OpenAIProvider initialized: {model_name}")

    def generate(self, user_prompt: str, system_prompt: str = "") -> str:
        """Generate a response using OpenAI."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        return response.choices[0].message.content.strip()

    @property
    def name(self) -> str:
        return f"openai-{self._model_name}"
