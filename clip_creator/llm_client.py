"""Talks to Claude or OpenAI. Simple if/else, no abstractions."""

from __future__ import annotations

from clip_creator.config import Config
from clip_creator.models import LLMError


class LLMClient:
    def __init__(self, config: Config) -> None:
        self.provider = config.llm.provider
        self.model = config.llm.model
        self.temperature = config.llm.temperature

        if self.provider == "anthropic":
            import anthropic

            self._anthropic = anthropic.Anthropic(api_key=config.anthropic_api_key)
        elif self.provider == "openai":
            import openai

            self._openai = openai.OpenAI(api_key=config.openai_api_key)
        else:
            raise LLMError(f"Unknown LLM provider: {self.provider}")

    @property
    def model_id(self) -> str:
        return f"{self.provider}/{self.model}"

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Send a prompt, get text back."""
        if self.provider == "anthropic":
            return self._complete_anthropic(system_prompt, user_prompt)
        else:
            return self._complete_openai(system_prompt, user_prompt)

    def _complete_anthropic(self, system_prompt: str, user_prompt: str) -> str:
        response = self._anthropic.messages.create(
            model=self.model,
            max_tokens=4096,
            temperature=self.temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text

    def _complete_openai(self, system_prompt: str, user_prompt: str) -> str:
        response = self._openai.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content
