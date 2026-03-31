from __future__ import annotations

import os
from typing import Any, Dict, List, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


class LLMClient:
    def __init__(self) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def build_chat_model(self, temperature: float = 0.0) -> ChatOpenAI:
        if not self.enabled:
            raise RuntimeError("OPENAI_API_KEY is not configured")
        return ChatOpenAI(
            model=self.model,
            temperature=temperature,
            api_key=cast(Any, self.api_key),
            base_url=self.base_url,
            timeout=30,
        )

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.0) -> str:
        lc_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            else:
                lc_messages.append(HumanMessage(content=content))

        llm = self.build_chat_model(temperature=temperature)
        response = llm.invoke(lc_messages)
        text = response.content
        if isinstance(text, list):
            text = "".join(str(x) for x in text)
        return str(text).strip()
