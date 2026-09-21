"""
The Target interface of the Adapter pattern.

    Client   -> agent.py, which knows only this interface
    Target   -> LLMAdapter (below)
    Adaptee  -> google-genai / openai / anthropic SDKs
    Adapter  -> GeminiAdapter, OpenAIAdapter, AnthropicAdapter

Deliberately narrow (Interface Segregation): this describes chat completion
with tools and nothing else. Embeddings are a different capability with a
different shape, so they would get their own `EmbeddingAdapter` rather than
being bolted on here.
"""
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import ClassVar

from src.llm.types import LLMResponse, Message, ToolSpec


class LLMAdapter(ABC):
    """Chat completion with optional tool calling, in provider-neutral terms."""

    #: Registry key, e.g. "gemini". Set by each concrete adapter.
    provider: ClassVar[str]

    def __init__(self, model: str, api_key: str) -> None:
        self.model = model
        self.api_key = api_key

    @abstractmethod
    def complete(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSpec] = (),
        system: str | None = None,
    ) -> LLMResponse:
        """
        Send a conversation, get one reply.

        Implementations must:
          - translate `messages` into the provider's own turn format
          - translate `tools` into the provider's tool envelope
          - return either text or tool_calls, never provider objects
        """

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} model={self.model!r}>"
