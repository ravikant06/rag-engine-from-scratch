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

from src import trace
from src.llm.types import LLMResponse, Message, Role, ToolSpec


class LLMAdapter(ABC):
    """Chat completion with optional tool calling, in provider-neutral terms."""

    #: Registry key, e.g. "gemini". Set by each concrete adapter.
    provider: ClassVar[str]

    def __init__(self, model: str, api_key: str) -> None:
        self.model = model
        self.api_key = api_key

    def complete(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSpec] = (),
        system: str | None = None,
    ) -> LLMResponse:
        """
        Template Method: trace the call, delegate translation to _complete().

        Every adapter gets identical logging for free, and the tracing code
        lives in one place rather than being copied per provider.
        """
        if not trace.is_on():
            return self._complete(messages, tools=tools, system=system)

        trace.section(f"LLM CALL -> {self.provider} / {self.model}")
        if system:
            trace.body("system instruction", system, limit=700)
        trace.bullets(
            f"messages ({len(messages)})",
            [self._describe(m) for m in messages],
        )
        trace.kv("tools offered", ", ".join(t.name for t in tools) or "(none)")

        with trace.timed() as elapsed:
            reply = self._complete(messages, tools=tools, system=system)

        if reply.wants_tools:
            summary = f"{len(reply.tool_calls)} tool call(s)"
            trace.result(summary, elapsed[0])
            for call in reply.tool_calls:
                trace.bullets(
                    "  requested",
                    [f"{call.name}({trace.compact_json(call.arguments)})"],
                )
        else:
            trace.result("text answer", elapsed[0])
            trace.body("  answer", reply.text or "", limit=700)

        usage = trace.usage_of(reply.raw)
        if usage:
            trace.kv("tokens", usage)
        return reply

    @staticmethod
    def _describe(message: Message) -> str:
        """One-line rendering of a turn, for the trace."""
        if message.role is Role.TOOL:
            result = message.tool_result
            return (
                f"[tool:{result.name}] -> "
                f"{result.content.get('count', '?')} result(s)"
            )
        if message.tool_calls:
            calls = ", ".join(
                f"{c.name}({trace.compact_json(c.arguments, limit=160)})"
                for c in message.tool_calls
            )
            return f"[{message.role.value}] calls: {calls}"
        text = (message.text or "").replace("\n", " ")
        return f"[{message.role.value}] {text[:120]}"

    @abstractmethod
    def _complete(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSpec] = (),
        system: str | None = None,
    ) -> LLMResponse:
        """
        Send a conversation, get one reply. Implementations must:
          - translate `messages` into the provider's own turn format
          - translate `tools` into the provider's tool envelope
          - return either text or tool_calls, never provider objects
        """

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} model={self.model!r}>"
