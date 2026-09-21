"""
Proof that the Adapter pattern actually bought us something.

`agent.answer()` is driven here by a FakeAdapter — no network, no API key, no
provider SDK. If agent.py contained any Gemini-specific code this test could
not exist, which is the point: the Client depends on the abstraction only.

Run:  python tests/test_adapter.py     (plain asserts; pytest not required)
"""
import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm import LLMAdapter, LLMResponse, Message, ToolCall, ToolSpec  # noqa: E402
from src.llm.registry import available  # noqa: E402


class FakeAdapter(LLMAdapter):
    """Replays a scripted list of LLMResponses and records what it was sent."""

    provider = "fake"

    def __init__(self, replies: list[LLMResponse]) -> None:
        super().__init__(model="fake-model", api_key="none")
        self._replies = list(replies)
        self.calls: list[list[Message]] = []

    def _complete(self, messages: Sequence[Message], *, tools: Sequence[ToolSpec] = (),
                 system: str | None = None) -> LLMResponse:
        self.calls.append(list(messages))
        return self._replies.pop(0)


def test_all_providers_registered():
    for name in ("gemini", "openai", "anthropic"):
        assert name in available(), f"{name} not registered"


def test_tool_spec_is_plain_json_schema():
    from src.agent import SEARCH_DOCS

    assert isinstance(SEARCH_DOCS, ToolSpec)
    assert SEARCH_DOCS.parameters["type"] == "object"          # lowercase = JSON Schema
    assert "query" in SEARCH_DOCS.parameters["required"]
    # tenant_id must never be model-selectable
    assert "tenant_id" not in SEARCH_DOCS.parameters["properties"]


def test_agent_runs_without_any_provider_sdk():
    """One tool call, then an answer — the whole loop, no network."""
    from src import agent

    fake = FakeAdapter([
        LLMResponse(tool_calls=(ToolCall(id="1", name="search_docs",
                                         arguments={"query": "replicas",
                                                    "doc_type": "guide"}),)),
        LLMResponse(text="order-service runs 4 replicas.\nSources: deployment.md"),
    ])
    chunks, answer, trace = agent.answer("how many replicas?", llm=fake)

    assert answer.startswith("order-service runs 4 replicas")
    assert len(trace) == 1
    assert trace[0]["where"] == {"doc_type": "guide"}
    assert len(chunks) == 1
    # Second call must carry the tool result back to the model.
    assert len(fake.calls[1]) == 3            # user, assistant(tool_call), tool
    assert fake.calls[1][2].tool_result.content["count"] == 1


def test_retry_after_empty_result():
    """Model sees zero hits and searches again — the agentic behaviour."""
    from src import agent

    fake = FakeAdapter([
        LLMResponse(tool_calls=(ToolCall(id="1", name="search_docs",
                                         arguments={"query": "x", "severity": "SEV-1"}),)),
        LLMResponse(tool_calls=(ToolCall(id="2", name="search_docs",
                                         arguments={"query": "x"}),)),
        LLMResponse(text="I could not find this in the documentation."),
    ])
    _, answer, trace = agent.answer("any SEV-1?", llm=fake)

    assert [step["count"] for step in trace] == [0, 1]
    assert "could not find" in answer


def _fake_chunk(i: int = 1) -> dict:
    return {"chunk_id": f"c{i}", "source": "deployment.md", "heading": "Other services",
            "chunk_index": i, "text": "order-service runs 4 replicas", "score": 0.8}


def main() -> None:
    from src import agent

    results = []

    test_all_providers_registered(); results.append("providers registered")
    test_tool_spec_is_plain_json_schema(); results.append("tool spec is plain JSON Schema")

    # Stub retrieval so no Qdrant or embedding call is made.
    agent._run_search = lambda args, tenant, k: [_fake_chunk()]
    test_agent_runs_without_any_provider_sdk(); results.append("loop runs with fake adapter")

    calls = {"n": 0}

    def empty_then_hit(args, tenant, k):
        calls["n"] += 1
        return [] if calls["n"] == 1 else [_fake_chunk()]

    agent._run_search = empty_then_hit
    test_retry_after_empty_result(); results.append("retries after empty result")

    for line in results:
        print(f"  PASS  {line}")
    print(f"\n{len(results)} passed")


if __name__ == "__main__":
    main()
