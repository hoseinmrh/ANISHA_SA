"""Provider-neutral contracts for chat models with tool support."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Protocol, Sequence


JsonObject = dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    """A normalized model request to invoke one named tool."""

    id: str
    name: str
    arguments: JsonObject


@dataclass(frozen=True)
class ChatMessage:
    """A provider-independent chat message."""

    role: str
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = field(default_factory=tuple)
    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None


@dataclass(frozen=True)
class LLMResponse:
    """Normalized output from an LLM provider."""

    content: str = ""
    tool_calls: tuple[ToolCall, ...] = field(default_factory=tuple)
    thinking: Optional[str] = None


class LLMClient(Protocol):
    """Interface implemented by every ANISHA model provider adapter."""

    model: str

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Optional[Sequence[Mapping[str, Any]]] = None,
        response_schema: Optional[Mapping[str, Any]] = None,
    ) -> LLMResponse:
        """Generate the next assistant message."""

