from __future__ import annotations

from typing import Any, AsyncIterator

from claude_code_sdk import AssistantMessage, ClaudeCodeOptions, query

# SDK v0.0.25 raises MessageParseError for unknown CLI event types (e.g. rate_limit_event
# emitted by CLI >= 2.1.x). Patch parse_message to return None instead so we can skip
# unknown events and continue consuming the stream rather than aborting the whole query.
try:
    from claude_code_sdk._internal import client as _sdk_client
    from claude_code_sdk._internal.message_parser import parse_message as _orig_parse
    from claude_code_sdk._errors import MessageParseError as _MPE

    def _lenient_parse(data: dict) -> object | None:  # type: ignore[misc]
        try:
            return _orig_parse(data)
        except _MPE:
            return None  # Unknown CLI event (e.g. rate_limit_event) — skip instead of crash

    _sdk_client.parse_message = _lenient_parse  # type: ignore[attr-defined]
except Exception:
    pass


async def _stream_prompt(text: str):
    """Wrap a string as an AsyncIterable so stdin stays open for control responses.

    In string mode the claude subprocess stdin is closed immediately, which
    breaks the bidirectional control channel needed for in-process MCP tools.
    Streaming mode keeps stdin alive so _handle_control_request can write back.
    """
    yield {"type": "user", "message": {"role": "user", "content": text}}


class SdkAgent:
    """Wraps claude-code-sdk query() with a LangGraph-compatible streaming interface.

    Provides .ainvoke() and .astream() that match the signatures used by
    analyst/server.py and technician/server.py, so no server changes are needed.
    """

    def __init__(self, options: ClaudeCodeOptions) -> None:
        self.options = options

    def _user_text(self, input: dict[str, Any]) -> str:
        messages = input.get("messages", [])
        if not messages:
            return ""
        last = messages[-1]
        if hasattr(last, "content"):
            content = last.content
            return content if isinstance(content, str) else str(content)
        if isinstance(last, dict):
            return last.get("content", "")
        return str(last)

    async def ainvoke(self, input: dict[str, Any], config: dict | None = None) -> dict[str, Any]:
        messages = input.get("messages", [])
        parts: list[str] = []
        gen = query(prompt=_stream_prompt(self._user_text(input)), options=self.options)
        try:
            async for msg in gen:
                if msg is None:  # Unknown event type patched to None — skip
                    continue
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if hasattr(block, "text") and block.text:
                            parts.append(block.text)
        finally:
            await gen.aclose()
        return {"messages": list(messages) + [{"role": "assistant", "content": "".join(parts)}]}

    async def astream(
        self, input: dict[str, Any], config: dict | None = None, **kwargs: Any
    ) -> AsyncIterator[tuple[Any, dict]]:
        from langchain_core.messages import AIMessageChunk, ToolMessage

        gen = query(prompt=_stream_prompt(self._user_text(input)), options=self.options)
        try:
            async for msg in gen:
                if msg is None:  # Unknown event type — skip
                    continue
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if hasattr(block, "text") and block.text:
                            yield (AIMessageChunk(content=block.text), {})
                        elif hasattr(block, "name"):  # ToolUseBlock
                            yield (ToolMessage(content=f"[tool: {block.name}]", tool_call_id="sdk"), {})
        finally:
            await gen.aclose()
