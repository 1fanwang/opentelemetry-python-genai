# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import claude_agent_sdk
import anyio
import pytest
from claude_agent_sdk import (
    CLIConnectionError,
    ClaudeAgentOptions,
    ResultMessage,
    Transport,
    UserMessage,
)
from opentelemetry.instrumentation.genai.claude_agent_sdk import (
    ClaudeAgentSDKInstrumentor,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.test_util_genai.instrumentor import instrument
from opentelemetry.trace import SpanKind, StatusCode


class _QueryTransport(Transport):
    def __init__(
        self,
        messages: list[dict[str, Any]],
        *,
        error: BaseException | None = None,
        wait_after_messages: bool = False,
    ) -> None:
        self._messages = messages
        self._error = error
        self._wait_after_messages = wait_after_messages
        self._writes: list[str] = []
        self._ready = False

    async def connect(self) -> None:
        self._ready = True

    async def write(self, data: str) -> None:
        self._writes.append(data)

    def read_messages(self) -> AsyncIterator[dict[str, Any]]:
        async def _read() -> AsyncIterator[dict[str, Any]]:
            while not self._writes:
                await anyio.sleep(0)

            request = json.loads(self._writes[0])
            yield {
                "type": "control_response",
                "response": {
                    "subtype": "success",
                    "request_id": request["request_id"],
                    "response": {},
                },
            }
            for message in self._messages:
                yield message
            if self._error is not None:
                raise self._error
            if self._wait_after_messages:
                await anyio.sleep_forever()

        return _read()

    async def close(self) -> None:
        self._ready = False

    def is_ready(self) -> bool:
        return self._ready

    async def end_input(self) -> None:
        return None


def _result_message() -> dict[str, Any]:
    return {
        "type": "result",
        "subtype": "success",
        "duration_ms": 12,
        "duration_api_ms": 10,
        "is_error": False,
        "num_turns": 1,
        "session_id": "session-123",
        "total_cost_usd": 0.001,
    }


def _user_message() -> dict[str, Any]:
    return {
        "type": "user",
        "message": {"role": "user", "content": "hello"},
        "parent_tool_use_id": None,
        "session_id": "session-123",
    }


async def _prompt() -> AsyncIterator[dict[str, Any]]:
    yield _user_message()


@pytest.mark.anyio
async def test_query_emits_parented_agent_span(
    tracer_provider,
    logger_provider,
    meter_provider,
    span_exporter,
) -> None:
    transport = _QueryTransport([_result_message()])
    tracer = tracer_provider.get_tracer(__name__)

    with instrument(
        ClaudeAgentSDKInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    ):
        with tracer.start_as_current_span("parent"):
            stream = claude_agent_sdk.query(
                prompt=_prompt(),
                options=ClaudeAgentOptions(model="sonnet"),
                transport=transport,
            )
            assert stream.__class__.__name__ == "async_generator"
            messages = [message async for message in stream]

    assert len(messages) == 1
    assert isinstance(messages[0], ResultMessage)

    spans = span_exporter.get_finished_spans()
    parent = next(span for span in spans if span.name == "parent")
    invocation = next(span for span in spans if span.name == "invoke_agent")
    assert invocation.kind == SpanKind.INTERNAL
    assert invocation.context.trace_id == parent.context.trace_id
    assert invocation.parent.span_id == parent.context.span_id
    assert invocation.attributes[GenAI.GEN_AI_OPERATION_NAME] == "invoke_agent"
    assert GenAI.GEN_AI_AGENT_NAME not in invocation.attributes
    assert invocation.attributes[GenAI.GEN_AI_REQUEST_MODEL] == "sonnet"
    assert invocation.attributes[GenAI.GEN_AI_CONVERSATION_ID] == "session-123"
    assert invocation.attributes[GenAI.GEN_AI_RESPONSE_FINISH_REASONS] == (
        "success",
    )
    assert invocation.status.status_code == StatusCode.UNSET


@pytest.mark.anyio
async def test_query_reraises_sdk_error_once(
    tracer_provider,
    logger_provider,
    meter_provider,
    span_exporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = CLIConnectionError("connection lost")

    async def _failing_query(**_: Any) -> AsyncIterator[Any]:
        raise error
        yield

    monkeypatch.setattr(claude_agent_sdk, "query", _failing_query)

    with instrument(
        ClaudeAgentSDKInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    ):
        with pytest.raises(CLIConnectionError) as exc_info:
            async for _ in claude_agent_sdk.query(prompt="hello"):
                pass

    assert exc_info.value is error
    spans = [
        span
        for span in span_exporter.get_finished_spans()
        if span.name == "invoke_agent"
    ]
    assert len(spans) == 1
    assert spans[0].status.status_code == StatusCode.ERROR
    assert (
        spans[0].attributes["error.type"]
        == "claude_agent_sdk._errors.CLIConnectionError"
    )


@pytest.mark.anyio
async def test_query_aclose_finalizes_once(
    tracer_provider,
    logger_provider,
    meter_provider,
    span_exporter,
) -> None:
    transport = _QueryTransport(
        [_user_message()],
        wait_after_messages=True,
    )

    with instrument(
        ClaudeAgentSDKInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    ):
        stream = claude_agent_sdk.query(prompt=_prompt(), transport=transport)
        assert isinstance(await anext(stream), UserMessage)
        await stream.aclose()
        await stream.aclose()

    spans = [
        span
        for span in span_exporter.get_finished_spans()
        if span.name == "invoke_agent"
    ]
    assert len(spans) == 1
    assert spans[0].status.status_code == StatusCode.UNSET
