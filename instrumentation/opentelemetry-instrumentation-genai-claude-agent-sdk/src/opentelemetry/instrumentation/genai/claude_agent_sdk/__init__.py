# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""
OpenTelemetry Claude Agent SDK Instrumentation
===============================================

Instrumentation for the `Claude Agent SDK
<https://github.com/anthropics/claude-agent-sdk-python>`_.

Usage
-----

.. code-block:: python

    from opentelemetry.instrumentation.genai.claude_agent_sdk import (
        ClaudeAgentSDKInstrumentor,
    )

    # Enable instrumentation before importing query.
    ClaudeAgentSDKInstrumentor().instrument()

    from claude_agent_sdk import (
        ClaudeAgentOptions,
        AgentDefinition,
        AssistantMessage,
        TextBlock,
        query,
    )

    # Use Claude Agent SDK normally
    import anyio


    async def main():
        options = ClaudeAgentOptions(
            agents={
                "assistant": AgentDefinition(
                    description="A helpful assistant",
                    prompt="You are a helpful assistant.",
                    tools=["Read"],
                    model="sonnet",
                ),
            },
        )

        async for message in query(prompt="Hello!", options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(block.text)


    anyio.run(main)

API
---
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Collection
from typing import Any

from claude_agent_sdk import Message, ResultMessage
from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.genai.claude_agent_sdk.package import (
    _instruments,
)
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.util.genai.completion_hook import load_completion_hook
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import AgentInvocation
from opentelemetry.util.genai.stream import AsyncStreamWrapper


class _ClaudeQueryStream(AsyncStreamWrapper[Message]):
    def __init__(
        self,
        stream: AsyncIterator[Message],
        invocation: AgentInvocation,
    ) -> None:
        super().__init__(stream)
        self._self_invocation = invocation

    def _process_chunk(self, chunk: Message) -> None:
        if not isinstance(chunk, ResultMessage):
            return

        self._self_invocation.conversation_id = chunk.session_id
        terminal_status = (
            getattr(chunk, "terminal_reason", None)
            or getattr(chunk, "stop_reason", None)
            or chunk.subtype
        )
        self._self_invocation.finish_reasons = [terminal_status]

    def _on_stream_end(self) -> None:
        self._self_invocation.stop()

    def _on_stream_error(self, error: BaseException) -> None:
        self._self_invocation.fail(error)


class ClaudeAgentSDKInstrumentor(BaseInstrumentor):
    """Trace one-shot Claude Agent SDK queries as agent invocations."""

    def __init__(self) -> None:
        super().__init__()
        self._handler: TelemetryHandler | None = None

    # pylint: disable=no-self-use
    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        """Enable Claude Agent SDK instrumentation.

        Args:
            **kwargs: Optional arguments
                - tracer_provider: TracerProvider instance
                - meter_provider: MeterProvider instance
                - logger_provider: LoggerProvider instance
        """

        # Get providers from kwargs
        tracer_provider = kwargs.get("tracer_provider")
        logger_provider = kwargs.get("logger_provider")
        meter_provider = kwargs.get("meter_provider")

        handler = TelemetryHandler(
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            logger_provider=logger_provider,
            completion_hook=kwargs.get("completion_hook")
            or load_completion_hook(),
        )
        self._handler = handler

        def _wrap_query(
            wrapped: Callable[..., AsyncIterator[Message]],
            instance: object | None,
            args: tuple[Any, ...],
            call_kwargs: dict[str, Any],
        ) -> AsyncIterator[Message]:
            stream = wrapped(*args, **call_kwargs)
            options = call_kwargs.get("options")
            invocation = handler.invoke_local_agent(
                request_model=getattr(options, "model", None),
            )
            return _ClaudeQueryStream(stream, invocation)

        wrap_function_wrapper("claude_agent_sdk", "query", _wrap_query)

    def _uninstrument(self, **kwargs: Any) -> None:
        """Disable Claude Agent SDK instrumentation.

        This removes all patches applied during instrumentation.
        """
        import claude_agent_sdk

        unwrap(claude_agent_sdk, "query")
        self._handler = None
