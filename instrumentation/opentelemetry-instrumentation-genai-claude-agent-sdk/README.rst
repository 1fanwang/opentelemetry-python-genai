OpenTelemetry Claude Agent SDK Instrumentation
==============================================

|pypi|

.. |pypi| image:: https://badge.fury.io/py/opentelemetry-instrumentation-genai-claude-agent-sdk.svg
   :target: https://pypi.org/project/opentelemetry-instrumentation-genai-claude-agent-sdk/

This library traces one-shot ``query()`` calls made by the
`Claude Agent SDK <https://github.com/anthropics/claude-agent-sdk-python>`_
as ``invoke_agent`` spans. Claude Code's native telemetry is unchanged.

Installation
------------

::

    pip install opentelemetry-instrumentation-genai-claude-agent-sdk

If you don't have a Claude Agent SDK application yet, try our `examples <examples>`_
which only need a valid Anthropic API key.

Check out the `zero-code example <examples/zero-code>`_ for a quick start.

Usage
-----

This section describes how to set up Claude Agent SDK instrumentation if you're setting OpenTelemetry up manually.
Check out the `manual example <examples/manual>`_ for more details.

.. code-block:: python

    from opentelemetry.instrumentation.genai.claude_agent_sdk import ClaudeAgentSDKInstrumentor

    # Instrument Claude Agent SDK
    ClaudeAgentSDKInstrumentor().instrument()

    from claude_agent_sdk import ClaudeAgentOptions, AgentDefinition, AssistantMessage, TextBlock, query

    # Use Claude Agent SDK as normal
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

        async for message in query(
            prompt="Hello, Claude!",
            options=options,
        ):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(block.text)

    anyio.run(main)

References
----------

* `OpenTelemetry Project <https://opentelemetry.io/>`_
* `OpenTelemetry GenAI semantic conventions <https://opentelemetry.io/docs/specs/semconv/gen-ai/>`_
* `Claude Agent SDK (Python) <https://github.com/anthropics/claude-agent-sdk-python>`_
* `Anthropic Documentation <https://docs.anthropic.com/>`_
