# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Focused regression corpus for LangChain create_agent classification."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Self
from unittest import mock

import langchain.agents
import pytest
from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import tool

from opentelemetry.instrumentation.genai.langchain.callback_handler import (
    OpenTelemetryLangChainCallbackHandler,
)
from opentelemetry.util.genai.invocation import (
    AgentInvocation,
    ToolInvocation,
    WorkflowInvocation,
)

create_agent = getattr(langchain.agents, "create_agent", None)
if create_agent is None:
    pytest.skip(
        "create_agent requires a newer langchain version",
        allow_module_level=True,
    )

langgraph_graph = import_module("langgraph.graph")
END = langgraph_graph.END
START = langgraph_graph.START
StateGraph = langgraph_graph.StateGraph


class FakeModel(FakeMessagesListChatModel):
    def bind_tools(
        self,
        tools: Any,
        *,
        tool_choice: Any = None,
        **kwargs: Any,
    ) -> Self:
        return self


@tool
def noop() -> str:
    """Do nothing."""
    return "ok"


def _handler() -> tuple[OpenTelemetryLangChainCallbackHandler, mock.MagicMock]:
    telemetry = mock.MagicMock()
    workflow = mock.MagicMock(spec=WorkflowInvocation)
    workflow.span = mock.MagicMock()
    workflow.span.is_recording.return_value = False
    telemetry.workflow.return_value = workflow

    def make_agent(*args: Any, **kwargs: Any) -> mock.MagicMock:
        invocation = mock.MagicMock(spec=AgentInvocation)
        invocation.agent_name = kwargs.get("agent_name")
        invocation.span = mock.MagicMock()
        invocation.span.is_recording.return_value = False
        return invocation

    telemetry.invoke_local_agent.side_effect = make_agent

    tool_invocation = mock.MagicMock(spec=ToolInvocation)
    tool_invocation.span = mock.MagicMock()
    tool_invocation.span.is_recording.return_value = False
    telemetry.tool.return_value = tool_invocation
    return OpenTelemetryLangChainCallbackHandler(telemetry), telemetry


def _agent_names(telemetry: mock.MagicMock) -> list[str]:
    return [
        call.kwargs["agent_name"]
        for call in telemetry.invoke_local_agent.call_args_list
    ]


@pytest.mark.parametrize(
    ("name", "expected_name"),
    [(None, "LangGraph"), ("named_agent", "named_agent")],
)
def test_create_agent_root(
    name: str | None,
    expected_name: str,
    span_exporter,
    start_instrumentation,
) -> None:
    agent_kwargs = {"name": name} if name is not None else {}
    create_agent(
        FakeModel(responses=[AIMessage(content="done")]),
        [noop],
        **agent_kwargs,
    ).invoke({"messages": [("user", "hi")]})

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [f"invoke_agent {expected_name}"]
    agent_span = spans[0]
    assert agent_span.parent is None
    assert agent_span.attributes["gen_ai.agent.name"] == expected_name


def test_create_agent_name_wins_over_run_name_override(
    span_exporter, start_instrumentation
) -> None:
    agent = create_agent(
        FakeModel(responses=[AIMessage(content="done")]),
        [noop],
        name="planner_agent",
    )
    RunnableLambda(
        lambda _: agent.invoke(
            {"messages": [("user", "hi")]}, {"run_name": "step1"}
        )
    ).with_config(run_name="planner").invoke({})

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "invoke_agent planner_agent",
        "invoke_workflow planner",
    ]
    agent_span, workflow_span = spans
    assert agent_span.parent.span_id == workflow_span.context.span_id
    assert workflow_span.parent is None
    assert agent_span.attributes["gen_ai.agent.name"] == "planner_agent"
    assert "gen_ai.agent.name" not in workflow_span.attributes


def test_create_agent_internal_nodes_are_not_agents(
    span_exporter, start_instrumentation
) -> None:
    create_agent(
        FakeModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[{"name": "noop", "args": {}, "id": "1"}],
                ),
                AIMessage(content="done"),
            ]
        ),
        [noop],
        name="agent",
    ).invoke({"messages": [("user", "hi")]})

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "execute_tool noop",
        "invoke_agent agent",
    ]
    agent_span = spans[-1]
    assert agent_span.parent is None
    assert agent_span.attributes["gen_ai.agent.name"] == "agent"
    tool_span = spans[0]
    assert tool_span.parent.span_id == agent_span.context.span_id
    assert "gen_ai.agent.name" not in tool_span.attributes


def test_create_agent_with_configured_agent_name_emits_one_agent() -> None:
    handler, telemetry = _handler()
    create_agent(
        FakeModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[{"name": "noop", "args": {}, "id": "1"}],
                ),
                AIMessage(content="done"),
            ]
        ),
        [noop],
    ).with_config(metadata={"agent_name": "ordinary"}).invoke(
        {"messages": [("user", "hi")]}, {"callbacks": [handler]}
    )

    assert _agent_names(telemetry) == ["ordinary"]
    assert telemetry.tool.call_count == 1


def test_ordinary_runnable_is_not_an_agent() -> None:
    handler, telemetry = _handler()
    RunnableLambda(lambda value: value).with_config(
        run_name="ordinary"
    ).invoke("value", {"callbacks": [handler]})

    telemetry.invoke_local_agent.assert_not_called()


def test_plain_state_graph_is_not_an_agent() -> None:
    handler, telemetry = _handler()
    builder = StateGraph(dict[str, int])
    builder.add_node("increment", lambda state: {"value": state["value"] + 1})
    builder.add_edge(START, "increment")
    builder.add_edge("increment", END)
    builder.compile().invoke({"value": 1}, {"callbacks": [handler]})

    telemetry.invoke_local_agent.assert_not_called()


@pytest.mark.xfail(
    strict=True,
    reason=(
        "LangChain callbacks do not identify a nested create_agent root "
        "separately from metadata inherited from its enclosing agent"
    ),
)
def test_nested_create_agent_is_known_limitation() -> None:
    handler, telemetry = _handler()
    inner = create_agent(
        FakeModel(responses=[AIMessage(content="inner done")]),
        [noop],
        name="inner",
    )

    @tool
    def delegate(config: RunnableConfig) -> str:
        """Delegate to the inner agent."""
        result = inner.invoke({"messages": [("user", "work")]}, config)
        return str(result["messages"][-1].content)

    outer = create_agent(
        FakeModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "delegate", "args": {}, "id": "call-1"}
                    ],
                ),
                AIMessage(content="outer done"),
            ]
        ),
        [delegate],
        name="outer",
    )
    outer.invoke({"messages": [("user", "start")]}, {"callbacks": [handler]})

    assert _agent_names(telemetry) == ["outer", "inner"]
