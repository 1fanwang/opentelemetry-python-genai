# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, TypedDict

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from opentelemetry.instrumentation.genai.langchain import LangChainInstrumentor
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.test_util_genai.instrumentor import instrument

langgraph_graph = pytest.importorskip("langgraph.graph")
END = langgraph_graph.END
START = langgraph_graph.START
StateGraph = langgraph_graph.StateGraph


class _State(TypedDict):
    messages: list[BaseMessage]


def _respond(_: _State) -> _State:
    return {"messages": [AIMessage(content="done")]}


def _graph(node: Any, *, name: str | None = None) -> Any:
    builder = StateGraph(_State)
    builder.add_node("step", node)
    builder.add_edge(START, "step")
    builder.add_edge("step", END)
    return builder.compile(name=name)


def _nested_graph() -> Any:
    return _graph(_graph(_respond, name="named_subgraph"))


def _workflow_spans(span_exporter: Any) -> list[Any]:
    return [
        span
        for span in span_exporter.get_finished_spans()
        if span.attributes
        and span.attributes.get(GenAI.GEN_AI_OPERATION_NAME)
        == "invoke_workflow"
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "async_mode",
    [False, True],
    ids=["sync", "async"],
)
async def test_nested_graph_emits_workflow_span(
    tracer_provider,
    meter_provider,
    logger_provider,
    span_exporter,
    async_mode: bool,
) -> None:
    with instrument(
        LangChainInstrumentor(),
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        logger_provider=logger_provider,
        content_capture="SPAN_ONLY",
    ):
        graph = _nested_graph()
        inputs = {"messages": [HumanMessage(content="hello")]}
        result = (
            await graph.ainvoke(inputs) if async_mode else graph.invoke(inputs)
        )

    assert result == {"messages": [AIMessage(content="done")]}
    workflow_spans = _workflow_spans(span_exporter)
    assert len(workflow_spans) == 2
    workflow_spans_by_name = {
        span.attributes[GenAI.GEN_AI_WORKFLOW_NAME]: span
        for span in workflow_spans
    }
    assert set(workflow_spans_by_name) == {"LangGraph", "named_subgraph"}
    inner_span = workflow_spans_by_name["named_subgraph"]
    outer_span = workflow_spans_by_name["LangGraph"]
    if not async_mode:
        assert inner_span.parent.span_id == outer_span.context.span_id
    assert all(
        GenAI.GEN_AI_INPUT_MESSAGES in span.attributes
        for span in workflow_spans
    )
    assert all(
        GenAI.GEN_AI_OUTPUT_MESSAGES in span.attributes
        for span in workflow_spans
    )


def test_nested_runnable_sequence_is_not_workflow(
    tracer_provider,
    meter_provider,
    logger_provider,
    span_exporter,
) -> None:
    sequence = RunnableLambda(_respond) | RunnableLambda(_respond)

    with instrument(
        LangChainInstrumentor(),
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        logger_provider=logger_provider,
    ):
        _graph(sequence).invoke({"messages": [HumanMessage(content="hello")]})

    assert len(_workflow_spans(span_exporter)) == 1


def test_nested_graph_under_agent_metadata_is_workflow(
    tracer_provider,
    meter_provider,
    logger_provider,
    span_exporter,
) -> None:
    with instrument(
        LangChainInstrumentor(),
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        logger_provider=logger_provider,
    ):
        _nested_graph().invoke(
            {"messages": [HumanMessage(content="hello")]},
            config={"metadata": {"agent_type": "outer"}},
        )

    spans = span_exporter.get_finished_spans()
    span_names = {span.name for span in spans}
    assert "invoke_workflow named_subgraph" in span_names
    assert "invoke_agent named_subgraph" not in span_names
    nested_span = next(
        span for span in spans if span.name == "invoke_workflow named_subgraph"
    )
    assert nested_span.parent is not None
