# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

from opentelemetry.instrumentation.genai.llama_index import (
    LlamaIndexInstrumentor,
)

LlamaIndexInstrumentor().instrument()


async def main() -> None:
    agent = FunctionAgent(
        name="assistant",
        llm=OpenAI(model="gpt-4o-mini"),
        streaming=False,
    )
    response = await agent.run(user_msg="Hello!")
    print(response)


asyncio.run(main())
