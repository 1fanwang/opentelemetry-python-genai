OpenTelemetry LlamaIndex Zero-Code Instrumentation Example
==========================================================

This example uses the ``opentelemetry-instrument`` CLI to configure and
instrument a LlamaIndex application without telemetry setup in the application.

Setup
-----

Run an OTLP-compatible collector on ``http://localhost:4317``, or set
``OTEL_EXPORTER_OTLP_ENDPOINT`` to another endpoint. Then create an environment
and install the example dependencies:

::

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    opentelemetry-bootstrap -a install
    export OPENAI_API_KEY=your_api_key_here

Run
---

::

    opentelemetry-instrument python agent.py

Set ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_ONLY`` to capture
prompt and completion content on spans.
