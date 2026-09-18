from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor

from app.config import settings


def setup_tracing():
    tracer_provider = register(
        project_name="rag",
        endpoint=f"{settings.phoenix_url}/v1/traces",
    )

    LangChainInstrumentor().instrument(
        tracer_provider=tracer_provider
    )

    return tracer_provider