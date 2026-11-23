from autogen_ext.models.openai import OpenAIChatCompletionClient

from config import get_settings

MODEL_INFO = {
    "vision": False,
    "function_calling": True,
    "json_output": True,
    "family": "unknown",
    "structured_output": True,
}

DEFAULT_OPENROUTER_MODEL = "meta-llama/llama-3.3-70b-instruct"


def build_client(model: str = None):
    settings = get_settings()
    target_model = model or getattr(settings, "DEFAULT_MODEL", None) or DEFAULT_OPENROUTER_MODEL

    headers = {
        "HTTP-Referer": "https://agentlytics.ai",
        "X-Title": "DataAnalyzer",
    }

    return OpenAIChatCompletionClient(
        model=target_model,
        api_key=settings.get("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        model_info=MODEL_INFO,
        parallel_tool_calls=False,
        default_headers=headers,
    )
