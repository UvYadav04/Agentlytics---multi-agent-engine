from autogen_ext.models.openai import OpenAIChatCompletionClient

from config import get_settings

MODEL_INFO = {
    "vision": False,
    "function_calling": True,
    "json_output": True,
    "family": "unknown",
    "structured_output": True,
}

DEFAULT_CEREBRAS_MODEL = "gpt-oss-120b"


def build_client(model: str = None):
    settings = get_settings()
    configured_model = getattr(settings, "CEREBRAS_MODEL", None)
    target_model = model or configured_model or DEFAULT_CEREBRAS_MODEL
    # Guard against inheriting foreign provider models
    if ":free" in target_model or "nemotron" in target_model.lower() or "/" in target_model:
        target_model = configured_model or DEFAULT_CEREBRAS_MODEL
    if target_model.startswith("openai/"):
        target_model = target_model[len("openai/"):]

    return OpenAIChatCompletionClient(
        model=target_model,
        api_key=settings.get("CEREBRAS_API_KEY"),
        base_url="https://api.cerebras.ai/v1",
        model_info=MODEL_INFO,
        parallel_tool_calls=False,
    )
