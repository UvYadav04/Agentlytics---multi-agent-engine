from autogen_ext.models.openai import OpenAIChatCompletionClient

from config import get_settings

MODEL_INFO = {
    "vision": False,
    "function_calling": True,
    "json_output": True,
    "family": "unknown",
    "structured_output": True,
}


DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"

def build_client(model: str = None):
    settings = get_settings()
    target_model = model or getattr(settings, "DEFAULT_MODEL", None) or DEFAULT_GROQ_MODEL
    if "llama-3.3-70b" in target_model.lower():
        target_model = DEFAULT_GROQ_MODEL
    elif target_model == "gpt-oss-120b":
        target_model = "openai/gpt-oss-120b"

    return OpenAIChatCompletionClient(
        model=target_model,
        api_key=settings.GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
        model_info=MODEL_INFO,
        parallel_tool_calls=False,
    )
