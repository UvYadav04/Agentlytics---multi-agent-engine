from autogen_ext.models.openai import OpenAIChatCompletionClient

from config import get_settings

MODEL_INFO = {
    "vision": True,
    "function_calling": True,
    "json_output": True,
    "family": "unknown",
    "structured_output": True,
}


DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"


def build_client(model: str = None):
    settings = get_settings()
    api_key = (settings.get("GEMINI_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not configured")

    configured_model = settings.get("GEMINI_MODEL")
    target_model = model or configured_model or DEFAULT_GEMINI_MODEL
    if target_model.startswith("google/"):
        target_model = target_model[len("google/"):]
    # Guard against inheriting foreign provider model IDs (e.g. openai/gpt-oss-120b, Qwen/..., :free)
    if "/" in target_model or ":free" in target_model:
        target_model = configured_model or DEFAULT_GEMINI_MODEL

    return OpenAIChatCompletionClient(
        model=target_model,
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        model_info=MODEL_INFO,
        parallel_tool_calls=False,
    )
