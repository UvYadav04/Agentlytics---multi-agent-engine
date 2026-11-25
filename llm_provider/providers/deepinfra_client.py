from autogen_ext.models.openai import OpenAIChatCompletionClient

from config import get_settings

MODEL_INFO = {
    "vision": False,
    "function_calling": True,
    "json_output": True,
    "family": "unknown",
    "structured_output": True,
}

DEFAULT_MAX_TOKENS = 2048
DEFAULT_DEEPINFRA_MODEL = "openai/gpt-oss-120b"

DISABLE_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}


def build_client(model: str = None):
    settings = get_settings()
    max_tokens = int(settings.get("DEEPINFRA_MAX_TOKENS", DEFAULT_MAX_TOKENS))
    configured_model = settings.get("DEEPINFRA_MODEL")
    target_model = model or configured_model or DEFAULT_DEEPINFRA_MODEL
    if target_model == "gpt-oss-120b":
        target_model = "openai/gpt-oss-120b"
    # Guard against inheriting foreign provider model names (e.g. OpenRouter free models)
    if ":free" in target_model or "nemotron" in target_model.lower():
        target_model = configured_model or DEFAULT_DEEPINFRA_MODEL
    service_tier = settings.get("DEEPINFRA_SERVICE_TIER", "priority")
    extra_body = {**DISABLE_THINKING}
    if service_tier:
        extra_body["service_tier"] = service_tier

    return OpenAIChatCompletionClient(
        model=target_model,
        api_key=settings.DEEPINFRA_API_KEY,
        base_url="https://api.deepinfra.com/v1/openai",
        model_info=MODEL_INFO,
        parallel_tool_calls=False,
        max_tokens=max_tokens,
        extra_body=extra_body,
    )
