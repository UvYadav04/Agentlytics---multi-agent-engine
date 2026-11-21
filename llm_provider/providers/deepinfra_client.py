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

DISABLE_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}


def build_client(model: str = None):
    settings = get_settings()
    max_tokens = int(settings.get("DEEPINFRA_MAX_TOKENS", DEFAULT_MAX_TOKENS))
    target_model = model or settings.get("DEFAULT_MODEL", "Qwen/Qwen2.5-72B-Instruct")
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
