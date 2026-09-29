_AGENT_TOOLS = {"invoke_tabular_agent", "invoke_document_agent", "invoke_document_processor"}
_WRITER_TOOLS = {"generate_report"}


def _result_status_message(names: list[str], any_error: bool) -> str:
    if any_error:
        return "Something went wrong there - figuring out how to recover..."
    if any(n in _AGENT_TOOLS for n in names):
        return "Reading the agent's result..."
    if any(n in _WRITER_TOOLS for n in names):
        return "Reading the writer's result..."
    return "Reading the tool's result..."


def make_tool_event_translator(friendly_names: dict[str, str], *, include_result_status: bool = False):
    def _label(name: str) -> str:
        return friendly_names.get(name, name)

    def translate(event) -> list[dict]:
        event_type = type(event).__name__

        if event_type == "ToolCallRequestEvent":
            message = "; ".join(_label(call.name) for call in event.content)
            return [{"type": "tool_call", "message": message}]

        if event_type == "ToolCallExecutionEvent":
            names = [res.name for res in event.content]
            any_error = any(getattr(res, "is_error", False) for res in event.content)
            message = "; ".join(_label(n) for n in names)
            events = [{"type": "tool_error" if any_error else "tool_result", "message": message}]
            if include_result_status:
                events.append({
                    "type": "status",
                    "message": _result_status_message(names, any_error),
                })
            return events

        return []

    return translate
