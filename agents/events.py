def make_tool_event_translator(friendly_names: dict[str, str]):
    def _label(name: str) -> str:
        return friendly_names.get(name, name)

    def translate(event) -> list[dict]:
        event_type = type(event).__name__

        if event_type == "ToolCallRequestEvent":
            message = "; ".join(_label(call.name) for call in event.content)
            return [{"type": "tool_call", "message": message}]

        if event_type == "ToolCallExecutionEvent":
            any_error = any(getattr(res, "is_error", False) for res in event.content)
            message = "; ".join(_label(res.name) for res in event.content)
            events = [{"type": "tool_error" if any_error else "tool_result", "message": message}]
            # The event above only flips the paired tool_call row's icon (see
            # InvestigationTrail.tsx's buildRows) - it says nothing about what happens next. The
            # model still has to read the tool's output and decide its next move (another tool
            # call, or the final answer), which can take several seconds with nothing else
            # visible in the trail - without this, the UI looks frozen right after a tool
            # finishes. This appends a plain status row to cover that gap; it gets naturally
            # superseded the moment the next tool_call/status/completed event arrives.
            events.append({
                "type": "status",
                "message": (
                    "Something went wrong there - figuring out how to recover..."
                    if any_error else "Reviewing the results..."
                ),
            })
            return events

        return []

    return translate
