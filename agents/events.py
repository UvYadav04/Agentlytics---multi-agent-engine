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
            # Only the Orchestrator gets the extra status row below - it's the one whose tool
            # results (an entire sub-agent run, a file export, a report write) can take long
            # enough after returning that the trail looks frozen while it decides its next move.
            # Sub-agents (Tabular/Document) already have their own "Assigning an agent"/"Executing
            # a Python script" rows covering that same gap one level up, so adding this here too
            # just repeats the same message several times per investigation - see
            # orchestrator/agent.py and tabular|document/agent.py's _translate_event wiring.
            if include_result_status:
                events.append({
                    "type": "status",
                    "message": _result_status_message(names, any_error),
                })
            return events

        return []

    return translate
