CORE_TOOLS = [
    "get_current_date",
    "recall_user_info",
    "list_files",
    "search_files",
    "get_file_details",
    "list_tables",
    "list_file_formats",
    "generate_hypotheses",
    "invoke_tabular_agent",
    "invoke_document_agent",
    "invoke_document_processor",
    # A cheap, always-available way to unlock a capability-gated tool (see CAPABILITY_TOOLS
    # below) the moment the model realizes it needs one - e.g. right after seeing an agent's
    # findings and deciding a report is warranted, with nothing else it needs to call at that
    # exact moment to piggyback the request onto. See its docstring in
    # tools/orchestrator/orchestrator_tools.py and OrchestratorTools.request_capabilities.
    "request_capabilities",
]

# Deliverable tools kept out of CORE_TOOLS so their (fairly long) descriptions don't ride along
# on every single turn's prompt - exposed only once the model requests the matching capability,
# either by piggybacking `next_capabilities` onto a real tool call it's already making (e.g.
# invoke_tabular_agent), or via the standalone request_capabilities tool above when it has
# nothing else to call at the moment it realizes it needs one. Either way, the capability is
# available for exactly one call after being requested.
CAPABILITY_TOOLS: dict[str, dict] = {
    "csv": {
        "tools": ["generate_csv"],
        "description": "export an existing data artifact as a CSV file",
    },
    "report": {
        "tools": ["generate_report"],
        "description": "write an LLM-composed markdown report file from findings you already have",
    },
}


def all_tool_names() -> list[str]:
    names = list(CORE_TOOLS)
    for meta in CAPABILITY_TOOLS.values():
        for name in meta["tools"]:
            if name not in names:
                names.append(name)
    return names


def tools_for_capabilities(capability_names) -> list[str]:
    names: list[str] = []
    for cap in capability_names or []:
        for name in CAPABILITY_TOOLS.get(cap, {}).get("tools", []):
            if name not in names:
                names.append(name)
    return names


def capability_catalog_text() -> str:
    return "\n".join(f'- "{name}": {meta["description"]}' for name, meta in CAPABILITY_TOOLS.items())
