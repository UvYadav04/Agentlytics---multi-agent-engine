import json
import logging
import time
from typing import Any, Optional

from autogen_agentchat.agents import AssistantAgent
from autogen_core import CancellationToken

from agents.database.config import SYSTEM_MESSAGE, get_model_config
from agents.events import make_tool_event_translator
from agents.final_answer import split_follow_up_questions
from agents.logger import get_agent_logger, log_event
from agents.thread_context import thread_context_brief
from agents.timing import ToolCallTimer
from llm_provider import LLMProvider, get_settings
from tools.database.connection_manager import ConnectionManager
from tools.database.database_tools import DatabaseTools
from tools.orchestrator.models import DatabaseFindings, InvestigationCancelled

try:
    from monitoring.wrappers.agents import trace_agent
except Exception:
    def trace_agent(*args, **kwargs):
        return lambda fn: fn


class DatabaseAgent:
    def __init__(
        self,
        connection_manager: ConnectionManager,
        available_connections: Optional[list[dict]] = None,
        visualization_service=None,
        dashboard_service=None,
        workspace_id: str = "default",
        **kwargs,
    ):
        self.logger = get_agent_logger("database_agent")
        self.connection_manager = connection_manager
        self.tools = DatabaseTools(
            connection_manager=connection_manager,
            available_connections=available_connections or [],
            visualization_service=visualization_service,
            dashboard_service=dashboard_service,
            workspace_id=workspace_id,
        )

        model_config = get_model_config()
        fallback_provider = get_settings().get("FALLBACK_LLM_PROVIDER", "groq")
        client = LLMProvider(
            model_config["provider"], fallback_provider=fallback_provider,
        ).get_client(model_config["model"])

        self.agent = AssistantAgent(
            name="database_agent",
            model_client=client,
            tools=[
                self.tools.list_available_connections,
                self.tools.inspect_schema,
                self.tools.execute_query,
                self.tools.propose_visualization,
                self.tools.ask_user,
            ],
            system_message=SYSTEM_MESSAGE,
            reflect_on_tool_use=False,
            max_tool_iterations=10,
        )

    @trace_agent(name="database_agent_run", component="agent.database")
    async def run(
        self,
        objective: str,
        connection_id: Optional[str] = None,
        constraints: Optional[dict] = None,
        on_event=None,
        thread_context: Optional[dict] = None,
        cancel_check=None,
        ask_user=None,
    ) -> DatabaseFindings:
        await self.agent.on_reset(CancellationToken())
        self.tools.charts_proposed = []
        self.tools.query_history = []
        self.tools.on_event = on_event
        self.tools.ask_user_callback = ask_user

        constraints = constraints or {}
        conn_info = None
        if connection_id:
            conn_info = next(
                (c for c in self.tools.available_connections if c.get("connection_id") == connection_id),
                None,
            )

        task_lines = [f"Objective: {objective}"]
        if connection_id:
            task_lines.append(f"Target connection_id: {connection_id}")
            if conn_info:
                task_lines.append(f"Database Name: {conn_info.get('name')}")
                task_lines.append(f"Database Type: {conn_info.get('db_type')}")
                objects = conn_info.get("objects", [])
                if objects:
                    task_lines.append(f"Known tables/collections: {', '.join(objects[:30])}")
        else:
            connections = self.tools.list_available_connections()
            task_lines.append(f"Available database connections: {connections}")

        if constraints:
            task_lines.append(f"Constraints: {constraints}")
        if thread_context:
            task_lines.append(f"{thread_context_brief(thread_context)}")

        task = "\n".join(task_lines)
        self.logger.info("objective sent to database agent: %s", task)

        run_start = time.perf_counter()
        tool_timer = ToolCallTimer(self.logger)
        transcript = []
        final_text = ""
        stream = self.agent.run_stream(task=task)

        try:
            async for event in stream:
                if not hasattr(event, "messages"):
                    log_event(self.logger, event)
                    tool_timer.record(event)
                    line = self._transcript_line(event)
                    if line:
                        transcript.append(line)
                    if type(event).__name__ == "TextMessage" and event.source == self.agent.name:
                        final_text = event.content
                    if on_event is not None:
                        for translated in self._translate_event(event):
                            await on_event(translated)

                if cancel_check is not None and await cancel_check():
                    await stream.aclose()
                    if on_event is not None:
                        await on_event({"type": "cancelled", "message": "Investigation cancelled."})
                    raise InvestigationCancelled()
        finally:
            aclose = getattr(stream, "aclose", None)
            if aclose is not None:
                try:
                    await aclose()
                except Exception:
                    pass

        self.logger.info("database agent run took %.3fs", time.perf_counter() - run_start)

        summary, follow_up_questions = split_follow_up_questions(final_text)
        db_type = conn_info.get("db_type", "unknown") if conn_info else "unknown"
        if not connection_id and self.tools.query_history:
            connection_id = self.tools.query_history[-1].get("connection_id", "")
            db_type = self.tools.query_history[-1].get("db_type", db_type)

        return DatabaseFindings(
            summary=summary,
            connection_id=connection_id or "",
            database_type=db_type,
            query_results=list(self.tools.query_history),
            charts=list(self.tools.charts_proposed),
            follow_up_questions=follow_up_questions,
        )

    @staticmethod
    def _transcript_line(event) -> str:
        event_type = type(event).__name__
        if event_type == "ToolCallRequestEvent":
            return "\n".join(f"CALL {call.name}({call.arguments})" for call in event.content)
        if event_type == "ToolCallExecutionEvent":
            return "\n".join(f"RESULT {res.name} -> {res.content}" for res in event.content)
        return ""

    _FRIENDLY_TOOL_NAMES = {
        "list_available_connections": "Listing database connections",
        "inspect_schema": "Inspecting database schema",
        "execute_query": "Executing database query",
        "propose_visualization": "Configuring visualization",
        "ask_user": "Asking you a question",
    }

    _translate_event = staticmethod(make_tool_event_translator(_FRIENDLY_TOOL_NAMES))

