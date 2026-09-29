from typing import Awaitable, Callable, Optional

AskUserCallback = Callable[[str, list, str], Awaitable[str]]

MAX_OPTIONS = 4

UNAVAILABLE_REPLY = (
    "Asking the user is not available right now. Proceed with the most reasonable assumption "
    "and state that assumption clearly in your answer."
)

ASK_USER_TOOL_DESCRIPTION = """Ask the user ONE short clarifying question and wait for their reply.

Only use this when a decision is genuinely ambiguous, would materially change the result, and
cannot be resolved from the data, the file metadata, or the conversation so far - for example
two equally plausible date columns, an undefined business term, or an unclear time period.

- question: one concise, specific question in plain language.
- options: optional list of 2-4 short answer choices when the answer is one of a few known
  values (e.g. column names). The user may still reply with free text.

Returns the user's answer as text. If the reply says no answer was received, continue with the
most reasonable assumption and state it in your answer. Never ask for permission to proceed,
never ask something you can answer yourself, and never repeat a question already answered."""


class AskUserMixin:
    ask_user_callback: Optional[AskUserCallback] = None
    ask_user_source: str = "agent"

    async def ask_user(self, question: str, options: Optional[list[str]] = None) -> str:
        if self.ask_user_callback is None:
            return UNAVAILABLE_REPLY
        cleaned = []
        for option in options or []:
            text = str(option).strip()
            if text and text not in cleaned:
                cleaned.append(text)
        return await self.ask_user_callback(question.strip(), cleaned[:MAX_OPTIONS], self.ask_user_source)


AskUserMixin.ask_user.__doc__ = ASK_USER_TOOL_DESCRIPTION
