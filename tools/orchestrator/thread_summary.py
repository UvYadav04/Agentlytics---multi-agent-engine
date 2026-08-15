from agents.orchestrator.config import get_model_config
from llm_provider import LLMProvider, get_settings
from tools.llm_call import ask_llm_async

_SUMMARY_MARKER = "SUMMARY:"
_PREFERENCES_MARKER = "NEW_PREFERENCES:"

DEFAULT_TOKEN_BUDGET = 12000
PRUNE_THRESHOLD_RATIO = 0.85

COMPRESSION_PROMPT = """Compress the following conversation history into a compact plain-language
summary a data-analysis agent can use as background context. Fold the previous summary and all
the older turns below into one updated summary, a few sentences at most. Preserve concrete facts
(file names, columns, numbers, decisions) and drop pleasantries and redundant detail. If a later
turn corrects or contradicts an earlier one, keep only the corrected version.

Previous summary:
{previous_summary}

Older turns to fold in:
{fold_turns}

Reply with the updated summary text only, nothing else."""

TURN_ANALYSIS_PROMPT = """You maintain two things for an ongoing data-analysis conversation:

1. A rolling summary of everything that happened BEFORE the conversation's raw recent-turns
window (only relevant once the chat has grown past that window - see below).
2. A short list of durable user preferences/facts worth remembering across future conversations -
things that would still be true and useful in a completely different chat, about a different
topic, weeks from now (e.g. "works in the finance team", "always wants dollar amounts rounded to
2 decimals", "prefers short answers without lengthy caveats").

Be conservative here. On most turns the right answer is nothing at all - that is the normal,
expected outcome, not a failure to find something. Only extract a fact when the user has clearly
stated it, or unmistakably implied something general about how they always want things done -
never when you're merely inferring a "preference" from the one specific choice they made in this
one turn's specific context.

In particular: what chart type, column, file, or analysis the user asked for IN THIS TURN is a
one-off task detail, not a preference, even if it resembles one on the surface. "Show a bar chart
of sales by region" tells you nothing about what they'd want for a totally different question
next week - do not record it as "prefers bar charts" or anything similar. Only record a
chart/format preference if the user says something explicitly general, like "I always want bar
charts" or "from now on, round to 2 decimals" - never infer generality from a single instance.

{fold_section}

Latest turn (for preference extraction only - this turn is still shown to the agent verbatim
elsewhere, so do NOT fold it into the summary yourself):
User: {query}
Assistant: {response}

Reply in exactly this format, both section headers always present:

{summary_marker}
{summary_instruction}

{preferences_marker}
One durable user fact/preference per line, extracted ONLY from the latest turn above, and ONLY if
it's clearly general/lasting per the rules above - not a detail specific to this one question.
When in doubt, leave it out. If there's nothing new (the common case), write the single word
None."""

_NO_FOLD_SECTION = (
    "Nothing needs folding into the summary this time - the raw recent-turns window still "
    "covers everything so far. Repeat the previous summary back UNCHANGED, word for word:\n"
    "Previous summary:\n{previous_summary}"
)

_FOLD_SECTION = (
    "Previous summary:\n{previous_summary}\n\n"
    "Fold the following older turn(s) into it - these are about to age out of the raw "
    "recent-turns window, so anything worth keeping must be captured here or it's lost. If a "
    "folded-in turn corrects or contradicts something already in the summary, keep only the "
    "corrected version, don't preserve both:\n{fold_turns}"
)


def _format_fold_turns(turns: list[dict]) -> str:
    return "\n\n".join(
        f"User: {t.get('query', '')}\nAssistant: {t.get('response', '')}" for t in turns
    )


def _parse_response(raw: str, fallback_summary: str) -> tuple[str, list[str]]:
    if _PREFERENCES_MARKER not in raw:
        return raw.replace(_SUMMARY_MARKER, "", 1).strip() or fallback_summary, []

    summary_part, _, preferences_part = raw.partition(_PREFERENCES_MARKER)
    summary = summary_part.replace(_SUMMARY_MARKER, "", 1).strip() or fallback_summary

    preferences = []
    for line in preferences_part.strip().splitlines():
        line = line.strip().lstrip("-*").strip()
        if line and line.lower() != "none":
            preferences.append(line)
    return summary, preferences


async def analyze_turn(
    previous_summary: str, query: str, response: str, turns_to_fold: list[dict] | None = None,
) -> tuple[str, list[str]]:
    model_config = get_model_config()
    fallback_provider = get_settings().get("FALLBACK_LLM_PROVIDER", "groq")
    client = LLMProvider(model_config["provider"], fallback_provider=fallback_provider).get_client(model_config["model"])

    fallback_summary = previous_summary or ""
    previous_summary_display = previous_summary or "(none yet - this is the first turn in this chat)"

    if turns_to_fold:
        fold_section = _FOLD_SECTION.format(
            previous_summary=previous_summary_display, fold_turns=_format_fold_turns(turns_to_fold),
        )
        summary_instruction = "The updated summary, with the turn(s) above folded in. Keep it compact plain language - a few sentences, not a transcript."
    else:
        fold_section = _NO_FOLD_SECTION.format(previous_summary=previous_summary_display)
        summary_instruction = "(repeat the previous summary back unchanged, per the instruction above)"

    prompt = TURN_ANALYSIS_PROMPT.format(
        fold_section=fold_section, query=query, response=response,
        summary_marker=_SUMMARY_MARKER, preferences_marker=_PREFERENCES_MARKER,
        summary_instruction=summary_instruction,
    )
    raw = (await ask_llm_async(client, prompt)).strip()
    return _parse_response(raw, fallback_summary)


def estimate_tokens(text: str) -> int:
    return max(1, len(text or "") // 4)


def estimate_thread_context_tokens(summary: str, recent_turns: list[dict], query: str = "") -> int:
    total = estimate_tokens(summary) + estimate_tokens(query)
    for turn in recent_turns or []:
        total += estimate_tokens(turn.get("query", "")) + estimate_tokens(turn.get("response", ""))
    return total


def token_budget() -> int:
    raw = get_settings().get("MEMORY_TOKEN_BUDGET")
    try:
        return int(raw) if raw else DEFAULT_TOKEN_BUDGET
    except (TypeError, ValueError):
        return DEFAULT_TOKEN_BUDGET


def should_prune(summary: str, recent_turns: list[dict], query: str = "") -> bool:
    budget = token_budget()
    used = estimate_thread_context_tokens(summary, recent_turns, query)
    return len(recent_turns or []) > 1 and used >= budget * PRUNE_THRESHOLD_RATIO


async def compress_context(previous_summary: str, recent_turns: list[dict]) -> tuple[str, list[dict]]:
    if len(recent_turns or []) <= 1:
        return previous_summary, recent_turns

    turns_to_fold = recent_turns[:-1]
    kept_turns = recent_turns[-1:]

    model_config = get_model_config()
    fallback_provider = get_settings().get("FALLBACK_LLM_PROVIDER", "groq")
    client = LLMProvider(model_config["provider"], fallback_provider=fallback_provider).get_client(model_config["model"])

    prompt = COMPRESSION_PROMPT.format(
        previous_summary=previous_summary or "(none yet)",
        fold_turns=_format_fold_turns(turns_to_fold),
    )
    try:
        raw = (await ask_llm_async(client, prompt)).strip()
    except Exception:
        return previous_summary, kept_turns
    return raw or previous_summary, kept_turns
