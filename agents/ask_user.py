ASK_USER_INSTRUCTION = """
You can call ask_user to ask the user one clarifying question mid-task and wait for the reply.
Use it sparingly: only when a choice is genuinely ambiguous, would materially change the result,
and cannot be resolved from the data, file metadata, or conversation context. Prefer offering
2-4 short options. Do not ask for permission or confirmation, do not ask anything you can check
yourself, and never ask the same thing twice. If no answer arrives, proceed with the most
reasonable assumption and state it in your answer.
"""
