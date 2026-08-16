_SIZE_KEYWORDS = (
    "too large",
    "exceeds",
    "exceeded",
    "maximum batch size",
    "max batch size",
    "payload size",
    "payload too large",
    "size limit",
    "request entity too large",
)


def is_size_related_error(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if status == 413:
        return True
    message = str(exc).lower()
    return any(keyword in message for keyword in _SIZE_KEYWORDS)
