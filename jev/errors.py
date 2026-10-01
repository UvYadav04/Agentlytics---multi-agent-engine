class JevError(Exception):
    pass


class JevConfigError(JevError):
    pass


class JevValidationError(JevError):
    pass


class JevAPIError(JevError):
    def __init__(self, message: str, status_code: int | None = None, detail=None):
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail


class JevAuthenticationError(JevAPIError):
    pass


class JevRequestError(JevAPIError):
    pass


class JevUnavailableError(JevAPIError):
    pass


class JevTimeoutError(JevUnavailableError):
    pass
