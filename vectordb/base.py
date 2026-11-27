from abc import ABC, abstractmethod
from typing import Optional


class VectorStoreError(RuntimeError):
    """Raised when a vector store operation fails."""

    def __init__(self, message: str, *, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class BaseVectorStore(ABC):
    @abstractmethod
    def upsert(self, chunks: list) -> list:
        raise NotImplementedError

    @abstractmethod
    def query(self, query_text: str, top_k: int, filters: Optional[dict] = None) -> list:
        raise NotImplementedError

    @abstractmethod
    def delete(self, ids: list) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_by_id(self, ids: list) -> list:
        raise NotImplementedError

    @abstractmethod
    def get_by_filter(self, filters: dict) -> list:
        raise NotImplementedError
