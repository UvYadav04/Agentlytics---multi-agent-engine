import os
from typing import Type

from ingestion.file_types.base import BaseIngestor
from ingestion.file_types.csv.csv_ingestor import CSVIngestor
from ingestion.file_types.xlsx.xlsx_ingestor import XLSXIngestor

# Document ingestion (PDF/TXT) intentionally removed - blocked at api_service/routers/files.py's
# presign_upload (the real, authoritative gate: no File/Investigation doc for one of these ever
# gets created), and again here as a defense-in-depth backstop so a leftover/pre-existing pdf or
# txt File doc can never be picked up by run_ingestion either. PDFIngestor/TXTIngestor are left in
# place under ingestion/file_types/ (not deleted) in case document support is restored later - see
# ingestion/file_types/pdf/ and ingestion/file_types/txt/.
EXTENSION_REGISTRY: dict[str, Type[BaseIngestor]] = {
    ".csv": CSVIngestor,
    ".xlsx": XLSXIngestor,
}


def get_ingestor_for(file_path: str) -> Type[BaseIngestor]:

    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    ingestor_cls = EXTENSION_REGISTRY.get(ext)
    if ingestor_cls is None:
        supported = ", ".join(sorted(EXTENSION_REGISTRY.keys()))
        raise ValueError(
            f"Unsupported file type '{ext or '<none>'}' for '{file_path}'. "
            f"Supported types: {supported}"
        )
    return ingestor_cls
