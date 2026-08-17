import os
import uuid
from typing import Callable, Optional

import pypdf

from ingestion.errors import is_size_related_error
from ingestion.file_types.base import BaseIngestor
from ingestion.file_types.pdf.chunker import BaseChunker, DoclingChunker
from ingestion.file_types.pdf.llamaparse_client import parse_pdf_pages
from ingestion.file_types.pdf.utils import extract_tables, is_scanned
from ingestion.models import IngestionResult
from vectordb.schema import ChunkRecord

# Kept in sync with Client/src/components/chat/UploadModal.tsx's MAX_PDF_PAGES - the client check
# is what most users see, this is the server-side backstop for anything that reaches ingestion
# anyway (API calls, a stale client, etc.) and also keeps oversized PDFs from ever reaching the
# vector_store.upsert call below, which is what a page count this high tends to blow up.
MAX_PDF_PAGES = 30


class PDFIngestor(BaseIngestor):
    def __init__(self, storage=None, vector_store=None, chunker: BaseChunker = None):
        super().__init__(storage=storage, vector_store=vector_store)
        self.chunker = chunker or DoclingChunker()
        self.errors = []
        self._pages_cache: dict[str, tuple] = {}

    def _pages_cached(self, file_path: str) -> tuple:
        if file_path not in self._pages_cache:
            self._pages_cache[file_path] = parse_pdf_pages(file_path)
        return self._pages_cache[file_path]

    def validate(self, file_path: str) -> bool:
        if not os.path.isfile(file_path) or os.path.getsize(file_path) == 0:
            self.errors = ["file does not exist or is empty"]
            return False
        try:
            reader = pypdf.PdfReader(file_path)
            page_count = len(reader.pages)
            if page_count == 0:
                self.errors = ["PDF has 0 pages"]
                return False
            if page_count > MAX_PDF_PAGES:
                self.errors = [f"PDF has {page_count} pages - exceeds the {MAX_PDF_PAGES}-page limit"]
                return False
            self.errors = []
            return True
        except Exception as exc:
            self.errors = [str(exc)]
            return False

    def extract_metadata(self, file_path: str) -> dict:
        pages, _ = self._pages_cached(file_path)
        scanned = all(is_scanned(page_doc) for _, page_doc in pages) if pages else True
        return {"page_count": len(pages), "is_scanned": scanned}

    def ingest(
        self, file_path: str, workspace_id: str, file_id: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> IngestionResult:
        try:
            if self.vector_store is None:
                raise RuntimeError("no vector store provided")

            pages, errors = self._pages_cached(file_path)
            errors = list(errors)
            total_pages = len(pages)

            chunk_records = []
            extracted_tables = []
            table_chunk_records = []
            chunk_index = 0
            table_index = 0

            if progress_callback and total_pages:
                # Fires once up front with done=0 so the client learns pages_total (and that
                # ingestion has actually started) before the first page finishes chunking -
                # otherwise a slow first page would show no progress bar at all for a while.
                try:
                    progress_callback(0, total_pages)
                except Exception:
                    pass

            for pages_done, (page_no, page_doc) in enumerate(pages, start=1):
                page_chunks = self.chunker.chunk_document(page_doc)
                for chunk in page_chunks:
                    chunk_records.append(ChunkRecord(
                        chunk_id=f"{file_id}_{chunk_index}_{uuid.uuid4().hex[:8]}",
                        file_id=file_id,
                        workspace_id=workspace_id,
                        text=chunk.text,
                        metadata={"page": page_no, "chunk_index": chunk_index, "section": chunk.section},
                    ))
                    chunk_index += 1

                page_tables, page_table_records, page_candidate_count = self._extract_tables(
                    page_doc, page_chunks, page_no, workspace_id, file_id, errors, table_index,
                )
                table_index += page_candidate_count
                extracted_tables.extend(page_tables)
                table_chunk_records.extend(page_table_records)

                if progress_callback:
                    try:
                        progress_callback(pages_done, total_pages)
                    except Exception:
                        pass

            all_records = chunk_records + table_chunk_records
            if all_records:
                try:
                    self.vector_store.upsert(all_records)
                except Exception as exc:
                    return IngestionResult(
                        file_id=file_id,
                        workspace_id=workspace_id,
                        status="failed",
                        output_ref="",
                        schema_summary={},
                        errors=[f"Failed to index in the vector store: {exc}"],
                        error_kind="vector_store_size_exceeded" if is_size_related_error(exc) else None,
                    )

            status = "success" if not errors else "partial"

            return IngestionResult(
                file_id=file_id,
                workspace_id=workspace_id,
                status=status,
                output_ref=f"workspace_{workspace_id}",
                schema_summary={"page_count": len(pages)},
                chunk_count=len(chunk_records),
                extracted_tables=extracted_tables,
                errors=errors,
            )
        except Exception as exc:
            return IngestionResult(
                file_id=file_id,
                workspace_id=workspace_id,
                status="failed",
                output_ref="",
                schema_summary={},
                errors=[str(exc)],
            )

    def _extract_tables(
        self, document, chunks: list, page_no: int, workspace_id: str, file_id: str,
        errors: list, start_index: int,
    ) -> tuple:
        tables = extract_tables(document, chunks, page_override=page_no)
        if not tables:
            return [], [], 0

        if self.storage is None:
            errors.append(f"no storage provided, skipped {len(tables)} table(s)")
            return [], [], len(tables)

        extracted_tables = []
        table_chunk_records = []
        for offset, table in enumerate(tables):
            table_file_id = f"{file_id}_table_{start_index + offset}"
            dataframe = table["dataframe"]
            try:
                self.storage.write(dataframe, f"{workspace_id}/{table_file_id}.parquet")
                output_ref = table_file_id
            except Exception as exc:
                errors.append(f"table on page {table['page']} skipped - failed to write: {exc}")
                continue
            columns = [str(c) for c in dataframe.columns]
            row_count = len(dataframe)

            extracted_tables.append({
                "file_id": table_file_id,
                "output_ref": output_ref,
                "page": table["page"],
                "row_count": row_count,
                "columns": columns,
            })

            caption = table["caption"]
            text = f"{caption}\nColumns: {', '.join(columns)}"
            table_chunk_records.append(ChunkRecord(
                chunk_id=f"{table_file_id}_{uuid.uuid4().hex[:8]}",
                file_id=file_id,
                workspace_id=workspace_id,
                text=text,
                metadata={
                    "page": table["page"],
                    "type": "table",
                    "table_ref": table_file_id,
                    "row_count": row_count,
                    "columns": ", ".join(columns),
                },
            ))

        return extracted_tables, table_chunk_records, len(tables)
