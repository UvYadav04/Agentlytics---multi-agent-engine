from datetime import datetime, timezone

from tools.orchestrator.models import FileCatalogEntry


class FileCatalog:
    def __init__(self):
        self.entries = {}

    def add_entry(self, entry) -> None:
        self.entries[entry.file_id] = entry

    def remove_entry(self, file_id: str) -> None:
        self.entries.pop(file_id, None)

    def all(self) -> list:
        return list(self.entries.values())

    def browsable(self) -> list:
        return [e for e in self.entries.values() if is_browsable(e)]

    def get_by_id(self, file_id: str):
        """Direct lookup of file entry by unique file_id."""
        return self.entries.get(file_id)

    def get_by_name(self, filename: str, exact: bool = False) -> list:
        """Find file(s) by filename (supports case-insensitive or exact)."""
        target = filename.strip()
        matches = []
        for e in self.entries.values():
            if exact and e.filename == target:
                matches.append(e)
            elif not exact and (e.filename.lower() == target.lower() or target.lower() in e.filename.lower()):
                matches.append(e)
        return matches

    def find_by_prefix(self, prefix: str) -> list:
        """Find file(s) matching prefix on filename or file_id."""
        pref = prefix.strip().lower()
        return [
            e for e in self.entries.values()
            if e.filename.lower().startswith(pref) or e.file_id.lower().startswith(pref)
        ]


def is_tabular_output_ref(output_ref: str) -> bool:
    return bool(output_ref) and not output_ref.startswith("workspace_")


def is_browsable(entry) -> bool:
    if entry.file_type == "table":
        return "from_xlsx" in (entry.tags or [])
    if entry.file_type in ("pdf", "txt"):
        return True
    return is_tabular_output_ref(entry.output_ref)


def table_catalog_entry(table: dict, *, source_id: str, source_filename: str,
                         source_file_type: str, uploaded_at) -> FileCatalogEntry:
    location = table.get("location") or (
        f"page {table['page']}" if table.get("page") is not None else f"table {table.get('index', '?')}"
    )
    return FileCatalogEntry(
        file_id=table["file_id"],
        filename=f"{source_filename} ({location})",
        file_type="table",
        uploaded_at=uploaded_at,
        size_bytes=0,
        output_ref=table["output_ref"],
        row_count=table.get("row_count"),
        columns=table.get("columns"),
        tags=[f"from_{source_file_type}", f"source:{source_id}"],
    )


def entries_from_ingestion(result, filename: str, file_type: str, size_bytes: int = 0) -> list:
    now = datetime.now(timezone.utc)
    entries = [
        FileCatalogEntry(
            file_id=result.file_id,
            filename=filename,
            file_type=file_type,
            uploaded_at=now,
            size_bytes=size_bytes,
            output_ref=result.output_ref,
            row_count=result.row_count,
            page_count=result.schema_summary.get("page_count"),
            columns=result.schema_summary.get("columns"),
        )
    ]

    entries.extend(
        table_catalog_entry(
            table,
            source_id=result.file_id,
            source_filename=filename,
            source_file_type=file_type,
            uploaded_at=now,
        )
        for table in result.extracted_tables
    )

    return entries
