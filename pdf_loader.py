from io import BytesIO
from pypdf import PdfReader
from rag_store import ingest_document


def ingest_pdf_bytes(file_bytes: bytes, source_name: str) -> dict:
    """PDF байттарын оқып, әр беттің нөмірін сақтай отырып Qdrant-қа жүктейді."""
    reader = PdfReader(BytesIO(file_bytes))
    total_chunks = 0
    pages_processed = 0

    for page_idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        cleaned_text = " ".join(text.split())
        if cleaned_text:
            added = ingest_document(
                text=cleaned_text,
                source_name=source_name,
                page=page_idx,
            )
            total_chunks += added
            pages_processed += 1

    return {
        "source": source_name,
        "pages_processed": pages_processed,
        "chunks_added": total_chunks,
    }