
from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from pypdf import PdfReader

from app.chunking import split_text_into_chunks


app = FastAPI(
    title="Document Q&A API",
    description="A RAG application for answering questions from documents",
    version="0.1.0",
)


@app.get("/")
def home():
    return {"message": "Welcome to Document Q&A API"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    # 1. Validate the filename
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF file.",
        )

    # 2. Read the uploaded file
    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    # 3. Extract text and create chunks
    try:
        reader = PdfReader(BytesIO(contents))

        pages = []
        all_chunks = []

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""

            if not text.strip():
                continue

            pages.append(
                {
                    "page_number": page_number,
                    "text": text,
                }
            )

            chunks = split_text_into_chunks(
                text,
                chunk_size=500,
                overlap=50,
            )

            for chunk_index, chunk in enumerate(chunks, start=1):
                all_chunks.append(
                    {
                        "chunk_id": f"page_{page_number}_chunk_{chunk_index}",
                        "filename": file.filename,
                        "page_number": page_number,
                        "text": chunk,
                    }
                )

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Could not read or process this PDF file.",
        ) from exc

    # 4. Reject PDFs without extractable text
    if not pages:
        raise HTTPException(
            status_code=400,
            detail="No extractable text found in the PDF.",
        )

    # 5. Return the extracted text and chunks
    return {
        "filename": file.filename,
        "total_pages": len(reader.pages),
        "pages_with_text": len(pages),
        "total_chunks": len(all_chunks),
        "chunks": all_chunks,
    }
