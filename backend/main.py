
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pdf_processor import extract_pdf_text, PDFProcessingError
from rag_pipeline import add_pdf_to_index, answer_question


app = FastAPI(
    title="Personal Knowledge AI",
    description="Chat with your PDFs using RAG and Gemini",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

MAX_FILE_SIZE = 15 * 1024 * 1024


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


@app.get("/")
def home():
    return {
        "message": "Personal Knowledge AI API is running",
        "docs": "/docs",
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/api/pdf/upload")
async def upload_pdf(file: UploadFile = File(...)):
    filename = file.filename or "uploaded.pdf"

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed.",
        )

    if file.content_type not in (
        "application/pdf",
        "application/octet-stream",
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid content type. Upload a PDF file.",
        )

    try:
        pdf_bytes = await file.read(MAX_FILE_SIZE + 1)

        if len(pdf_bytes) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="File exceeds the 15 MB upload limit.",
            )

        try:
            result = extract_pdf_text(pdf_bytes)
        except PDFProcessingError as exc:
            raise HTTPException(
                status_code=422,
                detail=str(exc),
            ) from exc

        if result["is_likely_scanned"]:
            return {
                "filename": filename,
                "message": "No selectable text found. OCR is needed.",
                "page_count": result["page_count"],
                "pages_with_text": 0,
                "chunks_indexed": 0,
                "is_likely_scanned": True,
            }

        try:
            chunk_count = add_pdf_to_index(
                result["pages"],
                filename,
            )
        except Exception as exc:
            print(f"Indexing error: {type(exc).__name__}")
            raise HTTPException(
                status_code=500,
                detail="Could not index this PDF. Check server logs.",
            ) from exc

        if chunk_count == 0:
            raise HTTPException(
                status_code=422,
                detail="No usable text was found in this PDF.",
            )

        return {
            "filename": filename,
            "message": "PDF processed and indexed successfully.",
            "page_count": result["page_count"],
            "pages_with_text": result["pages_with_text"],
            "chunks_indexed": chunk_count,
            "is_likely_scanned": False,
        }

    finally:
        await file.close()


@app.post("/api/chat")
def chat(request: ChatRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    try:
        result = answer_question(question)
        return result

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        print(f"Chat error: {type(exc).__name__}")
        raise HTTPException(
            status_code=502,
            detail=(
                "Could not generate an answer. Check the Gemini model, "
                "API key, quota, network connection, and server logs."
            ),
        ) from exc
