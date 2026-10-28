
# import fitz  # PyMuPDF
import pymupdf


MAX_PDF_PAGES = 200


class PDFProcessingError(Exception):
    """Raised when a PDF cannot be processed safely."""


def extract_pdf_text(pdf_bytes: bytes) -> dict:
    """
    Extract text from a PDF and preserve page numbers.

    Args:
        pdf_bytes: Raw bytes of the uploaded PDF.

    Returns:
        Dictionary containing page-wise text and full text.
    """
    if not pdf_bytes:
        raise PDFProcessingError("The uploaded file is empty.")

    if not pdf_bytes.startswith(b"%PDF-"):
        raise PDFProcessingError(
            "Invalid PDF file. Please upload a valid PDF."
        )

    try:
        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
            if document.is_encrypted:
                raise PDFProcessingError(
                    "Password-protected PDFs are not supported yet."
                )

            if document.page_count == 0:
                raise PDFProcessingError("The PDF contains no pages.")

            if document.page_count > MAX_PDF_PAGES:
                raise PDFProcessingError(
                    f"PDF exceeds the {MAX_PDF_PAGES}-page limit."
                )

            pages = []

            for page_number, page in enumerate(document, start=1):
                text = page.get_text("text").strip()

                pages.append({
                    "page_number": page_number,
                    "text": text,
                    "character_count": len(text),
                })

            full_text = "\n\n".join(
                f"[Page {page['page_number']}]\n{page['text']}"
                for page in pages
                if page["text"]
            )

            pages_with_text = sum(
                1 for page in pages if page["text"]
            )

            return {
                "page_count": document.page_count,
                "pages_with_text": pages_with_text,
                "full_text": full_text,
                "pages": pages,
                "is_likely_scanned": (
                    pages_with_text == 0 and document.page_count > 0
                ),
            }

    except PDFProcessingError:
        raise

    except Exception as exc:
        raise PDFProcessingError(
            "Unable to read this PDF. It may be damaged or unsupported."
        ) from exc
