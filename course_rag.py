# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "fastapi>=0.115",
#     "uvicorn[standard]>=0.30",
#     "python-dotenv>=1.0",
#     "openai>=1.40",
#     "pydantic>=2.8",
#     "chromadb>=0.5",
#     "pypdf>=4.3",
#     "python-multipart>=0.0.9",
# ]
# ///

"""
Course Assistant RAG API
========================

A small FastAPI RAG application for course assistance.

Workflow:
1. Upload course PDF(s).
2. Extract text.
3. Split the course into overlapping chunks.
4. Create embeddings for each chunk.
5. Store chunks + embeddings in ChromaDB.
6. Ask a question.
7. Retrieve the most relevant course chunks.
8. Ask the LLM to answer ONLY from the retrieved course content.
9. Return the answer and source/page references.

The application uses OpenRouter's OpenAI-compatible API for:
- embeddings
- chat generation

Required:
    OPENROUTER_API_KEY

Optional:
    OPENROUTER_MODEL
    OPENROUTER_EMBED_MODEL
    CHROMA_DB_DIR
    TOP_K
"""

import logging
import os
import re
import uuid
from pathlib import Path
from functools import lru_cache
from typing import Any

import chromadb
from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, File, Form, HTTPException, Query, UploadFile
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, Field
from pypdf import PdfReader

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("course_rag")


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

class Settings:
    def __init__(self) -> None:
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "")

        # Change these in .env if you prefer other OpenRouter models.
        self.openrouter_model = os.getenv(
            "OPENROUTER_MODEL",
            "openai/gpt-4o-mini",
        )
        self.openrouter_embed_model = os.getenv(
            "OPENROUTER_EMBED_MODEL",
            "openai/text-embedding-3-small",
        )

        self.chroma_db_dir = os.getenv("CHROMA_DB_DIR", "./course_chroma_db")
        self.collection_name = os.getenv(
            "CHROMA_COLLECTION",
            "course_documents",
        )

        self.chunk_size = int(os.getenv("CHUNK_SIZE", "1200"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
        self.default_top_k = int(os.getenv("TOP_K", "5"))

        self.app_host = os.getenv("APP_HOST", "127.0.0.1")
        self.app_port = int(os.getenv("APP_PORT", "8000"))

    def require_api_key(self) -> None:
        if not self.openrouter_api_key:
            raise RuntimeError(
                "Missing OPENROUTER_API_KEY. Add it to your .env file."
            )


settings = Settings()


# --------------------------------------------------------------------------
# PDF loading and chunking
# --------------------------------------------------------------------------

class CourseDocumentLoader:
    """Loads a PDF and keeps page information for source citations."""

    @staticmethod
    def load_pdf(path: str) -> list[dict[str, Any]]:
        try:
            reader = PdfReader(path)
        except Exception as exc:
            raise ValueError(f"Could not read PDF: {exc}") from exc

        pages: list[dict[str, Any]] = []

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = re.sub(r"\s+", " ", text).strip()

            if text:
                pages.append(
                    {
                        "page": page_number,
                        "text": text,
                    }
                )

        if not pages:
            raise ValueError(
                "No extractable text was found. "
                "The PDF may be scanned/image-only."
            )

        return pages


class TextChunker:
    """Creates overlapping chunks while preserving page information."""

    def __init__(self, chunk_size: int, overlap: int) -> None:
        if overlap >= chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []

        for page in pages:
            text = page["text"]
            page_number = page["page"]

            start = 0
            while start < len(text):
                end = min(start + self.chunk_size, len(text))
                chunk_text = text[start:end].strip()

                if chunk_text:
                    chunks.append(
                        {
                            "text": chunk_text,
                            "page": page_number,
                        }
                    )

                if end >= len(text):
                    break

                start = end - self.overlap

        return chunks


# --------------------------------------------------------------------------
# OpenRouter clients
# --------------------------------------------------------------------------

class EmbeddingModel:
    def __init__(self, client: OpenAI, model: str) -> None:
        self.client = client
        self.model = model

    def encode(self, text: str) -> list[float]:
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text,
            )
            return response.data[0].embedding
        except OpenAIError as exc:
            logger.exception("Embedding call failed")
            raise RuntimeError(f"Embedding generation failed: {exc}") from exc


class LLMClient:
    def __init__(self, client: OpenAI, model: str) -> None:
        self.client = client
        self.model = model

    def generate(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a course assistant. "
                            "Answer using only the supplied course context. "
                            "If the answer is not supported by the context, "
                            "say that the information was not found in the "
                            "uploaded course material. Do not invent facts."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0.2,
            )
        except OpenAIError as exc:
            logger.exception("LLM call failed")
            raise RuntimeError(f"LLM generation failed: {exc}") from exc

        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("LLM returned an empty response.")

        return content


# --------------------------------------------------------------------------
# Chroma vector store
# --------------------------------------------------------------------------

class CourseVectorStore:
    def __init__(self, persist_dir: str, collection_name: str) -> None:
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        chunks: list[dict[str, Any]],
        embeddings: list[list[float]],
        course_name: str,
        filename: str,
    ) -> int:
        if not chunks:
            return 0

        ids = [str(uuid.uuid4()) for _ in chunks]

        documents = [chunk["text"] for chunk in chunks]

        metadatas = [
            {
                "course_name": course_name,
                "filename": filename,
                "page": int(chunk["page"]),
            }
            for chunk in chunks
        ]

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        return len(chunks)

    def search(
        self,
        embedding: list[float],
        top_k: int,
        course_name: str | None = None,
    ) -> list[dict[str, Any]]:
        where = {"course_name": course_name} if course_name else None

        kwargs: dict[str, Any] = {
            "query_embeddings": [embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }

        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        retrieved: list[dict[str, Any]] = []

        for document, metadata, distance in zip(
            documents,
            metadatas,
            distances,
        ):
            retrieved.append(
                {
                    "text": document,
                    "course_name": metadata.get("course_name", ""),
                    "filename": metadata.get("filename", ""),
                    "page": metadata.get("page"),
                    "distance": distance,
                }
            )

        return retrieved

    def count(self) -> int:
        return self.collection.count()


# --------------------------------------------------------------------------
# RAG service
# --------------------------------------------------------------------------

class CourseRAGService:
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        llm: LLMClient,
        vector_store: CourseVectorStore,
        chunker: TextChunker,
    ) -> None:
        self.embedding_model = embedding_model
        self.llm = llm
        self.vector_store = vector_store
        self.chunker = chunker

    def ingest(
        self,
        pdf_path: str,
        course_name: str,
        filename: str,
    ) -> dict[str, Any]:
        pages = CourseDocumentLoader.load_pdf(pdf_path)
        chunks = self.chunker.split(pages)

        embeddings: list[list[float]] = []

        for index, chunk in enumerate(chunks, start=1):
            logger.info(
                "Embedding chunk %s/%s for %s",
                index,
                len(chunks),
                filename,
            )
            embeddings.append(self.embedding_model.encode(chunk["text"]))

        count = self.vector_store.add_chunks(
            chunks=chunks,
            embeddings=embeddings,
            course_name=course_name,
            filename=filename,
        )

        return {
            "course_name": course_name,
            "filename": filename,
            "pages": len(pages),
            "chunks": count,
        }

    def ask(
        self,
        question: str,
        course_name: str | None,
        top_k: int,
    ) -> dict[str, Any]:
        question_embedding = self.embedding_model.encode(question)

        retrieved = self.vector_store.search(
            embedding=question_embedding,
            top_k=top_k,
            course_name=course_name,
        )

        if not retrieved:
            return {
                "answer": (
                    "I could not find relevant information in the "
                    "uploaded course material."
                ),
                "sources": [],
            }

        context_parts = []

        for index, item in enumerate(retrieved, start=1):
            context_parts.append(
                f"[Source {index} | Course: {item['course_name']} | "
                f"File: {item['filename']} | Page: {item['page']}]\n"
                f"{item['text']}"
            )

        context = "\n\n".join(context_parts)

        prompt = f"""
Student question:
{question}

Retrieved course context:
{context}

Instructions:
- Answer the student's question using the retrieved course context.
- Do not use unsupported information.
- If the context does not contain the answer, explicitly say so.
- Explain concepts clearly and in a student-friendly way.
- When useful, use short bullet points.
- At the end, include a "Sources" section listing the source number,
  filename, and page number used.
"""

        answer = self.llm.generate(prompt)

        sources = [
            {
                "source": index,
                "course_name": item["course_name"],
                "filename": item["filename"],
                "page": item["page"],
                "distance": item["distance"],
                "snippet": item["text"][:300],
            }
            for index, item in enumerate(retrieved, start=1)
        ]

        return {
            "answer": answer,
            "sources": sources,
        }


# --------------------------------------------------------------------------
# API schemas
# --------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    course_name: str | None = None
    top_k: int = Field(default=settings.default_top_k, ge=1, le=20)


class AskResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]


class IngestResponse(BaseModel):
    course_name: str
    filename: str
    pages: int
    chunks: int


class HealthResponse(BaseModel):
    status: str
    indexed_chunks: int


# --------------------------------------------------------------------------
# App wiring
# --------------------------------------------------------------------------

@lru_cache
def get_rag_service() -> CourseRAGService:
    settings.require_api_key()

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=settings.openrouter_api_key,
    )

    embedding_model = EmbeddingModel(
        client,
        settings.openrouter_embed_model,
    )

    llm = LLMClient(
        client,
        settings.openrouter_model,
    )

    vector_store = CourseVectorStore(
        settings.chroma_db_dir,
        settings.collection_name,
    )

    chunker = TextChunker(
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
    )

    return CourseRAGService(
        embedding_model=embedding_model,
        llm=llm,
        vector_store=vector_store,
        chunker=chunker,
    )


router = APIRouter()


@router.post("/courses/upload", response_model=IngestResponse)
async def upload_course(
    course_name: str = Form(..., min_length=1),
    course_pdf: UploadFile = File(...),
) -> IngestResponse:
    if course_pdf.content_type not in (
        "application/pdf",
        "application/x-pdf",
    ):
        raise HTTPException(
            status_code=415,
            detail="course_pdf must be a PDF file.",
        )

    service = get_rag_service()

    # Keep the uploaded file in memory; this avoids leaving course files
    # permanently on the server.
    content = await course_pdf.read()

    temp_path = Path(f"./.tmp_{uuid.uuid4()}.pdf")

    try:
        temp_path.write_bytes(content)

        result = service.ingest(
            pdf_path=str(temp_path),
            course_name=course_name.strip(),
            filename=course_pdf.filename or "course.pdf",
        )

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    finally:
        temp_path.unlink(missing_ok=True)

    return IngestResponse(**result)


@router.post("/ask", response_model=AskResponse)
def ask_course(request: AskRequest) -> AskResponse:
    service = get_rag_service()

    try:
        result = service.ask(
            question=request.question,
            course_name=request.course_name,
            top_k=request.top_k,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return AskResponse(**result)


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    service = get_rag_service()

    return HealthResponse(
        status="ok",
        indexed_chunks=service.vector_store.count(),
    )


@router.get("/")
def root() -> dict[str, str]:
    return {
        "message": "Course Assistant RAG API is running",
        "docs": "/docs",
    }


app = FastAPI(
    title="Course Assistant RAG",
    description=(
        "RAG application that answers student questions from "
        "uploaded course PDFs."
    ),
    version="1.0.0",
)

app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=settings.app_host,
        port=settings.app_port,
    )
