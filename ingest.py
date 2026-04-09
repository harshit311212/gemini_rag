try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import os
import io
import time
import fitz  # PyMuPDF
from PIL import Image
import chromadb
from google import genai
from google.genai import types
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

# ── Gemini client ──────────────────────────────────────────────────────────────
try:
    client = genai.Client()
except Exception as e:
    print(f"Error initializing Gemini client: {e}")
    print("Please ensure GEMINI_API_KEY is set in your environment.")
    exit(1)

EMBED_MODEL = "gemini-embedding-2-preview"

# ── Helpers ────────────────────────────────────────────────────────────────────

def _image_to_part(image: Image.Image) -> types.Part:
    """Encode a PIL image as an inline JPEG Part (no disk I/O)."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")


# Errors that are worth retrying (rate-limit OR transient server-side failures)
_RETRYABLE = ("429", "500", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "INTERNAL")

def _embed_with_retry(contents, label: str, max_retries: int = 3):
    """Call embed_content with exponential back-off on rate-limit and transient errors."""
    for attempt in range(max_retries):
        try:
            response = client.models.embed_content(model=EMBED_MODEL, contents=contents)
            # Guard: API should always return embeddings; if not, treat as transient error
            if not response.embeddings:
                raise ValueError("API returned an empty embeddings list")
            return response
        except Exception as e:
            err = str(e)
            if any(code in err for code in _RETRYABLE):
                wait = 30 * (2 ** attempt)          # 30 s → 60 s → 120 s
                print(f"  Retryable error on {label} (attempt {attempt + 1}/{max_retries}). Waiting {wait}s... [{err[:80]}]")
                time.sleep(wait)
            else:
                print(f"  Non-retryable embedding error on {label}: {e}")
                return None
    print(f"  Gave up embedding {label} after {max_retries} retries.")
    return None


# ── Phase 1 — PDF extraction (sequential; PyMuPDF is not thread-safe) ─────────

def extract_content_from_pdf(pdf_path: str) -> tuple[list, list]:
    """
    Returns two separate lists:
      text_chunks  – dicts with content (str)
      image_chunks – dicts with part (types.Part) + placeholder content
    """
    doc = fitz.open(pdf_path)
    text_chunks: list[dict] = []
    image_chunks: list[dict] = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)

        # --- Text ---
        text = page.get_text("text").strip()
        if text:
            text_chunks.append({
                "page_number": page_num + 1,
                "type": "text",
                "content": text,
            })

        # --- Images ---
        for img_info in page.get_images(full=True):
            xref = img_info[0]
            base_image = doc.extract_image(xref)

            # Skip decorative/small images
            if base_image["width"] < 300 or base_image["height"] < 100:
                continue

            image = Image.open(io.BytesIO(base_image["image"]))
            if image.mode != "RGB":
                image = image.convert("RGB")

            image_chunks.append({
                "page_number": page_num + 1,
                "type": "image",
                "content": f"[Image on page {page_num + 1}]",
                "part": _image_to_part(image),
            })

    print(f"Extraction complete: {len(text_chunks)} text chunks, {len(image_chunks)} image chunks.")
    return text_chunks, image_chunks


# ── Phase 2a — Batch-embed ALL text in a SINGLE API call ─────────────────────

# Gemini embedding API accepts at most 100 items per batch request
_TEXT_BATCH_SIZE = 100

def embed_text_chunks(text_chunks: list[dict]) -> list[dict]:
    """
    Sends text chunks in sub-batches of _TEXT_BATCH_SIZE (API limit).
    Returns only the chunks that were successfully embedded.
    """
    if not text_chunks:
        return []

    embedded: list[dict] = []
    total_batches = (len(text_chunks) + _TEXT_BATCH_SIZE - 1) // _TEXT_BATCH_SIZE
    print(f"Embedding {len(text_chunks)} text chunks in {total_batches} batch(es) of up to {_TEXT_BATCH_SIZE}...")

    for batch_num in range(total_batches):
        start = batch_num * _TEXT_BATCH_SIZE
        end = start + _TEXT_BATCH_SIZE
        batch = text_chunks[start:end]
        texts = [c["content"] for c in batch]

        label = f"text batch {batch_num + 1}/{total_batches} (chunks {start}–{end - 1})"
        response = _embed_with_retry(texts, label=label)
        if response is None:
            print(f"  WARNING: {label} failed entirely — skipping {len(batch)} chunks.")
            continue

        for chunk, emb in zip(batch, response.embeddings):
            chunk["embedding"] = emb.values
            embedded.append(chunk)

        print(f"  ✓ {label} done ({len(batch)} embeddings received).")

    print(f"  Text embedding complete: {len(embedded)}/{len(text_chunks)} chunks embedded.")
    return embedded


# ── Phase 2b — Concurrently embed all images ──────────────────────────────────

def _embed_single_image(chunk: dict, index: int, total: int) -> dict | None:
    """Worker: embed one image chunk and return the chunk with its embedding."""
    response = _embed_with_retry(
        contents=[chunk["part"]],
        label=f"image {index + 1}/{total} (page {chunk['page_number']})",
    )
    if response is None:
        return None
    chunk["embedding"] = response.embeddings[0].values
    return chunk


def embed_image_chunks(image_chunks: list[dict], max_workers: int = 8) -> list[dict]:
    """
    Fires one concurrent API call per image chunk.
    max_workers=8 gives good throughput without overloading free-tier quotas.
    """
    if not image_chunks:
        return []

    total = len(image_chunks)
    print(f"Embedding {total} image chunks in parallel (max_workers={max_workers})...")

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_chunk = {
            executor.submit(_embed_single_image, chunk, i, total): chunk
            for i, chunk in enumerate(image_chunks)
        }
        for future in as_completed(future_to_chunk):
            result = future.result()
            if result is not None:
                results.append(result)
                print(f"  ✓ Image embedded — page {result['page_number']} ({len(results)}/{total})")

    return results


# ── Phase 3 — Store in ChromaDB ───────────────────────────────────────────────

def create_vector_db(
    embedded_chunks: list[dict],
    db_path: str = "./chroma_db",
    collection_name: str = "dp600_rag",
):
    print(f"\nInitializing ChromaDB at '{db_path}'...")
    chroma_client = chromadb.PersistentClient(path=db_path)

    try:
        chroma_client.delete_collection(name=collection_name)
    except Exception:
        pass

    collection = chroma_client.create_collection(name=collection_name)

    ids, documents, metadatas, embeddings = [], [], [], []
    for i, chunk in enumerate(embedded_chunks):
        ids.append(f"chunk_{i}")
        documents.append(chunk["content"])
        metadatas.append({
            "page_number": chunk["page_number"],
            "type": chunk["type"],
            "document_name": "Multimodal_RAG.pdf",
        })
        embeddings.append(chunk["embedding"])

    if ids:
        collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        print(f"Successfully stored {len(ids)} chunks in ChromaDB.")

    print("Ingestion completed successfully.")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pdf_file = "Multimodal_RAG.pdf"
    if not os.path.exists(pdf_file):
        print(f"Error: '{pdf_file}' not found in the current directory.")
    else:
        t0 = time.perf_counter()
        print("Starting ingestion pipeline...\n")

        # Phase 1 — Extract
        text_chunks, image_chunks = extract_content_from_pdf(pdf_file)

        if not text_chunks and not image_chunks:
            print("No content could be extracted.")
        else:
            # Phase 2 — Embed (text batch + image parallel, can overlap if desired)
            embedded_text   = embed_text_chunks(text_chunks)
            embedded_images = embed_image_chunks(image_chunks)

            all_chunks = embedded_text + embedded_images

            if not all_chunks:
                print("No chunks were successfully embedded.")
            else:
                # Phase 3 — Store
                create_vector_db(all_chunks)

        elapsed = time.perf_counter() - t0
        print(f"\nTotal ingestion time: {elapsed:.1f}s")
