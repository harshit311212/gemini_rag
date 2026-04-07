try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import os
import io
import fitz  # PyMuPDF
from PIL import Image
import chromadb
from google import genai
from dotenv import load_dotenv

load_dotenv()

# Initialize Gemini Client
# Ensure GEMINI_API_KEY is available in your .env
try:
    client = genai.Client()
except Exception as e:
    print(f"Error initializing Gemini client: {e}")
    print("Please ensure your GEMINI_API_KEY is set in your environment.")
    exit(1)

def extract_content_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    chunks = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        
        # 1. Extract Text
        text = page.get_text("text").strip()
        if text:
            # We can do further sub-chunking here if the text is huge, 
            # but since page-level is a good start for a short brochure:
            chunks.append({
                "page_number": page_num + 1,
                "type": "text",
                "content": text
            })
            
        # 2. Extract Images
        image_list = page.get_images(full=True)
        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            
            # Filter out extremely small images (like icons or lines) to save API calls
            width = base_image["width"]
            height = base_image["height"]
            if width < 50 or height < 50:
                continue
                
            image = Image.open(io.BytesIO(image_bytes))
            image_ext = base_image["ext"]
            
            # Convert image to RGB format if it's not (e.g., RGBA or CMYK)
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # Use Gemini to summarize the image
            summary = summarize_image(image, page_num + 1)
            if summary:
                chunks.append({
                    "page_number": page_num + 1,
                    "type": "image",
                    "content": f"Image Description: {summary}"
                })
                
    return chunks

def summarize_image(image: Image.Image, page_num: int) -> str:
    print(f"Summarizing an image on page {page_num} using Gemini 1.5 Flash...")
    prompt = "Describe this image in detail. If it is a chart or dashboard, summarize the key data points or visual elements. If it contains text, accurately transcribe the text. Provide a comprehensive summary useful for answering questions about it."
    
    try:
        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents=[prompt, image]
        )
        return response.text.strip()
    except Exception as e:
        print(f"Error summarizing image: {e}")
        return ""

def create_vector_db(chunks, db_path="./chroma_db", collection_name="dp600_rag"):
    print(f"Initializing ChromaDB at {db_path}...")
    chroma_client = chromadb.PersistentClient(path=db_path)
    
    # Delete existing collection if it exists for a fresh start
    try:
        chroma_client.delete_collection(name=collection_name)
    except Exception:
        pass
        
    collection = chroma_client.create_collection(name=collection_name)
    
    print("Generating embeddings using gemini-embedding-2-preview (Gemini 2)...")
    
    batch_ids = []
    batch_embeddings = []
    batch_documents = []
    batch_metadatas = []
    
    for i, chunk in enumerate(chunks):
        content = chunk["content"]
        # Generate embedding using Gemini 2 Model
        response = client.models.embed_content(
            model='gemini-embedding-2-preview',
            contents=content
        )
        embedding = response.embeddings[0].values
        
        batch_ids.append(f"chunk_{i}")
        batch_embeddings.append(embedding)
        batch_documents.append(content)
        batch_metadatas.append({
            "page_number": chunk["page_number"],
            "type": chunk["type"],
            "document_name": "Multimodal_RAG.pdf"
        })
        print(f"Embedded chunk {i+1}/{len(chunks)}")
        
    # Add to ChromaDB in one batch
    if batch_ids:
        collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings,
            documents=batch_documents,
            metadatas=batch_metadatas
        )
        
    print("Ingestion completed successfully.")

if __name__ == "__main__":
    pdf_file = "Multimodal_RAG.pdf"
    if not os.path.exists(pdf_file):
        print(f"Error: '{pdf_file}' not found in the current directory.")
    else:
        print("Starting ingestion pipeline...")
        extracted_chunks = extract_content_from_pdf(pdf_file)
        if not extracted_chunks:
            print("No text or image content could be extracted.")
        else:
            print(f"Extracted {len(extracted_chunks)} chunks in total.")
            create_vector_db(extracted_chunks)
