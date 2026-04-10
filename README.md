🚀 Gemini Multimodal RAG Application
A Streamlit-based Retrieval-Augmented Generation (RAG) system that combines:
🔷 Google Gemini Embedding (Multimodal) for rich context understanding
⚡ Groq LLM Engine for ultra-fast response generation
🗂️ ChromaDB for efficient vector storage
This application enables accurate, context-aware responses over text and image-based data.
📌 Overview
Traditional LLM systems:
❌ Lack external knowledge
❌ Prone to hallucination
❌ Limited multimodal capabilities
This project implements a Multimodal RAG pipeline:
📥 Retrieve relevant context → 🧠 Enhance query → ⚡ Generate response using Groq LLM
⚙️ Key Features
🧠 Multimodal embeddings using Gemini Embedding 2 (Preview)
⚡ Fast inference powered by Groq
🔍 Semantic search with ChromaDB
📄 PDF processing using PyMuPDF
🖼️ Image handling via Pillow
🌐 Interactive UI with Streamlit
🔐 Environment-based config using python-dotenv
🏗️ Project Structure
Bash
gemini_rag/
│── main.py                 # Streamlit app entry point
│── navigation.py           # Page routing logic
│── pages/                  # UI pages (chat, upload, etc.)
│── .streamlit/             # Streamlit configuration
│── requirements.txt        # Project dependencies
│── README.md               # Documentation
🔄 How It Works
1. Multimodal Data Processing
Extract text from PDFs using PyMuPDF
Load images using Pillow
Convert all inputs into embeddings via Gemini API
2. Vector Storage
Store embeddings in ChromaDB
Efficient similarity-based retrieval
3. Retrieval
User query → embedding
Retrieve most relevant chunks from vector DB
4. Generation (Groq)
Combine:
User query
Retrieved context
Send to Groq LLM
Generate accurate and fast responses
🛠️ Tech Stack
Language: Python
Frontend: Streamlit
LLM Engine: Groq
Embeddings: Google Gemini (Multimodal)
Vector DB: ChromaDB
PDF Processing: PyMuPDF
Image Processing: Pillow
Environment Management: python-dotenv
Database Fix: pysqlite3-binary (for compatibility)
📦 Requirements
Plain text
groq
google-genai
pymupdf
chromadb
streamlit
python-dotenv
pillow
pysqlite3-binary
🚀 Getting Started
1. Clone Repository
Bash
git clone https://github.com/harshit311212/gemini_rag.git
cd gemini_rag
2. Install Dependencies
Bash
pip install -r requirements.txt
3. Setup Environment Variables
Create a .env file:
Environment
GOOGLE_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key
4. Run the Application
Bash
streamlit run main.py
💡 Use Cases
📄 Chat with PDFs
🖼️ Multimodal search (text + images)
🧑‍💻 Developer knowledge assistant
🏢 Enterprise document Q&A
🎓 AI-powered study assistant
