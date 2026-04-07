try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import os
import streamlit as st
import chromadb
from google import genai
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Streamlit Page Config
st.set_page_config(page_title="Data Bear DP-600 RAG", page_icon="🤖", layout="centered")

st.title("Data Bear DP-600 Multimodal RAG Assistant")

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    try:
        gemini_api_key = st.secrets["GEMINI_API_KEY"]
    except (KeyError, FileNotFoundError):
        pass

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    try:
        groq_api_key = st.secrets["GROQ_API_KEY"]
    except (KeyError, FileNotFoundError):
        pass

if not gemini_api_key or not groq_api_key:
    st.error("⚠️ **API Keys Missing!** Please ensure `GEMINI_API_KEY` and `GROQ_API_KEY` are set in your `.env` file locally or in **Streamlit Advanced Settings -> Secrets** for cloud deployment.")
    st.stop()

# Ensure keys are in os.environ for subprocesses like ingest.py
os.environ["GEMINI_API_KEY"] = gemini_api_key
os.environ["GROQ_API_KEY"] = groq_api_key

# Initialize API Clients
try:
    gemini_client = genai.Client(api_key=gemini_api_key)
    groq_client = Groq(api_key=groq_api_key)
except Exception as e:
    st.error(f"Error initializing API clients: {e}")
    st.stop()

system_prompt_template = """You are a precise, highly accurate assistant for the Data Bear DP-600 training course.
You will be provided with contextual information extracted from the course brochure.

Context:
{retrieved_context}

Instructions:
1. Answer the user's question USING ONLY the provided context.
2. If the answer is present in the context, provide a clear, concise response.
3. If the answer is NOT entirely contained within the context, or if the question is completely unrelated to the provided information, you MUST output exactly this string: "The question is irrelevant to the doc". Do not include any apologies, explanations, or conversational filler.
"""

@st.cache_resource
def get_chroma_collection():
    try:
        if not os.path.exists("./chroma_db"):
            st.info("First run detected: Parsing document and building the knowledge base using Gemini 2.0 Flash. This might take a minute...")
            import subprocess
            import sys
            try:
                result = subprocess.run([sys.executable, "ingest.py"], capture_output=True, text=True)
                if result.returncode != 0:
                    st.error(f"Failed to process document: {result.stderr}")
                    return None
                st.success("Knowledge base built successfully!")
            except Exception as e:
                st.error(f"Failed to process document: {e}")
                return None
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        return chroma_client.get_collection(name="dp600_rag")
    except Exception as e:
        st.error(f"Error loading ChromaDB collection: {e}")
        return None

collection = get_chroma_collection()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_query = st.chat_input("Ask a question about the DP-600 training...")

if user_query:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    if collection is None:
        st.error("System is not initialized. Please run the ingestion script.")
    else:
        with st.spinner("Searching the brochure & generating response..."):
            # 1. Embed user query with gemini-embedding-2-preview
            try:
                response = gemini_client.models.embed_content(
                    model='gemini-embedding-2-preview',
                    contents=user_query
                )
                query_embedding = response.embeddings[0].values
                
                # 2. Retrieve top K chunks from Chroma
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=5
                )
                
                retrieved_context = ""
                if results['documents'] and results['documents'][0]:
                    for i, doc in enumerate(results['documents'][0]):
                        metadata = results['metadatas'][0][i]
                        retrieved_context += f"--- Chunk {i+1} [Page {metadata['page_number']} | Type: {metadata['type']}] ---\n{doc}\n\n"
                else:
                    retrieved_context = "No relevant context found."
                    
                # 3. Generate response with Groq
                formatted_prompt = system_prompt_template.format(retrieved_context=retrieved_context)
                
                groq_response = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": formatted_prompt},
                        {"role": "user", "content": f"User Question: {user_query}"}
                    ],
                    temperature=0.0
                )
                
                answer = groq_response.choices[0].message.content
                
                # Add assistant message to chat history
                st.session_state.messages.append({"role": "assistant", "content": answer})
                with st.chat_message("assistant"):
                    st.markdown(answer)
                    with st.expander("Show Retrieved Context (For Debugging)"):
                        st.text(retrieved_context)
                        
            except Exception as e:
                st.error(f"An error occurred: {e}")
