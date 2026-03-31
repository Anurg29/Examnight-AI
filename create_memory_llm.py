"""
create_memory_llm.py
====================
Step 1 — Run this ONCE to build the FAISS vector store from your PDF files.
Put your medical PDFs inside the /data folder, then run:
    python create_memory_llm.py
"""

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

DATA_PATH      = "data/"
DB_FAISS_PATH  = "vectorstore/db_faiss"


# ── Step 1: Load all PDFs from /data ────────────────────────────────────────
def load_pdf_files(data_path):
    loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} pages from PDFs in '{data_path}'")
    return documents


# ── Step 2: Split into chunks ────────────────────────────────────────────────
def create_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} chunks")
    return chunks


# ── Step 3: Embed using sentence-transformers (free, no API key needed) ──────
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# ── Step 4: Save FAISS vector store ─────────────────────────────────────────
def build_vectorstore():
    if not os.path.exists(DATA_PATH) or not os.listdir(DATA_PATH):
        print(f"❌ No PDFs found in '{DATA_PATH}'. Add PDF files and retry.")
        return

    documents      = load_pdf_files(DATA_PATH)
    chunks         = create_chunks(documents)
    embedding_model = get_embedding_model()

    os.makedirs("vectorstore", exist_ok=True)
    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(DB_FAISS_PATH)
    print(f"✅ Vector store saved to '{DB_FAISS_PATH}'")


if __name__ == "__main__":
    build_vectorstore()
