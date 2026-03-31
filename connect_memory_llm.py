"""
connect_memory_llm.py
=====================
Quick CLI test — run this to verify your RAG pipeline works before launching the UI.
Usage:
    python connect_memory_llm.py
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.language_models.llms import LLM
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from huggingface_hub import InferenceClient
from typing import Optional, List, Any

import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

HF_TOKEN           = os.getenv("HF_TOKEN")
HUGGINGFACE_REPO_ID = "Qwen/Qwen2.5-7B-Instruct"
DB_FAISS_PATH      = "vectorstore/db_faiss"

CUSTOM_PROMPT_TEMPLATE = """
You are ExamNight AI, a medical exam assistant. Use ONLY the context below to answer.
If the answer is not in the context, say "I don't have information on that in my knowledge base."
Never make up medical information.

Context: {context}
Question: {question}

Answer:"""


class ExamNightLLM(LLM):
    repo_id: str
    hf_token: str
    max_new_tokens: int = 512
    temperature: float = 0.5

    @property
    def _llm_type(self) -> str:
        return "examnight_hf_chat"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        client = InferenceClient(model=self.repo_id, token=self.hf_token)
        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content


def load_llm():
    return ExamNightLLM(
        repo_id=HUGGINGFACE_REPO_ID,
        hf_token=HF_TOKEN,
        max_new_tokens=512,
        temperature=0.5,
    )


def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)


def build_qa_chain(llm, vectorstore):
    prompt = PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )


if __name__ == "__main__":
    print("🔄 Loading model and vector store...")
    llm         = load_llm()
    vectorstore = load_vectorstore()
    qa_chain    = build_qa_chain(llm, vectorstore)
    print("✅ ExamNight AI ready!\n")

    while True:
        user_query = input("You: ").strip()
        if user_query.lower() in ["exit", "quit"]:
            break

        response  = qa_chain.invoke({"query": user_query})
        result    = response["result"]
        sources   = response["source_documents"]

        print(f"\nExamNight AI: {result}\n")
        print("📄 Sources:")
        for i, doc in enumerate(sources, 1):
            src  = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "?")
            print(f"  [{i}] {os.path.basename(src)} — Page {page + 1}")
        print()
