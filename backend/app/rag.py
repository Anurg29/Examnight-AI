from __future__ import annotations

import os
import tempfile
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any
from uuid import uuid4

from huggingface_hub import InferenceClient
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.language_models.llms import LLM
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import DB_FAISS_PATH, HF_TOKEN, HUGGINGFACE_REPO_ID

STRICT_PROMPT_TEMPLATE = """\
You are ExamNight AI, a helpful and accurate medical exam assistant.
Use ONLY the context provided below to answer the user's question.
If the answer is not in the context, say clearly: "I don't have information on that in my knowledge base."
Never guess or fabricate medical information.
Follow the response instructions exactly.

Response Instructions:
{response_instructions}

Chat History:
{chat_history}

Context:
{context}

Question: {question}

Answer:"""

HYBRID_PROMPT_TEMPLATE = """\
You are ExamNight AI, a helpful and accurate medical exam assistant.
Use the provided context first. If context is partial, you may add clearly labeled supplementary knowledge.
Do not fabricate facts.
Follow the response instructions exactly.

Response format:
Document-Based Answer:
- Grounded answer from context with concise clarity.

Supplementary (General Knowledge):
- Optional. Add only if needed to complete the explanation.
- If not needed, write: "Not required."

Response Instructions:
{response_instructions}

Chat History:
{chat_history}

Context:
{context}

Question: {question}

Answer:"""


class ExamNightLLM(LLM):
    repo_id: str
    hf_token: str
    max_new_tokens: int = 768
    temperature: float = 0.4

    @property
    def _llm_type(self) -> str:
        return "examnight_hf_chat"

    def _call(self, prompt: str, stop=None, **kwargs: Any) -> str:
        client = InferenceClient(model=self.repo_id, token=self.hf_token)
        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content


@dataclass
class SessionState:
    uploaded_vectorstore: FAISS | None = None
    uploaded_file_names: list[str] = field(default_factory=list)
    messages: list[dict[str, str]] = field(default_factory=list)


class SessionStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, SessionState] = {}

    def create_session(self) -> str:
        session_id = uuid4().hex
        with self._lock:
            self._sessions[session_id] = SessionState()
        return session_id

    def get_session(self, session_id: str) -> SessionState:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(session_id)
        return session

    def set_uploads(self, session_id: str, vectorstore: FAISS, file_names: list[str]) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(session_id)
            session.uploaded_vectorstore = vectorstore
            session.uploaded_file_names = file_names
            session.messages = []

    def set_messages(self, session_id: str, messages: list[dict[str, str]]) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(session_id)
            session.messages = messages

    def reset(self, session_id: str, clear_documents: bool) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(session_id)
            session.messages = []
            if clear_documents:
                session.uploaded_vectorstore = None
                session.uploaded_file_names = []


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def has_default_vectorstore() -> bool:
    return (DB_FAISS_PATH / "index.faiss").exists() and (DB_FAISS_PATH / "index.pkl").exists()


@lru_cache(maxsize=1)
def load_default_vectorstore() -> FAISS:
    if not has_default_vectorstore():
        raise FileNotFoundError("Default vector store has not been built yet.")
    return FAISS.load_local(
        str(DB_FAISS_PATH),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )


@lru_cache(maxsize=1)
def load_llm() -> ExamNightLLM:
    if not HF_TOKEN:
        raise RuntimeError("Missing HF_TOKEN. Add it to the root .env file.")
    return ExamNightLLM(
        repo_id=HUGGINGFACE_REPO_ID,
        hf_token=HF_TOKEN,
        max_new_tokens=768,
        temperature=0.4,
    )


def build_vectorstore_from_uploads(files: list[tuple[str, bytes]]) -> tuple[FAISS, int, int]:
    all_documents = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for file_name, file_bytes in files:
            tmp_path = os.path.join(tmpdir, file_name)
            with open(tmp_path, "wb") as handle:
                handle.write(file_bytes)
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()
            for document in documents:
                document.metadata["source"] = file_name
            all_documents.extend(documents)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(all_documents)
    vectorstore = FAISS.from_documents(chunks, get_embeddings())
    return vectorstore, len(all_documents), len(chunks)


def resolve_active_vectorstores(session: SessionState, source_mode: str) -> list[tuple[str, FAISS]]:
    vectorstores: list[tuple[str, FAISS]] = []

    if source_mode in {"uploaded", "combined"}:
        if session.uploaded_vectorstore is None and source_mode == "uploaded":
            raise ValueError("Upload PDFs first before using uploaded mode.")
        if session.uploaded_vectorstore is not None:
            vectorstores.append(("Uploaded PDFs", session.uploaded_vectorstore))

    # Removed built-in encyclopedia (Gale Encyclopedia) from vectorstores

    if not vectorstores:
        raise ValueError("No knowledge source is available for the selected mode.")

    return vectorstores


def _doc_key(document: Any) -> str:
    source = str(document.metadata.get("source", ""))
    page = str(document.metadata.get("page", ""))
    return f"{source}|{page}|{document.page_content[:120]}"


def retrieve_ranked_documents(
    query: str,
    vectorstores: list[tuple[str, FAISS]],
    k_per_store: int = 4,
    top_k: int = 6,
):
    ranked: list[tuple[Any, float]] = []

    for kb_label, store in vectorstores:
        for document, score in store.similarity_search_with_score(query, k=k_per_store):
            document.metadata["kb_label"] = kb_label
            ranked.append((document, float(score)))

    ranked.sort(key=lambda item: item[1])

    deduped: dict[str, tuple[Any, float]] = {}
    for document, score in ranked:
        key = _doc_key(document)
        if key not in deduped:
            deduped[key] = (document, score)
        if len(deduped) >= top_k:
            break

    return [item[0] for item in deduped.values()]


def build_chat_history(messages: list[dict[str, str]], max_turns: int = 4) -> str:
    if not messages:
        return "No prior conversation."

    recent_messages = messages[-(max_turns * 2) :]
    lines = []
    for message in recent_messages:
        role = "User" if message["role"] == "user" else "Assistant"
        lines.append(f"{role}: {message['content']}")
    return "\n".join(lines)


def build_context_block(source_documents) -> str:
    if not source_documents:
        return "No relevant document context retrieved."

    blocks = []
    for index, document in enumerate(source_documents, start=1):
        source = os.path.basename(str(document.metadata.get("source", "Unknown")))
        page = document.metadata.get("page")
        kb_label = str(document.metadata.get("kb_label", "Knowledge Base"))
        page_label = f"Page {page + 1}" if isinstance(page, int) else "Page N/A"
        blocks.append(
            f"[{index}] Source: {source} | {page_label} | KB: {kb_label}\n{document.page_content.strip()}"
        )
    return "\n\n".join(blocks)


def resolve_exam_profile(selected_profile: str, query: str) -> str:
    if selected_profile != "auto":
        return selected_profile

    lowered = query.lower()
    if any(token in lowered for token in ["6 mark", "6 marks", "six mark", "six marks"]):
        return "six_mark"
    if any(token in lowered for token in ["7 mark", "7 marks", "seven mark", "seven marks"]):
        return "seven_mark"
    if any(token in lowered for token in ["compare", "difference", "differentiate", "distinguish", "versus", "vs"]):
        return "comparison"
    if any(token in lowered for token in ["viva", "oral", "interview"]):
        return "viva"
    if any(token in lowered for token in ["define", "what is", "who is", "list", "name"]):
        return "two_mark"
    if any(token in lowered for token in ["short note", "brief note", "write a note", "enumerate", "classify"]):
        return "five_mark"
    if any(token in lowered for token in ["explain", "describe", "discuss", "elaborate", "in detail"]):
        return "ten_mark"
    return "five_mark"


def build_response_instructions(
    presentation_mode: str,
    exam_profile: str,
    query: str,
) -> tuple[str, str]:
    if presentation_mode == "standard":
        return (
            "Give a direct, concise answer. Use short paragraphs or bullets only when they improve clarity. "
            "Avoid unnecessary filler and keep the response easy to study.",
            "standard",
        )

    resolved_profile = resolve_exam_profile(exam_profile, query)
    instructions = {
        "two_mark": (
            "Answer like a 2-mark university exam response. Start with a one-line definition or direct answer, "
            "then give 2 to 4 short bullet points. Keep it around 50 to 90 words."
        ),
        "five_mark": (
            "Answer like a 5-mark exam response. Use headings: Definition, Key Points, and Summary. "
            "Cover 4 to 6 important points and keep it around 120 to 180 words."
        ),
        "six_mark": (
            "Answer like a 6-mark exam response. Use headings: Definition, Explanation, and Summary. "
            "Cover 5 to 6 important points with brief explanation and keep it around 150 to 220 words."
        ),
        "seven_mark": (
            "Answer like a 7-mark exam response. Use headings: Introduction, Main Points, and Conclusion. "
            "Cover 6 to 7 important points with slightly more detail and keep it around 180 to 260 words."
        ),
        "ten_mark": (
            "Answer like a 10-mark long answer. Use headings: Introduction, Main Points, and Conclusion. "
            "Present 5 to 8 well-structured points and keep it around 220 to 350 words."
        ),
        "comparison": (
            "Answer in a comparison format. Use a compact table or paired bullets to show at least 4 clear differences, "
            "then end with a one-line summary."
        ),
        "viva": (
            "Answer like a viva preparation response. Use headings: Direct Answer, Important Points, and Possible Viva Follow-Ups. "
            "Keep it concise, high-yield, and easy to speak aloud."
        ),
    }
    return instructions[resolved_profile], resolved_profile


def generate_answer(
    query: str,
    llm: ExamNightLLM,
    source_documents,
    answer_mode: str,
    presentation_mode: str,
    exam_profile: str,
    chat_messages: list[dict[str, str]],
) -> tuple[str, str]:
    if answer_mode == "strict" and not source_documents:
        return "I don't have information on that in my knowledge base.", "none"

    response_instructions, resolved_profile = build_response_instructions(
        presentation_mode,
        exam_profile,
        query,
    )

    prompt = PromptTemplate(
        template=STRICT_PROMPT_TEMPLATE if answer_mode == "strict" else HYBRID_PROMPT_TEMPLATE,
        input_variables=["chat_history", "context", "question", "response_instructions"],
    ).format(
        chat_history=build_chat_history(chat_messages),
        context=build_context_block(source_documents),
        question=query,
        response_instructions=response_instructions,
    )

    answer = llm.invoke(prompt)
    return str(answer).strip(), resolved_profile


def serialise_sources(source_documents) -> list[dict[str, Any]]:
    serialised = []
    for document in source_documents:
        page = document.metadata.get("page")
        serialised.append(
            {
                "source": os.path.basename(str(document.metadata.get("source", "Unknown"))),
                "page": page + 1 if isinstance(page, int) else None,
                "kb_label": str(document.metadata.get("kb_label", "Knowledge Base")),
                "content": document.page_content.strip(),
            }
        )
    return serialised
