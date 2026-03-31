"""
examnight_ai.py
===============
ExamNight AI — Medical Exam Assistant
Built with LangChain + FAISS + HuggingFace + Streamlit

Features:
  • Upload your own PDFs and ask questions instantly
  • Fallback to built-in Gale Medical Encyclopedia
  • Combined retrieval mode (uploaded + built-in)
  • Source transparency — every answer cites document + page
  • Strict RAG and Hybrid answer modes
  • Exam mode for 2-mark, 5-mark, 6-mark, 7-mark, 10-mark, comparison, and viva answers

Run:
    streamlit run examnight_ai.py
"""

import os
import tempfile
from typing import Optional, List, Any, Tuple

import streamlit as st
from dotenv import load_dotenv, find_dotenv
from huggingface_hub import InferenceClient
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.language_models.llms import LLM
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(find_dotenv())

# ── Constants ─────────────────────────────────────────────────────────────────
DB_FAISS_PATH = "vectorstore/db_faiss"
HUGGINGFACE_REPO_ID = "Qwen/Qwen2.5-7B-Instruct"
HF_TOKEN = os.getenv("HF_TOKEN")

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


# ── Custom LLM via HF InferenceClient.chat_completion ────────────────────────
class ExamNightLLM(LLM):
    """Uses HuggingFace InferenceClient chat_completion."""

    repo_id: str
    hf_token: str
    max_new_tokens: int = 512
    temperature: float = 0.4

    @property
    def _llm_type(self) -> str:
        return "examnight_hf_chat"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        client = InferenceClient(model=self.repo_id, token=self.hf_token)
        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content


# ── Cached resource loaders ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="🔗 Loading embedding model...")
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource(show_spinner="📚 Loading built-in knowledge base...")
def load_default_vectorstore():
    embeddings = get_embeddings()
    db = FAISS.load_local(
        DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True
    )
    return db


@st.cache_resource(show_spinner="🤖 Loading AI model...")
def load_llm():
    if not HF_TOKEN:
        raise ValueError("Missing HF_TOKEN. Add HF_TOKEN in your .env file.")
    return ExamNightLLM(
        repo_id=HUGGINGFACE_REPO_ID,
        hf_token=HF_TOKEN,
        max_new_tokens=512,
        temperature=0.4,
    )


def build_vectorstore_from_pdfs(uploaded_files) -> Tuple[FAISS, int, int]:
    """Process uploaded PDF files and return a FAISS vectorstore."""
    all_documents = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for uploaded_file in uploaded_files:
            tmp_path = os.path.join(tmpdir, uploaded_file.name)
            with open(tmp_path, "wb") as f:
                f.write(uploaded_file.read())
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()

            for doc in docs:
                doc.metadata["source"] = uploaded_file.name
            all_documents.extend(docs)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(all_documents)
    embeddings = get_embeddings()
    db = FAISS.from_documents(chunks, embeddings)
    return db, len(all_documents), len(chunks)


def _doc_key(doc: Any) -> str:
    source = str(doc.metadata.get("source", ""))
    page = str(doc.metadata.get("page", ""))
    text_head = doc.page_content[:120]
    return f"{source}|{page}|{text_head}"


def retrieve_ranked_documents(
    query: str,
    vectorstores: List[Tuple[str, FAISS]],
    k_per_store: int = 4,
    top_k: int = 6,
):
    """Retrieve and merge nearest chunks from multiple vector stores."""
    ranked = []

    for kb_label, store in vectorstores:
        docs_with_score = store.similarity_search_with_score(query, k=k_per_store)
        for doc, score in docs_with_score:
            doc.metadata["kb_label"] = kb_label
            ranked.append((doc, float(score)))

    ranked.sort(key=lambda item: item[1])  # Lower FAISS distance is better.

    dedup = {}
    for doc, score in ranked:
        key = _doc_key(doc)
        if key not in dedup:
            dedup[key] = (doc, score)
        if len(dedup) >= top_k:
            break

    return [entry[0] for entry in dedup.values()]


def build_chat_history(messages, max_turns: int = 4) -> str:
    if not messages:
        return "No prior conversation."

    selected = messages[-(max_turns * 2):]
    lines = []
    for msg in selected:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def build_context_block(source_documents) -> str:
    if not source_documents:
        return "No relevant document context retrieved."

    blocks = []
    for i, doc in enumerate(source_documents, 1):
        source = os.path.basename(str(doc.metadata.get("source", "Unknown")))
        page = doc.metadata.get("page", None)
        kb_label = str(doc.metadata.get("kb_label", "Knowledge Base"))
        page_label = f"Page {page + 1}" if isinstance(page, int) else "Page N/A"

        blocks.append(
            f"[{i}] Source: {source} | {page_label} | KB: {kb_label}\n"
            f"{doc.page_content.strip()}"
        )

    return "\n\n".join(blocks)


def resolve_exam_profile(selected_profile: str, query: str) -> str:
    """Infer an exam-style answer shape when auto mode is selected."""
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
) -> Tuple[str, str]:
    """Return prompt instructions and the resolved exam profile label."""
    if presentation_mode == "standard":
        return (
            "Give a direct, concise answer. Use short paragraphs or bullets only when they improve clarity. "
            "Avoid unnecessary filler and keep the response easy to study.",
            "standard",
        )

    resolved_profile = resolve_exam_profile(exam_profile, query)

    exam_instructions = {
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

    return exam_instructions[resolved_profile], resolved_profile


def format_profile_label(profile: str) -> str:
    labels = {
        "standard": "Standard",
        "auto": "Auto",
        "two_mark": "2-Mark",
        "five_mark": "5-Mark",
        "six_mark": "6-Mark",
        "seven_mark": "7-Mark",
        "ten_mark": "10-Mark",
        "comparison": "Comparison",
        "viva": "Viva",
        "none": "None",
    }
    return labels.get(profile, profile.replace("_", " ").title())


def generate_answer(
    query: str,
    llm: ExamNightLLM,
    source_documents,
    answer_mode: str,
    presentation_mode: str,
    exam_profile: str,
    chat_messages,
) -> Tuple[str, str]:
    if answer_mode == "strict" and not source_documents:
        return "I don't have information on that in my knowledge base.", "none"

    chat_history = build_chat_history(chat_messages)
    context_block = build_context_block(source_documents)
    response_instructions, resolved_profile = build_response_instructions(
        presentation_mode,
        exam_profile,
        query,
    )

    template = (
        STRICT_PROMPT_TEMPLATE
        if answer_mode == "strict"
        else HYBRID_PROMPT_TEMPLATE
    )

    prompt = PromptTemplate(
        template=template,
        input_variables=["chat_history", "context", "question", "response_instructions"],
    ).format(
        chat_history=chat_history,
        context=context_block,
        question=query,
        response_instructions=response_instructions,
    )

    answer = llm.invoke(prompt)
    if isinstance(answer, str):
        return answer.strip(), resolved_profile

    return str(answer).strip(), resolved_profile


def render_source_cards(source_documents):
    if not source_documents:
        return

    st.markdown("##### 📄 Sources Used")
    for i, doc in enumerate(source_documents, 1):
        source = os.path.basename(str(doc.metadata.get("source", "Unknown")))
        page = doc.metadata.get("page", None)
        kb_label = str(doc.metadata.get("kb_label", "Knowledge Base"))

        label = f"📘 [{i}] {source}"
        if page is not None:
            label += f" — Page {page + 1}"
        label += f" | {kb_label}"

        with st.expander(label, expanded=False):
            st.markdown(
                f"""
                <div style="
                    background:#F0F4FF;
                    border-left:4px solid #1565C0;
                    padding:10px 14px;
                    border-radius:6px;
                    font-size:0.88rem;
                    color:#1F2937;
                    line-height:1.6;
                ">
                {doc.page_content.strip()}
                </div>
                """,
                unsafe_allow_html=True,
            )


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ExamNight AI — Medical Exam Assistant",
    page_icon="🏥",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .status-card {
        border-radius: 10px;
        padding: 10px 14px;
        margin: 8px 0;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .badge-custom {
        background: #E8F5E9;
        color: #2E7D32;
        border-left: 4px solid #2E7D32;
    }
    .badge-upload {
        background: #E3F2FD;
        color: #1565C0;
        border-left: 4px solid #1565C0;
    }
    .badge-hybrid {
        background: #FFF8E1;
        color: #9A6700;
        border-left: 4px solid #E3A008;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ── Session state init ────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_vectorstore" not in st.session_state:
    st.session_state.uploaded_vectorstore = None
if "uploaded_file_names" not in st.session_state:
    st.session_state.uploaded_file_names = []
if "active_source" not in st.session_state:
    st.session_state.active_source = "default"
if "answer_mode" not in st.session_state:
    st.session_state.answer_mode = "strict"
if "presentation_mode" not in st.session_state:
    st.session_state.presentation_mode = "standard"
if "exam_profile" not in st.session_state:
    st.session_state.exam_profile = "auto"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/caduceus.png", width=64)
    st.title("🏥 ExamNight AI")
    st.markdown("**AI-Powered Medical Q&A**")
    st.divider()

    st.markdown("#### 📤 Upload Your PDFs")
    st.caption("Upload semester PDFs (notes/slides/books) for exam-focused answers.")

    uploaded_files = st.file_uploader(
        label="Drop PDF files here",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        new_names = [f.name for f in uploaded_files]
        if new_names != st.session_state.uploaded_file_names:
            with st.spinner("⚙️ Processing PDFs and building your vector index..."):
                try:
                    db, pages, chunks = build_vectorstore_from_pdfs(uploaded_files)
                    st.session_state.uploaded_vectorstore = db
                    st.session_state.uploaded_file_names = new_names
                    st.session_state.active_source = "uploaded"
                    st.session_state.messages = []
                    st.success(f"✅ Processed {pages} pages → {chunks} chunks")
                except Exception as e:
                    st.error(f"❌ Error processing PDFs: {e}")

        if st.session_state.uploaded_vectorstore:
            st.markdown(
                f"""<div class="status-card badge-upload">
                    📂 <b>{len(st.session_state.uploaded_file_names)} PDF(s) loaded</b><br>
                    {'<br>'.join(f'• {n}' for n in st.session_state.uploaded_file_names)}
                </div>""",
                unsafe_allow_html=True,
            )

    st.divider()

    st.markdown("#### 🗂️ Active Knowledge Source")

    has_uploaded = st.session_state.uploaded_vectorstore is not None
    has_default = os.path.exists(DB_FAISS_PATH)

    source_options = []
    if has_uploaded:
        source_options.append("📤 Uploaded PDFs")
    if has_default:
        source_options.append("📚 Built-in Encyclopedia")
    if has_uploaded and has_default:
        source_options.append("🧠 Combined (Uploaded + Built-in)")

    if not source_options:
        st.warning("No knowledge base available. Upload PDFs or run create_memory_llm.py")
    else:
        selected_source = st.radio(
            label="Choose source:",
            options=source_options,
            label_visibility="collapsed",
        )

        if "Combined" in selected_source:
            st.session_state.active_source = "combined"
        elif "Uploaded" in selected_source:
            st.session_state.active_source = "uploaded"
        else:
            st.session_state.active_source = "default"

    st.divider()

    st.markdown("#### 🎯 Answer Mode")
    selected_mode = st.radio(
        label="Choose answer mode:",
        options=[
            "📘 Strict RAG (documents only)",
            "⚖️ Hybrid (documents + model knowledge)",
        ],
        label_visibility="collapsed",
    )

    st.session_state.answer_mode = "strict" if "Strict" in selected_mode else "hybrid"

    if st.session_state.answer_mode == "strict":
        st.markdown(
            '<div class="status-card badge-custom">🟢 Strict mode: answers only from retrieved text</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-card badge-hybrid">🟡 Hybrid mode: retrieval first + supplementary knowledge</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    st.markdown("#### 📝 Answer Presentation")
    selected_presentation = st.radio(
        label="Choose answer presentation:",
        options=[
            "Standard Answer",
            "Exam Answer",
        ],
        label_visibility="collapsed",
    )
    st.session_state.presentation_mode = (
        "standard" if "Standard" in selected_presentation else "exam"
    )

    if st.session_state.presentation_mode == "exam":
        exam_profile_map = {
            "Auto Detect": "auto",
            "2-Mark Answer": "two_mark",
            "5-Mark Answer": "five_mark",
            "6-Mark Answer": "six_mark",
            "7-Mark Answer": "seven_mark",
            "10-Mark Answer": "ten_mark",
            "Comparison Answer": "comparison",
            "Viva Answer": "viva",
        }
        selected_exam_profile = st.selectbox(
            "Exam format",
            options=list(exam_profile_map.keys()),
            index=0,
        )
        st.session_state.exam_profile = exam_profile_map[selected_exam_profile]
        st.markdown(
            '<div class="status-card badge-upload">📚 Exam mode: structured semester-style answers</div>',
            unsafe_allow_html=True,
        )
    else:
        st.session_state.exam_profile = "auto"
        st.markdown(
            '<div class="status-card badge-custom">✍️ Standard mode: normal study assistant answers</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    st.markdown("#### ℹ️ How it works")
    st.markdown(
        """
    1. Upload semester PDFs or use built-in encyclopedia
    2. Choose strict or hybrid answering mode
    3. Choose standard or exam answer presentation
    4. ExamNight AI retrieves top relevant chunks (RAG)
    5. Qwen2.5 generates the final answer
    6. Source pages are shown for verification
    """
    )
    st.divider()

    st.warning(
        "⚠️ ExamNight AI is for informational use only. "
        "Always consult a qualified doctor for medical advice."
    )

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown(
    """
<h2 style='color:#1565C0; margin-bottom:0'>🏥 ExamNight AI</h2>
<p style='color:#6B7280; margin-top:4px'>
    Upload semester PDFs or use the built-in encyclopedia.
    Answers are grounded in retrieved documents with source citations.
</p>
<hr style='border:1px solid #E5E7EB; margin-bottom:20px'>
""",
    unsafe_allow_html=True,
)

# ── Determine active vectorstores ─────────────────────────────────────────────
active_vectorstores: List[Tuple[str, FAISS]] = []
source_label = ""

if st.session_state.active_source == "uploaded" and st.session_state.uploaded_vectorstore:
    active_vectorstores = [("Uploaded PDFs", st.session_state.uploaded_vectorstore)]
    source_label = "📤 **Uploaded PDFs**"
elif st.session_state.active_source == "combined":
    if st.session_state.uploaded_vectorstore:
        active_vectorstores.append(("Uploaded PDFs", st.session_state.uploaded_vectorstore))
    if os.path.exists(DB_FAISS_PATH):
        active_vectorstores.append(("Gale Encyclopedia", load_default_vectorstore()))
    source_label = "🧠 **Combined Source**"
elif os.path.exists(DB_FAISS_PATH):
    active_vectorstores = [("Gale Encyclopedia", load_default_vectorstore())]
    source_label = "📚 **Gale Medical Encyclopedia**"

if not active_vectorstores:
    st.info("👆 Upload a PDF in the sidebar or build the default FAISS index to get started.")
    st.stop()

st.markdown(
    f"<span style='background:#E3F2FD;color:#1565C0;border-radius:20px;"
    f"padding:4px 12px;font-size:0.82rem;font-weight:600'>"
    f"Active source: {source_label} | Retrieval: {st.session_state.answer_mode.upper()} | "
    f"Presentation: {st.session_state.presentation_mode.upper()}"
    f"</span>",
    unsafe_allow_html=True,
)
st.markdown("<br>", unsafe_allow_html=True)

# ── Render chat history ───────────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and "resolved_profile" in message:
            st.caption(f"Answer format: {format_profile_label(message['resolved_profile'])}")
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            render_source_cards(message["sources"])

# ── Chat input ────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask a question about your document...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("🔍 Retrieving context and generating answer..."):
            try:
                llm = load_llm()
                source_documents = retrieve_ranked_documents(
                    user_input,
                    active_vectorstores,
                    k_per_store=4,
                    top_k=6,
                )
                result, resolved_profile = generate_answer(
                    query=user_input,
                    llm=llm,
                    source_documents=source_documents,
                    answer_mode=st.session_state.answer_mode,
                    presentation_mode=st.session_state.presentation_mode,
                    exam_profile=st.session_state.exam_profile,
                    chat_messages=st.session_state.messages,
                )

                st.caption(f"Answer format: {format_profile_label(resolved_profile)}")
                st.markdown(result)
                render_source_cards(source_documents)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result,
                        "resolved_profile": resolved_profile,
                        "sources": source_documents,
                    }
                )

            except Exception as e:
                err_msg = f"❌ Error: {str(e)}"
                st.error(err_msg)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": err_msg,
                    }
                )
