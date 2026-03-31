from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .auth import router as auth_router
from .config import CORS_ORIGINS
from .rag import (
    SessionStore,
    build_vectorstore_from_uploads,
    generate_answer,
    has_default_vectorstore,
    load_llm,
    resolve_active_vectorstores,
    retrieve_ranked_documents,
    serialise_sources,
)
from .schemas import (
    ChatRequest,
    ChatResponse,
    ConfigResponse,
    ResetResponse,
    SessionCreateResponse,
    UploadResponse,
)

app = FastAPI(title='ExamNight AI API', version='1.0.0')
session_store = SessionStore()

app.include_router(auth_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS if CORS_ORIGINS != ['*'] else ['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.get('/api/health')
def health_check() -> dict[str, str]:
    return {'status': 'ok'}


@app.get('/api/config', response_model=ConfigResponse)
def get_config() -> ConfigResponse:
    return ConfigResponse(
        source_modes=['default', 'uploaded', 'combined'],
        answer_modes=['strict', 'hybrid'],
        presentation_modes=['standard', 'exam'],
        exam_profiles=[
            'auto',
            'two_mark',
            'five_mark',
            'six_mark',
            'seven_mark',
            'ten_mark',
            'comparison',
            'viva',
        ],
        default_knowledge_base_ready=has_default_vectorstore(),
    )


@app.post('/api/sessions', response_model=SessionCreateResponse)
def create_session() -> SessionCreateResponse:
    return SessionCreateResponse(session_id=session_store.create_session())


@app.post('/api/sessions/{session_id}/documents', response_model=UploadResponse)
async def upload_documents(session_id: str, files: list[UploadFile] = File(...)) -> UploadResponse:
    try:
        session_store.get_session(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail='Session not found.') from exc

    if not files:
        raise HTTPException(status_code=400, detail='Upload at least one PDF file.')

    prepared_files: list[tuple[str, bytes]] = []
    for upload in files:
        if not upload.filename or not upload.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail='Only PDF uploads are supported.')
        content = await upload.read()
        if not content:
            raise HTTPException(status_code=400, detail=f'{upload.filename} is empty.')
        prepared_files.append((upload.filename, content))

    try:
        vectorstore, pages, chunks = build_vectorstore_from_uploads(prepared_files)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f'Unable to process PDFs: {exc}') from exc

    session_store.set_uploads(
        session_id,
        vectorstore,
        [file_name for file_name, _ in prepared_files],
    )
    return UploadResponse(
        session_id=session_id,
        file_names=[file_name for file_name, _ in prepared_files],
        pages=pages,
        chunks=chunks,
    )


@app.post('/api/chat', response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    try:
        session = session_store.get_session(payload.session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail='Session not found.') from exc

    try:
        vectorstores = resolve_active_vectorstores(session, payload.source_mode)
        source_documents = retrieve_ranked_documents(payload.query, vectorstores)
        pending_messages = session.messages + [{'role': 'user', 'content': payload.query}]
        answer, resolved_profile = generate_answer(
            query=payload.query,
            llm=load_llm(),
            source_documents=source_documents,
            answer_mode=payload.answer_mode,
            presentation_mode=payload.presentation_mode,
            exam_profile=payload.exam_profile,
            chat_messages=pending_messages,
        )
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'Chat generation failed: {exc}') from exc

    session_store.set_messages(
        payload.session_id,
        pending_messages + [{'role': 'assistant', 'content': answer, 'resolved_profile': resolved_profile}],
    )
    return ChatResponse(
        answer=answer,
        resolved_profile=resolved_profile,
        sources=serialise_sources(source_documents),
        file_names=session.uploaded_file_names,
        source_mode=payload.source_mode,
        answer_mode=payload.answer_mode,
        presentation_mode=payload.presentation_mode,
    )


@app.post('/api/sessions/{session_id}/reset', response_model=ResetResponse)
def reset_session(session_id: str, clear_documents: bool = False) -> ResetResponse:
    try:
        session_store.reset(session_id, clear_documents)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail='Session not found.') from exc

    return ResetResponse(
        session_id=session_id,
        cleared_chat=True,
        cleared_documents=clear_documents,
    )
