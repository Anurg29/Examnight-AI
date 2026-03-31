# ExamNight AI

ExamNight AI is now structured as a full-stack project with a FastAPI backend and a React frontend.
It keeps the original document-grounded RAG flow, adds exam-style answer shaping, and exposes everything through a proper API and interactive web UI.

## Stack

- `frontend/`: React + Vite chat interface
- `backend/`: FastAPI API for sessions, uploads, and exam-style chat
- `examnight_ai.py`: legacy Streamlit prototype kept for reference
- `create_memory_llm.py`: builds the built-in FAISS index from `data/`

## Features

- Upload subject PDFs and build a temporary session knowledge base
- Use built-in knowledge, uploaded PDFs, or both together
- Switch between `Strict RAG` and `Hybrid` answer modes
- Switch between `Standard` and `Exam` presentation modes
- Exam formats: `2-Mark`, `5-Mark`, `6-Mark`, `7-Mark`, `10-Mark`, `Comparison`, `Viva`
- Source citation cards for every answer

## Project Structure

```
Examnight-ai/
├── backend/
│   ├── app/
│   │   ├── config.py
│   │   ├── main.py
│   │   ├── rag.py
│   │   └── schemas.py
│   ├── requirements.txt
│   └── runtime.txt
├── frontend/
│   ├── src/
│   │   ├── api/client.js
│   │   ├── components/ChatMessage.jsx
│   │   ├── components/ControlPanel.jsx
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── styles.css
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
├── data/
├── vectorstore/
├── examnight_ai.py
├── create_memory_llm.py
└── render.yaml
```

## Frontend

The frontend is now a dedicated React app inside `frontend/`.
It is no longer only a Streamlit UI.

## Backend API

Main endpoints:

- `POST /api/sessions`: create a chat session
- `POST /api/sessions/{session_id}/documents`: upload PDFs for that session
- `POST /api/chat`: ask a question with source mode and answer mode controls
- `POST /api/sessions/{session_id}/reset`: clear chat or clear uploaded documents too
- `GET /api/config`: available modes and knowledge-base readiness
- `GET /api/health`: health check

## Run Locally

1. Build the default knowledge base if you want built-in encyclopedia answers:
```bash
python create_memory_llm.py
```

2. Start the backend:
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

3. Start the frontend in another terminal:
```bash
cd frontend
npm install
npm run dev
```

Backend default URL: `http://localhost:8000`
Frontend default URL: `http://localhost:5173`

## Environment Variables

Root `.env`:

```bash
HF_TOKEN=your_huggingface_token_here
```

Backend optional:

```bash
CORS_ORIGINS=http://localhost:5173
```

Frontend optional:

```bash
VITE_API_BASE_URL=http://localhost:8000
```

## Render Deployment

`render.yaml` now defines two services:

- `examnight-api`: Python web service for FastAPI
- `examnight-frontend`: static site for React

Set these environment variables in Render:

- Backend: `HF_TOKEN`
- Backend: `CORS_ORIGINS`
- Frontend: `VITE_API_BASE_URL`

For production, set `VITE_API_BASE_URL` to your Render backend URL and set `CORS_ORIGINS` to your frontend URL.

## Legacy Streamlit App

The previous Streamlit prototype still exists and can be run with:

```bash
streamlit run examnight_ai.py
```
