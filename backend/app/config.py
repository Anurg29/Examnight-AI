from pathlib import Path
import os

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / '.env')

DB_FAISS_PATH = ROOT_DIR / 'vectorstore' / 'db_faiss'
HUGGINGFACE_REPO_ID = os.getenv('HUGGINGFACE_REPO_ID', 'Qwen/Qwen2.5-7B-Instruct')
HF_TOKEN = os.getenv('HF_TOKEN')
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv('CORS_ORIGINS', 'http://localhost:5173').split(',')
    if origin.strip()
]

# ── MongoDB Auth ──────────────────────────────────────────────────────────────
MONGODB_URI = os.getenv('MONGODB_URI', '')  # e.g. mongodb+srv://user:pass@cluster.mongodb.net/examnight
JWT_SECRET = os.getenv('JWT_SECRET', 'change-me-in-production')
JWT_EXPIRE_HOURS = int(os.getenv('JWT_EXPIRE_HOURS', '24'))
