import re
import shutil
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import HTTPException

from app.core.config import SESSIONS_DIR, SESSION_TTL_MINUTES


def sanitize_session_id(session_id: str) -> str:
    session_id = session_id.strip()

    if not session_id:
        raise HTTPException(
            status_code=400,
            detail="Session ID is required."
        )

    safe_session_id = re.sub(r"[^a-zA-Z0-9_-]", "_", session_id)

    return safe_session_id


def get_session_base_dir(session_id: str) -> Path:
    safe_session_id = sanitize_session_id(session_id)
    session_dir = SESSIONS_DIR / safe_session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def get_session_dataset_dir(session_id: str) -> Path:
    dataset_dir = get_session_base_dir(session_id) / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    return dataset_dir


def get_session_models_dir(session_id: str) -> Path:
    models_dir = get_session_base_dir(session_id) / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def get_session_model_path(session_id: str) -> Path:
    return get_session_models_dir(session_id) / "model.pkl"


def delete_session_data(session_id: str) -> dict:
    safe_session_id = sanitize_session_id(session_id)
    session_dir = SESSIONS_DIR / safe_session_id

    if session_dir.exists():
        shutil.rmtree(session_dir)

    return {
        "session_id": safe_session_id,
        "deleted": True
    }


def cleanup_old_sessions() -> dict:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    now = datetime.now()
    deleted_sessions = []

    for session_dir in SESSIONS_DIR.iterdir():
        if not session_dir.is_dir():
            continue

        modified_time = datetime.fromtimestamp(session_dir.stat().st_mtime)

        if now - modified_time > timedelta(minutes=SESSION_TTL_MINUTES):
            shutil.rmtree(session_dir)
            deleted_sessions.append(session_dir.name)

    return {
        "deleted_sessions": deleted_sessions,
        "deleted_count": len(deleted_sessions)
    }