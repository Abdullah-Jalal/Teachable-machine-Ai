from typing import Annotated, List
from fastapi.responses import FileResponse
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from app.ml_engine import predict_image, train_model
from app.utils.file_utils import (
    delete_class_dataset,
    get_dataset_summary,
    save_uploaded_images,
)
from app.utils.session_utils import cleanup_old_sessions, delete_session_data
from app.utils.session_utils import get_session_model_path
from app.utils.session_utils import (
    cleanup_old_sessions,
    delete_session_data,
    get_session_model_path,
)
router = APIRouter()


@router.get("/")
def root():
    return {
        "message": "Teachable Machine AI Backend is running.",
        "status": "success"
    }


@router.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "FastAPI ML Backend"
    }


@router.post("/upload-sample")
async def upload_sample(
    session_id: Annotated[str, Form(...)],
    class_name: Annotated[str, Form(...)],
    files: Annotated[List[UploadFile], File(...)]
):
    result = await save_uploaded_images(session_id, class_name, files)

    return {
        "message": "Images uploaded successfully.",
        "data": result
    }


@router.post("/train")
def train(session_id: Annotated[str, Form(...)]):
    result = train_model(session_id)

    return {
        "message": "Training completed successfully.",
        "data": result
    }


@router.post("/predict")
def predict(
    session_id: Annotated[str, Form(...)],
    file: Annotated[UploadFile, File(...)]
):
    result = predict_image(session_id, file)

    return {
        "message": "Prediction completed successfully.",
        "data": result
    }
https://github.com/Abdullah-Jalal/Teachable-machine-Ai

@router.get("/dataset-summary")
def dataset_summary(session_id: str):
    result = get_dataset_summary(session_id)

    return {
        "message": "Dataset summary fetched successfully.",
        "data": result
    }


@router.delete("/delete-class")
def delete_class(
    session_id: Annotated[str, Form(...)],
    class_name: Annotated[str, Form(...)]
):
    result = delete_class_dataset(session_id, class_name)

    return {
        "message": result["message"],
        "data": result
    }


@router.delete("/reset-session")
def reset_session(session_id: Annotated[str, Form(...)]):
    result = delete_session_data(session_id)

    return {
        "message": "Session data deleted successfully.",
        "data": result
    }


@router.post("/cleanup-old-sessions")
def cleanup_sessions():
    result = cleanup_old_sessions()

    return {
        "message": "Old sessions cleaned successfully.",
        "data": result
    }
@router.get("/export-model")
def export_model(session_id: str):
    model_path = get_session_model_path(session_id)

    if not model_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Trained model not found. Please train the model before exporting."
        )

    return FileResponse(
        path=model_path,
        filename="teachable_machine_model.pkl",
        media_type="application/octet-stream",
    )