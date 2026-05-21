import re
import uuid
from pathlib import Path
from typing import List

from fastapi import UploadFile, HTTPException
from PIL import Image

import shutil

from app.utils.session_utils import get_session_dataset_dir

from app.core.config import (
    ALLOWED_IMAGE_EXTENSIONS,
    DATASET_DIR,
    MAX_IMAGE_SIZE_BYTES,
)


def sanitize_class_name(class_name: str) -> str:
    cleaned_name = class_name.strip()

    if not cleaned_name:
        raise HTTPException(
            status_code=400,
            detail="Class name cannot be empty."
        )

    cleaned_name = re.sub(r"[^a-zA-Z0-9_-]", "_", cleaned_name)

    if len(cleaned_name) > 50:
        raise HTTPException(
            status_code=400,
            detail="Class name is too long. Maximum 50 characters allowed."
        )

    return cleaned_name


def validate_image_extension(filename: str) -> str:
    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image type '{extension}'. Allowed types are: jpg, jpeg, png, webp."
        )

    return extension


async def validate_image_size(file: UploadFile) -> bytes:
    contents = await file.read()

    if len(contents) > MAX_IMAGE_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail="Image size is too large. Maximum allowed size is 10 MB."
        )

    await file.seek(0)
    return contents


def verify_image_file(file_path: Path) -> None:
    try:
        with Image.open(file_path) as img:
            img.verify()
    except Exception:
        if file_path.exists():
            file_path.unlink()

        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid image."
        )


async def save_uploaded_images(session_id: str, class_name: str, files: List[UploadFile]) -> dict:
    safe_class_name = sanitize_class_name(class_name)

    if not files:
        raise HTTPException(
            status_code=400,
            detail="Please upload at least one image."
        )

    session_dataset_dir = get_session_dataset_dir(session_id)
    class_folder = session_dataset_dir / safe_class_name
    class_folder.mkdir(parents=True, exist_ok=True)

    saved_files = []

    for file in files:
        extension = validate_image_extension(file.filename)
        await validate_image_size(file)

        unique_filename = f"{uuid.uuid4()}{extension}"
        destination_path = class_folder / unique_filename

        with open(destination_path, "wb") as buffer:
            buffer.write(await file.read())

        verify_image_file(destination_path)

        saved_files.append(unique_filename)

    return {
        "class_name": safe_class_name,
        "saved_count": len(saved_files),
        "saved_files": saved_files,
        "folder_path": str(class_folder)
    }




def get_dataset_summary(session_id: str) -> dict:
    """
    Return a summary of all class folders and image counts inside the dataset directory.
    Purpose:
    The frontend can use this to know how many images exist in each class folder.
    This is better than only relying on Streamlit session state.
    """
    # DATASET_DIR.mkdir(parents=True, exist_ok=True)
    session_dataset_dir = get_session_dataset_dir(session_id)

    summary = {}

    for class_folder in session_dataset_dir.iterdir():
        if not class_folder.is_dir():
            continue

        image_count = 0

        for image_path in class_folder.iterdir():
            if (
                image_path.is_file()
                and image_path.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS
            ):
                image_count += 1

        summary[class_folder.name] = image_count

    return {
        "classes": summary,
        "total_classes": len(summary),
        "total_images": sum(summary.values())
    }


def delete_class_dataset(session_id: str, class_name: str) -> dict:
    """
    Delete a class folder from the dataset directory.
    Purpose:
    When the user deletes a class from the UI, the backend dataset should also be cleaned.
    """
    safe_class_name = sanitize_class_name(class_name)
    session_dataset_dir = get_session_dataset_dir(session_id)
    class_folder = session_dataset_dir / safe_class_name

    if not class_folder.exists():
        return {
            "class_name": safe_class_name,
            "deleted": False,
            "message": "Class folder does not exist."
        }

    if not class_folder.is_dir():
        raise HTTPException(
            status_code=400,
            detail="Invalid class path."
        )

    shutil.rmtree(class_folder)

    return {
        "class_name": safe_class_name,
        "deleted": True,
        "message": "Class dataset deleted successfully."
    }