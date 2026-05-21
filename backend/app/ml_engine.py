from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import torch
from fastapi import HTTPException
from PIL import Image, UnidentifiedImageError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torchvision import models, transforms

from app.core.config import (
    ALLOWED_IMAGE_EXTENSIONS,
    DATASET_DIR,
    IMAGE_SIZE,
    MIN_CLASSES_REQUIRED,
    MIN_IMAGES_PER_CLASS,
    MODEL_PATH,
    RANDOM_STATE,
    TEST_SIZE,
)
from app.utils.session_utils import get_session_dataset_dir, get_session_model_path


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_preprocessing_transform():
    """
    Create the exact image preprocessing pipeline used before feature extraction.
    This same transform must also be used during prediction.
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def load_feature_extractor(device: torch.device):
    """
    Load MobileNetV3 Small pre-trained on ImageNet.
    Remove the final classification layer so the model works as a feature extractor.
    """
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)

    model.classifier = torch.nn.Identity()
    model.eval()
    model.to(device)

    return model


def get_class_folders(session_id: str) -> List[Path]:
    """
    Return all valid class folders inside the dataset directory.
    """
    dataset_dir = get_session_dataset_dir(session_id)

    if not dataset_dir.exists():
        raise HTTPException(
            status_code=400,
            detail="Dataset folder does not exist. Please upload images first."
        )

    class_folders = [
        folder for folder in dataset_dir.iterdir()
        if folder.is_dir()
    ]

    if len(class_folders) < MIN_CLASSES_REQUIRED:
        raise HTTPException(
            status_code=400,
            detail=f"At least {MIN_CLASSES_REQUIRED} classes are required for training."
        )

    return class_folders


def get_image_paths_from_class_folder(class_folder: Path) -> List[Path]:
    """
    Return valid image paths from one class folder.
    """
    image_paths = [
        image_path for image_path in class_folder.iterdir()
        if image_path.is_file()
        and image_path.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS
    ]

    return image_paths


def validate_dataset(class_folders: List[Path]) -> Dict[str, int]:
    """
    Check that every class has enough images.
    Return class distribution.
    """
    class_distribution = {}

    for class_folder in class_folders:
        image_paths = get_image_paths_from_class_folder(class_folder)
        image_count = len(image_paths)

        if image_count < MIN_IMAGES_PER_CLASS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Class '{class_folder.name}' has only {image_count} image(s). "
                    f"Minimum {MIN_IMAGES_PER_CLASS} images per class are required."
                )
            )

        class_distribution[class_folder.name] = image_count

    return class_distribution


def load_image_as_tensor(image_path: Path, transform) -> torch.Tensor:
    """
    Load one image, convert to RGB, apply preprocessing, and return a tensor.
    """
    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image)
        return image_tensor

    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid or corrupted image found: {image_path.name}"
        )


def extract_features_for_image(
    image_tensor: torch.Tensor,
    feature_extractor,
    device: torch.device
) -> np.ndarray:
    """
    Pass one image tensor through MobileNetV3 and return extracted features.
    """
    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        features = feature_extractor(image_tensor)

    return features.cpu().numpy().flatten()


def build_feature_dataset(
    class_folders: List[Path],
    feature_extractor,
    transform,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert all dataset images into feature vectors and labels.
    """
    features = []
    labels = []

    for class_folder in class_folders:
        class_name = class_folder.name
        image_paths = get_image_paths_from_class_folder(class_folder)

        for image_path in image_paths:
            image_tensor = load_image_as_tensor(image_path, transform)
            image_features = extract_features_for_image(
                image_tensor=image_tensor,
                feature_extractor=feature_extractor,
                device=device
            )

            features.append(image_features)
            labels.append(class_name)

    return np.array(features), np.array(labels)


def train_model(session_id: str) -> Dict:
    """
    Main training function.
    It scans the dataset, extracts MobileNetV3 features,
    trains Logistic Regression, saves model.pkl, and returns metrics.
    """
    class_folders = get_class_folders(session_id)
    class_distribution = validate_dataset(class_folders)

    device = get_device()
    transform = get_preprocessing_transform()
    feature_extractor = load_feature_extractor(device)

    features, labels = build_feature_dataset(
        class_folders=class_folders,
        feature_extractor=feature_extractor,
        transform=transform,
        device=device
    )

    if len(features) == 0:
        raise HTTPException(
            status_code=400,
            detail="No valid images found for training."
        )

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            encoded_labels,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=encoded_labels
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            encoded_labels,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )

    classifier = LogisticRegression(
        max_iter=1000,
        solver="lbfgs"
    )

    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    session_model_path = get_session_model_path(session_id)
    session_model_path.parent.mkdir(parents=True, exist_ok=True)

    model_package = {
        "classifier": classifier,
        "label_encoder": label_encoder,
        "class_names": list(label_encoder.classes_),
        "class_distribution": class_distribution,
        "image_size": IMAGE_SIZE,
        "feature_extractor": "mobilenet_v3_small",
        "accuracy": float(accuracy),
        "total_images": int(len(labels)),
    }

    joblib.dump(model_package, session_model_path)

    return {
        "message": "Model trained successfully.",
        "model_path": str(session_model_path),
        "accuracy": round(float(accuracy), 4),
        "accuracy_percentage": round(float(accuracy) * 100, 2),
        "classes": list(label_encoder.classes_),
        "class_distribution": class_distribution,
        "total_images": int(len(labels)),
        "device": str(device),
    }


def load_trained_model_package(session_id: str) -> Dict:
    """
    Load the saved model package from model.pkl.
    This package contains the classifier, label encoder, class names, and metadata.
    """
    session_model_path = get_session_model_path(session_id)

    if not session_model_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Trained model not found. Please train the model first."
        )

    try:
        model_package = joblib.load(session_model_path)
        return model_package

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Failed to load trained model. Please retrain the model."
        )


def predict_image(session_id: str, image_file) -> Dict:
    """
    Predict the class of a single uploaded image.
    The image goes through the same preprocessing and feature extraction pipeline
    used during training.
    """
    model_package = load_trained_model_package(session_id)

    classifier = model_package["classifier"]
    label_encoder = model_package["label_encoder"]
    class_names = model_package["class_names"]

    device = get_device()
    transform = get_preprocessing_transform()
    feature_extractor = load_feature_extractor(device)

    try:
        image = Image.open(image_file.file).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid image."
        )

    image_tensor = transform(image)

    image_features = extract_features_for_image(
        image_tensor=image_tensor,
        feature_extractor=feature_extractor,
        device=device
    )

    image_features = image_features.reshape(1, -1)

    predicted_label_index = classifier.predict(image_features)[0]
    predicted_class = label_encoder.inverse_transform([predicted_label_index])[0]

    if hasattr(classifier, "predict_proba"):
        probabilities = classifier.predict_proba(image_features)[0]
    else:
        raise HTTPException(
            status_code=500,
            detail="The trained classifier does not support probability prediction."
        )

    probability_dict = {
        class_name: round(float(probability) * 100, 2)
        for class_name, probability in zip(class_names, probabilities)
    }

    confidence = probability_dict[predicted_class]

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": probability_dict,
        "model_accuracy": round(float(model_package.get("accuracy", 0)) * 100, 2),
        "trained_classes": class_names,
        "feature_extractor": model_package.get("feature_extractor", "mobilenet_v3_small"),
        "image_size": model_package.get("image_size", IMAGE_SIZE),
    }