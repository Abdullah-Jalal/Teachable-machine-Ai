# Teachable Machine AI — Full Stack Image Classification Platform

A full-stack AI web application inspired by Google Teachable Machine that allows users to create custom image classification models using their own dataset. Users can upload or capture images, train a model using transfer learning, and perform real-time predictions via image upload or webcam.

---

## 🚀 Overview

This project implements a complete end-to-end machine learning workflow in a production-style architecture, combining frontend UI, backend APIs, and a transfer learning-based ML pipeline.

Users can:

- Create custom image classes
- Upload training images
- Capture webcam samples
- Train an image classification model
- Perform real-time predictions
- View confidence scores
- Export trained models
- Reset training sessions

---

## ✨ Features

### 🖥️ Frontend (Streamlit)
- Teachable Machine-inspired UI
- Dynamic class management (add/edit/delete)
- Image upload per class
- Webcam-based dataset capture
- Real-time training status updates
- Image prediction interface
- Webcam prediction support
- Confidence score visualization
- Backend health monitoring
- Model export and reset controls

---

### ⚙️ Backend (FastAPI)
- RESTful API architecture
- Dataset upload and management
- Session-based storage system
- Training pipeline API
- Prediction API
- Model export endpoint
- Dataset summary endpoint
- Session reset functionality
- Automatic cleanup of old sessions

---

### 🧠 Machine Learning Pipeline
- Transfer Learning using MobileNetV3
- Feature extraction using ImageNet-pretrained weights
- Logistic Regression classifier
- Image preprocessing (224×224 normalization)
- Label encoding for class mapping
- Train/test split evaluation
- Accuracy calculation
- Probability-based predictions

---

## 🧰 Tech Stack

### Frontend
- Python
- Streamlit
- streamlit-webrtc
- OpenCV
- Pillow
- Requests

### Backend
- Python
- FastAPI
- Uvicorn
- PyTorch
- Torchvision
- Scikit-learn
- Joblib
- Pillow

### DevOps
- Docker
- Docker Compose

---

## 📁 Project Structure

```txt
teachable-machine-ai/
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── routes.py
│   │   ├── core/
│   │   │   └── config.py
│   │   ├── utils/
│   │   │   ├── file_utils.py
│   │   │   └── session_utils.py
│   │   ├── main.py
│   │   └── ml_engine.py
│   │
│   ├── Dockerfile
│   ├── requirements.txt
│   └── .dockerignore
│
├── frontend/
│   ├── app.py
│   ├── Dockerfile
│   ├── requirements.txt
│   └── .dockerignore
│
├── docker-compose.yml
├── .dockerignore
└── README.md

⚡ Installation & Setup
1. Clone Repository
git clone https://github.com/Abdullah-Jalal/Teachable-machine-Ai.git
cd Teachable-machine-Ai
2. Run with Docker Compose
docker compose up --build

🌐 Application URLs
| Service     | URL                                                      |
| ----------- | -------------------------------------------------------- |
| Frontend    | [http://localhost:8501](http://localhost:8501)           |
| Backend API | [http://localhost:8000](http://localhost:8000)           |
| API Docs    | [http://localhost:8000/docs](http://localhost:8000/docs) |

🔄 ML Workflow
Training Pipeline
Upload or capture images
Resize to 224×224
Extract features using MobileNetV3
Train Logistic Regression classifier
Evaluate accuracy
Save model using Joblib
Prediction Pipeline
Input image (upload/webcam)
Preprocess image
Extract features
Predict class probabilities
Return confidence scores

🔄 ML Workflow
Training Pipeline
Upload or capture images
Resize to 224×224
Extract features using MobileNetV3
Train Logistic Regression classifier
Evaluate accuracy
Save model using Joblib
Prediction Pipeline
Input image (upload/webcam)
Preprocess image
Extract features
Predict class probabilities
Return confidence scores
🚀 Future Improvements
User authentication system
Cloud deployment (AWS / Azure)
Dataset augmentation pipeline
Training visualization dashboard
GPU acceleration support
Model versioning system
Database integration
Mobile responsive UI
