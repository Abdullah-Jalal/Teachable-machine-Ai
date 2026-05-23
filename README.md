# 🧠 Teachable Machine AI — Full Stack Intelligent Image Classification Platform

A production-style full-stack AI web application inspired by Google Teachable Machine that allows users to build, train, evaluate, and deploy custom image classification models using their own dataset.

This project extends beyond basic image classification by adding dataset intelligence, AI insights, training analytics, and real-time performance evaluation.

---

## 🚀 Overview

This system implements a complete end-to-end machine learning pipeline with a modern web dashboard and backend API architecture.

Users can:

- Create custom image classes dynamically
- Upload or capture training images via webcam
- Train ML models using transfer learning (MobileNetV3)
- Perform real-time image and webcam predictions
- Analyze dataset quality before training
- View AI-generated insights after training
- Monitor training performance metrics
- Export trained models
- Reset and manage sessions

---

## ✨ Features

### 🖥️ Frontend (Streamlit Dashboard)

- Teachable Machine–style UI
- Dynamic class management (add/edit/delete)
- Image upload per class
- Webcam dataset capture support
- Real-time training progress visualization
- Dark mode support 🌙
- Prediction interface (image + webcam)
- Top-K prediction display (Explainable AI)
- Confidence status (High / Medium / Low)
- Recent predictions history
- Live FPS + inference time monitoring
- Dataset Health Score dashboard 📊
- AI Insights Panel after training 🧠
- Model export and reset controls

---

### ⚙️ Backend (FastAPI)

- RESTful API architecture
- Dataset session management
- Training pipeline API
- Prediction API with probability outputs
- Model export endpoint (Joblib)
- Dataset summary endpoint
- Session reset functionality
- Real-time inference support

---

### 🧠 Machine Learning Pipeline

- Transfer Learning using MobileNetV3 (ImageNet pretrained)
- Feature extraction for high performance classification
- Logistic Regression classifier
- Image preprocessing (224×224 normalization)
- Label encoding for dynamic class mapping
- Train/test evaluation pipeline
- Accuracy computation
- Probability-based predictions
- Confusion matrix evaluation 📊
- Precision, Recall, F1-score support

---

### 📊 AI Intelligence Layer

#### 📈 Dataset Health Score
- Dataset balance detection
- Class size validation
- Missing/weak class detection
- Dataset quality scoring system

#### 🧠 AI Insights Panel
- Best performing class detection
- Weak class identification
- Dataset imbalance detection
- Model confidence stability analysis

#### 🎯 Explainable AI Output
- Top-3 predictions breakdown
- Confidence status:
  - High Confidence (≥85%)
  - Medium Confidence (60–84%)
  - Low Confidence (<60%)

#### ⚡ Performance Metrics
- Live FPS counter (webcam mode)
- Inference time (ms)
- Training duration analytics

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
teachable-machine-ai/
│
├── backend/
│ ├── app/
│ │ ├── api/
│ │ │ └── routes.py
│ │ ├── core/
│ │ │ └── config.py
│ │ ├── utils/
│ │ │ ├── file_utils.py
│ │ │ └── session_utils.py
│ │ ├── main.py
│ │ └── ml_engine.py
│
├── frontend/
│ ├── app.py
│ ├── Dockerfile
│ ├── requirements.txt
│
├── docker-compose.yml
└── README.md

---

## ⚡ Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/Abdullah-Jalal/Teachable-machine-Ai.git
cd Teachable-machine-Ai

2️⃣ Run with Docker Compose
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
Dataset health analysis
Resize images to 224×224
Feature extraction using MobileNetV3
Train classifier
Evaluate performance
Generate AI insights
Save model using Joblib
Prediction Pipeline
Input image (upload/webcam)
Preprocess image
Extract features
Predict probabilities
Return:
Top predictions
Confidence score
Confidence status
Inference time
📊 Evaluation System
Confusion Matrix heatmap
Precision, Recall, F1-score
Dataset imbalance detection
Class-wise performance analysis
🚀 Future Improvements
User authentication system
Cloud deployment (AWS / Azure / GCP)
Dataset augmentation pipeline
Model versioning system (v1, v2, v3)
GPU acceleration support
Database integration (PostgreSQL / MongoDB)
Mobile responsive UI
One-click dataset import/export
