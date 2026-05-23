🧠 Teachable Machine AI — Full Stack Intelligent Image Classification Platform

A production-style full-stack AI web application inspired by Google Teachable Machine that enables users to build, train, evaluate, and deploy custom image classification models using their own dataset.

This project goes beyond standard image classification by adding:

📊 Dataset quality analysis
🧠 AI insights engine
📈 Training analytics
⚡ Real-time prediction performance metrics
🎯 Explainable AI outputs
🚀 Overview

This system implements a complete end-to-end machine learning pipeline with a modern web interface and backend API architecture.

Users can:

Create custom image classes dynamically
Upload or capture training images via webcam
Train ML models using transfer learning (MobileNetV3)
Perform real-time image & webcam predictions
Analyze dataset quality before training
View AI-generated insights after training
Monitor model performance metrics
Export trained models
Reset and manage training sessions

✨ Key Features
🖥️ Frontend (Streamlit Dashboard)
Teachable Machine–style interactive UI
Dynamic class creation and management
Image upload per class
Webcam-based dataset collection (frame-by-frame capture)
Real-time training progress visualization

Dark mode UI support 🌙
Prediction interface (image + webcam)
Top-K prediction display (Explainable AI)
Confidence status badges (High / Medium / Low)
Recent predictions history panel
Live FPS + inference time monitoring

Dataset Health Score dashboard 📊
AI Insights Panel (post-training analysis) 🧠
Model export and reset controls

⚙️ Backend (FastAPI Service)
RESTful API architecture
Dataset session management
Training pipeline orchestration
Prediction API with probability output
Model export (Joblib serialization)
Dataset summary & analytics endpoint
Session reset and cleanup system
Real-time inference support for webcam

🧠 Machine Learning Pipeline
Transfer Learning using MobileNetV3 (ImageNet pretrained)
Feature extraction for high-performance classification
Logistic Regression classifier
Image preprocessing (224×224 normalization)
Label encoding for dynamic class mapping
Train/test evaluation pipeline
Accuracy computation
Probability-based multi-class predictions
Confusion Matrix evaluation 📊
Precision, Recall, F1-score support
📊 AI Intelligence Layer (Advanced Additions)

This project includes an AI analysis engine that evaluates model and dataset quality:

📈 Dataset Health Score
Class balance analysis
Dataset size validation
Missing class detection
Data sufficiency warnings

🧠 AI Insights Panel
Best performing class detection
Weak class identification
Dataset imbalance detection
Model confidence stability analysis

🎯 Explainable Predictions
Top-3 predictions breakdown
Confidence classification:
High Confidence (≥85%)
Medium Confidence (60–84%)
Low Confidence (<60%)
⚡ Performance Metrics
Live FPS counter (webcam mode)
Inference time (ms)
Training duration analytics

🧰 Tech Stack
Frontend
Python
Streamlit
streamlit-webrtc
OpenCV
Pillow
Requests
Backend
Python
FastAPI
Uvicorn
PyTorch
Torchvision
Scikit-learn
Joblib
Pillow
DevOps
Docker
Docker Compose

📁 Project Structure
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
🏋️ Training Pipeline
Upload or capture images
Dataset health analysis
Resize images to 224×224
Feature extraction using MobileNetV3
Train Logistic Regression classifier
Evaluate model performance
Generate AI insights
Save model using Joblib
🔮 Prediction Pipeline
Input image (upload/webcam)
Preprocess image
Extract deep features
Predict class probabilities
Return:
Top predictions
Confidence score
Confidence status
Inference time
📊 Evaluation System

This project includes a full ML evaluation module:

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
One-click dataset export/import
