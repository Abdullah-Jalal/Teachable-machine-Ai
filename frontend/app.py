import base64
import io
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List
import os
import av
import cv2
import numpy as np
import requests
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer
import matplotlib.pyplot as plt


API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


@dataclass
class WebcamSampleProcessor(VideoProcessorBase):
    is_recording: bool = False
    capture_interval_seconds: float = 0.25
    last_capture_time: float = 0.0
    captured_frames: List[bytes] = field(default_factory=list)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")

        if self.is_recording:
            current_time = time.time()

            if current_time - self.last_capture_time >= self.capture_interval_seconds:
                self.last_capture_time = current_time

                processed_image = self.prepare_sample_image(image)

                success, encoded_image = cv2.imencode(
                    ".jpg",
                    processed_image,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 85],
                )

                if success:
                    self.captured_frames.append(encoded_image.tobytes())

        return av.VideoFrame.from_ndarray(image, format="bgr24")

    @staticmethod
    def prepare_sample_image(image: np.ndarray) -> np.ndarray:
        height, width, _ = image.shape
        crop_size = min(height, width)

        start_x = (width - crop_size) // 2
        start_y = (height - crop_size) // 2

        cropped = image[start_y:start_y + crop_size, start_x:start_x + crop_size]
        resized = cv2.resize(cropped, (224, 224))

        return resized


@dataclass
class PreviewWebcamProcessor(VideoProcessorBase):
    latest_frame: bytes | None = None
    capture_interval_seconds: float = 0.3
    last_capture_time: float = 0.0
    frame_times: List[float] = field(default_factory=list)
    fps: float = 0.0
    inference_time_ms: float = 0.0
    last_predictions: dict = field(default_factory=dict)
    session_id: str = ""

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        current_time = time.time()

        self.frame_times.append(current_time)
        self.frame_times = [t for t in self.frame_times if current_time - t <= 1.0]
        if len(self.frame_times) > 1:
            self.fps = len(self.frame_times) / (self.frame_times[-1] - self.frame_times[0] + 1e-5)
        else:
            self.fps = 0.0

        if current_time - self.last_capture_time >= self.capture_interval_seconds:
            self.last_capture_time = current_time

            processed_image = self.prepare_prediction_image(image)

            success, encoded_image = cv2.imencode(
                ".jpg",
                processed_image,
                [int(cv2.IMWRITE_JPEG_QUALITY), 90],
            )

            if success:
                self.latest_frame = encoded_image.tobytes()
                if self.session_id:
                    try:
                        start_inf = time.time()
                        files_payload = {
                            "file": (
                                "webcam_preview.jpg",
                                self.latest_frame,
                                "image/jpeg",
                            )
                        }
                        response = requests.post(
                            f"{API_BASE_URL}/predict",
                            data={"session_id": self.session_id},
                            files=files_payload,
                            timeout=2,
                        )
                        if response.status_code == 200:
                            self.last_predictions = response.json()["data"]
                            self.inference_time_ms = (time.time() - start_inf) * 1000
                    except Exception:
                        pass

        # Draw overlay HUD
        cv2.rectangle(image, (5, 5), (280, 85), (0, 0, 0), -1)
        cv2.putText(image, f"FPS: {self.fps:.1f}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(image, f"Inference: {self.inference_time_ms:.1f}ms", (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        if self.last_predictions:
            pred_class = self.last_predictions.get("predicted_class", "None")
            confidence = self.last_predictions.get("confidence", 0.0)
            cv2.putText(image, f"Pred: {pred_class} ({confidence}%)", (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        return av.VideoFrame.from_ndarray(image, format="bgr24")

    @staticmethod
    def prepare_prediction_image(image: np.ndarray) -> np.ndarray:
        height, width, _ = image.shape
        crop_size = min(height, width)

        start_x = (width - crop_size) // 2
        start_y = (height - crop_size) // 2

        cropped = image[start_y:start_y + crop_size, start_x:start_x + crop_size]
        resized = cv2.resize(cropped, (224, 224))

        return resized


# Load custom TB icon
icon_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "tb_icon.png"
)

if os.path.exists(icon_path):
    tb_icon = Image.open(icon_path)
else:
    tb_icon = "🧠"

st.set_page_config(
    page_title="Teachable Machine AI",
    page_icon=tb_icon,
    layout="wide",
    initial_sidebar_state="collapsed",
)


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: #f6f8f7;
        }

        header[data-testid="stHeader"] {
            display: none;
        }

        section[data-testid="stSidebar"] {
            display: none;
        }

        div[data-testid="InputInstructions"] {
            display: none !important;
        }

        div[data-testid="column"] {
            overflow: visible !important;
        }

        div[data-testid="stVerticalBlock"],
        div[data-testid="stVerticalBlockBorderWrapper"],
        div[data-testid="stElementContainer"] {
            overflow: visible !important;
        }

        div[data-testid="column"]:has(.tm-sticky-wrapper) {
            min-height: 300vh !important;
            overflow: visible !important;
        }

        .block-container {
            padding-top: 1.2rem;
            padding-left: 2rem;
            padding-right: 2rem;
            max-width: 1500px;
        }

        .tm-topbar {
            background: #ffffff;
            border-radius: 14px;
            padding: 14px 22px;
            display: flex;
            align-items: center;
            gap: 14px;
            box-shadow: none;
            border: 1px solid #dadce0;
            margin-bottom: 22px;
            width: fit-content;
            min-width: 340px;
        }

        .tm-menu-icon {
            font-size: 26px;
            line-height: 1;
            color: #202124;
        }

        .tm-logo-text {
            font-size: 26px;
            font-weight: 800;
            color: #1967d2;
            letter-spacing: -0.5px;
        }

        .tm-class-card-pro {
            border: 1px solid #dadce0 !important;
            border-radius: 10px !important;
            padding: 0 !important;
            overflow: hidden !important;
            background: #ffffff !important;
            box-shadow: 0 2px 4px rgba(60,64,67,0.18) !important;
            margin-bottom: 26px !important;
        }

        .tm-class-card-pro:hover {
            box-shadow: 0 3px 7px rgba(60,64,67,0.22) !important;
        }

        .tm-class-card-disabled {
            opacity: 0.58;
            background: #f8f9fa;
        }

        .tm-card-divider-pro {
            height: 1px;
            background: #dadce0;
            margin: 0 0 16px 0;
        }

        .tm-edit-icon {
            font-size: 16px;
            color: #9aa0a6;
            padding-top: 8px;
            text-align: center;
        }

        .tm-disabled-message {
            background: #f1f3f4;
            color: #5f6368;
            padding: 12px 14px;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 600;
            margin: 0 16px 16px 16px;
        }

        .tm-sample-label {
            margin-bottom: 12px;
            color: #3c4043;
            font-size: 16px;
            font-weight: 600;
        }

        .tm-gallery-panel {
            border-left: 1px solid #eef0ef;
            padding-left: 18px;
            min-height: 120px;
        }

        .tm-counter {
            color: #5f6368;
            font-size: 13px;
            font-weight: 600;
            margin-bottom: 8px;
        }

        .tm-empty-gallery {
            background: #f8f9fa;
            border: 1px dashed #dadce0;
            color: #80868b;
            border-radius: 8px;
            padding: 16px;
            font-size: 13px;
            line-height: 1.4;
            text-align: center;
        }

        .tm-gallery-grid {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            max-height: 220px;
            overflow-y: auto;
            align-content: flex-start;
        }

        .tm-gallery-image {
            width: 52px;
            height: 52px;
            object-fit: cover;
            border-radius: 6px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            border: 1px solid #eef0ef;
        }

        .tm-gallery-grid::-webkit-scrollbar {
            width: 6px;
        }

        .tm-gallery-grid::-webkit-scrollbar-track {
            background: transparent;
        }

        .tm-gallery-grid::-webkit-scrollbar-thumb {
            background: #dadce0;
            border-radius: 4px;
        }

        .tm-gallery-grid::-webkit-scrollbar-thumb:hover {
            background: #bdc1c6;
        }

        .tm-upload-mode-panel {
            background: #e8f0fe;
            border-radius: 0;
            padding: 14px;
            margin: -16px 0 14px -2px;
            min-height: 320px;
        }

        .tm-upload-mode-top {
            display: flex;
            justify-content: space-between;
            align-items: center;
            color: #1967d2;
            font-size: 15px;
            font-weight: 700;
            margin-bottom: 14px;
        }

        .tm-upload-close {
            color: #1967d2;
            font-size: 26px;
            line-height: 1;
            font-weight: 400;
        }

        .tm-upload-help-card {
            background: #d2e3fc;
            border-radius: 4px;
            padding: 20px 14px;
            text-align: center;
            color: #1967d2;
            font-size: 15px;
            line-height: 1.4;
            font-weight: 700;
            margin-bottom: 14px;
        }

        .tm-crop-info {
            margin-top: 22px;
            text-align: center;
            color: #4285f4;
            font-size: 14px;
            font-weight: 700;
        }

        .tm-crop-visual {
            font-size: 34px;
            margin-bottom: 8px;
        }

        .tm-panel {
            background: #ffffff;
            border-radius: 12px;
            border: 1px solid #eef0ef;
            box-shadow: 0 8px 24px rgba(20,20,20,0.06);
            overflow: hidden;
            margin-bottom: 22px;
        }

        .tm-sticky-wrapper {
            position: -webkit-sticky !important;
            position: sticky !important;
            top: 24px !important;
            z-index: 999 !important;
            align-self: flex-start !important;
        }

        .tm-panel-header {
            padding: 18px 20px;
            border-bottom: 1px solid #e0e0e0;
            font-size: 21px;
            font-weight: 700;
            color: #202124;
        }

        .tm-panel-body {
            padding: 20px;
        }

        .tm-muted {
            color: #5f6368;
            font-size: 14px;
            line-height: 1.5;
        }

        .tm-connector {
            height: 2px;
            background: #c4c7c5;
            margin-top: 145px;
            position: relative;
        }

        .tm-connector:before {
            content: "";
            width: 13px;
            height: 13px;
            border-top: 2px solid #c4c7c5;
            border-right: 2px solid #c4c7c5;
            transform: rotate(45deg);
            position: absolute;
            right: -1px;
            top: -6px;
        }

        .tm-status-success {
            background: #e6f4ea;
            color: #137333;
            padding: 12px 14px;
            border-radius: 10px;
            font-weight: 600;
            margin-top: 12px;
            font-size: 14px;
        }

        .tm-status-error {
            background: #fce8e6;
            color: #c5221f;
            padding: 12px 14px;
            border-radius: 10px;
            font-weight: 600;
            margin-top: 12px;
            font-size: 14px;
        }

        .tm-status-info {
            background: #e8f0fe;
            color: #1967d2;
            padding: 12px 14px;
            border-radius: 10px;
            font-weight: 600;
            margin-top: 12px;
            font-size: 14px;
        }

        .tm-preview-header-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 12px;
            padding: 16px 18px;
            border-bottom: 1px solid #e0e0e0;
        }

        .tm-preview-title {
            font-size: 21px;
            font-weight: 800;
            color: #202124;
        }

        .tm-export-button {
            background: #e8f0fe;
            color: #1967d2;
            border-radius: 6px;
            padding: 10px 14px;
            font-size: 14px;
            font-weight: 700;
            text-align: center;
            min-width: 145px;
        }

        .tm-input-control-row-clean {
            margin-bottom: 10px;
        }

        .tm-input-title {
            color: #5f6368;
            font-size: 17px;
            font-weight: 800;
            margin-bottom: 8px;
        }

        .tm-output-divider {
            height: 1px;
            background: #e0e0e0;
            margin: 18px -20px 14px -20px;
            position: relative;
        }

        .tm-output-divider::after {
            content: "↓";
            width: 24px;
            height: 24px;
            background: #dadce0;
            color: #80868b;
            border-radius: 50%;
            position: absolute;
            left: 50%;
            top: -12px;
            transform: translateX(-50%);
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
        }

        .tm-output-title {
            color: #5f6368;
            font-size: 18px;
            font-weight: 800;
            margin-bottom: 16px;
        }

        .tm-prediction-main {
            background: #f8fafd;
            border: 1px solid #d9e2f3;
            border-radius: 10px;
            padding: 14px;
            margin-bottom: 16px;
        }

        .tm-prediction-label {
            font-size: 13px;
            color: #5f6368;
            font-weight: 700;
            margin-bottom: 4px;
        }

        .tm-prediction-class {
            font-size: 26px;
            line-height: 1.2;
            font-weight: 900;
            color: #1967d2;
        }

        .tm-prediction-confidence {
            font-size: 14px;
            font-weight: 700;
            color: #3c4043;
            margin-top: 6px;
        }

        .tm-confidence-row {
            display: grid;
            grid-template-columns: 72px 1fr;
            align-items: center;
            gap: 10px;
            margin-bottom: 12px;
        }

        .tm-confidence-class {
            font-size: 14px;
            font-weight: 800;
            color: #5f6368;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .tm-confidence-track {
            height: 30px;
            background: #fce8e6;
            border-radius: 6px;
            position: relative;
            overflow: hidden;
        }

        .tm-confidence-fill {
            height: 100%;
            background: #ea4335;
            border-radius: 6px;
            min-width: 4px;
            display: flex;
            align-items: center;
            justify-content: flex-end;
            color: #ffffff;
            font-size: 12px;
            font-weight: 800;
            padding-right: 6px;
        }

        .tm-confidence-fill-top {
            background: #f29900;
        }

        .tm-file-preview-image {
            border-radius: 6px;
            border: 1px solid #dadce0;
            overflow: hidden;
            margin-bottom: 12px;
        }

        .tm-webcam-placeholder {
            background: #f1f3f4;
            border-radius: 8px;
            border: 1px dashed #bdc1c6;
            min-height: 180px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #5f6368;
            font-weight: 700;
            text-align: center;
            padding: 18px;
            margin-bottom: 14px;
        }

        div.stButton > button {
            width: 100%;
            min-height: 44px;
            border-radius: 6px !important;
            border: 1px solid #eef0ef;
            background: #f8f9fa;
            color: #1967d2;
            font-weight: 700;
            padding: 0.75rem 1rem;
            white-space: pre-line;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            text-align: center !important;
        }

        div.stButton > button:hover {
            background: #e8f0fe;
            color: #174ea6;
            border: 1px solid #d2e3fc;
        }

        div[data-testid="stFileUploader"] {
            background: transparent !important;
            border: none !important;
            padding: 0 !important;
            min-height: auto !important;
        }

        div[data-testid="stFileUploader"] section {
            background: #f1f5ff !important;
            border: 1px dashed #9bbcf9 !important;
            border-radius: 8px !important;
            padding: 16px 12px !important;
        }

        div[data-testid="stFileUploader"] button {
            background: #ffffff !important;
            color: #1967d2 !important;
            border: 1px solid #8ab4f8 !important;
            border-radius: 8px !important;
            font-weight: 700 !important;
        }

        div[data-testid="stFileUploader"] small {
            color: #6b7280 !important;
            line-height: 1.4 !important;
        }

        div[data-testid="stPopover"] button svg {
            display: none !important;
        }

        div[data-testid="stPopover"] button {
            min-width: 34px !important;
            width: 34px !important;
            height: 34px !important;
            padding: 0 !important;
            border-radius: 50% !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-size: 22px !important;
            background: transparent !important;
            color: #5f6368 !important;
            border: none !important;
        }

        div[data-testid="stTextInput"] input {
            border: 1px solid transparent;
            border-radius: 6px;
            font-size: 16px;
            font-weight: 700;
            color: #202124;
            padding: 6px 8px;
            background: transparent;
        }

        div[data-testid="stTextInput"] input:hover {
            border: 1px solid #eef0ef;
            background: #f9f9f9;
        }

        div[data-testid="stTextInput"] input:focus {
            border: 1px solid #8ab4f8;
            background: #ffffff;
            box-shadow: none;
        }



        .tm-backend-online {
            background: #e6f4ea;
            color: #137333;
            border-radius: 999px;
            padding: 6px 10px;
            font-size: 12px;
            font-weight: 800;
            margin-left: 10px;
            white-space: nowrap;
        }

        .tm-backend-offline {
            background: #fce8e6;
            color: #c5221f;
            border-radius: 999px;
            padding: 6px 10px;
            font-size: 12px;
            font-weight: 800;
            margin-left: 10px;
            white-space: nowrap;
        }

        @media (max-width: 1000px) {
            .tm-topbar {
                width: 100%;
                min-width: unset;
            }

            .tm-connector {
                display: none;
            }

            .tm-sticky-wrapper {
                position: static !important;
            }

            div[data-testid="column"]:has(.tm-sticky-wrapper) {
                min-height: auto !important;
            }

            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.get("dark_mode", True):
        st.markdown(
            """
            <style>
            .stApp {
                background: #000000 !important;
                color: #ffffff !important;
            }
            .tm-topbar {
                background: #0c0c0e !important;
                border: 1px solid #222224 !important;
                color: #ffffff !important;
            }
            .tm-logo-text {
                color: #8ab4f8 !important;
            }
            .tm-class-card-pro {
                background: #0c0c0e !important;
                border: 1px solid #222224 !important;
                box-shadow: 0 4px 12px rgba(0,0,0,0.7) !important;
                color: #ffffff !important;
            }
            .tm-class-card-pro:hover {
                box-shadow: 0 6px 16px rgba(0,0,0,0.9) !important;
            }
            .tm-card-divider-pro {
                background: #222224 !important;
            }
            .tm-sample-label {
                color: #ffffff !important;
            }
            .tm-gallery-panel {
                border-left: 1px solid #222224 !important;
            }
            .tm-counter {
                color: #9aa0a6 !important;
            }
            .tm-empty-gallery {
                background: #050506 !important;
                border: 1px dashed #222224 !important;
                color: #9aa0a6 !important;
            }
            .tm-panel {
                background: #0c0c0e !important;
                border: 1px solid #222224 !important;
                box-shadow: 0 8px 24px rgba(0,0,0,0.7) !important;
                color: #ffffff !important;
            }
            .tm-panel-header {
                border-bottom: 1px solid #222224 !important;
                color: #ffffff !important;
            }
            .tm-muted {
                color: #9aa0a6 !important;
            }
            .tm-status-success {
                background: #137333 !important;
                color: #e6f4ea !important;
            }
            .tm-status-error {
                background: #c5221f !important;
                color: #fce8e6 !important;
            }
            .tm-status-info {
                background: #1967d2 !important;
                color: #e8f0fe !important;
            }
            .tm-preview-title {
                color: #ffffff !important;
            }
            .tm-input-title {
                color: #9aa0a6 !important;
            }
            .tm-output-title {
                color: #9aa0a6 !important;
            }
            .tm-prediction-main {
                background: #0d1627 !important;
                border: 1px solid #1a3a6c !important;
            }
            .tm-prediction-class {
                color: #8ab4f8 !important;
            }
            .tm-prediction-confidence {
                color: #e0e0e0 !important;
            }
            .tm-confidence-class {
                color: #9aa0a6 !important;
            }
            .tm-confidence-track {
                background: #222224 !important;
            }
            .tm-webcam-placeholder {
                background: #050506 !important;
                border: 1px dashed #222224 !important;
                color: #9aa0a6 !important;
            }
            div.stButton > button {
                background: #0c0c0e !important;
                color: #8ab4f8 !important;
                border: 1px solid #222224 !important;
            }
            div.stButton > button:hover {
                background: #1a3a6c !important;
                color: #ffffff !important;
                border: 1px solid #1967d2 !important;
            }
            div[data-testid="stTextInput"] input {
                color: #ffffff !important;
                background: #050506 !important;
            }
            div[data-testid="stTextInput"] input:hover {
                background: #121214 !important;
                border: 1px solid #222224 !important;
            }
            div[data-testid="stTextInput"] input:focus {
                background: #0c0c0e !important;
                border: 1px solid #1967d2 !important;
            }
            div[data-testid="stFileUploader"] section {
                background: #080c18 !important;
                border: 1px dashed #1a3a6c !important;
            }
            div[data-testid="stFileUploader"] button {
                background: #1c1c1e !important;
                color: #8ab4f8 !important;
                border: 1px solid #333333 !important;
            }
            div[data-testid="stFileUploader"] small {
                color: #9aa0a6 !important;
            }
            .stMarkdown, p, span, label, li, ul {
                color: #e0e0e0 !important;
            }
            div[data-testid="stExpander"] {
                background-color: #0c0c0e !important;
                border: 1px solid #222224 !important;
            }
            .tm-health-card {
                border-radius: 10px;
                padding: 16px;
                margin-bottom: 22px;
                font-family: Inter, system-ui, sans-serif;
            }
            .tm-health-card span, .tm-health-card li, .tm-health-card p, .tm-health-card ul {
                color: inherit !important;
            }
            .tm-confidence-badge {
                padding: 6px 12px;
                border-radius: 20px;
                font-weight: bold;
                font-size: 13px;
                display: inline-block;
                margin-bottom: 12px;
                font-family: Inter, sans-serif;
            }
            .tm-confidence-badge, .tm-confidence-badge span, .tm-confidence-badge p {
                color: inherit !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )


def initialize_session_state() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    if "classes" not in st.session_state:
        st.session_state.classes = [
            {"id": 1, "name": "Class 1", "uploaded_count": 0, "enabled": True, "disabled": False, "images": [], "ui_state": "idle"},
            {"id": 2, "name": "Class 2", "uploaded_count": 0, "enabled": True, "disabled": False, "images": [], "ui_state": "idle"},
        ]

    if "next_class_id" not in st.session_state:
        st.session_state.next_class_id = 3

    if "model_trained" not in st.session_state:
        st.session_state.model_trained = False

    if "training_result" not in st.session_state:
        st.session_state.training_result = None

    if "last_prediction" not in st.session_state:
        st.session_state.last_prediction = None

    if "dataset_summary" not in st.session_state:
        st.session_state.dataset_summary = {}

    if "preview_input_enabled" not in st.session_state:
        st.session_state.preview_input_enabled = True

    if "preview_mode" not in st.session_state:
        st.session_state.preview_mode = "File"

    if "sample_gallery" not in st.session_state:
        st.session_state.sample_gallery = {}

    if "active_camera_class_id" not in st.session_state:
        st.session_state.active_camera_class_id = None

    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = True

    if "history" not in st.session_state:
        st.session_state.history = []


def sanitize_name(name: str) -> str:
    import re
    cleaned = name.strip()
    cleaned = re.sub(r"[^a-zA-Z0-9_-]", "_", cleaned)
    return cleaned


def sync_classes_with_backend_summary() -> None:
    summary = st.session_state.get("dataset_summary", {})
    if not summary:
        return
    classes_summary = summary.get("classes", {})
    for c in st.session_state.classes:
        name = c["name"]
        sanitized = sanitize_name(name)
        real_count = 0
        if name in classes_summary:
            real_count = classes_summary[name]
        elif sanitized in classes_summary:
            real_count = classes_summary[sanitized]
        c["images"] = [None] * real_count
        c["disabled"] = not c.get("enabled", True)


def calculate_dataset_health():
    classes = st.session_state.classes

    valid_classes = [
        c for c in classes
        if len(c.get("images", [])) > 0 and not c.get("disabled", False)
    ]

    if not valid_classes:
        return 0, ["No dataset available"]

    counts = [len(c.get("images", [])) for c in valid_classes]

    min_images = min(counts)
    max_images = max(counts)

    score = 100
    messages = []

    if min_images < 10:
        score -= 25
        messages.append("Some classes have very few samples")

    if max_images - min_images > 20:
        score -= 20
        messages.append("Dataset is imbalanced")

    if len(valid_classes) < 2:
        score -= 50
        messages.append("Need at least 2 classes")

    if score >= 80:
        messages.append("Dataset quality looks good")

    return max(score, 0), messages


def render_dataset_health_score() -> None:
    score, messages = calculate_dataset_health()
    dark_mode = st.session_state.get("dark_mode", True)
    if dark_mode:
        if score >= 80:
            color = "#81c784"
            bg_color = "#0a2212"
            border_color = "#1b5e20"
        elif score >= 50:
            color = "#ffb74d"
            bg_color = "#241909"
            border_color = "#e65100"
        else:
            color = "#e57373"
            bg_color = "#240a0c"
            border_color = "#b71c1c"
    else:
        if score >= 80:
            color = "#137333"
            bg_color = "#e6f4ea"
            border_color = "#a8dab5"
        elif score >= 50:
            color = "#b06000"
            bg_color = "#fef7e0"
            border_color = "#ffe0b2"
        else:
            color = "#c5221f"
            bg_color = "#fce8e6"
            border_color = "#f5c2c1"

    messages_html = "".join([f"<li style='margin-bottom: 2px;'>{msg}</li>" for msg in messages])

    st.markdown(
        f"""
        <div class="tm-health-card" style="
            background: {bg_color};
            border: 1px solid {border_color};
            color: {color};
        ">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <span style="font-weight: 800; font-size: 16px;">Dataset Health Score</span>
                <span style="font-size: 26px; font-weight: 900;">{score}/100</span>
            </div>
            <ul style="margin: 10px 0 0 16px; padding: 0; font-size: 13px; line-height: 1.4;">
                {messages_html}
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def plot_confusion_matrix(cm_data, class_names):
    import numpy as np
    cm = np.array(cm_data)
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    dark_mode = st.session_state.get("dark_mode", False)
    if dark_mode:
        fig.patch.set_facecolor('#1e1e1e')
        ax.set_facecolor('#1e1e1e')
        text_color = 'white'
        cmap = plt.cm.Blues_r
    else:
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        text_color = 'black'
        cmap = plt.cm.Blues

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, color=text_color)
    ax.set_yticklabels(class_names, color=text_color)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else text_color)

    ax.set_title("Confusion Matrix Heatmap", color=text_color, fontweight='bold', fontsize=11)
    ax.set_xlabel("Predicted Label", color=text_color, fontsize=9)
    ax.set_ylabel("True Label", color=text_color, fontsize=9)

    if dark_mode:
        ax.spines['bottom'].set_color('#333333')
        ax.spines['top'].set_color('#333333')
        ax.spines['right'].set_color('#333333')
        ax.spines['left'].set_color('#333333')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
    else:
        ax.spines['bottom'].set_color('#dadce0')
        ax.spines['top'].set_color('#dadce0')
        ax.spines['right'].set_color('#dadce0')
        ax.spines['left'].set_color('#dadce0')

    fig.tight_layout()
    return fig


def render_evaluation_metrics_panel(training_result) -> None:
    eval_metrics = training_result.get("eval_metrics")
    if not eval_metrics:
        return

    class_metrics = eval_metrics.get("class_metrics", {})
    class_names = list(class_metrics.keys())

    st.markdown('<div class="tm-panel" style="margin-top: 20px;">', unsafe_allow_html=True)
    st.markdown('<div class="tm-panel-header">Evaluation Metrics</div>', unsafe_allow_html=True)
    st.markdown('<div class="tm-panel-body">', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        try:
            fig = plot_confusion_matrix(eval_metrics["confusion_matrix"], class_names)
            st.pyplot(fig, clear_figure=True)
        except Exception as e:
            st.error(f"Error plotting confusion matrix: {str(e)}")

    with col2:
        st.markdown("**Metrics Summary**")
        table_rows = ""
        for name, metrics in class_metrics.items():
            table_rows += f"""
            <tr style="border-bottom: 1px solid #dadce0;">
                <td style="padding: 6px; font-weight: bold;">{name}</td>
                <td style="padding: 6px; text-align: right;">{metrics['precision']:.1f}%</td>
                <td style="padding: 6px; text-align: right;">{metrics['recall']:.1f}%</td>
                <td style="padding: 6px; text-align: right;">{metrics['f1_score']:.1f}%</td>
                <td style="padding: 6px; text-align: right;">{metrics['support']}</td>
            </tr>
            """
        
        avg_f1 = np.mean([m['f1_score'] for m in class_metrics.values()])
        avg_precision = np.mean([m['precision'] for m in class_metrics.values()])
        avg_recall = np.mean([m['recall'] for m in class_metrics.values()])
        total_support = sum([m['support'] for m in class_metrics.values()])

        table_rows += f"""
        <tr style="background: #e8f0fe; font-weight: bold; border-top: 2px solid #1967d2;">
            <td style="padding: 6px; color: #1967d2 !important;">Macro Avg</td>
            <td style="padding: 6px; text-align: right; color: #1967d2 !important;">{avg_precision:.1f}%</td>
            <td style="padding: 6px; text-align: right; color: #1967d2 !important;">{avg_recall:.1f}%</td>
            <td style="padding: 6px; text-align: right; color: #1967d2 !important;">{avg_f1:.1f}%</td>
            <td style="padding: 6px; text-align: right; color: #1967d2 !important;">{total_support}</td>
        </tr>
        """

        dark_mode = st.session_state.get("dark_mode", False)
        header_style = "background: #f1f3f4; color: #202124;" if not dark_mode else "background: #2a2a2a; color: #ffffff;"
        border_style = "border: 1px solid #dadce0;" if not dark_mode else "border: 1px solid #333333;"
        text_color_style = "color: #202124;" if not dark_mode else "color: #e0e0e0;"

        st.markdown(
            f"""
            <table style="width: 100%; border-collapse: collapse; font-family: Inter, sans-serif; font-size: 12px; {border_style} {text_color_style}">
                <thead>
                    <tr style="{header_style} font-weight: bold;">
                        <th style="padding: 6px; text-align: left; border-bottom: 2px solid #c4c7c5;">Class</th>
                        <th style="padding: 6px; text-align: right; border-bottom: 2px solid #c4c7c5;">P</th>
                        <th style="padding: 6px; text-align: right; border-bottom: 2px solid #c4c7c5;">R</th>
                        <th style="padding: 6px; text-align: right; border-bottom: 2px solid #c4c7c5;">F1</th>
                        <th style="padding: 6px; text-align: right; border-bottom: 2px solid #c4c7c5;">Supp</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_ai_insights(training_result) -> None:
    eval_metrics = training_result.get("eval_metrics")
    class_distribution = training_result.get("class_distribution", {})

    if not eval_metrics or not class_distribution:
        return

    class_metrics = eval_metrics.get("class_metrics", {})
    if not class_metrics:
        return

    st.markdown('<div class="tm-panel" style="margin-top: 20px;">', unsafe_allow_html=True)
    st.markdown('<div class="tm-panel-header">AI Insights Panel</div>', unsafe_allow_html=True)
    st.markdown('<div class="tm-panel-body">', unsafe_allow_html=True)

    insights = []

    # 1. Compare performance
    best_class = max(class_metrics, key=lambda k: class_metrics[k]["f1_score"])
    best_f1 = class_metrics[best_class]["f1_score"]
    insights.append(f"🎉 **{best_class} class performs best** with an F1-score of **{best_f1:.1f}%**.")

    worst_class = min(class_metrics, key=lambda k: class_metrics[k]["f1_score"])
    worst_f1 = class_metrics[worst_class]["f1_score"]
    if worst_f1 < 75 and worst_class != best_class:
        insights.append(f"⚠️ **{worst_class} class needs improvement** (F1-score: **{worst_f1:.1f}%**). Consider adding more diverse training images.")

    # 2. Imbalance detection
    counts = list(class_distribution.values())
    if len(counts) > 1:
        ratio = max(counts) / (min(counts) + 1e-5)
        min_class = min(class_distribution, key=class_distribution.get)
        max_class = max(class_distribution, key=class_distribution.get)
        if ratio > 2.0:
            insights.append(f"🚨 **Dataset Imbalance:** **{min_class}** has significantly fewer samples than **{max_class}** ({ratio:.1f}x imbalance ratio). Add more images to **{min_class}**.")
        elif ratio > 1.25:
            insights.append(f"⚠️ **Dataset Imbalance:** **{min_class}** has fewer samples than **{max_class}** (ratio: {ratio:.1f}x).")
        else:
            insights.append("✅ **Dataset Balance:** Your dataset distribution is well-balanced.")

    # 3. Stability comment
    avg_f1 = np.mean([m["f1_score"] for m in class_metrics.values()])
    if avg_f1 >= 85:
        insights.append("🛡️ **Model Stability:** Model confidence is **stable and highly reliable**.")
    elif avg_f1 >= 60:
        insights.append("⚖️ **Model Stability:** Model confidence is **moderately stable**. Borderline inputs might cause fluctuating predictions.")
    else:
        insights.append("❌ **Model Stability:** Model confidence is **unstable**. Add more clean training samples to stabilize predictions.")

    dark_mode = st.session_state.get("dark_mode", False)
    bg_style = "background: #f8fafd; border: 1px solid #d9e2f3;" if not dark_mode else "background: #181818; border: 1px solid #333333;"
    text_style = "color: #3c4043; line-height: 1.6; font-size: 13px;" if not dark_mode else "color: #e0e0e0; line-height: 1.6; font-size: 13px;"

    insights_html = "".join([f"<div style='margin-bottom: 8px;'>{ins}</div>" for ins in insights])

    st.markdown(
        f"""
        <div style="border-radius: 8px; padding: 16px; {bg_style} {text_style}">
            {insights_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def add_to_prediction_history(prediction) -> None:
    if "history" not in st.session_state:
        st.session_state.history = []
    
    prediction_info = {
        "class": prediction["predicted_class"],
        "confidence": prediction["confidence"]
    }
    st.session_state.history.append(prediction_info)
    st.session_state.history = st.session_state.history[-5:]


def render_topbar() -> None:
    backend_online = check_backend_health()
    status_text = "Backend Online" if backend_online else "Backend Offline"
    status_class = "tm-backend-online" if backend_online else "tm-backend-offline"

    st.markdown(
        f"""
        <div class="tm-topbar">
            <div class="tm-menu-icon">☰</div>
            <div class="tm-logo-text">Teachable Machine</div>
            <div class="{status_class}">{status_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    reset_clicked = st.button(
        "Reset Project",
        key="reset_project_button",
        use_container_width=False,
    )

    if reset_clicked:
        try:
            reset_session_on_backend()
        except Exception:
            pass

        reset_frontend_project_state()
        st.rerun()


def show_status(message: str, status_type: str = "info") -> None:
    css_class = {
        "success": "tm-status-success",
        "error": "tm-status-error",
        "info": "tm-status-info",
    }.get(status_type, "tm-status-info")

    st.markdown(
        f"<div class='{css_class}'>{message}</div>",
        unsafe_allow_html=True,
    )



def check_backend_health() -> bool:
    try:
        response = requests.get(
            f"{API_BASE_URL}/health",
            timeout=5,
        )
        return response.status_code == 200
    except Exception:
        return False


def cleanup_old_sessions_on_backend() -> None:
    try:
        requests.post(f"{API_BASE_URL}/cleanup-old-sessions", timeout=30)
    except Exception:
        pass


def upload_samples_to_backend(class_name: str, uploaded_files: List) -> Dict:
    files_payload = []

    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.getvalue()
        files_payload.append(
            (
                "files",
                (
                    uploaded_file.name,
                    file_bytes,
                    uploaded_file.type or "image/jpeg",
                ),
            )
        )

    response = requests.post(
        f"{API_BASE_URL}/upload-sample",
        data={
            "session_id": st.session_state.session_id,
            "class_name": class_name,
        },
        files=files_payload,
        timeout=120,
    )

    if response.status_code != 200:
        error_message = response.json().get("detail", "Failed to upload samples.")
        raise Exception(error_message)

    return response.json()


def upload_webcam_frames_to_backend(class_name: str, frame_bytes_list: List[bytes]) -> Dict:
    files_payload = []

    for index, frame_bytes in enumerate(frame_bytes_list):
        files_payload.append(
            (
                "files",
                (
                    f"webcam_sample_{index + 1}.jpg",
                    frame_bytes,
                    "image/jpeg",
                ),
            )
        )

    response = requests.post(
        f"{API_BASE_URL}/upload-sample",
        data={
            "session_id": st.session_state.session_id,
            "class_name": class_name,
        },
        files=files_payload,
        timeout=180,
    )

    if response.status_code != 200:
        error_message = response.json().get("detail", "Failed to upload webcam samples.")
        raise Exception(error_message)

    return response.json()


def train_model_on_backend() -> Dict:
    response = requests.post(
        f"{API_BASE_URL}/train",
        data={"session_id": st.session_state.session_id},
        timeout=600,
    )

    if response.status_code != 200:
        error_message = response.json().get("detail", "Training failed.")
        raise Exception(error_message)

    return response.json()


def predict_image_on_backend(uploaded_file) -> Dict:
    files_payload = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type or "image/jpeg",
        )
    }

    response = requests.post(
        f"{API_BASE_URL}/predict",
        data={"session_id": st.session_state.session_id},
        files=files_payload,
        timeout=180,
    )

    if response.status_code != 200:
        error_message = response.json().get("detail", "Prediction failed.")
        raise Exception(error_message)

    return response.json()


def predict_frame_bytes_on_backend(frame_bytes: bytes) -> Dict:
    files_payload = {
        "file": (
            "webcam_preview.jpg",
            frame_bytes,
            "image/jpeg",
        )
    }

    response = requests.post(
        f"{API_BASE_URL}/predict",
        data={"session_id": st.session_state.session_id},
        files=files_payload,
        timeout=180,
    )

    if response.status_code != 200:
        error_message = response.json().get("detail", "Webcam prediction failed.")
        raise Exception(error_message)

    return response.json()


def get_dataset_summary_from_backend() -> Dict:
    response = requests.get(
        f"{API_BASE_URL}/dataset-summary",
        params={"session_id": st.session_state.session_id},
        timeout=60,
    )

    if response.status_code != 200:
        error_message = response.json().get("detail", "Failed to fetch dataset summary.")
        raise Exception(error_message)

    return response.json()["data"]


def delete_class_from_backend(class_name: str) -> Dict:
    response = requests.delete(
        f"{API_BASE_URL}/delete-class",
        data={
            "session_id": st.session_state.session_id,
            "class_name": class_name,
        },
        timeout=60,
    )

    if response.status_code != 200:
        error_message = response.json().get("detail", "Failed to delete class.")
        raise Exception(error_message)

    return response.json()["data"]


def reset_session_on_backend() -> Dict:
    response = requests.delete(
        f"{API_BASE_URL}/reset-session",
        data={"session_id": st.session_state.session_id},
        timeout=60,
    )

    if response.status_code != 200:
        error_message = response.json().get("detail", "Failed to reset session.")
        raise Exception(error_message)

    return response.json()["data"]


def get_export_model_url() -> str:
    return f"{API_BASE_URL}/export-model?session_id={st.session_state.session_id}"


def get_real_uploaded_count(class_name: str, fallback_count: int = 0) -> int:
    dataset_classes = st.session_state.dataset_summary.get("classes", {})

    if class_name in dataset_classes:
        return dataset_classes[class_name]

    safe_name = class_name.replace(" ", "_")

    if safe_name in dataset_classes:
        return dataset_classes[safe_name]

    return fallback_count


def validate_classes_before_training() -> List[str]:
    warnings = []

    active_classes = [
        class_item for class_item in st.session_state.classes
        if class_item.get("enabled", True)
    ]

    if len(active_classes) < 2:
        warnings.append("At least two enabled classes are required for training.")

    seen_names = set()

    for class_item in active_classes:
        class_name = class_item["name"].strip()

        if not class_name:
            warnings.append("Every enabled class must have a valid name.")
            continue

        normalized_name = class_name.lower()

        if normalized_name in seen_names:
            warnings.append(f"Duplicate class name found: {class_name}")
        else:
            seen_names.add(normalized_name)

        uploaded_count = get_real_uploaded_count(
            class_name,
            class_item.get("uploaded_count", 0)
        )

        if uploaded_count <= 0:
            warnings.append(
                f"Please upload image samples for '{class_name}' or disable/delete this class."
            )

    return warnings


def add_new_class() -> None:
    class_id = st.session_state.next_class_id

    new_class = {
        "id": class_id,
        "name": f"Class {class_id}",
        "uploaded_count": 0,
        "enabled": True,
        "ui_state": "idle",
    }

    st.session_state.classes.append(new_class)
    st.session_state.sample_gallery[class_id] = []

    st.session_state.next_class_id += 1
    st.session_state.model_trained = False
    st.session_state.training_result = None
    st.session_state.last_prediction = None


def handle_class_action(action: str, class_item: Dict, index: int) -> None:
    class_id = class_item["id"]

    if action == "disable":
        st.session_state.classes[index]["enabled"] = False
        st.session_state.classes[index]["disabled"] = True
        st.session_state.classes[index]["ui_state"] = "idle"

        if st.session_state.active_camera_class_id == class_id:
            st.session_state.active_camera_class_id = None

        st.session_state.model_trained = False
        st.session_state.training_result = None
        st.session_state.last_prediction = None
        st.rerun()

    if action == "enable":
        st.session_state.classes[index]["enabled"] = True
        st.session_state.classes[index]["disabled"] = False
        st.session_state.classes[index]["ui_state"] = "idle"

        st.session_state.model_trained = False
        st.session_state.training_result = None
        st.session_state.last_prediction = None
        st.rerun()

    if action == "delete":
        try:
            delete_class_from_backend(class_item["name"])
        except Exception:
            pass

        if class_id in st.session_state.sample_gallery:
            del st.session_state.sample_gallery[class_id]

        if st.session_state.active_camera_class_id == class_id:
            st.session_state.active_camera_class_id = None

        st.session_state.classes.pop(index)

        st.session_state.model_trained = False
        st.session_state.training_result = None
        st.session_state.last_prediction = None
        st.rerun()


def cache_uploaded_previews(class_id: int, uploaded_files: List) -> None:
    if class_id not in st.session_state.sample_gallery:
        st.session_state.sample_gallery[class_id] = []

    for uploaded_file in uploaded_files:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            preview_buffer = io.BytesIO()
            image.thumbnail((120, 120))
            image.save(preview_buffer, format="JPEG")
            st.session_state.sample_gallery[class_id].append(preview_buffer.getvalue())
            uploaded_file.seek(0)
        except Exception:
            continue


def cache_webcam_previews(class_id: int, frame_bytes_list: List[bytes]) -> None:
    if class_id not in st.session_state.sample_gallery:
        st.session_state.sample_gallery[class_id] = []

    for frame_bytes in frame_bytes_list:
        st.session_state.sample_gallery[class_id].append(frame_bytes)


def render_sample_gallery(class_id: int, sample_count: int) -> None:
    st.markdown(
        f"""
        <div class="tm-counter">
            {sample_count} Image Sample{"s" if sample_count != 1 else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )

    gallery_items = st.session_state.sample_gallery.get(class_id, [])

    if not gallery_items:
        st.markdown(
            """
            <div class="tm-empty-gallery">
                Uploaded sample previews will appear here.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    image_html = ""

    for image_bytes in gallery_items[-16:]:
        encoded_image = base64.b64encode(image_bytes).decode("utf-8")
        image_html += f"""
        <img 
            src="data:image/jpeg;base64,{encoded_image}" 
            class="tm-gallery-image" 
            alt="sample"
        />
        """

    st.markdown(
        f"""
        <div class="tm-gallery-grid">
            {image_html}
        </div>
        """,
        unsafe_allow_html=True,
    )



def render_upload_mode_visual() -> None:
    """
    Render the Teachable-Machine-style upload panel as a real visual block.
    Using components.html prevents Streamlit from showing raw HTML text.
    """
    components.html(
        """
        <div style="
            background:#e8f0fe;
            border-radius:0;
            padding:14px;
            min-height:300px;
            font-family:Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
            box-sizing:border-box;
        ">
            <div style="
                display:flex;
                justify-content:space-between;
                align-items:center;
                color:#1967d2;
                font-size:15px;
                font-weight:700;
                margin-bottom:14px;
            ">
                <span>File</span>
                <span style="font-size:26px; line-height:1; font-weight:400;">×</span>
            </div>

            <div style="
                background:#d2e3fc;
                border-radius:6px;
                padding:22px 14px;
                text-align:center;
                color:#1967d2;
                font-size:15px;
                line-height:1.4;
                font-weight:700;
                margin-bottom:18px;
                box-sizing:border-box;
            ">
                <div style="font-size:20px; margin-bottom:8px;">▣</div>
                Choose images from your files,<br>
                or drag &amp; drop here
            </div>

            <div style="
                margin-top:24px;
                text-align:center;
                color:#4285f4;
                font-size:14px;
                font-weight:700;
            ">
                <div style="font-size:34px; margin-bottom:8px;">🖼️ → 🖼️</div>
                Images will be cropped to square
            </div>
        </div>
        """,
        height=330,
        scrolling=False,
    )

def render_webcam_capture_section(class_item: Dict, index: int, updated_name: str) -> None:
    class_id = class_item["id"]

    st.markdown(
        """
        <div class="tm-webcam-placeholder" style="min-height: 80px;">
            Webcam is active. Use Start/Stop Recording to collect samples.
        </div>
        """,
        unsafe_allow_html=True,
    )

    webrtc_ctx = webrtc_streamer(
        key=f"class_webcam_{class_id}",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=WebcamSampleProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False,
        },
        async_processing=True,
    )

    processor = webrtc_ctx.video_processor

    if processor:
        current_count = len(processor.captured_frames)

        st.markdown(
            f"""
            <div class="tm-counter">
                Captured in current session: {current_count}
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("Start Recording", key=f"start_recording_{class_id}", use_container_width=True):
            processor.is_recording = True
            show_status("Recording started. Move the object slowly in front of the camera.", "success")

        if st.button("Stop Recording", key=f"stop_recording_{class_id}", use_container_width=True):
            processor.is_recording = False
            show_status("Recording stopped. You can save the captured samples now.", "info")

        if st.button("Save Webcam Samples", key=f"save_webcam_{class_id}", use_container_width=True):
            if not updated_name.strip():
                show_status("Please enter a valid class name before saving webcam samples.", "error")
            elif not processor.captured_frames:
                show_status("No webcam samples captured yet. Start recording first.", "error")
            else:
                try:
                    processor.is_recording = False

                    with st.spinner(f"Uploading webcam samples for {updated_name}..."):
                        result = upload_webcam_frames_to_backend(
                            updated_name.strip(),
                            processor.captured_frames,
                        )

                    saved_count = result["data"]["saved_count"]

                    st.session_state.classes[index]["uploaded_count"] += saved_count
                    cache_webcam_previews(class_id, processor.captured_frames)

                    processor.captured_frames.clear()

                    st.session_state.model_trained = False
                    st.session_state.training_result = None
                    st.session_state.last_prediction = None

                    try:
                        st.session_state.dataset_summary = get_dataset_summary_from_backend()
                    except Exception:
                        pass

                    show_status(
                        f"{saved_count} webcam sample(s) saved successfully for {updated_name}.",
                        "success",
                    )

                    st.session_state.classes[index]["ui_state"] = "idle"

                    if st.session_state.active_camera_class_id == class_id:
                        st.session_state.active_camera_class_id = None
                        
                    st.rerun()

                except requests.exceptions.ConnectionError:
                    show_status(
                        "Backend is not running. Start FastAPI using: uvicorn app.main:app --reload",
                        "error",
                    )
                except Exception as error:
                    show_status(str(error), "error")

    close_clicked = st.button(
        "Close Webcam",
        key=f"close_webcam_{class_id}",
        use_container_width=True,
    )

    if close_clicked:
        st.session_state.classes[index]["ui_state"] = "idle"

        if st.session_state.active_camera_class_id == class_id:
            st.session_state.active_camera_class_id = None

        st.rerun()


def render_class_card(class_item: Dict, index: int) -> None:
    class_id = class_item["id"]
    is_enabled = class_item.get("enabled", True)
    ui_state = class_item.get("ui_state", "idle")

    disabled_class = "" if is_enabled else "tm-class-card-disabled"
    status_label = "Enabled" if is_enabled else "Disabled"

    real_count = get_real_uploaded_count(
        class_item["name"],
        class_item.get("uploaded_count", 0),
    )

    st.markdown(
        f'<div class="tm-class-card-pro {disabled_class}">',
        unsafe_allow_html=True,
    )

    header_col_1, header_col_2, header_col_3 = st.columns([7, 1, 1])

    with header_col_1:
        updated_name = st.text_input(
            "Class name",
            value=class_item["name"],
            key=f"class_name_{class_id}",
            label_visibility="collapsed",
            placeholder="Enter class name",
            disabled=not is_enabled,
        )

        if updated_name.strip() and updated_name != class_item["name"]:
            st.session_state.classes[index]["name"] = updated_name.strip()
            st.session_state.model_trained = False
            st.session_state.training_result = None
            st.session_state.last_prediction = None

    with header_col_2:
        st.markdown(
            """
            <div class="tm-edit-icon">✎</div>
            """,
            unsafe_allow_html=True,
        )

    with header_col_3:
        with st.popover("⋮", use_container_width=True):
            st.caption(f"Class status: {status_label}")

            if is_enabled:
                if st.button("Disable class", key=f"disable_{class_id}"):
                    handle_class_action("disable", class_item, index)
            else:
                if st.button("Enable class", key=f"enable_{class_id}"):
                    handle_class_action("enable", class_item, index)

            if st.button("Delete class", key=f"delete_{class_id}", type="secondary"):
                handle_class_action("delete", class_item, index)

    st.markdown('<div class="tm-card-divider-pro"></div>', unsafe_allow_html=True)

    if not is_enabled:
        st.markdown(
            """
            <div class="tm-disabled-message">
                This class is disabled and will be ignored during training.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return

    if ui_state == "camera_on":
        render_webcam_capture_section(class_item, index, updated_name)
        st.markdown("</div>", unsafe_allow_html=True)
        return

    body_left, body_right = st.columns([1.05, 1.45])

    with body_left:
        st.markdown(
            '<div class="tm-sample-label">Add Image Samples:</div>',
            unsafe_allow_html=True,
        )

        if ui_state == "idle":
            webcam_clicked = st.button(
                "🎥 Webcam",
                key=f"webcam_btn_{class_id}",
                use_container_width=True,
            )

            upload_clicked_mode = st.button(
                "⬆ Upload",
                key=f"upload_btn_{class_id}",
                use_container_width=True,
            )

            if webcam_clicked:
                if st.session_state.active_camera_class_id not in [None, class_id]:
                    show_status(
                        "Another class webcam is active. Please close it before opening a new one.",
                        "error",
                    )
                else:
                    st.session_state.active_camera_class_id = class_id
                    st.session_state.classes[index]["ui_state"] = "camera_on"
                    st.rerun()

            if upload_clicked_mode:
                st.session_state.classes[index]["ui_state"] = "upload"
                st.rerun()

        if ui_state == "upload":
            render_upload_mode_visual()

            uploaded_files = st.file_uploader(
                "Upload training images",
                type=["jpg", "jpeg", "png", "webp"],
                accept_multiple_files=True,
                key=f"uploader_{class_id}",
                label_visibility="collapsed",
            )

            upload_clicked = st.button(
                "Save Samples",
                key=f"save_uploads_{class_id}",
                use_container_width=True,
            )

            cancel_clicked = st.button(
                "Close File Mode",
                key=f"cancel_upload_{class_id}",
                use_container_width=True,
            )

            if cancel_clicked:
                st.session_state.classes[index]["ui_state"] = "idle"
                st.rerun()

            if upload_clicked:
                if not updated_name.strip():
                    show_status("Please enter a valid class name before uploading.", "error")
                elif not uploaded_files:
                    show_status("Please select at least one image for this class.", "error")
                else:
                    try:
                        with st.spinner(f"Uploading samples for {updated_name}..."):
                            result = upload_samples_to_backend(
                                updated_name.strip(),
                                uploaded_files,
                            )

                        saved_count = result["data"]["saved_count"]

                        st.session_state.classes[index]["uploaded_count"] += saved_count
                        cache_uploaded_previews(class_id, uploaded_files)

                        st.session_state.model_trained = False
                        st.session_state.training_result = None
                        st.session_state.last_prediction = None

                        try:
                            st.session_state.dataset_summary = get_dataset_summary_from_backend()
                        except Exception:
                            pass

                        show_status(
                            f"{saved_count} image sample(s) uploaded successfully for {updated_name}.",
                            "success",
                        )

                        st.session_state.classes[index]["ui_state"] = "idle"
                        st.rerun()

                    except requests.exceptions.ConnectionError:
                        show_status(
                            "Backend is not running. Start FastAPI using: uvicorn app.main:app --reload",
                            "error",
                        )
                    except Exception as error:
                        show_status(str(error), "error")

    with body_right:
        st.markdown('<div class="tm-gallery-panel">', unsafe_allow_html=True)
        render_sample_gallery(class_id, real_count)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def render_classes_area() -> None:
    for index, class_item in enumerate(st.session_state.classes):
        render_class_card(class_item, index)

    if st.button("⊞ Add a class", key="add_class_button", use_container_width=True):
        add_new_class()
        st.rerun()


def render_training_panel() -> None:
    st.markdown('<div class="tm-sticky-wrapper">', unsafe_allow_html=True)
    st.markdown('<div class="tm-panel">', unsafe_allow_html=True)
    st.markdown('<div class="tm-panel-header">Training</div>', unsafe_allow_html=True)
    st.markdown('<div class="tm-panel-body">', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="tm-muted">
            Upload samples for at least two enabled classes, then train your model.
        </div>
        """,
        unsafe_allow_html=True,
    )

    train_clicked = st.button("Train Model", key="train_model_button")

    if train_clicked:
        try:
            try:
                st.session_state.dataset_summary = get_dataset_summary_from_backend()
            except Exception:
                pass

            validation_warnings = validate_classes_before_training()

            if validation_warnings:
                for warning in validation_warnings:
                    show_status(warning, "error")
                st.stop()

            # Multi-stage Progress bar animation
            status_text = st.empty()
            progress = st.progress(0)
            
            # Stage 1: Data Loading
            status_text.markdown("🔄 **Stage 1/4: Data loading...**")
            for i in range(25):
                time.sleep(0.01)
                progress.progress(i + 1)

            # Stage 2: Feature Extraction (Make actual backend request)
            status_text.markdown("🧠 **Stage 2/4: Feature extraction...**")
            for i in range(25, 50):
                time.sleep(0.01)
                progress.progress(i + 1)

            result = train_model_on_backend()

            # Stage 3: Training
            status_text.markdown("⚡ **Stage 3/4: Model training...**")
            for i in range(50, 75):
                time.sleep(0.01)
                progress.progress(i + 1)

            # Stage 4: Evaluation
            status_text.markdown("📈 **Stage 4/4: Model evaluation...**")
            for i in range(75, 100):
                time.sleep(0.01)
                progress.progress(i + 1)

            progress.progress(100)
            time.sleep(0.1)
            status_text.empty()
            progress.empty()

            st.session_state.model_trained = True
            st.session_state.training_result = result["data"]
            st.session_state.last_prediction = None

            show_status("Model trained successfully. Preview is now unlocked.", "success")
            
            # Sync session states classes
            try:
                st.session_state.dataset_summary = get_dataset_summary_from_backend()
                sync_classes_with_backend_summary()
            except Exception:
                pass

        except requests.exceptions.ConnectionError:
            show_status(
                "Backend is not running. Start FastAPI using: uvicorn app.main:app --reload",
                "error",
            )
        except Exception as error:
            st.session_state.model_trained = False
            show_status(str(error), "error")

    if st.session_state.training_result:
        training_data = st.session_state.training_result

        st.markdown(
            f"""
            <div style="background: #e6f4ea; border: 1px solid #a8dab5; border-radius: 10px; padding: 16px; margin-top: 12px; font-family: Inter, sans-serif;">
                <div style="font-weight: 800; color: #137333; font-size: 15px; margin-bottom: 10px;">⚡ Training Analytics</div>
                <div style="font-size: 13px; color: #137333; line-height: 1.6;">
                    • <b>Accuracy:</b> {training_data.get("accuracy_percentage", 0)}%<br>
                    • <b>Training completed time:</b> {training_data.get("training_time_seconds", 0)}s<br>
                    • <b>Images processed:</b> {training_data.get("images_processed", 0)}<br>
                    • <b>Classes count:</b> {training_data.get("classes_count", 0)}<br>
                    • <b>Device:</b> {training_data.get("device", "cpu")}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        show_status("Model is not trained yet. Preview will unlock after training.", "info")

    st.markdown(
        """
        <div class="tm-muted" style="margin-top: 16px; border-top: 1px solid #f3f4f3; padding-top: 14px;">
            Advanced ⌄
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_probability_bars(probabilities: Dict[str, float]) -> None:
    sorted_probabilities = dict(
        sorted(
            probabilities.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    )

    top_class = next(iter(sorted_probabilities), None)

    for class_name, score in sorted_probabilities.items():
        safe_width = max(2, min(float(score), 100))
        top_class_style = "tm-confidence-fill-top" if class_name == top_class else ""

        st.markdown(
            f"""
            <div class="tm-confidence-row">
                <div class="tm-confidence-class">{class_name}</div>
                <div class="tm-confidence-track">
                    <div class="tm-confidence-fill {top_class_style}" style="width: {safe_width}%;">
                        {score}%
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_preview_header() -> None:
    st.markdown(
        """
        <div class="tm-preview-header-row">
            <div class="tm-preview-title">Preview</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.model_trained:
        st.link_button(
            "⬆ Export Model",
            get_export_model_url(),
            use_container_width=True,
        )
    else:
        st.button(
            "⬆ Export Model",
            disabled=True,
            use_container_width=True,
        )


def render_preview_input_controls() -> None:
    st.markdown(
        """
        <div class="tm-input-control-row-clean">
            <div class="tm-input-title">Input</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.session_state.preview_input_enabled = st.toggle(
        "Input Enabled",
        value=st.session_state.preview_input_enabled,
        key="preview_input_toggle",
    )

    st.session_state.preview_mode = st.selectbox(
        "Input Mode",
        ["File", "Webcam"],
        index=0 if st.session_state.preview_mode == "File" else 1,
        key="preview_mode_select",
    )


def render_webcam_prediction_preview() -> None:
    st.markdown(
        """
        <div class="tm-webcam-placeholder" style="min-height: 80px; margin-bottom: 5px;">
            Webcam Preview Mode (Live Inference HUD Overlay)
        </div>
        """,
        unsafe_allow_html=True,
    )

    webrtc_ctx = webrtc_streamer(
        key="preview_webcam_predictor",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=PreviewWebcamProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False,
        },
        async_processing=True,
    )

    processor = webrtc_ctx.video_processor

    if processor:
        processor.session_id = st.session_state.session_id
        
        # Display live metrics from processor
        fps_val = getattr(processor, "fps", 0.0)
        inf_val = getattr(processor, "inference_time_ms", 0.0)
        
        st.markdown(
            f"""
            <div style="display: flex; gap: 15px; margin-top: 5px; margin-bottom: 10px;">
                <div style="background: #e8f0fe; padding: 8px; border-radius: 8px; flex: 1; text-align: center; border: 1px solid #d2e3fc;">
                    <div style="font-size: 11px; color: #5f6368; font-weight: 800;">LIVE FPS</div>
                    <div style="font-size: 16px; font-weight: 900; color: #1967d2;">{fps_val:.1f}</div>
                </div>
                <div style="background: #e8f0fe; padding: 8px; border-radius: 8px; flex: 1; text-align: center; border: 1px solid #d2e3fc;">
                    <div style="font-size: 11px; color: #5f6368; font-weight: 800;">INFERENCE</div>
                    <div style="font-size: 16px; font-weight: 900; color: #1967d2;">{inf_val:.1f} ms</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        # If live predictions exist, update session state last prediction
        if processor.last_predictions:
            st.session_state.last_prediction = processor.last_predictions
            # To avoid spamming, we only append to history on class change or manual trigger
            if not st.session_state.history or st.session_state.history[-1]["class"] != processor.last_predictions["predicted_class"]:
                add_to_prediction_history(processor.last_predictions)

    predict_webcam_clicked = st.button(
        "Capture & Predict Frame",
        key="predict_webcam_frame_button",
        use_container_width=True,
    )

    if predict_webcam_clicked:
        if not processor or not processor.latest_frame:
            show_status("No webcam frame available yet. Please wait a moment.", "error")
        else:
            try:
                with st.spinner("Predicting webcam frame..."):
                    result = predict_frame_bytes_on_backend(processor.latest_frame)

                st.session_state.last_prediction = result["data"]
                add_to_prediction_history(result["data"])

            except requests.exceptions.ConnectionError:
                show_status(
                    "Backend is not running. Start FastAPI using: uvicorn app.main:app --reload",
                    "error",
                )
            except Exception as error:
                show_status(str(error), "error")


def render_preview_panel() -> None:
    st.markdown('<div class="tm-sticky-wrapper">', unsafe_allow_html=True)
    st.markdown('<div class="tm-panel">', unsafe_allow_html=True)

    render_preview_header()

    st.markdown('<div class="tm-panel-body">', unsafe_allow_html=True)

    if not st.session_state.model_trained:
        st.markdown(
            """
            <div class="tm-muted">
                You must train a model on the left before you can preview it here.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        return

    render_preview_input_controls()

    if not st.session_state.preview_input_enabled:
        show_status("Preview input is currently OFF. Turn it ON to test predictions.", "info")

        st.markdown('<div class="tm-output-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="tm-output-title">Output</div>', unsafe_allow_html=True)

        if st.session_state.last_prediction:
            render_probability_bars(st.session_state.last_prediction["probabilities"])

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        return

    test_image = None

    if st.session_state.preview_mode == "File":
        test_image = st.file_uploader(
            "Choose image from your files",
            type=["jpg", "jpeg", "png", "webp"],
            key="prediction_uploader",
        )

        if test_image:
            image = Image.open(test_image).convert("RGB")
            st.markdown('<div class="tm-file-preview-image">', unsafe_allow_html=True)
            st.image(image, caption="Selected Input Image", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        predict_clicked = st.button("Predict Image", key="predict_button")

        if predict_clicked:
            if not test_image:
                show_status("Please upload a test image first.", "error")
            else:
                try:
                    with st.spinner("Predicting image..."):
                        result = predict_image_on_backend(test_image)

                    st.session_state.last_prediction = result["data"]
                    add_to_prediction_history(result["data"])

                except requests.exceptions.ConnectionError:
                    show_status(
                        "Backend is not running. Start FastAPI using: uvicorn app.main:app --reload",
                        "error",
                    )
                except Exception as error:
                    show_status(str(error), "error")

    else:
        render_webcam_prediction_preview()

    st.markdown('<div class="tm-output-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="tm-output-title">Output</div>', unsafe_allow_html=True)

    if st.session_state.last_prediction:
        prediction = st.session_state.last_prediction

        # Confidence status badge
        confidence = prediction["confidence"]
        dark_mode = st.session_state.get("dark_mode", True)
        if dark_mode:
            if confidence >= 85:
                status = "High Confidence"
                badge_color = "#0a2212"
                text_color = "#81c784"
                border_color = "#1b5e20"
            elif confidence >= 60:
                status = "Medium Confidence"
                badge_color = "#241909"
                text_color = "#ffb74d"
                border_color = "#e65100"
            else:
                status = "Low Confidence"
                badge_color = "#240a0c"
                text_color = "#e57373"
                border_color = "#b71c1c"
        else:
            if confidence >= 85:
                status = "High Confidence"
                badge_color = "#e6f4ea"
                text_color = "#137333"
                border_color = "#a8dab5"
            elif confidence >= 60:
                status = "Medium Confidence"
                badge_color = "#fef7e0"
                text_color = "#b06000"
                border_color = "#ffe0b2"
            else:
                status = "Low Confidence"
                badge_color = "#fce8e6"
                text_color = "#c5221f"
                border_color = "#f5c2c1"

        st.markdown(
            f"""
            <div class="tm-confidence-badge" style="
                background: {badge_color};
                border: 1px solid {border_color};
                color: {text_color};
            ">
                {status}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="tm-prediction-main">
                <div class="tm-prediction-label">Predicted Class</div>
                <div class="tm-prediction-class">{prediction["predicted_class"]}</div>
                <div class="tm-prediction-confidence">Confidence: {prediction["confidence"]}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Top predictions explainable AI
        st.markdown("**Top Predictions:**")
        top_predictions = prediction.get("top_predictions", [])
        if not top_predictions:
            probs = prediction.get("probabilities", {})
            sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            top_predictions = sorted_probs[:3]
            
        for cls, prob in top_predictions:
            st.markdown(f"• **{cls}** &nbsp;→&nbsp; **{prob:.1f}%**")

        st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
        render_probability_bars(prediction["probabilities"])

        # History display
        if st.session_state.history:
            st.markdown('<div style="height: 15px; border-top: 1px solid #dadce0; margin-top: 15px; padding-top: 10px;"></div>', unsafe_allow_html=True)
            st.markdown("**Recent Predictions History:**")
            for h in reversed(st.session_state.history):
                st.markdown(f"• **{h['class']}** → {h['confidence']}%")

        with st.expander("Model Info"):
            st.write(f"**Model Accuracy:** {prediction.get('model_accuracy', 0)}%")
            st.write(f"**Feature Extractor:** {prediction.get('feature_extractor', 'MobileNetV3')}")
            st.write(
                f"**Image Size:** {prediction.get('image_size', 224)} x {prediction.get('image_size', 224)}"
            )
    else:
        st.markdown(
            """
            <div class="tm-muted">
                Prediction output will appear here after you provide an input image.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)



def reset_frontend_project_state() -> None:
    st.session_state.session_id = str(uuid.uuid4())

    st.session_state.classes = [
        {"id": 1, "name": "Class 1", "uploaded_count": 0, "enabled": True, "ui_state": "idle"},
        {"id": 2, "name": "Class 2", "uploaded_count": 0, "enabled": True, "ui_state": "idle"},
    ]

    st.session_state.next_class_id = 3
    st.session_state.model_trained = False
    st.session_state.training_result = None
    st.session_state.last_prediction = None
    st.session_state.dataset_summary = {}
    st.session_state.sample_gallery = {}
    st.session_state.active_camera_class_id = None
    st.session_state.preview_input_enabled = True
    st.session_state.preview_mode = "File"


def main() -> None:
    initialize_session_state()
    cleanup_old_sessions_on_backend()

    try:
        st.session_state.dataset_summary = get_dataset_summary_from_backend()
        sync_classes_with_backend_summary()
    except Exception:
        pass

    inject_custom_css()
    render_topbar()

    left_column, connector_column, middle_column, connector_column_2, right_column = st.columns(
        [4.5, 0.3, 3.7, 0.3, 3.7]
    )

    with left_column:
        st.markdown("## Dataset Section")
        render_dataset_health_score()
        render_classes_area()

    with connector_column:
        st.markdown('<div class="tm-connector" style="margin-top: 250px;"></div>', unsafe_allow_html=True)

    with middle_column:
        st.markdown("## Training & Eval")
        render_training_panel()
        if st.session_state.training_result:
            render_evaluation_metrics_panel(st.session_state.training_result)
            render_ai_insights(st.session_state.training_result)

    with connector_column_2:
        st.markdown('<div class="tm-connector" style="margin-top: 250px;"></div>', unsafe_allow_html=True)

    with right_column:
        st.markdown("## Prediction Section")
        render_preview_panel()


if __name__ == "__main__":
    main()