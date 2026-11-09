"""6종 벌 탐지 Flask 웹앱.

- models/best.pt가 있으면 시작 시 자동 로드
- 웹에서 다른 .pt 모델 업로드/교체 가능
- 단일 이미지 추론, test 데이터 mAP 평가
"""
import base64
import io
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, render_template, request
from PIL import Image
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (CLASS_NAMES, CONF_THRESHOLD, DATA_DIR, DEFAULT_MODEL,  # noqa: E402
                    IMG_SIZE, IOU_THRESHOLD, MODELS_DIR, get_device)
from evaluation.evaluate import evaluate as run_evaluation  # noqa: E402

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 256 * 1024 * 1024  # 모델 업로드 허용 크기

TEST_DIR = DATA_DIR / "test"
BOX_COLOR = (0, 200, 255)

state = {"model": None, "name": None}


def load_model(path: Path):
    state["model"] = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    state["model"] = YOLO(str(path))
    state["name"] = path.name


def error(message):
    return jsonify({"success": False, "error": message})


def gpu_info():
    cuda = torch.cuda.is_available()
    return {"cuda": cuda, "gpu": torch.cuda.get_device_name(0) if cuda else "None"}


def draw_boxes_to_base64(pil_img, detections):
    img = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = f"{det['class']} {det['confidence']:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), BOX_COLOR, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y_text = max(y1 - 8, th + 8)
        cv2.rectangle(img, (x1, y_text - th - 8), (x1 + tw + 8, y_text), BOX_COLOR, -1)
        cv2.putText(img, label, (x1 + 4, y_text - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    buffer = io.BytesIO()
    Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/model_status")
def model_status():
    return jsonify({"loaded": state["model"] is not None, "model_name": state["name"],
                    "classes": CLASS_NAMES, **gpu_info()})


@app.route("/upload_model", methods=["POST"])
def upload_model():
    file = request.files.get("model")
    if file is None or file.filename == "":
        return error("모델 파일이 선택되지 않았습니다.")
    if not file.filename.lower().endswith(".pt"):
        return error(".pt 모델 파일만 업로드할 수 있습니다.")

    save_path = MODELS_DIR / Path(file.filename).name
    file.save(save_path)
    try:
        load_model(save_path)
    except Exception as e:  # 잘못된 가중치 파일 등
        state.update(model=None, name=None)
        return error(f"모델 로드 실패: {e}")

    return jsonify({"success": True, "message": "모델 업로드 및 로드 완료",
                    "model_name": state["name"], **gpu_info()})


@app.route("/predict", methods=["POST"])
def predict():
    if state["model"] is None:
        return error("먼저 모델을 업로드하세요.")
    file = request.files.get("image")
    if file is None or file.filename == "":
        return error("선택된 이미지가 없습니다.")
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        return error("jpg, jpeg, png 이미지만 사용할 수 있습니다.")

    try:
        img = Image.open(file.stream).convert("RGB")
        result = state["model"].predict(source=img, imgsz=IMG_SIZE, conf=CONF_THRESHOLD,
                                        iou=IOU_THRESHOLD, device=get_device(), verbose=False)[0]
    except Exception as e:
        return error(f"예측 실패: {e}")

    detections = []
    for box in (result.boxes or []):
        cls_id = int(box.cls[0])
        detections.append({
            "class": CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}",
            "class_id": cls_id,
            "confidence": round(float(box.conf[0]), 4),
            "bbox": box.xyxy[0].cpu().numpy().astype(int).tolist(),
        })
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    top = detections[0] if detections else None

    return jsonify({
        "success": True,
        "model_name": state["name"],
        "num_detections": len(detections),
        "top_class": top["class"] if top else "탐지 없음",
        "top_confidence": top["confidence"] if top else 0,
        "detections": detections,
        "result_image": draw_boxes_to_base64(img, detections),
    })


@app.route("/evaluate", methods=["POST"])
def evaluate():
    if state["model"] is None:
        return error("먼저 모델을 업로드하세요.")
    if not (TEST_DIR / "test.json").exists():
        return error(f"평가 데이터가 없습니다: {TEST_DIR / 'test.json'}")
    try:
        metrics = run_evaluation(state["model"], TEST_DIR)
    except Exception as e:
        return error(f"mAP 계산 실패: {e}")
    return jsonify({"success": True, "model_name": state["name"], **metrics})


if DEFAULT_MODEL.exists():
    load_model(DEFAULT_MODEL)
