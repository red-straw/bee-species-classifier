from flask import Flask, render_template, request, jsonify
from pathlib import Path
from ultralytics import YOLO
import torch
from PIL import Image
import io
import base64
import cv2
import numpy as np
import json
import shutil

app = Flask(__name__)

# 큰 모델 업로드 가능하게 설정
app.config["MAX_CONTENT_LENGTH"] = 256 * 1024 * 1024  # 256MB

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)
TEST_DIR = BASE_DIR / "test"
MAP_RESULTS_DIR = BASE_DIR / "map_results"
CONVERTED_TEST_DIR = BASE_DIR / "converted_test"

MAP_RESULTS_DIR.mkdir(exist_ok=True)
CONVERTED_TEST_DIR.mkdir(exist_ok=True)

CLASS_NAMES = ["apis", "black", "crabro", "ggoma", "jangsu", "simil"]

current_model = None
current_model_name = None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/model_status", methods=["GET"])
def model_status():
    return jsonify({
        "loaded": current_model is not None,
        "model_name": current_model_name,
        "classes": CLASS_NAMES,
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
    })


@app.route("/upload_model", methods=["POST"])
def upload_model():
    global current_model, current_model_name

    if "model" not in request.files:
        return jsonify({
            "success": False,
            "error": "모델 파일이 업로드되지 않았습니다."
        })

    file = request.files["model"]

    if file.filename == "":
        return jsonify({
            "success": False,
            "error": "선택된 파일이 없습니다."
        })

    if not file.filename.lower().endswith(".pt"):
        return jsonify({
            "success": False,
            "error": ".pt 모델 파일만 업로드할 수 있습니다."
        })

    try:
        model_filename = Path(file.filename).name
        save_path = MODELS_DIR / model_filename
        file.save(save_path)

        # 기존 모델 제거
        current_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 새 모델 로드
        current_model = YOLO(str(save_path))
        current_model_name = model_filename

        return jsonify({
            "success": True,
            "message": "모델 업로드 및 로드 완료",
            "model_name": current_model_name,
            "model_path": str(save_path),
            "cuda": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
        })

    except Exception as e:
        current_model = None
        current_model_name = None

        return jsonify({
            "success": False,
            "error": f"모델 로드 실패: {str(e)}"
        })

def draw_boxes_to_base64(pil_img, detections):
    img_cv = np.array(pil_img.convert("RGB"))
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cls_name = det["class"]
        conf = det["confidence"]

        color = (0, 200, 255)

        cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)

        label = f"{cls_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

        y_text = max(y1 - 8, th + 8)
        cv2.rectangle(img_cv, (x1, y_text - th - 8), (x1 + tw + 8, y_text), color, -1)
        cv2.putText(
            img_cv,
            label,
            (x1 + 4, y_text - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )

    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    result_img = Image.fromarray(img_rgb)

    buffer = io.BytesIO()
    result_img.save(buffer, format="PNG")
    buffer.seek(0)

    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@app.route("/predict", methods=["POST"])
def predict():
    if current_model is None:
        return jsonify({
            "success": False,
            "error": "먼저 모델을 업로드하세요."
        })

    if "image" not in request.files:
        return jsonify({
            "success": False,
            "error": "이미지 파일이 업로드되지 않았습니다."
        })

    file = request.files["image"]

    if file.filename == "":
        return jsonify({
            "success": False,
            "error": "선택된 이미지가 없습니다."
        })

    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        return jsonify({
            "success": False,
            "error": "jpg, jpeg, png 이미지만 사용할 수 있습니다."
        })

    try:
        img = Image.open(file.stream).convert("RGB")

        device = 0 if torch.cuda.is_available() else "cpu"

        results = current_model.predict(
            source=img,
            imgsz=768,
            conf=0.4,
            iou=0.45,
            device=device,
            verbose=False
        )

        result = results[0]
        detections = []

        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()

                cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"

                detections.append({
                    "class": cls_name,
                    "class_id": cls_id,
                    "confidence": round(conf, 4),
                    "bbox": [x1, y1, x2, y2]
                })

        detections.sort(key=lambda x: x["confidence"], reverse=True)

        result_image_base64 = draw_boxes_to_base64(img, detections)

        top = detections[0] if detections else None

        return jsonify({
            "success": True,
            "model_name": current_model_name,
            "num_detections": len(detections),
            "top_class": top["class"] if top else "탐지 없음",
            "top_confidence": top["confidence"] if top else 0,
            "detections": detections,
            "result_image": result_image_base64
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"예측 실패: {str(e)}"
        })

def convert_coco_json_to_yolo_dataset():
    """
    test/test.json + test 폴더 이미지들을
    converted_test/images/val, converted_test/labels/val 구조로 변환
    test.json 원본은 수정하지 않음
    """

    json_path = TEST_DIR / "test.json"

    if not json_path.exists():
        raise FileNotFoundError(f"test.json 파일이 없습니다: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])

    if not images:
        raise ValueError("test.json 안에 images 정보가 없습니다.")

    if not annotations:
        raise ValueError("test.json 안에 annotations 정보가 없습니다.")

    # 기존 변환 폴더 삭제 후 새로 생성
    if CONVERTED_TEST_DIR.exists():
        shutil.rmtree(CONVERTED_TEST_DIR)

    images_val_dir = CONVERTED_TEST_DIR / "images" / "val"
    labels_val_dir = CONVERTED_TEST_DIR / "labels" / "val"

    images_val_dir.mkdir(parents=True, exist_ok=True)
    labels_val_dir.mkdir(parents=True, exist_ok=True)

    # category_id → YOLO class_id 매핑
    category_id_to_class_id = {}

    if categories:
        name_to_class_id = {name: i for i, name in enumerate(CLASS_NAMES)}

        for cat in categories:
            cat_id = cat.get("id")
            cat_name = cat.get("name")

            if cat_name in name_to_class_id:
                category_id_to_class_id[cat_id] = name_to_class_id[cat_name]
            elif isinstance(cat_id, int):
                # category_id가 0~5이면 그대로 사용
                if 0 <= cat_id < len(CLASS_NAMES):
                    category_id_to_class_id[cat_id] = cat_id
                # category_id가 1~6이면 0~5로 보정
                elif 1 <= cat_id <= len(CLASS_NAMES):
                    category_id_to_class_id[cat_id] = cat_id - 1
    else:
        # categories가 없으면 0~5라고 가정
        category_id_to_class_id = {i: i for i in range(len(CLASS_NAMES))}

    # image_id → image 정보
    image_info_by_id = {}

    for img in images:
        image_id = img["id"]
        file_name = img["file_name"]

        image_info_by_id[image_id] = img

        src_img = TEST_DIR / file_name
        dst_img = images_val_dir / file_name

        if not src_img.exists():
            raise FileNotFoundError(f"이미지 파일이 없습니다: {src_img}")

        shutil.copy2(src_img, dst_img)

    # image_id별 annotation 모으기
    annotations_by_image_id = {}

    for ann in annotations:
        image_id = ann["image_id"]
        annotations_by_image_id.setdefault(image_id, []).append(ann)

    # YOLO txt 라벨 생성
    for image_id, img in image_info_by_id.items():
        file_name = img["file_name"]
        width = img["width"]
        height = img["height"]

        label_path = labels_val_dir / f"{Path(file_name).stem}.txt"
        lines = []

        for ann in annotations_by_image_id.get(image_id, []):
            category_id = ann["category_id"]

            if category_id not in category_id_to_class_id:
                raise ValueError(f"알 수 없는 category_id입니다: {category_id}")

            class_id = category_id_to_class_id[category_id]

            # COCO bbox: [x_min, y_min, width, height]
            x, y, w, h = ann["bbox"]

            x_center = (x + w / 2) / width
            y_center = (y + h / 2) / height
            box_w = w / width
            box_h = h / height

            # 0~1 범위 보정
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            box_w = max(0, min(1, box_w))
            box_h = max(0, min(1, box_h))

            lines.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}"
            )

        label_path.write_text("\n".join(lines), encoding="utf-8")

    # data.yaml 생성
    names_text = "\n".join([f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES)])

    data_yaml = CONVERTED_TEST_DIR / "data.yaml"

    yaml_content = f"""train: images/val
val: images/val

names:
{names_text}
"""

    data_yaml.write_text(yaml_content, encoding="utf-8")

    return data_yaml, len(images), len(annotations)


@app.route("/evaluate", methods=["POST"])
def evaluate():
    if current_model is None:
        return jsonify({
            "success": False,
            "error": "먼저 모델을 업로드하세요."
        })

    if not TEST_DIR.exists():
        return jsonify({
            "success": False,
            "error": f"test 폴더가 없습니다: {TEST_DIR}"
        })

    try:
        data_yaml, image_count, annotation_count = convert_coco_json_to_yolo_dataset()

        device = 0 if torch.cuda.is_available() else "cpu"

        results = current_model.val(
            data=str(data_yaml),
            split="val",
            imgsz=768,
            batch=8,
            conf=0.001,
            iou=0.60,
            max_det=300,
            device=device,
            project=str(MAP_RESULTS_DIR),
            name="latest",
            exist_ok=True,
            plots=True,
            verbose=True
        )

        map50 = float(results.box.map50)
        map5095 = float(results.box.map)
        precision = float(getattr(results.box, "mp", 0.0))
        recall = float(getattr(results.box, "mr", 0.0))

        return jsonify({
            "success": True,
            "model_name": current_model_name,
            "images": image_count,
            "annotations": annotation_count,
            "map50": round(map50, 4),
            "map5095": round(map5095, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "result_dir": str(MAP_RESULTS_DIR / "latest")
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"mAP 계산 실패: {str(e)}"
        })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5002, debug=True)
