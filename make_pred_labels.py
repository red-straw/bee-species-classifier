from pathlib import Path
from ultralytics import YOLO
import shutil
import torch

# ===== 경로 설정 =====
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model_storage" / "best_yolo26m_8276_img768_map873.pt"
TEST_DIR = BASE_DIR / "test"

OUT_DIR = BASE_DIR / "test_pred_markmaster"
IMG_OUT_DIR = OUT_DIR / "images"
LBL_OUT_DIR = OUT_DIR / "labels"

CLASS_NAMES = ["apis", "black", "crabro", "ggoma", "jangsu", "simil"]

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

if not TEST_DIR.exists():
    raise FileNotFoundError(f"Test directory not found: {TEST_DIR}")

# ===== 폴더 초기화 =====
if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)

IMG_OUT_DIR.mkdir(parents=True, exist_ok=True)
LBL_OUT_DIR.mkdir(parents=True, exist_ok=True)

# ===== 모델 로드 =====
model = YOLO(str(MODEL_PATH))
device = 0 if torch.cuda.is_available() else "cpu"

# ===== 이미지 목록 =====
image_exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
image_paths = [
    p for p in TEST_DIR.iterdir()
    if p.suffix.lower() in image_exts
]

print(f"이미지 개수: {len(image_paths)}")

# ===== 이미지 복사 + 예측 txt 생성 =====
for idx, img_path in enumerate(image_paths, start=1):
    # 이미지 복사
    dst_img = IMG_OUT_DIR / img_path.name
    shutil.copy2(img_path, dst_img)

    # 예측
    results = model.predict(
        source=str(img_path),
        imgsz=768,
        conf=0.25,   # 눈으로 볼 때 기준. 너무 많이 나오면 0.35~0.4로 올리기
        iou=0.6,
        device=device,
        verbose=False
    )

    result = results[0]
    h, w = result.orig_shape

    txt_path = LBL_OUT_DIR / f"{img_path.stem}.txt"

    lines = []

    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            cls_id = int(box.cls[0].item())

            # xyxy 절대좌표
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # YOLO 정규화 좌표 변환
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h

            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    txt_path.write_text("\n".join(lines), encoding="utf-8")

    if idx % 50 == 0:
        print(f"{idx}/{len(image_paths)} 완료")

print("완료")
print(f"MarkMaster 확인 폴더: {OUT_DIR}")
