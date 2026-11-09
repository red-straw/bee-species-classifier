"""폴더 단위 추론. 박스를 그린 이미지와(선택) YOLO 형식 예측 라벨을 저장.

    python evaluation/predict_folder.py --src data/samples --dst runs/predict
    python evaluation/predict_folder.py --src data/test --dst runs/pred_labels --save-labels
"""
import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CONF_THRESHOLD, DEFAULT_MODEL, IMG_SIZE, get_device, is_image  # noqa: E402
from data_tools.auto_label import to_yolo_lines  # noqa: E402

from ultralytics import YOLO  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--src", required=True, type=Path, help="이미지 폴더 (하위 폴더 포함)")
    p.add_argument("--dst", required=True, type=Path)
    p.add_argument("--imgsz", type=int, default=IMG_SIZE)
    p.add_argument("--conf", type=float, default=CONF_THRESHOLD)
    p.add_argument("--save-labels", action="store_true")
    args = p.parse_args()

    model = YOLO(args.model)
    device = get_device()
    images = [f for f in args.src.rglob("*") if is_image(f)]
    print(f"{len(images)}장 추론")

    for i, img in enumerate(images, 1):
        result = model.predict(str(img), imgsz=args.imgsz, conf=args.conf,
                               device=device, verbose=False)[0]
        out = args.dst / img.relative_to(args.src)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), result.plot())
        if args.save_labels:
            out.with_suffix(".txt").write_text("\n".join(to_yolo_lines(result)), encoding="utf-8")
        if i % 100 == 0:
            print(f"  {i}/{len(images)}")

    print(f"저장 위치: {args.dst}")


if __name__ == "__main__":
    main()
