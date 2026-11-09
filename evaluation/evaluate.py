"""COCO 형식 테스트셋(test.json + 이미지)으로 mAP 평가.

    python evaluation/evaluate.py --model models/best.pt --test-dir data/test
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CLASS_NAMES, DEFAULT_MODEL, IMG_SIZE, ROOT, get_device  # noqa: E402
from data_tools.coco_to_yolo import convert  # noqa: E402

from ultralytics import YOLO  # noqa: E402


def evaluate(model, test_dir: Path, imgsz=IMG_SIZE, out_root: Path = ROOT / "runs" / "eval"):
    data_yaml, n_images, n_anns = convert(test_dir / "test.json", test_dir,
                                          out_root / "converted_test", CLASS_NAMES)
    res = model.val(data=str(data_yaml), split="val", imgsz=imgsz, batch=8, conf=0.001,
                    iou=0.6, max_det=300, device=get_device(), project=str(out_root),
                    name="latest", exist_ok=True, plots=True, verbose=False)
    return {
        "images": n_images,
        "annotations": n_anns,
        "map50": round(float(res.box.map50), 4),
        "map5095": round(float(res.box.map), 4),
        "precision": round(float(res.box.mp), 4),
        "recall": round(float(res.box.mr), 4),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--test-dir", default=str(ROOT / "data" / "test"), type=Path)
    p.add_argument("--imgsz", type=int, default=IMG_SIZE)
    args = p.parse_args()

    metrics = evaluate(YOLO(args.model), args.test_dir, args.imgsz)
    for k, v in metrics.items():
        print(f"{k:12s}: {v}")


if __name__ == "__main__":
    main()
