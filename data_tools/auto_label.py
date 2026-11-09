"""학습된 모델로 새 이미지에 YOLO 라벨을 자동 생성 (반자동 라벨링).

입력:  <src>/<class>/*.jpg
출력:  <dst>/<class>/*.jpg + 같은 이름의 .txt (YOLO 형식)
생성된 라벨은 사람이 검수한 뒤 clean_labels.py로 빈 라벨을 정리한다.

    python data_tools/auto_label.py --model models/best.pt --src data/raw --dst data/auto_labels
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CLASS_NAMES, get_device, is_image  # noqa: E402

from ultralytics import YOLO  # noqa: E402


def to_yolo_lines(result):
    h, w = result.orig_shape
    lines = []
    if result.boxes is None:
        return lines
    for box in result.boxes:
        cls_id = int(box.cls[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        bw, bh = (x2 - x1) / w, (y2 - y1) / h
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--src", required=True, type=Path)
    p.add_argument("--dst", required=True, type=Path)
    p.add_argument("--imgsz", type=int, default=832)
    p.add_argument("--conf", type=float, default=0.25)
    args = p.parse_args()

    model = YOLO(args.model)
    device = get_device()

    for cls_name in CLASS_NAMES:
        src_dir, dst_dir = args.src / cls_name, args.dst / cls_name
        if not src_dir.exists():
            print(f"[skip] 폴더 없음: {src_dir}")
            continue
        dst_dir.mkdir(parents=True, exist_ok=True)
        images = [f for f in src_dir.iterdir() if is_image(f)]
        print(f"[{cls_name}] {len(images)}장 자동 라벨링")

        for i, img in enumerate(images, 1):
            result = model.predict(str(img), imgsz=args.imgsz, conf=args.conf,
                                   device=device, verbose=False)[0]
            shutil.copy2(img, dst_dir / img.name)
            (dst_dir / f"{img.stem}.txt").write_text("\n".join(to_yolo_lines(result)), encoding="utf-8")
            if i % 100 == 0:
                print(f"  {i}/{len(images)}")


if __name__ == "__main__":
    main()
