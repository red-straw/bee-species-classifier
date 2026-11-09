"""YOLO 학습 스크립트.

예시 (최종 모델 설정):
    python training/train.py --model yolo26m.pt --imgsz 768 --batch 5 --epochs 100 --name yolo26m_img768
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import ROOT, get_device  # noqa: E402

from ultralytics import YOLO  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Bee YOLO training")
    p.add_argument("--model", default="yolo26m.pt", help="사전학습 가중치 또는 이어서 학습할 .pt")
    p.add_argument("--data", default=str(ROOT / "training" / "data.yaml"))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=768)
    p.add_argument("--batch", type=float, default=5, help="정수 또는 0~1 (GPU 메모리 비율)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--cache", default="disk", choices=["disk", "ram", "false"])
    p.add_argument("--close-mosaic", type=int, default=10)
    p.add_argument("--name", default="exp")
    return p.parse_args()


def resolve_data_yaml(data_path: str) -> str:
    """data.yaml의 상대 path를 yaml 파일 위치 기준 절대경로로 바꿔 runs/에 저장."""
    import yaml

    data_file = Path(data_path).resolve()
    cfg = yaml.safe_load(data_file.read_text(encoding="utf-8"))
    if cfg.get("path") and not Path(cfg["path"]).is_absolute():
        cfg["path"] = (data_file.parent / cfg["path"]).resolve().as_posix()
        out = ROOT / "runs" / "data_resolved.yaml"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
        return str(out)
    return str(data_file)


def main():
    args = parse_args()
    batch = int(args.batch) if args.batch >= 1 else args.batch
    cache = False if args.cache == "false" else args.cache

    model = YOLO(args.model)
    model.train(
        data=resolve_data_yaml(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=batch,
        device=get_device(),
        workers=args.workers,
        cache=cache,
        close_mosaic=args.close_mosaic,
        project=str(ROOT / "runs"),
        name=args.name,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
