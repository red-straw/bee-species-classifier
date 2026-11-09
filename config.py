"""프로젝트 공통 설정."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

CLASS_NAMES = ["apis", "black", "crabro", "ggoma", "jangsu", "simil"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

MODELS_DIR = ROOT / "models"
DEFAULT_MODEL = MODELS_DIR / "best.pt"
DATA_DIR = ROOT / "data"

# 추론 기본값 (최종 모델 학습 해상도 기준)
IMG_SIZE = 768
CONF_THRESHOLD = 0.4
IOU_THRESHOLD = 0.45


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def get_device():
    import torch
    return 0 if torch.cuda.is_available() else "cpu"
