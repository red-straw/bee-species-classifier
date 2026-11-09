"""클래스 폴더 형태의 데이터를 하나로 모아 train/val(8:2)로 분할.

1차 데이터 1,701장과 자동 라벨 6,575장을 합쳐 8,276장 데이터셋을 만들 때 사용했다.

입력:  각 --src 아래 <class>/*.jpg + *.txt
출력:  <dst>/images/{train,val}, <dst>/labels/{train,val}

    python data_tools/build_dataset.py --src data/first data/auto_labels --dst data/dataset
"""
import argparse
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CLASS_NAMES, is_image  # noqa: E402


def collect(src_roots):
    items = []
    for root in src_roots:
        for cls_name in CLASS_NAMES:
            folder = root / cls_name
            if not folder.exists():
                continue
            for img in folder.iterdir():
                txt = folder / f"{img.stem}.txt"
                if is_image(img) and txt.exists():
                    items.append((img, txt))
    return items


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, nargs="+", type=Path)
    p.add_argument("--dst", required=True, type=Path)
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    items = collect(args.src)
    random.Random(args.seed).shuffle(items)
    n_train = int(len(items) * args.train_ratio)
    splits = {"train": items[:n_train], "val": items[n_train:]}

    for split, split_items in splits.items():
        img_dir = args.dst / "images" / split
        lbl_dir = args.dst / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for img, txt in split_items:
            name = img.stem
            if (img_dir / img.name).exists():  # 파일명 충돌 시 클래스명 접두어
                name = f"{img.parent.name}_{img.stem}"
            shutil.copy2(img, img_dir / f"{name}{img.suffix}")
            shutil.copy2(txt, lbl_dir / f"{name}.txt")

    print(f"총 {len(items)}장 -> train {len(splits['train'])} / val {len(splits['val'])}")


if __name__ == "__main__":
    main()
