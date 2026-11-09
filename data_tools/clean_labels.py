"""클래스 폴더(<root>/<class>/)에서 라벨 문제가 있는 파일 정리.

- 라벨(.txt)이 없는 이미지 삭제
- 라벨이 비어 있는(탐지 없음) 이미지 + 라벨 삭제
- 짝이 되는 이미지가 없는 라벨 삭제

기본은 미리보기(dry-run)이며, --apply를 붙여야 실제로 삭제한다.

    python data_tools/clean_labels.py --root data/auto_labels --apply
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CLASS_NAMES, is_image  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--apply", action="store_true", help="실제로 삭제")
    args = p.parse_args()

    stats = {"no_label": 0, "empty_label": 0, "orphan_label": 0}

    def remove(path: Path):
        if args.apply:
            path.unlink()

    for cls_name in CLASS_NAMES:
        folder = args.root / cls_name
        if not folder.exists():
            continue
        images = {f.stem: f for f in folder.iterdir() if is_image(f)}

        for stem, img in images.items():
            txt = folder / f"{stem}.txt"
            if not txt.exists():
                remove(img)
                stats["no_label"] += 1
            elif not txt.read_text(encoding="utf-8").strip():
                remove(img)
                remove(txt)
                stats["empty_label"] += 1

        for txt in folder.glob("*.txt"):
            if txt.stem not in images:
                remove(txt)
                stats["orphan_label"] += 1

    mode = "삭제" if args.apply else "삭제 예정(dry-run)"
    print(f"라벨 없는 이미지 {stats['no_label']}개, 빈 라벨 {stats['empty_label']}개, "
          f"이미지 없는 라벨 {stats['orphan_label']}개 {mode}")


if __name__ == "__main__":
    main()
