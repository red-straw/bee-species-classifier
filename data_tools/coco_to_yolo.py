"""COCO 형식 평가 데이터(test.json + 이미지)를 YOLO 검증용 구조로 변환."""
import json
import shutil
from pathlib import Path


def convert(coco_json: Path, image_dir: Path, out_dir: Path, class_names):
    coco = json.loads(coco_json.read_text(encoding="utf-8"))
    images, annotations = coco.get("images", []), coco.get("annotations", [])
    if not images or not annotations:
        raise ValueError("test.json에 images 또는 annotations가 없습니다.")

    # category_id -> YOLO class_id (이름 우선, 없으면 0~5 / 1~6 숫자 규칙)
    name_to_id = {n: i for i, n in enumerate(class_names)}
    cat_map = {}
    for cat in coco.get("categories", []):
        cid, cname = cat.get("id"), cat.get("name")
        if cname in name_to_id:
            cat_map[cid] = name_to_id[cname]
        elif isinstance(cid, int) and 0 <= cid < len(class_names):
            cat_map[cid] = cid
        elif isinstance(cid, int) and 1 <= cid <= len(class_names):
            cat_map[cid] = cid - 1
    if not cat_map:
        cat_map = {i: i for i in range(len(class_names))}

    if out_dir.exists():
        shutil.rmtree(out_dir)
    img_out = out_dir / "images" / "val"
    lbl_out = out_dir / "labels" / "val"
    img_out.mkdir(parents=True)
    lbl_out.mkdir(parents=True)

    anns_by_image = {}
    for ann in annotations:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    for img in images:
        src = image_dir / img["file_name"]
        if not src.exists():
            raise FileNotFoundError(f"이미지 없음: {src}")
        shutil.copy2(src, img_out / img["file_name"])

        w, h = img["width"], img["height"]
        lines = []
        for ann in anns_by_image.get(img["id"], []):
            if ann["category_id"] not in cat_map:
                raise ValueError(f"알 수 없는 category_id: {ann['category_id']}")
            x, y, bw, bh = ann["bbox"]
            vals = [(x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h]
            vals = [min(1, max(0, v)) for v in vals]
            lines.append(f"{cat_map[ann['category_id']]} " + " ".join(f"{v:.6f}" for v in vals))
        (lbl_out / f"{Path(img['file_name']).stem}.txt").write_text("\n".join(lines), encoding="utf-8")

    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(class_names))
    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(f"path: {out_dir.resolve().as_posix()}\ntrain: images/val\nval: images/val\n\nnames:\n{names}\n",
                         encoding="utf-8")
    return data_yaml, len(images), len(annotations)
