import json
from hashlib import sha256
from pathlib import Path
from shutil import copyfile

from PIL import Image


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "FPDS_dataset" / "FPDS"
OUTPUT = ROOT / "data"
SPLITS = {
    "train": (1, 2),
    "val": (3,),
    "test": (4, 5, 6, 7, 8),
}
CLASS_IDS = {-1: 0, 1: 1}


def convert_label(annotation: Path, width: int, height: int) -> str:
    lines = []
    for line in annotation.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        label, left, right, top, bottom = map(int, line.split())
        if label not in CLASS_IDS:
            raise ValueError(f"Неизвестный класс в {annotation}: {label}")
        if not (0 <= left < right <= width and 0 <= top < bottom <= height):
            raise ValueError(f"Неверные координаты в {annotation}: {line}")

        x_center = (left + right) / (2 * width)
        y_center = (top + bottom) / (2 * height)
        box_width = (right - left) / width
        box_height = (bottom - top) / height
        lines.append(
            f"{CLASS_IDS[label]} {x_center:.6f} {y_center:.6f} "
            f"{box_width:.6f} {box_height:.6f}\n"
        )
    return "".join(lines)


def main() -> None:
    seen = {}
    summary = {}
    for split, source_groups in SPLITS.items():
        images_dir = OUTPUT / "images" / split
        labels_dir = OUTPUT / "labels" / split
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        counts = {"frames": 0, "person": 0, "fallen_person": 0, "empty_frames": 0}
        for group in source_groups:
            source_dir = SOURCE / f"split{group}"
            images = sorted(source_dir.glob("*.png"))
            if not images:
                raise FileNotFoundError(f"Нет изображений в {source_dir}")

            for image_path in images:
                annotation = image_path.with_suffix(".txt")
                with Image.open(image_path) as image:
                    label = convert_label(annotation, *image.size)
                    digest = sha256(image.convert("RGB").tobytes()).hexdigest()

                previous_split, previous_image = seen.setdefault(digest, (split, image_path.name))
                if previous_split != split:
                    raise ValueError(f"Одинаковые кадры в разных выборках: {previous_image}, {image_path.name}")

                copyfile(image_path, images_dir / image_path.name)
                (labels_dir / annotation.name).write_text(label, encoding="utf-8")
                counts["frames"] += 1
                counts["empty_frames"] += not label
                for row in label.splitlines():
                    class_name = "fallen_person" if row.startswith("1 ") else "person"
                    counts[class_name] += 1

        summary[split] = {"groups": list(source_groups), **counts}
        print(f"{split}: {counts}")

    (OUTPUT / "dataset.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
