from argparse import ArgumentParser
from pathlib import Path
from shutil import copyfile

from ultralytics import YOLO, settings


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = ArgumentParser(description="Обучение YOLO11s на FPDS")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu", help="cpu или номер видеокарты")
    parser.add_argument("--pretrained", default="yolo11s.pt")
    args = parser.parse_args()

    settings.update({"sync": False})
    dataset = str(ROOT / "data.yaml")
    model = YOLO(args.pretrained)
    model.train(
        data=dataset,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=0,
        freeze=10,
        seed=42,
        project=str(ROOT / "runs"),
        name="fpds",
    )

    best = model.trainer.best
    report = YOLO(str(best)).val(
        data=dataset,
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=0,
        project=str(model.trainer.save_dir),
        name="test",
        plots=True,
    )
    for class_id, average_precision in zip(report.box.ap_class_index, report.box.ap):
        print(f"{report.names[class_id]} AP50-95: {average_precision:.3f}")

    target = ROOT / "weights" / "best.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    copyfile(best, target)
    print(f"Веса для FastAPI: {target}")


if __name__ == "__main__":
    main()
