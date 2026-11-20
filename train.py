import json
import os
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
from shutil import copyfile

import mlflow
from ultralytics import YOLO, settings


ROOT = Path(__file__).resolve().parent
METRIC_NAMES = ("precision", "recall", "ap50", "ap50_95")


def log_epoch(trainer) -> None:
    losses = trainer.label_loss_items(trainer.tloss, prefix="train")
    metrics = {**losses, **trainer.metrics, **trainer.lr}
    mlflow.log_metrics(
        {name.replace("(B)", ""): float(value) for name, value in metrics.items()},
        step=trainer.epoch + 1,
    )


def test_model(model, args, dataset: str) -> None:
    run_dir = model.trainer.save_dir
    test_dir = run_dir / "test"
    report = YOLO(str(model.trainer.best)).val(
        data=dataset,
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=0,
        project=str(run_dir),
        name="test",
        exist_ok=True,
        plots=True,
    )
    rows = {"all": report.box.mean_results()}
    for index, class_id in enumerate(report.box.ap_class_index):
        rows[report.names[class_id]] = report.box.class_result(index)

    scores = {
        name: dict(zip(METRIC_NAMES, map(float, values)))
        for name, values in rows.items()
    }
    mlflow.log_metrics({
        f"test/{name}/{metric}": value
        for name, values in scores.items()
        for metric, value in values.items()
    })
    (test_dir / "metrics.json").write_text(json.dumps(scores, indent=2) + "\n", encoding="utf-8")
    mlflow.log_artifacts(str(test_dir), artifact_path="test")


def main() -> None:
    parser = ArgumentParser(description="Обучение YOLO11s на FPDS")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu", help="cpu или номер видеокарты")
    parser.add_argument("--pretrained", default=str(ROOT / "runs" / "yolo11s.pt"))
    args = parser.parse_args()

    settings.update({"sync": False, "mlflow": False})
    dataset = str(ROOT / "data.yaml")
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment("FPDS-YOLO11s")
    with mlflow.start_run(
        run_name="Дообучение YOLO11s на FPDS v1",
        description="Готовая YOLO11s с весами COCO. Выбор весов по mAP50-95 на split3.",
    ) as run:
        mlflow.log_params({
            **vars(args), "model": "YOLO11s", "dataset": "FPDS v1",
            "freeze": 10, "seed": 42, "main_metric": "val mAP50-95",
        })
        mlflow.log_artifact(dataset, artifact_path="dataset")
        mlflow.log_artifact(str(ROOT / "data" / "dataset.json"), artifact_path="dataset")

        model = YOLO(args.pretrained)
        model.add_callback("on_model_save", log_epoch)
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
        run_dir = model.trainer.save_dir
        for filename in ("args.yaml", "results.csv", "results.png"):
            mlflow.log_artifact(str(run_dir / filename), artifact_path="training")
        test_model(model, args, dataset)

        best = model.trainer.best
        digest = sha256(best.read_bytes()).hexdigest()
        mlflow.log_artifact(str(best), artifact_path="weights")
        mlflow.set_tag("weights_sha256", digest)

    target = ROOT / "weights" / "best.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp.pt")
    copyfile(best, temporary)
    temporary.replace(target)
    metadata = {"experiment_id": run.info.experiment_id, "run_id": run.info.run_id, "sha256": digest}
    target.with_name("run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Веса для FastAPI: {target}. Запуск MLflow: {run.info.run_id}")


if __name__ == "__main__":
    main()
