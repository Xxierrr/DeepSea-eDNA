"""Model-quality monitoring and retraining workflow for DeepSea eDNA.

The workflow validates a labelled incoming batch, evaluates the champion,
retrains a separate candidate from the combined data, and applies a quality gate.
"""

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))


def parse_args():
    parser = argparse.ArgumentParser(description="Monitor and retrain the eDNA classifier")
    parser.add_argument("--incoming-csv", type=Path, default=None,
                        help="CSV with sequence,label and optional id columns")
    parser.add_argument("--training-data", type=Path,
                        default=ROOT / "data" / "training" / "marine_diverse")
    parser.add_argument("--champion-dir", type=Path,
                        default=ROOT / "ml_models" / "marine_model")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "mlops" / "retraining_demo")
    parser.add_argument("--incoming-per-class", type=int, default=10)
    parser.add_argument("--existing-per-class", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--min-improvement", type=float, default=0.0)
    parser.add_argument("--mlflow-tracking-uri", default="")
    parser.add_argument("--mlflow-experiment", default="marine-edna-monitoring")
    parser.add_argument("--mlflow-model-name", default="marine-edna-classifier")
    return parser.parse_args()


def load_csv_batch(path):
    sequences = []
    labels = []
    ids = []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"sequence", "label"}
        missing_columns = sorted(required - set(reader.fieldnames or []))
        if missing_columns:
            raise ValueError(f"Missing required CSV columns: {missing_columns}")
        for row_number, row in enumerate(reader, start=2):
            sequence = (row.get("sequence") or "").strip().upper()
            label_text = (row.get("label") or "").strip()
            if not sequence:
                raise ValueError(f"Missing sequence at CSV row {row_number}")
            if not label_text:
                raise ValueError(f"Missing label at CSV row {row_number}")
            try:
                label = int(label_text)
            except ValueError as exc:
                raise ValueError(f"Label must be an integer at CSV row {row_number}") from exc
            sequences.append(sequence)
            labels.append(label)
            ids.append((row.get("id") or "").strip() or f"row-{row_number}")
    return sequences, labels, ids


def generate_incoming_batch(training_data, incoming_per_class):
    from app.training_compat import load_fasta_dataset

    sequences, labels, taxonomy_map = load_fasta_dataset(str(training_data))
    grouped = {}
    remaining = {"sequences": [], "labels": []}
    for sequence, label in zip(sequences, labels):
        grouped.setdefault(label, []).append(sequence)

    incoming_sequences = []
    incoming_labels = []
    incoming_ids = []
    incoming_set = set()
    for label in sorted(grouped):
        selected = grouped[label][:incoming_per_class]
        for index, sequence in enumerate(selected):
            incoming_sequences.append(sequence)
            incoming_labels.append(label)
            incoming_ids.append(f"incoming-{label}-{index + 1}")
            incoming_set.add(sequence)
        remaining["sequences"].extend(grouped[label][incoming_per_class:])
        remaining["labels"].extend([label] * (len(grouped[label]) - incoming_per_class))

    return incoming_sequences, incoming_labels, incoming_ids, taxonomy_map, remaining, incoming_set


def validate_batch(sequences, labels, ids):
    if not sequences:
        raise ValueError("Incoming batch is empty")
    missing_values = sum(not sequence or label is None for sequence, label in zip(sequences, labels))
    duplicate_sequences = len(sequences) - len(set(sequences))
    duplicate_ids = len(ids) - len(set(ids))
    invalid_bases = sum(len(set(sequence) - set("ACGTN")) for sequence in sequences)
    if missing_values:
        raise ValueError(f"Incoming batch contains {missing_values} missing values")
    if duplicate_ids:
        raise ValueError(f"Incoming batch contains {duplicate_ids} duplicate IDs")
    if invalid_bases:
        raise ValueError("Incoming batch contains invalid DNA characters")
    return {
        "size": len(sequences),
        "missing_values": missing_values,
        "duplicate_sequences": duplicate_sequences,
        "duplicate_ids": duplicate_ids,
        "target_distribution": dict(sorted(Counter(labels).items(), key=lambda item: str(item[0])))
    }


def write_fasta_dataset(path, sequences, labels, taxonomy_map):
    path.mkdir(parents=True, exist_ok=True)
    for label in sorted(set(labels)):
        class_dir = path / f"class_{label}"
        class_dir.mkdir(parents=True, exist_ok=True)
        with (class_dir / "sequences.fasta").open("w", encoding="utf-8") as handle:
            for index, (sequence, sequence_label) in enumerate(zip(sequences, labels)):
                if sequence_label == label:
                    handle.write(f">sequence_{index}\n{sequence}\n")
    (path / "taxonomy.json").write_text(json.dumps(taxonomy_map, indent=2), encoding="utf-8")


def evaluate(loader, sequences, labels):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    predictions = loader.predict(sequences)
    predicted_labels = [prediction["predicted_class"] for prediction in predictions]
    return {
        "accuracy": float(accuracy_score(labels, predicted_labels)),
        "precision_weighted": float(precision_score(labels, predicted_labels, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(labels, predicted_labels, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(labels, predicted_labels, average="weighted", zero_division=0)),
    }, predicted_labels


def load_local_champion(champion_dir):
    from app.services.model_loader import ModelLoader

    loader = ModelLoader()
    loader.load_from_files(
        config_file=champion_dir / "model_config.json",
        taxonomy_file=champion_dir / "taxonomy_map.json",
        weights_file=champion_dir / "best_model.pth",
    )
    return loader


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.incoming_csv:
        sequences, labels, ids = load_csv_batch(args.incoming_csv)
        taxonomy_map = json.loads((args.training_data / "taxonomy.json").read_text())
        if len(set(labels)) == 0:
            raise ValueError("No target labels found")
        remaining = None
    else:
        sequences, labels, ids, taxonomy_map, remaining, _ = generate_incoming_batch(
            args.training_data, args.incoming_per_class
        )

    validation = validate_batch(sequences, labels, ids)

    champion = load_local_champion(args.champion_dir)
    champion_metrics, champion_predictions = evaluate(champion, sequences, labels)

    combined_dir = args.output_dir / "combined_training_data"
    if combined_dir.exists():
        shutil.rmtree(combined_dir)

    if remaining is None:
        from app.training_compat import load_fasta_dataset

        existing_sequences, existing_labels, _ = load_fasta_dataset(str(args.training_data))
        grouped = {}
        for sequence, label in zip(existing_sequences, existing_labels):
            grouped.setdefault(label, []).append(sequence)
        existing_sequences = []
        existing_labels = []
        for label in sorted(grouped):
            selected = grouped[label][:args.existing_per_class]
            existing_sequences.extend(selected)
            existing_labels.extend([label] * len(selected))
        remaining = {"sequences": existing_sequences, "labels": existing_labels}
    else:
        grouped = {}
        for sequence, label in zip(remaining["sequences"], remaining["labels"]):
            grouped.setdefault(label, []).append(sequence)
        existing_sequences = []
        existing_labels = []
        for label in sorted(grouped):
            selected = grouped[label][:args.existing_per_class]
            existing_sequences.extend(selected)
            existing_labels.extend([label] * len(selected))
        remaining = {"sequences": existing_sequences, "labels": existing_labels}

    combined_sequences = remaining["sequences"] + sequences
    combined_labels = remaining["labels"] + labels
    write_fasta_dataset(combined_dir, combined_sequences, combined_labels, taxonomy_map)

    candidate_dir = args.output_dir / "candidate_model"
    if candidate_dir.exists():
        shutil.rmtree(candidate_dir)

    from app.training_compat import train_classifier_compat

    training_results = train_classifier_compat(
        data_dir=str(combined_dir),
        output_dir=str(candidate_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=0.001,
        device="cuda" if __import__("torch").cuda.is_available() else "cpu",
        validation_split=0.2,
        use_augmentation=True,
        use_class_weights=True,
        early_stopping_patience=args.epochs,
        max_per_class=args.existing_per_class,
    )

    candidate = load_local_champion(candidate_dir)
    candidate_metrics, candidate_predictions = evaluate(candidate, sequences, labels)

    gate_metric = "f1_weighted"
    eligible = candidate_metrics[gate_metric] >= champion_metrics[gate_metric] + args.min_improvement
    decision = {
        "metric": gate_metric,
        "reason": "Weighted F1 is selected because this is an imbalanced multiclass eDNA problem.",
        "minimum_improvement": args.min_improvement,
        "champion_score": champion_metrics[gate_metric],
        "candidate_score": candidate_metrics[gate_metric],
        "eligible_for_promotion": eligible,
        "champion_overwritten": False,
        "decision": "PROMOTE_CANDIDATE" if eligible else "RETAIN_CHAMPION",
    }

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task_a_data_validation": validation,
        "task_b_champion_evaluation": {
            "model_directory": str(args.champion_dir),
            "metrics": champion_metrics,
        },
        "task_c_candidate_training": {
            "combined_training_size": len(combined_sequences),
            "candidate_directory": str(candidate_dir),
            "training_results": training_results,
        },
        "task_d_quality_gate": decision,
        "predictions": {
            "champion": champion_predictions,
            "candidate": candidate_predictions,
        },
    }

    report_path = args.output_dir / "evaluation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    with (args.output_dir / "incoming_batch_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "sequence", "label"])
        writer.writerows(zip(ids, sequences, labels))

    mlflow_info = None
    if args.mlflow_tracking_uri:
        import mlflow
        from app.ml import registry

        registry.configure(args.mlflow_tracking_uri)
        mlflow.set_experiment(args.mlflow_experiment)
        with mlflow.start_run(run_name="incoming-data-monitoring") as run:
            mlflow.log_params({
                "incoming_size": validation["size"],
                "combined_training_size": len(combined_sequences),
                "epochs": args.epochs,
                "gate_metric": gate_metric,
                "min_improvement": args.min_improvement,
            })
            mlflow.log_metrics({
                f"champion_{gate_metric}": champion_metrics[gate_metric],
                f"candidate_{gate_metric}": candidate_metrics[gate_metric],
                "champion_accuracy": champion_metrics["accuracy"],
                "candidate_accuracy": candidate_metrics["accuracy"],
                "val_accuracy": training_results["val_accuracy"],
            })
            mlflow.log_artifact(str(report_path), artifact_path="reports")
            version = registry.log_model_version(
                candidate_dir, args.mlflow_model_name, alias=registry.CHALLENGER
            )
            mlflow_info = {
                "run_id": run.info.run_id,
                "candidate_model_version": version.version,
            }
            if eligible:
                promoted, gate_message = registry.promote_if_better(
                    args.mlflow_model_name,
                    version.version,
                    metric=gate_metric,
                    min_improvement=args.min_improvement,
                )
                decision["mlflow_promotion"] = {"promoted": promoted, "message": gate_message}

    report["mlflow"] = mlflow_info
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps({
        "incoming_batch": validation,
        "champion_metrics": champion_metrics,
        "candidate_metrics": candidate_metrics,
        "quality_gate": decision,
        "report": str(report_path),
        "candidate_model": str(candidate_dir),
        "mlflow": mlflow_info,
    }, indent=2))
    return 0 if not eligible or candidate_dir.exists() else 1


if __name__ == "__main__":
    raise SystemExit(main())
