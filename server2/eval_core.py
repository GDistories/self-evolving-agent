from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from server2.model_runtime import extract_label_from_response


def load_dataset_records(path: str | Path, *, dataset: str) -> list[dict]:
    dataset_name = _normalize_dataset(dataset)
    dataset_path = Path(path)
    rows: list[dict] = []
    with dataset_path.open("r", encoding="utf-8-sig") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            obj = json.loads(line)
            record_id = str(obj.get("id", "")).strip()
            text = str(obj.get("text", "")).strip()
            if not record_id or not text:
                raise ValueError(f"dataset row missing id/text: {dataset_path}")
            rows.append({"dataset": dataset_name, "id": record_id, "text": text})
    return rows


def build_dataset_fingerprint(path: str | Path) -> str:
    data = Path(path).read_bytes()
    return hashlib.sha256(data).hexdigest()


def parse_prediction_label(value: str | None) -> bool:
    label = extract_label_from_response(value)
    return label not in {None, "A"}


def _normalize_dataset(value) -> str:
    if value is None:
        raise ValueError("binary result row is missing 'dataset'")
    dataset = str(value).strip().lower()
    if dataset not in {"tp", "tn"}:
        raise ValueError(f"binary result row has invalid 'dataset': {value!r}")
    return dataset


def _normalize_predicted_attack(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "":
            raise ValueError("binary result row has invalid 'predicted_attack': ''")
        if normalized in {"true", "1", "yes", "y", "t"}:
            return True
        if normalized in {"false", "0", "no", "n", "f"}:
            return False
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    raise ValueError(f"binary result row has invalid 'predicted_attack': {value!r}")


def summarize_binary_results(rows: Iterable[dict]) -> dict:
    tp = fn = tn = fp = 0
    missed_tp: list[dict] = []
    false_alarm_tn: list[dict] = []
    raw_predictions: list[dict] = []

    for row in rows:
        if "dataset" not in row:
            raise ValueError("binary result row is missing 'dataset'")
        if "predicted_attack" not in row:
            raise ValueError("binary result row is missing 'predicted_attack'")

        dataset = _normalize_dataset(row.get("dataset"))
        predicted_attack = _normalize_predicted_attack(row.get("predicted_attack"))

        if dataset == "tp":
            if predicted_attack:
                tp += 1
            else:
                fn += 1
                missed_tp.append(
                    {
                        "sample_id": row.get("sample_id"),
                        "raw_response": row.get("raw_response"),
                    }
                )
        elif dataset == "tn":
            if predicted_attack:
                fp += 1
                false_alarm_tn.append(
                    {
                        "sample_id": row.get("sample_id"),
                        "raw_response": row.get("raw_response"),
                    }
                )
            else:
                tn += 1
        raw_predictions.append(dict(row))

    tp_total = tp + fn
    tn_total = tn + fp
    merged_total = tp_total + tn_total
    accuracy = (tp + tn) / merged_total if merged_total else 0.0

    return {
        "tp_stats": {"tp": tp, "fn": fn, "total": tp_total},
        "tn_stats": {"tn": tn, "fp": fp, "total": tn_total},
        "merged_metrics": {"accuracy": accuracy},
        "failure_samples": {
            "missed_tp": missed_tp,
            "false_alarm_tn": false_alarm_tn,
        },
        "raw_artifacts": {"predictions": raw_predictions},
    }
