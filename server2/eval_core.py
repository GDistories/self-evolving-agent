from __future__ import annotations

from typing import Iterable


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
        elif dataset == "tn":
            if predicted_attack:
                fp += 1
            else:
                tn += 1

    tp_total = tp + fn
    tn_total = tn + fp
    merged_total = tp_total + tn_total
    accuracy = (tp + tn) / merged_total if merged_total else 0.0

    return {
        "tp_stats": {"tp": tp, "fn": fn, "total": tp_total},
        "tn_stats": {"tn": tn, "fp": fp, "total": tn_total},
        "merged_metrics": {"accuracy": accuracy},
    }
