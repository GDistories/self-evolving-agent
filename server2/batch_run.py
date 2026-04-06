from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from server2.eval_core import (
    build_dataset_fingerprint,
    load_dataset_records,
    summarize_binary_results,
)
from server2.model_runtime import ModelRuntime


def run_batch_job(
    *,
    candidate_id: str,
    system_prompt: str,
    user_template: str,
    tp_dataset: str | Path,
    tn_dataset: str | Path,
    model_runtime: Any | None = None,
    inference_config: dict[str, Any] | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    tp_path = Path(tp_dataset)
    tn_path = Path(tn_dataset)
    runtime = model_runtime or ModelRuntime()
    predictions = runtime.evaluate(
        system_prompt=system_prompt,
        user_template=user_template,
        dataset_records=load_dataset_records(tp_path, dataset="tp")
        + load_dataset_records(tn_path, dataset="tn"),
        inference_config=dict(inference_config or {}),
    )
    result = summarize_binary_results(predictions)
    result["candidate_id"] = candidate_id
    result["dataset_fingerprint"] = {
        "tp": build_dataset_fingerprint(tp_path),
        "tn": build_dataset_fingerprint(tn_path),
    }
    result["cache_hit"] = False

    if output_path is not None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one local server2 evaluation job.")
    parser.add_argument("--candidate-id", default="manual-batch-run")
    parser.add_argument("--system-prompt-file", required=True)
    parser.add_argument("--user-template-file")
    parser.add_argument("--user-template")
    parser.add_argument("--tp-dataset", required=True)
    parser.add_argument("--tn-dataset", required=True)
    parser.add_argument("--output")
    parser.add_argument("--model-lifecycle-mode", choices=("lazy_reuse", "per_job"), default="lazy_reuse")
    parser.add_argument("--model-path", default="Qwen3-8B")
    return parser.parse_args()


def _load_text_arg(inline_value: str | None, file_value: str | None, *, field_name: str) -> str:
    if inline_value and file_value:
        raise ValueError(f"{field_name} cannot use both inline and file input")
    if file_value:
        return Path(file_value).read_text(encoding="utf-8").strip()
    if inline_value:
        return inline_value
    raise ValueError(f"{field_name} is required")


def main() -> None:
    args = parse_args()
    system_prompt = Path(args.system_prompt_file).read_text(encoding="utf-8").strip()
    user_template = _load_text_arg(
        args.user_template,
        args.user_template_file,
        field_name="user_template",
    )
    runtime = ModelRuntime(
        mode=args.model_lifecycle_mode,
        backend_config={"model_path": args.model_path},
    )
    result = run_batch_job(
        candidate_id=args.candidate_id,
        system_prompt=system_prompt,
        user_template=user_template,
        tp_dataset=args.tp_dataset,
        tn_dataset=args.tn_dataset,
        model_runtime=runtime,
        output_path=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
