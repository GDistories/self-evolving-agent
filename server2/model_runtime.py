from __future__ import annotations

import gc
import math
import re
from threading import Lock
from typing import Any

LABELS = tuple("ABCDEFGHIJ")
LABEL_PATTERN = re.compile(r"\b([A-J])\b", flags=re.IGNORECASE)


def extract_label_from_response(response_text: str | None) -> str | None:
    if response_text is None:
        return None
    text = str(response_text).strip()
    if not text:
        return None
    if text in LABELS:
        return text
    patterns = (
        r"^\s*答案[:：]?\s*([A-J])\s*$",
        r"^\s*输出[:：]?\s*([A-J])\s*$",
        r"^\s*分类[:：]?\s*([A-J])\s*$",
        r"^\s*label[:：]?\s*([A-J])\s*$",
        LABEL_PATTERN.pattern,
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None


def render_user_content(user_template: str, text: str) -> str:
    return user_template.format(text=text)


class VLLMBackend:
    def __init__(self, **config: Any) -> None:
        from vllm import LLM, SamplingParams

        self._sampling_params_cls = SamplingParams
        self._config = dict(config)
        model_path = self._config.pop("model_path")
        self._llm = LLM(model=model_path, **self._config)

    def _build_sampling_params(self, inference_config: dict[str, Any]) -> Any:
        temperature = float(inference_config.get("temperature", 0.0))
        top_p = float(inference_config.get("top_p", 1.0))
        top_k = int(inference_config.get("top_k", -1))
        max_tokens = int(inference_config.get("max_tokens", 16))
        return self._sampling_params_cls(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            logprobs=20,
            stop=["<|im_end|>"],
        )

    def _build_prompt(self, system_prompt: str, user_content: str) -> str:
        tokenizer = self._llm.get_tokenizer()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    def evaluate_dataset(
        self,
        *,
        system_prompt: str,
        user_template: str,
        dataset_records: list[dict[str, Any]],
        inference_config: dict[str, Any],
    ) -> list[dict[str, Any]]:
        prompts = [
            self._build_prompt(
                system_prompt,
                render_user_content(user_template, row["text"]),
            )
            for row in dataset_records
        ]
        outputs = self._llm.generate(prompts, self._build_sampling_params(inference_config))
        results: list[dict[str, Any]] = []
        for row, output in zip(dataset_records, outputs):
            generated_text = (output.outputs[0].text or "").strip()
            label = extract_label_from_response(generated_text)
            predicted_attack = label not in {None, "A"}
            probability = None
            logprobs = output.outputs[0].logprobs or []
            if logprobs and label is not None:
                tokenizer = self._llm.get_tokenizer()
                label_ids = tokenizer.encode(label, add_special_tokens=False)
                if len(label_ids) == 1:
                    info = (logprobs[0] or {}).get(label_ids[0])
                    logprob = getattr(info, "logprob", None) if info is not None else None
                    if logprob is not None:
                        try:
                            probability = math.exp(logprob)
                        except OverflowError:
                            probability = None
            results.append(
                {
                    "dataset": row["dataset"],
                    "sample_id": row["id"],
                    "predicted_attack": predicted_attack,
                    "raw_response": generated_text,
                    "predicted_label": label,
                    "probability": probability,
                }
            )
        return results

    def close(self) -> None:
        llm = getattr(self, "_llm", None)
        self._llm = None
        if llm is not None:
            del llm
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


class ModelRuntime:
    def __init__(
        self,
        *,
        mode: str = "lazy_reuse",
        backend_factory: type[Any] = VLLMBackend,
        backend_config: dict[str, Any] | None = None,
    ) -> None:
        if mode not in {"per_job", "lazy_reuse"}:
            raise ValueError(f"unsupported model lifecycle mode: {mode}")
        self.mode = mode
        self.backend_factory = backend_factory
        self.backend_config = dict(backend_config or {})
        self._shared_backend: Any | None = None
        self._lock = Lock()

    def _get_shared_backend(self) -> Any:
        with self._lock:
            if self._shared_backend is None:
                self._shared_backend = self.backend_factory(**self.backend_config)
            return self._shared_backend

    def evaluate(
        self,
        *,
        system_prompt: str,
        user_template: str,
        dataset_records: list[dict[str, Any]],
        inference_config: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if self.mode == "per_job":
            backend = self.backend_factory(**self.backend_config)
            try:
                return backend.evaluate_dataset(
                    system_prompt=system_prompt,
                    user_template=user_template,
                    dataset_records=dataset_records,
                    inference_config=inference_config,
                )
            finally:
                close = getattr(backend, "close", None)
                if callable(close):
                    close()

        backend = self._get_shared_backend()
        return backend.evaluate_dataset(
            system_prompt=system_prompt,
            user_template=user_template,
            dataset_records=dataset_records,
            inference_config=inference_config,
        )
