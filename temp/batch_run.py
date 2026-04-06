from __future__ import annotations

import os
import gc
import json
import re
import math
from pathlib import Path
from typing import List, Iterable, Dict, Tuple, Optional
from collections import Counter, defaultdict

import pandas as pd
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams


# ===================== 配置区（只改这里） =====================
DATASET_DIRS = ["test5"]
# DATASET_DIRS = ["test4"]

TEST_ROOT = "test_data"
OUTPUT_ROOT = "eval_output"
MODEL_PATH = "Qwen3-8B"
PROMPT1_PATH = "skills/skills11.md"

# 是否把 .json 自动改名为 .jsonl
RENAME_JSON_TO_JSONL = True
RENAME_RECURSIVE = False

# ---------- 推理配置 ----------
# MAX_MODEL_LEN = 8192
MAX_MODEL_LEN = 24576
MAX_NEW_TOKENS = 16
BATCH_SIZE = 256
TEMPERATURE = 0.0

USE_ALLOWED_TOKEN_IDS = False
TRY_ENABLE_THINKING_FALSE = True

ENGINE_KW = dict(
    model=MODEL_PATH,
    tensor_parallel_size=1,
    distributed_executor_backend="mp",
    max_model_len=MAX_MODEL_LEN,
    max_num_seqs=64,
    max_num_batched_tokens=65536,
    enable_prefix_caching=True,
)

LABELS = list("ABCDEFGHIJ")
LABEL_REGEX = r"\b([A-J])\b"
WRITE_STATS_CSV = True
WRITE_DETAILS_CSV = True
WRITE_CONFUSION_CSV = True

# ============================================================


with open(PROMPT1_PATH, "r", encoding="utf-8-sig") as f:
    SYSTEM_PROMPT = f.read().strip()

_label_re = re.compile(LABEL_REGEX)


def _must_single_token(tokenizer, s: str) -> int:
    ids = tokenizer.encode(s, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"label string {s!r} is not single-token: ids={ids}")
    return ids[0]


def get_label_token_ids(tokenizer):
    label_to_id = {}
    for l in LABELS:
        try:
            tid = _must_single_token(tokenizer, l)
            label_to_id[l] = tid
        except Exception as e:
            print(f"[WARN] 无法获取 {l} 的单 token id: {e}")
    return label_to_id


def build_sampling_params_constrained(allowed_ids):
    try:
        sp = SamplingParams(
            temperature=TEMPERATURE,
            top_p=1.0,
            top_k=-1,
            max_tokens=1,
            logprobs=len(allowed_ids),
            stop=["<|im_end|>", "\n"],
            allowed_token_ids=allowed_ids,
        )
        print("[info] using allowed_token_ids constrained decoding (max_tokens=1)")
        return sp
    except TypeError:
        print("[warn] 当前 vLLM 版本不支持 allowed_token_ids，自动回退到普通生成")
        return build_sampling_params_free()


def build_sampling_params_free():
    return SamplingParams(
        temperature=TEMPERATURE,
        top_p=1.0,
        top_k=-1,
        max_tokens=MAX_NEW_TOKENS,
        logprobs=20,
        stop=["<|im_end|>"],
    )


def get_lp(info):
    if info is None:
        return None
    if hasattr(info, "logprob"):
        return info.logprob
    if isinstance(info, dict):
        return info.get("logprob")
    return None


def truncate_to_tokens(tokenizer, text: str, max_tokens: int) -> str:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) > max_tokens:
        ids = ids[:max_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)


def build_user_content(text: str) -> str:
    return f"待分类文本：\n{text}\n"


def build_prompt_with_chat_template(llm: LLM, text: str) -> str:
    tokenizer = llm.get_tokenizer()
    content_data = build_user_content(text)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content_data},
    ]

    try:
        if TRY_ENABLE_THINKING_FALSE:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def prepare_prompts(llm: LLM, raw_texts: Iterable[str]) -> List[str]:
    tokenizer = llm.get_tokenizer()
    safe_margin = 32
    max_input_tokens = max(1, MAX_MODEL_LEN - MAX_NEW_TOKENS - safe_margin)

    prompts = []
    for t in raw_texts:
        truncated_user = truncate_to_tokens(tokenizer, t, max_input_tokens // 2)
        prompt = build_prompt_with_chat_template(llm, truncated_user)

        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) > MAX_MODEL_LEN - MAX_NEW_TOKENS:
            ids = ids[: MAX_MODEL_LEN - MAX_NEW_TOKENS]
            prompt = tokenizer.decode(ids, skip_special_tokens=True)
        prompts.append(prompt)
    return prompts


def extract_label_from_response(resp: str) -> Optional[str]:
    if resp is None:
        return None
    s = str(resp).strip()
    if not s:
        return None

    # 优先精确匹配：只要单个字母
    if s in LABELS:
        return s

    # 常见格式兜底
    patterns = [
        r"^\s*答案[:：]?\s*([A-J])\s*$",
        r"^\s*输出[:：]?\s*([A-J])\s*$",
        r"^\s*分类[:：]?\s*([A-J])\s*$",
        r"^\s*label[:：]?\s*([A-J])\s*$",
        r"\b([A-J])\b",
    ]
    for p in patterns:
        m = re.search(p, s, flags=re.IGNORECASE)
        if m:
            lab = m.group(1).upper()
            if lab in LABELS:
                return lab
    return None


def infer_batch(
    llm: LLM,
    texts: List[str],
    sampling_params: SamplingParams,
    label_to_id: Dict[str, int],
) -> List[Tuple[str, Optional[str], Optional[float]]]:
    prompts = prepare_prompts(llm, texts)
    outputs = llm.generate(prompts, sampling_params)

    id_to_label = {v: k for k, v in label_to_id.items()}
    results = []

    for idx, out in enumerate(outputs):
        o = out.outputs[0]
        gen = (o.text or "").strip()
        final_label = None
        prob = None

        token_ids = getattr(o, "token_ids", [])
        lps = o.logprobs or []

        # 1) 如果启用了约束且第一 token 正好是标签，优先取 token_id
        if token_ids:
            first_tid = token_ids[0]
            if first_tid in id_to_label:
                final_label = id_to_label[first_tid]

        # 2) 未微调模型更常见：从文本中抽标签
        if final_label is None:
            final_label = extract_label_from_response(gen)

        # 3) 若能从 step0 logprobs 拿到该标签概率，则记录
        if final_label is not None and lps:
            step0 = lps[0] or {}
            target_tid = label_to_id.get(final_label)
            if target_tid is not None:
                info = step0.get(target_tid)
                logprob = get_lp(info)
                if logprob is not None:
                    try:
                        prob = math.exp(logprob)
                    except OverflowError:
                        prob = None

        results.append((gen, final_label, prob))

    return results


def rename_json_to_jsonl(dir_path: str, recursive: bool = False):
    base = Path(dir_path)
    if not base.exists():
        return
    it = base.rglob("*.json") if recursive else base.glob("*.json")
    for p in it:
        target = p.with_suffix(".jsonl")
        if target.exists():
            continue
        p.rename(target)


def parse_record(obj: dict) -> Optional[dict]:
    # 兼容多种输入格式
    # 1) 推荐格式：{"id":..., "text":..., "label":...}
    for key in ["text", "usercontent", "input", "query", "prompt"]:
        if key in obj and str(obj.get(key, "")).strip():
            return {
                "id": obj.get("id", ""),
                "usercontent": str(obj[key]).strip(),
                "label": str(obj.get("label", "")).strip(),
            }

    # 2) ShareGPT/SFT 格式：messages
    msgs = obj.get("messages", [])
    if isinstance(msgs, list) and msgs:
        user_text, label = "", ""
        for m in msgs:
            role = m.get("role")
            if role == "user" and not user_text:
                user_text = str(m.get("content", "")).strip()
            elif role == "assistant" and not label:
                label = str(m.get("content", "")).strip()
        if user_text:
            return {
                "id": obj.get("id", ""),
                "usercontent": user_text,
                "label": label,
            }

    return None


def load_jsonl_as_df(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            rec = parse_record(obj)
            if rec:
                rows.append(rec)
    return pd.DataFrame(rows)


def write_df_to_jsonl(df: pd.DataFrame, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8-sig") as f:
        for _, r in df.iterrows():
            obj = {
                "id": r.get("id", ""),
                "usercontent": r["usercontent"],
                "label": r.get("label", ""),
                "response": r.get("response", ""),
                "pred_label": r.get("pred_label", ""),
                "prob": r.get("prob", None),
                "done": int(r.get("done", 1)),
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def compute_metrics(df: pd.DataFrame) -> dict:
    # 允许没有 label，只做推理不做监督评测
    has_gold = "label" in df.columns and df["label"].astype(str).str.strip().ne("").any()

    total = len(df)
    parsed = int(df["pred_label"].notna().sum()) if "pred_label" in df.columns else 0
    parsed_rate = safe_div(parsed, total)

    out = {
        "total": total,
        "parsed": parsed,
        "parsed_rate": parsed_rate,
        "accuracy": None,
        "macro_f1": None,
    }

    if not has_gold:
        return out

    gold = df["label"].fillna("").astype(str).str.strip().str.upper()
    pred = df["pred_label"].fillna("__NONE__").astype(str).str.strip().str.upper()

    correct = int((gold == pred).sum())
    out["accuracy"] = safe_div(correct, total)

    per_label = {}
    f1s = []
    for lab in LABELS:
        tp = int(((gold == lab) & (pred == lab)).sum())
        fp = int(((gold != lab) & (pred == lab)).sum())
        fn = int(((gold == lab) & (pred != lab)).sum())
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        per_label[lab] = {
            "support": int((gold == lab).sum()),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        f1s.append(f1)
    out["macro_f1"] = sum(f1s) / len(f1s) if f1s else 0.0
    out["per_label"] = per_label
    return out


def build_confusion(df: pd.DataFrame) -> pd.DataFrame:
    gold = df["label"].fillna("").astype(str).str.strip().str.upper()
    pred = df["pred_label"].fillna("__NONE__").astype(str).str.strip().str.upper()

    labels = LABELS + ["__NONE__"]
    mat = pd.DataFrame(0, index=labels, columns=labels)
    for g, p in zip(gold, pred):
        if g not in mat.index:
            continue
        if p not in mat.columns:
            p = "__NONE__"
        mat.loc[g, p] += 1
    return mat


def write_stats_csv(metrics: dict, csv_path: str):
    out = Path(csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = [{
        "total": metrics.get("total", 0),
        "parsed": metrics.get("parsed", 0),
        "parsed_rate": metrics.get("parsed_rate", 0.0),
        "accuracy": metrics.get("accuracy", None),
        "macro_f1": metrics.get("macro_f1", None),
    }]
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")


def write_per_label_csv(metrics: dict, csv_path: str):
    per_label = metrics.get("per_label")
    if not per_label:
        return
    out = Path(csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for lab in LABELS:
        row = {"label": lab}
        row.update(per_label.get(lab, {}))
        rows.append(row)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")


def write_details_csv(all_detail_rows: List[Dict], csv_path: str):
    out = Path(csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_detail_rows)
    if not df.empty:
        df["usercontent"] = df["usercontent"].astype(str).str.slice(0, 500)
    df.to_csv(out, index=False, encoding="utf-8-sig")


def run_infer_on_dir(
    llm: LLM,
    input_dir: str,
    output_dir: str,
    sampling_params: SamplingParams,
    label_to_id: Dict[str, int],
) -> List[Dict]:
    os.makedirs(output_dir, exist_ok=True)
    jsonl_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".jsonl")])
    print(f"📂 {input_dir} -> 找到 {len(jsonl_files)} 个 .jsonl")

    all_detail_rows = []

    for i, filename in enumerate(tqdm(jsonl_files, desc=f"处理文件夹 {os.path.basename(input_dir)}"), 1):
        in_path = os.path.join(input_dir, filename)
        out_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".out.jsonl")

        src_df = load_jsonl_as_df(in_path)
        if src_df.empty:
            print(f"⚠️ 空/不可解析，跳过: {filename}")
            continue

        total = len(src_df)
        responses: List[str] = []
        pred_labels: List[Optional[str]] = []
        probs: List[Optional[float]] = []

        for start in tqdm(range(0, total, BATCH_SIZE), desc=f"推理 {filename}", leave=False):
            batch = src_df.iloc[start : start + BATCH_SIZE]
            batch_results = infer_batch(
                llm,
                batch["usercontent"].tolist(),
                sampling_params,
                label_to_id,
            )
            for gen_text, label, prob in batch_results:
                responses.append(gen_text)
                pred_labels.append(label)
                probs.append(prob)

        src_df["response"] = responses
        src_df["pred_label"] = pred_labels
        src_df["prob"] = probs
        src_df["done"] = 1
        write_df_to_jsonl(src_df, out_path)

        for _, r in src_df.iterrows():
            all_detail_rows.append({
                "file": filename,
                "id": r.get("id", ""),
                "usercontent": r["usercontent"],
                "label": r.get("label", ""),
                "pred_label": r.get("pred_label", ""),
                "prob": r.get("prob", None),
                "response": r.get("response", ""),
            })

    return all_detail_rows


def main():
    if RENAME_JSON_TO_JSONL:
        for d in DATASET_DIRS:
            in_dir = f"{TEST_ROOT}/{d}"
            rename_json_to_jsonl(in_dir, recursive=RENAME_RECURSIVE)

    print("🚀 正在加载 vLLM 模型...")
    llm = LLM(**ENGINE_KW)
    print("✅ 模型加载完成。")

    tokenizer = llm.get_tokenizer()
    label_to_id = get_label_token_ids(tokenizer)

    if USE_ALLOWED_TOKEN_IDS and label_to_id:
        allowed_ids = list(label_to_id.values())
        sampling_params = build_sampling_params_constrained(allowed_ids)
    else:
        print("[info] 使用普通生成模式")
        sampling_params = build_sampling_params_free()

    print(f"[info] label_to_id: {label_to_id}")

    try:
        for d in DATASET_DIRS:
            input_dir = f"{TEST_ROOT}/{d}"
            output_dir = f"{OUTPUT_ROOT}/{d}"

            if not os.path.isdir(input_dir):
                print(f"⚠️ 跳过：输入目录不存在: {input_dir}")
                continue

            print(f"\n==================== RUN {d} ====================")
            print(f"IN : {input_dir}")
            print(f"OUT: {output_dir}")

            detail_rows = run_infer_on_dir(
                llm, input_dir, output_dir,
                sampling_params, label_to_id,
            )

            details_df = pd.DataFrame(detail_rows)
            if details_df.empty:
                print(f"[warn] {d} 没有有效输出")
                continue

            metrics = compute_metrics(details_df)

            if WRITE_STATS_CSV:
                write_stats_csv(metrics, f"{output_dir}/stats.csv")
                write_per_label_csv(metrics, f"{output_dir}/per_label_metrics.csv")

            if WRITE_DETAILS_CSV:
                write_details_csv(detail_rows, f"{output_dir}/details.csv")

            if WRITE_CONFUSION_CSV and "label" in details_df.columns and details_df["label"].astype(str).str.strip().ne("").any():
                cm = build_confusion(details_df)
                cm.to_csv(f"{output_dir}/confusion_matrix.csv", encoding="utf-8-sig")

            print("[summary]", json.dumps({
                "dataset": d,
                "total": metrics.get("total"),
                "parsed": metrics.get("parsed"),
                "parsed_rate": metrics.get("parsed_rate"),
                "accuracy": metrics.get("accuracy"),
                "macro_f1": metrics.get("macro_f1"),
            }, ensure_ascii=False))

        print("\n🎉 runner 全部完成")

    finally:
        print("🧹 清理资源...")
        try:
            del llm
        except Exception:
            pass
        try:
            from vllm.distributed.parallel_state import (
                destroy_distributed_environment,
                destroy_model_parallel,
            )
            destroy_model_parallel()
            destroy_distributed_environment()
        except Exception as e:
            print(f"   - vllm清理非致命错误: {e}")

        gc.collect()
        if hasattr(torch, "npu"):
            torch.npu.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("✅ 资源清理完成")


if __name__ == "__main__":
    main()