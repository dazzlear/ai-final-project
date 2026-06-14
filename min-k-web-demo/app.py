from __future__ import annotations

import csv
import gc
import math
import os
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from flask import Flask, jsonify, request, send_from_directory
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = Path(os.getenv("OUTPUTS_DIR", str(BASE_DIR / "outputs"))).resolve()
FIGURES_DIR = Path(os.getenv("FIGURES_DIR", str(BASE_DIR / "figures"))).resolve()
DEFAULT_DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_THRESHOLD = float(os.getenv("DEFAULT_THRESHOLD", "-7.0"))
DEFAULT_MAX_LENGTH = int(os.getenv("MAX_LENGTH", "128"))
DEFAULT_K_PERCENT = float(os.getenv("K_PERCENT", "20"))

MODEL_ALIASES = {
    "tiny-gpt2": "sshleifer/tiny-gpt2",
    "sshleifer/tiny-gpt2": "sshleifer/tiny-gpt2",
    "pythia-70m": "EleutherAI/pythia-70m",
    "EleutherAI/pythia-70m": "EleutherAI/pythia-70m",
    "pythia-160m": "EleutherAI/pythia-160m",
    "EleutherAI/pythia-160m": "EleutherAI/pythia-160m",
    "pythia-410m": "EleutherAI/pythia-410m",
    "EleutherAI/pythia-410m": "EleutherAI/pythia-410m",
    "pythia-2.8b": "EleutherAI/pythia-2.8b",
    "EleutherAI/pythia-2.8b": "EleutherAI/pythia-2.8b",
    "gpt-neo-1.3b": "EleutherAI/gpt-neo-1.3B",
    "EleutherAI/gpt-neo-1.3B": "EleutherAI/gpt-neo-1.3B",
    "opt-1.3b": "facebook/opt-1.3b",
    "facebook/opt-1.3b": "facebook/opt-1.3b",
}

app = Flask(__name__)

@dataclass
class ModelBundle:
    model_name: str
    model: Any
    tokenizer: Any

_ACTIVE_BUNDLE: ModelBundle | None = None


def clean_number(value: float | int | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def average(values: list[float]) -> float:
    if not values:
        return float("nan")
    return sum(float(v) for v in values) / len(values)


def stable_log_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Manual stable log-softmax using log-sum-exp."""
    largest = logits.max(dim=dim, keepdim=True).values
    shifted = logits - largest
    denominator = torch.log(torch.exp(shifted).sum(dim=dim, keepdim=True))
    return shifted - denominator


def resolve_model_name(model_key: str) -> str:
    return MODEL_ALIASES.get(model_key, model_key)


def unload_active_model() -> None:
    global _ACTIVE_BUNDLE
    if _ACTIVE_BUNDLE is not None:
        del _ACTIVE_BUNDLE
        _ACTIVE_BUNDLE = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_model_bundle(model_key: str) -> ModelBundle:
    """Load one real model at a time. No fallback/fake scoring."""
    global _ACTIVE_BUNDLE
    model_name = resolve_model_name(model_key)

    if _ACTIVE_BUNDLE is not None and _ACTIVE_BUNDLE.model_name == model_name:
        return _ACTIVE_BUNDLE

    unload_active_model()

    use_cuda = DEFAULT_DEVICE == "cuda" and torch.cuda.is_available()
    print(f"[backend] Loading model: {model_name}")
    print(f"[backend] Device: {'cuda' if use_cuda else 'cpu'}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_cuda:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
        )
        model.to("cpu")

    model.eval()
    _ACTIVE_BUNDLE = ModelBundle(model_name=model_name, model=model, tokenizer=tokenizer)
    return _ACTIVE_BUNDLE


def token_logprobs(text: str, bundle: ModelBundle, max_length: int) -> tuple[list[float], list[dict[str, Any]]]:
    encoded = bundle.tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )

    device = next(bundle.model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    input_ids = encoded["input_ids"]

    if input_ids.shape[1] < 2:
        raise ValueError("Text must contain at least two tokens.")

    with torch.no_grad():
        outputs = bundle.model(**encoded)
        logits = outputs.logits

    # Causal LM alignment: position i predicts token i+1.
    prediction_logits = logits[:, :-1, :]
    expected_ids = input_ids[:, 1:]

    # This is intentionally shown manually for the project algorithm flow.
    normalized_logprobs = stable_log_softmax(prediction_logits.float(), dim=-1)

    picked_logprobs = (
        normalized_logprobs.gather(dim=2, index=expected_ids.unsqueeze(-1))
        .squeeze(-1)[0]
        .detach()
        .cpu()
        .tolist()
    )

    token_ids = input_ids[0].detach().cpu().tolist()
    token_rows = []

    for index, logprob in enumerate(picked_logprobs):
        token_id = token_ids[index + 1]
        token_text = bundle.tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        token_text = token_text.replace(" ", "·").replace("\n", "\\n").replace("\t", "\\t")
        token_rows.append({
            "index": index + 1,
            "token": token_text,
            "token_id": int(token_id),
            "logprob": clean_number(logprob),
            "selected": False,
            "rank": None,
        })

    return [float(value) for value in picked_logprobs], token_rows


def min_k_score(logprobs: list[float], k_percent: float) -> tuple[float, set[int], dict[int, int]]:
    """Manual MIN-K% implementation: sort, select lowest K%, average."""
    if not logprobs:
        return float("nan"), set(), {}

    selected_count = max(1, math.ceil(len(logprobs) * k_percent / 100.0))
    ranked_indices = sorted(range(len(logprobs)), key=lambda index: logprobs[index])
    selected_indices = ranked_indices[:selected_count]
    selected_set = set(selected_indices)
    rank_map = {index: rank + 1 for rank, index in enumerate(ranked_indices)}
    selected_values = [logprobs[index] for index in selected_indices]

    return average(selected_values), selected_set, rank_map


def loss_score(logprobs: list[float]) -> float:
    return average(logprobs)


def ppl_score(logprobs: list[float]) -> float:
    if not logprobs:
        return float("nan")
    return math.exp(-average(logprobs))


def zlib_score(text: str, logprobs: list[float]) -> float:
    if not logprobs:
        return float("nan")
    compressed_length = len(zlib.compress(text.encode("utf-8")))
    if compressed_length <= 0:
        return float("nan")
    return average(logprobs) / compressed_length


def analyze_text(text: str, model_key: str, k_percent: float, threshold: float, max_length: int) -> dict[str, Any]:
    started_at = time.time()
    bundle = get_model_bundle(model_key)
    logprobs, token_rows = token_logprobs(text, bundle=bundle, max_length=max_length)
    min_score, selected_indices, rank_map = min_k_score(logprobs, k_percent=k_percent)

    for zero_based_index, row in enumerate(token_rows):
        row["selected"] = zero_based_index in selected_indices
        row["rank"] = rank_map.get(zero_based_index)

    is_member = min_score > threshold

    return {
        "status": "ok",
        "model_name": bundle.model_name,
        "metrics": {
            "min_k_score": clean_number(min_score),
            "loss_score": clean_number(loss_score(logprobs)),
            "zlib_score": clean_number(zlib_score(text, logprobs)),
            "ppl": clean_number(ppl_score(logprobs)),
        },
        "prediction": {
            "label": "Likely member / seen" if is_member else "Likely non-member / unseen",
            "tone": "member" if is_member else "non-member",
        },
        "tokens": token_rows,
        "runtime": {
            "elapsed_seconds": round(time.time() - started_at, 3),
            "k_percent": k_percent,
            "threshold": threshold,
            "max_length": max_length,
        },
    }


@app.get("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.get("/css/<path:filename>")
def css_files(filename: str):
    return send_from_directory(BASE_DIR / "css", filename)


@app.get("/js/<path:filename>")
def js_files(filename: str):
    return send_from_directory(BASE_DIR / "js", filename)


@app.get("/api/health")
def health():
    return jsonify({
        "ok": True,
        "device": DEFAULT_DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "model_aliases": MODEL_ALIASES,
    })


@app.post("/api/analyze")
def analyze_api():
    payload = request.get_json(silent=True) or {}
    text = str(payload.get("text", "")).strip()

    if not text:
        return jsonify({"error": "Text is required."}), 400

    models = payload.get("models") or ["EleutherAI/pythia-70m"]
    if isinstance(models, str):
        models = [models]

    models = [str(model).strip() for model in models if str(model).strip()]
    if not models:
        return jsonify({"error": "At least one model is required."}), 400

    k_percent = float(payload.get("k_percent", DEFAULT_K_PERCENT))
    threshold = float(payload.get("threshold", DEFAULT_THRESHOLD))
    max_length = int(payload.get("max_length", DEFAULT_MAX_LENGTH))

    if not (0 < k_percent <= 100):
        return jsonify({"error": "k_percent must be in the range (0, 100]."}), 400
    if not (2 <= max_length <= 2048):
        return jsonify({"error": "max_length must be between 2 and 2048."}), 400

    results = []

    for model_key in models:
        try:
            results.append(analyze_text(
                text=text,
                model_key=model_key,
                k_percent=k_percent,
                threshold=threshold,
                max_length=max_length,
            ))
        except BaseException as exc:
            unload_active_model()
            message = str(exc)
            if "out of memory" in message.lower():
                message = "Out of memory. Select one smaller model or run this model in Colab/GPU. Details: " + message[:500]
            results.append({
                "status": "error",
                "model_name": resolve_model_name(str(model_key)),
                "error": message[:900],
            })

    return jsonify({
        "results": results,
        "request": {
            "n_models": len(models),
            "k_percent": k_percent,
            "threshold": threshold,
            "max_length": max_length,
        },
        "implementation_note": {
            "fallback": "Disabled. No mock or fake model scores are used.",
            "backend_source": "Integrated using web_app.zip as a guide, but keeping the min-k-web-demo frontend.",
        },
    })


# ============================================================
# Dashboard file APIs
# ============================================================

@app.get("/outputs/<path:filename>")
def serve_outputs(filename: str):
    return send_from_directory(OUTPUTS_DIR, filename)


@app.get("/figures/<path:filename>")
def serve_figures(filename: str):
    return send_from_directory(FIGURES_DIR, filename)


def read_csv_rows(csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        columns = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    return columns, rows


def to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        text = str(value).strip()
        if not text or text.upper() in {"N/A", "NA", "NAN"}:
            return None
        return float(text)
    except Exception:
        return None


@app.get("/api/dashboard/table1")
def dashboard_table1():
    csv_path = OUTPUTS_DIR / "table1_results.csv"

    if not csv_path.exists():
        return jsonify({
            "exists": False,
            "error": f"Missing file: {csv_path}",
            "expected": "outputs/table1_results.csv",
        }), 404

    columns, rows = read_csv_rows(csv_path)

    return jsonify({
        "exists": True,
        "filename": "table1_results.csv",
        "columns": columns,
        "rows": rows,
    })


@app.get("/api/dashboard/evaluation")
def dashboard_evaluation():
    csv_path = OUTPUTS_DIR / "evaluation_summary.csv"

    if not csv_path.exists():
        return jsonify({
            "exists": False,
            "error": f"Missing file: {csv_path}",
            "expected": "outputs/evaluation_summary.csv",
        }), 404

    columns, rows = read_csv_rows(csv_path)

    return jsonify({
        "exists": True,
        "filename": "evaluation_summary.csv",
        "columns": columns,
        "rows": rows,
    })


@app.get("/api/dashboard/summary")
def dashboard_summary():
    table_path = OUTPUTS_DIR / "table1_results.csv"
    eval_path = OUTPUTS_DIR / "evaluation_summary.csv"

    payload: dict[str, Any] = {
        "exists": table_path.exists() or eval_path.exists(),
        "cards": [],
        "table1_available": table_path.exists(),
        "evaluation_available": eval_path.exists(),
    }

    if table_path.exists():
        _, table_rows = read_csv_rows(table_path)

        best_method = None
        min_k_row = None

        for row in table_rows:
            method = row.get("method", "")
            avg_value = to_float(row.get("avg"))

            if method == "Min-K%":
                min_k_row = row

            if avg_value is not None:
                if best_method is None or avg_value > best_method["value"]:
                    best_method = {
                        "label": method,
                        "value": avg_value,
                    }

        if min_k_row:
            payload["cards"].append({
                "title": "Min-K% Avg AUC",
                "value": min_k_row.get("avg", "N/A"),
                "detail": "Average AUC across the compared models in table1_results.csv.",
            })

        if best_method:
            payload["cards"].append({
                "title": "Best Overall Method",
                "value": best_method["label"],
                "detail": f"Highest average AUC: {best_method['value']:.3f}",
            })

    if eval_path.exists():
        _, eval_rows = read_csv_rows(eval_path)

        payload["cards"].append({
            "title": "Evaluation Rows",
            "value": str(len(eval_rows)),
            "detail": "Rows in evaluation_summary.csv across methods, models, lengths, and settings.",
        })

        best_auc = None
        best_tpr = None

        for row in eval_rows:
            auc = to_float(row.get("auc"))
            tpr = to_float(row.get("tpr_at_5fpr"))

            if auc is not None:
                if best_auc is None or auc > best_auc["value"]:
                    best_auc = {
                        "value": auc,
                        "method": row.get("method", ""),
                        "model": row.get("model_key", ""),
                        "length": row.get("length", ""),
                        "setting": row.get("setting", ""),
                    }

            if tpr is not None:
                if best_tpr is None or tpr > best_tpr["value"]:
                    best_tpr = {
                        "value": tpr,
                        "method": row.get("method", ""),
                        "model": row.get("model_key", ""),
                        "length": row.get("length", ""),
                        "setting": row.get("setting", ""),
                    }

        if best_auc:
            payload["cards"].append({
                "title": "Best AUC Row",
                "value": f"{best_auc['value']:.3f}",
                "detail": f"{best_auc['method']} · {best_auc['model']} · len{best_auc['length']} · {best_auc['setting']}",
            })

        if best_tpr:
            payload["cards"].append({
                "title": "Best TPR@5%FPR Row",
                "value": f"{best_tpr['value']:.3f}",
                "detail": f"{best_tpr['method']} · {best_tpr['model']} · len{best_tpr['length']} · {best_tpr['setting']}",
            })

    return jsonify(payload)


@app.get("/api/dashboard/roc-image")
def dashboard_roc_image():
    figure_path = FIGURES_DIR / "roc_curve_min_k.png"

    if not figure_path.exists():
        return jsonify({
            "exists": False,
            "error": f"Missing file: {figure_path}",
            "expected": "figures/roc_curve_min_k.png",
        }), 404

    return jsonify({
        "exists": True,
        "filename": "roc_curve_min_k.png",
        "url": "/figures/roc_curve_min_k.png",
    })


@app.get("/api/dashboard/roc")
def dashboard_roc_alias():
    return dashboard_roc_image()



if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    app.run(host="127.0.0.1", port=port, debug=False)
