# SPDX-License-Identifier: MPL-2.0

"""Closed-vocabulary proposer baseline for micro-search hole states."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F


DEFAULT_TOPK = (1, 3, 5)


@dataclass(frozen=True)
class MicroSearchProposerConfig:
    """Feature and model configuration for the closed-vocabulary proposer."""

    n_probe_points: int = 16
    max_input_vars: int = 4
    max_basis_dims: int = 4
    hidden_dim: int = 128


class _MicroSearchClosedVocabNet(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        if value.ndim == 0:
            return float(value.item())
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return float(value)
        return None
    return str(value)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _coerce_rows(payload: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if isinstance(payload, Mapping):
        rows = [dict(row) for row in list(payload.get("rows", []) or []) if isinstance(row, Mapping)]
        meta = dict(payload)
        meta["rows"] = None
        return rows, meta
    return [dict(row) for row in list(payload or []) if isinstance(row, Mapping)], {}


def _json_ast_root_op(node: Any) -> str:
    if isinstance(node, (list, tuple)) and len(node) >= 1:
        return str(node[0])
    return ""


def _json_ast_depth(node: Any) -> int:
    if not isinstance(node, (list, tuple)) or len(node) == 0:
        return 0
    op = str(node[0])
    if op in ("var", "const", "hparam"):
        return 1
    if op in ("sin", "cos", "exp", "log", "sqrt", "sqr", "neg"):
        return 1 + _json_ast_depth(node[1] if len(node) > 1 else None)
    if op in ("add", "sub", "mul", "div"):
        left = _json_ast_depth(node[1] if len(node) > 1 else None)
        right = _json_ast_depth(node[2] if len(node) > 2 else None)
        return 1 + max(left, right)
    return 1


def _json_ast_size(node: Any) -> int:
    if not isinstance(node, (list, tuple)) or len(node) == 0:
        return 0
    op = str(node[0])
    if op in ("var", "const", "hparam"):
        return 1
    if op in ("sin", "cos", "exp", "log", "sqrt", "sqr", "neg"):
        return 1 + _json_ast_size(node[1] if len(node) > 1 else None)
    if op in ("add", "sub", "mul", "div"):
        return 1 + _json_ast_size(node[1] if len(node) > 1 else None) + _json_ast_size(node[2] if len(node) > 2 else None)
    return 1


def _flatten_matrix(raw: Any, *, rows: int, cols: int) -> list[float]:
    out = [0.0] * (rows * cols)
    if not isinstance(raw, list):
        return out
    for i in range(min(rows, len(raw))):
        row = raw[i]
        if not isinstance(row, list):
            continue
        for j in range(min(cols, len(row))):
            out[i * cols + j] = _safe_float(row[j], 0.0)
    return out


def _flatten_vector(raw: Any, *, rows: int) -> list[float]:
    out = [0.0] * rows
    if not isinstance(raw, list):
        return out
    for i in range(min(rows, len(raw))):
        item = raw[i]
        if isinstance(item, list):
            out[i] = _safe_float(item[0] if item else 0.0, 0.0)
        else:
            out[i] = _safe_float(item, 0.0)
    return out


def _pad_vector(raw: Any, *, size: int) -> list[float]:
    out = [0.0] * size
    if not isinstance(raw, list):
        return out
    for i in range(min(size, len(raw))):
        out[i] = _safe_float(raw[i], 0.0)
    return out


def _one_hot(items: Sequence[str], value: str) -> list[float]:
    val = str(value or "")
    return [1.0 if token == val else 0.0 for token in items]


def build_micro_search_vocabulary(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Build the closed vocabulary from exact completion tables."""

    vocab: set[str] = set()
    for row in rows:
        truth_expr = str(row.get("hole_truth_expr", "") or "")
        if truth_expr:
            vocab.add(truth_expr)
        rankings = dict(row.get("rankings", {}) or {})
        for key in ("inverse", "residual"):
            for cand in list(rankings.get(key, []) or []):
                if not isinstance(cand, Mapping):
                    continue
                expr = str(cand.get("expr", "") or "")
                if expr:
                    vocab.add(expr)
    return sorted(vocab)


def valid_completion_exprs(row: Mapping[str, Any]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    rankings = dict(row.get("rankings", {}) or {})
    for key in ("inverse", "residual"):
        for cand in list(rankings.get(key, []) or []):
            if not isinstance(cand, Mapping):
                continue
            expr = str(cand.get("expr", "") or "")
            if expr == "" or expr in seen:
                continue
            seen.add(expr)
            out.append(expr)
    return out


def featurize_micro_search_state(
    row: Mapping[str, Any],
    *,
    config: MicroSearchProposerConfig | None = None,
) -> tuple[torch.Tensor, list[str]]:
    """Turn a micro-search row into a fixed-size numeric feature vector."""

    cfg = MicroSearchProposerConfig() if config is None else config
    samples = dict(row.get("samples", {}) or {})
    inverse_target = dict(row.get("inverse_target", {}) or {})
    residual_target = dict(row.get("residual_target", {}) or {})
    grammar = dict(row.get("grammar", {}) or {})
    units = dict(row.get("units", {}) or {})
    score = dict(row.get("candidate_score", {}) or {})

    current_ast = row.get("hole_current_expr_ast", None)
    parent_ast = row.get("candidate_expr_ast", None)
    hole_path = list(row.get("hole_path", []) or [])
    current_root = _json_ast_root_op(current_ast)
    parent_root = _json_ast_root_op(parent_ast)
    op_vocab = ("", "var", "const", "add", "sub", "mul", "div", "sin", "cos", "exp", "log", "sqrt", "sqr", "neg")

    features: list[float] = []
    features.extend([
        float(_safe_int(len(hole_path), 0)),
        float(_json_ast_depth(current_ast)),
        float(_json_ast_size(current_ast)),
        float(_json_ast_depth(parent_ast)),
        float(_json_ast_size(parent_ast)),
        float(_safe_float(score.get("probe_mse", 0.0), 0.0)),
        float(_safe_float(score.get("fit_mse", 0.0), 0.0)),
        float(_safe_float(inverse_target.get("confidence", 0.0), 0.0)),
        float(_safe_float(inverse_target.get("valid_fraction_fit", 0.0), 0.0)),
        float(_safe_float(inverse_target.get("valid_fraction_probe", 0.0), 0.0)),
        float(_safe_float(residual_target.get("valid_fraction_fit", 0.0), 0.0)),
        float(_safe_float(residual_target.get("valid_fraction_probe", 0.0), 0.0)),
        float(_safe_int(grammar.get("n_candidates", 0), 0)),
        1.0 if bool(grammar.get("truth_in_grammar", False)) else 0.0,
        1.0 if bool(grammar.get("current_in_grammar", False)) else 0.0,
    ])
    features.extend(_one_hot(op_vocab, current_root))
    features.extend(_one_hot(op_vocab, parent_root))
    features.extend(_pad_vector(units.get("hole_dim", []), size=int(cfg.max_basis_dims)))

    var_dims = list(units.get("var_dims", []) or [])
    for i in range(int(cfg.max_input_vars)):
        dim = var_dims[i] if i < len(var_dims) else []
        features.extend(_pad_vector(dim, size=int(cfg.max_basis_dims)))

    n_points = int(cfg.n_probe_points)
    max_vars = int(cfg.max_input_vars)
    features.extend(_flatten_matrix(samples.get("x_probe", []), rows=n_points, cols=max_vars))
    features.extend(_flatten_vector(samples.get("hole_current_probe", []), rows=n_points))
    features.extend(_flatten_vector(samples.get("inverse_target_probe", []), rows=n_points))
    features.extend(_flatten_vector(samples.get("residual_target_probe", []), rows=n_points))
    features.extend(_flatten_vector(samples.get("inverse_valid_mask_probe", []), rows=n_points))
    features.extend(_flatten_vector(samples.get("residual_valid_mask_probe", []), rows=n_points))
    return torch.as_tensor(features, dtype=torch.float32), valid_completion_exprs(row)


def prepare_micro_search_supervised_examples(
    payload: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    config: MicroSearchProposerConfig | None = None,
    splits: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Prepare tensors, labels, and masks for closed-vocabulary training."""

    rows, meta = _coerce_rows(payload)
    cfg = MicroSearchProposerConfig() if config is None else config
    split_filter = None if splits is None else {str(split) for split in splits}
    selected_rows = [
        row for row in rows
        if (split_filter is None or str(row.get("split", "")) in split_filter)
        and bool((row.get("grammar", {}) or {}).get("truth_in_grammar", False))
        and str(row.get("hole_truth_expr", "") or "") != ""
        and bool((row.get("samples", {}) or {}))
    ]
    vocabulary = build_micro_search_vocabulary(rows)
    vocab_to_idx = {expr: idx for idx, expr in enumerate(vocabulary)}

    xs: list[torch.Tensor] = []
    ys: list[int] = []
    valid_masks: list[torch.Tensor] = []
    split_names: list[str] = []
    state_ids: list[str] = []
    row_refs: list[dict[str, Any]] = []

    for row in selected_rows:
        truth_expr = str(row.get("hole_truth_expr", "") or "")
        if truth_expr not in vocab_to_idx:
            continue
        x, valid_exprs = featurize_micro_search_state(row, config=cfg)
        valid_mask = torch.zeros(len(vocabulary), dtype=torch.bool)
        for expr in valid_exprs:
            idx = vocab_to_idx.get(str(expr), None)
            if idx is not None:
                valid_mask[idx] = True
        target_idx = vocab_to_idx.get(truth_expr, None)
        if target_idx is None or not bool(valid_mask[target_idx]):
            continue
        xs.append(x)
        ys.append(int(target_idx))
        valid_masks.append(valid_mask)
        split_names.append(str(row.get("split", "")))
        state_ids.append(str(row.get("state_id", "")))
        row_refs.append(dict(row))

    if not xs:
        raise ValueError("No usable rows found for micro-search proposer training. Ensure samples and completion tables are present.")

    x_tensor = torch.stack(xs, dim=0)
    y_tensor = torch.as_tensor(ys, dtype=torch.long)
    valid_tensor = torch.stack(valid_masks, dim=0)
    return {
        "rows": row_refs,
        "meta": meta,
        "config": cfg,
        "vocabulary": vocabulary,
        "vocab_to_idx": vocab_to_idx,
        "x": x_tensor,
        "y": y_tensor,
        "valid_mask": valid_tensor,
        "split": split_names,
        "state_id": state_ids,
    }


def _masked_cross_entropy(logits: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    masked_logits = logits.masked_fill(~valid_mask, -1.0e9)
    return F.cross_entropy(masked_logits, target)


def _accuracy_at_k(logits: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor, *, k: int) -> float:
    if int(logits.shape[0]) == 0:
        return float("nan")
    masked_logits = logits.masked_fill(~valid_mask, -1.0e9)
    kk = max(1, min(int(k), int(masked_logits.shape[1])))
    topk = masked_logits.topk(kk, dim=1).indices
    hits = (topk == target.unsqueeze(1)).any(dim=1).float().mean()
    return float(hits.item())


def _split_indices(split_names: Sequence[str], split: str) -> list[int]:
    return [idx for idx, name in enumerate(split_names) if str(name) == str(split)]


def _compute_metrics(
    model: _MicroSearchClosedVocabNet,
    x: torch.Tensor,
    y: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    topk: Sequence[int] = DEFAULT_TOPK,
) -> dict[str, Any]:
    if int(x.shape[0]) == 0:
        return {
            "n_rows": 0,
            "loss": None,
            "top1_accuracy": None,
            "topk_accuracy": {int(k): None for k in topk},
        }
    model.eval()
    with torch.no_grad():
        logits = model(x)
        loss = _masked_cross_entropy(logits, y, valid_mask)
        metrics = {
            "n_rows": int(x.shape[0]),
            "loss": float(loss.item()),
            "top1_accuracy": _accuracy_at_k(logits, y, valid_mask, k=1),
            "topk_accuracy": {int(k): _accuracy_at_k(logits, y, valid_mask, k=int(k)) for k in topk},
        }
    return metrics


def train_micro_search_closed_vocab_proposer(
    payload: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    config: MicroSearchProposerConfig | None = None,
    epochs: int = 200,
    lr: float = 1.0e-3,
    weight_decay: float = 1.0e-4,
    seed: int = 0,
    topk: Sequence[int] = DEFAULT_TOPK,
) -> dict[str, Any]:
    """Train a simple closed-vocabulary proposer directly on micro-search rows."""

    torch.manual_seed(int(seed))
    prepared = prepare_micro_search_supervised_examples(payload, config=config)
    cfg: MicroSearchProposerConfig = prepared["config"]
    x = prepared["x"]
    y = prepared["y"]
    valid_mask = prepared["valid_mask"]
    splits = list(prepared["split"])
    vocabulary = list(prepared["vocabulary"])

    train_idx = _split_indices(splits, "train")
    val_idx = _split_indices(splits, "val")
    test_idx = _split_indices(splits, "test")
    if not train_idx:
        train_idx = list(range(int(x.shape[0])))
    if not val_idx:
        val_idx = list(train_idx)

    feature_mean = x[train_idx].mean(dim=0)
    feature_std = x[train_idx].std(dim=0)
    feature_std = torch.where(feature_std > 1.0e-6, feature_std, torch.ones_like(feature_std))

    x_norm = (x - feature_mean.unsqueeze(0)) / feature_std.unsqueeze(0)
    model = _MicroSearchClosedVocabNet(
        in_dim=int(x.shape[1]),
        hidden_dim=int(cfg.hidden_dim),
        out_dim=int(len(vocabulary)),
    ).to(dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    best_val = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    history: list[dict[str, Any]] = []
    for epoch in range(1, max(1, int(epochs)) + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        logits = model(x_norm[train_idx])
        loss = _masked_cross_entropy(logits, y[train_idx], valid_mask[train_idx])
        loss.backward()
        opt.step()

        val_metrics = _compute_metrics(
            model,
            x_norm[val_idx],
            y[val_idx],
            valid_mask[val_idx],
            topk=topk,
        )
        train_metrics = _compute_metrics(
            model,
            x_norm[train_idx],
            y[train_idx],
            valid_mask[train_idx],
            topk=topk,
        )
        history.append({
            "epoch": int(epoch),
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "train_top1": train_metrics["top1_accuracy"],
            "val_top1": val_metrics["top1_accuracy"],
        })
        val_loss = float(val_metrics["loss"]) if val_metrics["loss"] is not None else float("inf")
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state, strict=True)
    model.eval()
    train_metrics = _compute_metrics(model, x_norm[train_idx], y[train_idx], valid_mask[train_idx], topk=topk)
    val_metrics = _compute_metrics(model, x_norm[val_idx], y[val_idx], valid_mask[val_idx], topk=topk)
    test_metrics = _compute_metrics(model, x_norm[test_idx], y[test_idx], valid_mask[test_idx], topk=topk) if test_idx else {
        "n_rows": 0,
        "loss": None,
        "top1_accuracy": None,
        "topk_accuracy": {int(k): None for k in topk},
    }

    bundle = {
        "model_kind": "micro_search_closed_vocab_proposer",
        "trained": True,
        "model": model,
        "config": {
            "n_probe_points": int(cfg.n_probe_points),
            "max_input_vars": int(cfg.max_input_vars),
            "max_basis_dims": int(cfg.max_basis_dims),
            "hidden_dim": int(cfg.hidden_dim),
        },
        "vocabulary": vocabulary,
        "feature_mean": feature_mean.detach().cpu(),
        "feature_std": feature_std.detach().cpu(),
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "metrics": {
            "n_rows": int(x.shape[0]),
            "n_train_rows": int(len(train_idx)),
            "n_val_rows": int(len(val_idx)),
            "n_test_rows": int(len(test_idx)),
            "best_val_loss": None if not math.isfinite(best_val) else float(best_val),
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
        },
        "history": history,
    }
    return bundle


def save_micro_search_proposer_bundle(bundle: Mapping[str, Any], path: str | pathlib.Path) -> None:
    out_path = pathlib.Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(bundle)
    payload.pop("model", None)
    torch.save(payload, out_path)


def load_micro_search_proposer_bundle(path: str | pathlib.Path) -> dict[str, Any]:
    payload = dict(torch.load(pathlib.Path(path), map_location="cpu", weights_only=False))
    cfg_raw = dict(payload.get("config", {}) or {})
    cfg = MicroSearchProposerConfig(
        n_probe_points=int(cfg_raw.get("n_probe_points", 16)),
        max_input_vars=int(cfg_raw.get("max_input_vars", 4)),
        max_basis_dims=int(cfg_raw.get("max_basis_dims", 4)),
        hidden_dim=int(cfg_raw.get("hidden_dim", 128)),
    )
    vocabulary = [str(expr) for expr in list(payload.get("vocabulary", []) or [])]
    feature_mean = torch.as_tensor(payload.get("feature_mean", []), dtype=torch.float32)
    feature_std = torch.as_tensor(payload.get("feature_std", []), dtype=torch.float32)
    model = _MicroSearchClosedVocabNet(
        in_dim=int(feature_mean.numel()),
        hidden_dim=int(cfg.hidden_dim),
        out_dim=int(len(vocabulary)),
    ).to(dtype=torch.float32)
    model.load_state_dict(payload.get("model_state_dict", {}), strict=True)
    model.eval()
    payload["config"] = cfg
    payload["vocabulary"] = vocabulary
    payload["feature_mean"] = feature_mean
    payload["feature_std"] = feature_std
    payload["model"] = model
    return payload


def predict_micro_search_proposer(
    bundle: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    topk: int | None = None,
) -> list[dict[str, Any]]:
    """Rank valid row completions under the learned closed-vocabulary proposer."""

    model = bundle.get("model", None)
    if model is None:
        raise ValueError("Loaded proposer bundle is missing a materialized model")
    cfg = bundle.get("config", MicroSearchProposerConfig())
    if not isinstance(cfg, MicroSearchProposerConfig):
        cfg = MicroSearchProposerConfig(**dict(cfg))
    x, valid_exprs = featurize_micro_search_state(row, config=cfg)
    vocabulary = list(bundle.get("vocabulary", []) or [])
    vocab_to_idx = {expr: idx for idx, expr in enumerate(vocabulary)}
    valid_idx = [idx for expr in valid_exprs if (idx := vocab_to_idx.get(expr, None)) is not None]
    if not valid_idx:
        return []
    feature_mean = torch.as_tensor(bundle.get("feature_mean", torch.zeros_like(x)), dtype=torch.float32)
    feature_std = torch.as_tensor(bundle.get("feature_std", torch.ones_like(x)), dtype=torch.float32)
    x_norm = (x - feature_mean) / feature_std
    with torch.no_grad():
        logits = model(x_norm.unsqueeze(0)).squeeze(0)
        masked = torch.full_like(logits, -1.0e9)
        masked[valid_idx] = logits[valid_idx]
        probs = torch.softmax(masked, dim=0)

    candidates = {}
    rankings = dict(row.get("rankings", {}) or {})
    for key in ("inverse", "residual"):
        for cand in list(rankings.get(key, []) or []):
            if isinstance(cand, Mapping):
                candidates[str(cand.get("expr", "") or "")] = dict(cand)

    rows: list[dict[str, Any]] = []
    for idx in valid_idx:
        expr = str(vocabulary[idx])
        cand = dict(candidates.get(expr, {"expr": expr}))
        cand["learned_logit"] = float(logits[idx].item())
        cand["learned_prob"] = float(probs[idx].item())
        rows.append(cand)
    rows.sort(key=lambda cand: (-float(cand.get("learned_prob", 0.0)), str(cand.get("expr", ""))))
    if topk is None:
        return rows
    return rows[: max(1, int(topk))]


def run_micro_search_proposer_pipeline(
    *,
    dataset_path: str | pathlib.Path,
    output_dir: str | pathlib.Path,
    n_probe_points: int = 16,
    max_input_vars: int = 4,
    max_basis_dims: int = 4,
    hidden_dim: int = 128,
    epochs: int = 200,
    lr: float = 1.0e-3,
    weight_decay: float = 1.0e-4,
    seed: int = 0,
    verbose: bool = True,
) -> dict[str, Any]:
    payload = json.loads(pathlib.Path(dataset_path).read_text(encoding="utf-8"))
    cfg = MicroSearchProposerConfig(
        n_probe_points=int(n_probe_points),
        max_input_vars=int(max_input_vars),
        max_basis_dims=int(max_basis_dims),
        hidden_dim=int(hidden_dim),
    )
    bundle = train_micro_search_closed_vocab_proposer(
        payload,
        config=cfg,
        epochs=int(epochs),
        lr=float(lr),
        weight_decay=float(weight_decay),
        seed=int(seed),
    )
    out_dir = pathlib.Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = out_dir / "micro_search_closed_vocab_proposer.pt"
    save_micro_search_proposer_bundle(bundle, bundle_path)
    summary = {
        "dataset_path": str(dataset_path),
        "bundle_path": str(bundle_path),
        "metrics": dict(bundle.get("metrics", {}) or {}),
        "config": dict(bundle.get("config", {}) or {}),
    }
    (out_dir / "micro_search_closed_vocab_proposer_summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2),
        encoding="utf-8",
    )
    if verbose:
        print(
            f"[micro-search-proposer] vocab={len(bundle['vocabulary'])} "
            f"train_top1={bundle['metrics']['train']['top1_accuracy']} "
            f"val_top1={bundle['metrics']['val']['top1_accuracy']}"
        )
    return _jsonable(summary)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a closed-vocabulary proposer on micro-search data")
    parser.add_argument("--dataset", required=True, help="Path to micro-search dataset JSON")
    parser.add_argument("--output_dir", required=True, help="Directory for bundle and summary")
    parser.add_argument("--n_probe_points", type=int, default=16)
    parser.add_argument("--max_input_vars", type=int, default=4)
    parser.add_argument("--max_basis_dims", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight_decay", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return run_micro_search_proposer_pipeline(
        dataset_path=str(args.dataset),
        output_dir=str(args.output_dir),
        n_probe_points=int(args.n_probe_points),
        max_input_vars=int(args.max_input_vars),
        max_basis_dims=int(args.max_basis_dims),
        hidden_dim=int(args.hidden_dim),
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        seed=int(args.seed),
    )


if __name__ == "__main__":  # pragma: no cover
    main()
