# SPDX-License-Identifier: MPL-2.0
"""Canonical, slice-trained Stage-A fit tournaments for CoE search."""

from __future__ import annotations

import copy
import hashlib
import math
import multiprocessing as mp
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, TimeoutError, as_completed
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from .coe_witness import _thread_cap_initializer
from .training import (
    _sr_canonical_init_provider,
    train_candidate_model,
    train_initial_model,
)


@dataclass
class StageAFitLane:
    slice_id: int
    status: str
    accepted: bool = False
    fit_val_loss: float = float("inf")
    train_loss: float = float("inf")
    comparison_mse: float = float("inf")
    finite_fraction: float = 0.0
    train_start: Optional[int] = None
    train_stop: Optional[int] = None
    canonical_init_applied: bool = False
    canonical_fingerprint: Optional[str] = None
    elapsed_seconds: float = 0.0
    error: Optional[str] = None
    state_dict: Optional[dict[str, Any]] = None
    best_params: Optional[torch.Tensor] = None
    full_parameters: Optional[list[torch.Tensor]] = None
    parameter_layout_fingerprint: Optional[str] = None
    comparison_predictions: Optional[list[float]] = None
    row_losses: Optional[list[float]] = None

    def summary(self) -> dict[str, Any]:
        return {
            "slice_id": int(self.slice_id),
            "status": self.status,
            "accepted": bool(self.accepted),
            "fit_val_loss": float(self.fit_val_loss),
            "train_loss": float(self.train_loss),
            "comparison_mse": float(self.comparison_mse),
            "finite_fraction": float(self.finite_fraction),
            "train_start": self.train_start,
            "train_stop": self.train_stop,
            "canonical_init_applied": bool(self.canonical_init_applied),
            "canonical_fingerprint": self.canonical_fingerprint,
            "parameter_count": (
                None
                if self.full_parameters is None
                else sum(int(value.numel()) for value in self.full_parameters)
            ),
            "parameter_layout_fingerprint": self.parameter_layout_fingerprint,
            "elapsed_seconds": float(self.elapsed_seconds),
            "error": self.error,
        }


class _AppliedStateOptimizer:
    """Compatibility shim: the selected worker state is already applied."""

    def _update_param_groups(self, _params) -> None:
        return None


@dataclass(frozen=True)
class StageAModelRecipe:
    """Pickle-safe instructions for rebuilding one pure-NN Stage-A model."""

    ast_root: Any
    state_dict: dict[str, torch.Tensor]
    full_parameters: tuple[torch.Tensor, ...]
    parameter_layout_fingerprint: str
    model_base_name: str
    model_scale: float
    nout_size: int
    block_size_target: Optional[int]
    dtype_name: str
    global_input_dim: Optional[int]
    fit_y_link: Optional[str]
    fit_y_link_scale: float


def _make_model_recipe(model: torch.nn.Module) -> StageAModelRecipe:
    leaves = list(getattr(model, "leaf", ()))
    ast_root = getattr(model, "ast_root", None)
    if len(leaves) != 1 or ast_root is None:
        raise TypeError("Stage-A fit recipe requires a one-leaf AST model")
    leaf = leaves[0]
    stage0 = getattr(leaf, "stage0", None)
    stage1 = getattr(leaf, "stage1", None)
    first = stage0 if stage0 is not None else leaf
    last = stage1 if stage1 is not None else leaf
    first_base = getattr(first, "base_model", None)
    last_base = getattr(last, "base_model", None)
    if first_base is None or last_base is None:
        raise TypeError("Stage-A fit recipe requires segmented NN leaves")
    try:
        dtype_name = str(next(model.parameters()).dtype).removeprefix("torch.")
    except StopIteration as exc:
        raise TypeError("Stage-A fit recipe requires trainable parameters") from exc
    full_parameters, parameter_layout_fingerprint = _full_parameter_snapshot(model)
    recipe = StageAModelRecipe(
        ast_root=copy.deepcopy(ast_root),
        state_dict={
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        },
        full_parameters=tuple(full_parameters),
        parameter_layout_fingerprint=parameter_layout_fingerprint,
        model_base_name=type(first_base).__name__,
        model_scale=float(getattr(first_base, "model_scale")),
        nout_size=int(getattr(last_base, "Nout_size")),
        block_size_target=getattr(first, "_block_size_target", None),
        dtype_name=dtype_name,
        global_input_dim=getattr(model, "_global_input_dim", None),
        fit_y_link=getattr(model, "fit_y_link", None),
        fit_y_link_scale=float(getattr(model, "fit_y_link_scale", 1.0)),
    )
    # Fail locally before launching a pool if a future AST/config extension is
    # not portable under multiprocessing spawn.
    pickle.dumps(recipe, protocol=pickle.HIGHEST_PROTOCOL)
    return recipe


def _rebuild_model(recipe: StageAModelRecipe) -> torch.nn.Module:
    from .model_builders import LeafBuilder, build_composite_ast

    dtype = getattr(torch, recipe.dtype_name)
    hp = SimpleNamespace(
        model_base_name=recipe.model_base_name,
        Gmodel_scale=recipe.model_scale,
        Nout_size=recipe.nout_size,
        block_size_target=recipe.block_size_target,
    )
    device = torch.device("cpu")
    model, _nparam, _ast = build_composite_ast(
        copy.deepcopy(recipe.ast_root),
        None,
        None,
        LeafBuilder(hp, device, dtype),
        device,
        dtype,
    )
    # Segmented LM registers its current block as transient ``*_fit``
    # parameters. Their presence and shapes depend on where a fit stopped and
    # are neither topology nor persistent model state; the authoritative full
    # TSOP vectors below restore their underlying values. Load every other
    # parameter/buffer strictly, while allowing only those transient views to
    # differ between the source and freshly rebuilt model.
    transient_names = {"a_fit", "b_fit", "c_fit", "K_fit"}
    persistent_state = {
        name: value
        for name, value in recipe.state_dict.items()
        if name.rsplit(".", 1)[-1] not in transient_names
    }
    incompatible = model.load_state_dict(persistent_state, strict=False)
    invalid_missing = [
        name
        for name in incompatible.missing_keys
        if name.rsplit(".", 1)[-1] not in transient_names
    ]
    invalid_unexpected = [
        name
        for name in incompatible.unexpected_keys
        if name.rsplit(".", 1)[-1] not in transient_names
    ]
    if invalid_missing or invalid_unexpected:
        raise ValueError(
            "rebuilt Stage-A model has incompatible persistent state: "
            f"missing={invalid_missing}, unexpected={invalid_unexpected}"
        )
    _rebuilt_parameters, rebuilt_layout = _full_parameter_snapshot(model)
    if rebuilt_layout != recipe.parameter_layout_fingerprint:
        raise ValueError(
            "rebuilt Stage-A model has a different full TSOP parameter layout "
            f"({rebuilt_layout} != {recipe.parameter_layout_fingerprint})"
        )
    # ``state_dict`` preserves registered tensors and is useful for restoring
    # module buffers, but segmented models can also hold fixed TSOP pieces
    # outside the currently registered LM block. Canonical initialization is
    # sensitive to those values, so restore the complete pre-canonical state.
    _load_full_parameter_snapshot(model, recipe.full_parameters)
    if recipe.global_input_dim is not None:
        model.declare_global_input_dim(int(recipe.global_input_dim))
    model.fit_y_link = recipe.fit_y_link
    model.fit_y_link_scale = float(recipe.fit_y_link_scale)
    return model


def _state_fingerprint(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _segmented_base_models(model: torch.nn.Module) -> list[torch.nn.Module]:
    leaves = list(getattr(model, "leaf", ()))
    if len(leaves) != 1:
        raise TypeError("portable Stage-A state requires a one-leaf AST model")
    leaf = leaves[0]
    stage0 = getattr(leaf, "stage0", None)
    stage1 = getattr(leaf, "stage1", None)
    adaptors = [stage0, stage1] if stage0 is not None and stage1 is not None else [leaf]
    bases = [getattr(adaptor, "base_model", None) for adaptor in adaptors]
    if any(base is None for base in bases):
        raise TypeError("portable Stage-A state requires segmented base models")
    return bases


def _full_parameter_snapshot(
    model: torch.nn.Module,
) -> tuple[list[torch.Tensor], str]:
    """Capture the invariant full TSOP state, independent of LM block masks."""
    vectors: list[torch.Tensor] = []
    layout = hashlib.sha256()
    for base_index, base in enumerate(_segmented_base_models(model)):
        groups = base.get_parameters()
        pieces = [piece for group in groups for piece in group]
        vector = torch.cat(
            [piece.detach().cpu().reshape(-1) for piece in pieces]
        ).clone()
        vectors.append(vector)
        layout.update(f"{base_index}:{type(base).__name__}".encode())
        for group_index, group in enumerate(groups):
            for piece in group:
                layout.update(
                    f"{group_index}:{tuple(piece.shape)}:{piece.dtype}".encode()
                )
    return vectors, layout.hexdigest()


def _load_full_parameter_snapshot(
    model: torch.nn.Module, vectors: Sequence[torch.Tensor]
) -> None:
    bases = _segmented_base_models(model)
    if len(bases) != len(vectors):
        raise ValueError(
            f"portable state has {len(vectors)} stages; model has {len(bases)}"
        )
    for base, vector in zip(bases, vectors):
        loader = getattr(base, "load_full_parameters_tsop_", None)
        if not callable(loader):
            raise TypeError(f"{type(base).__name__} cannot load full TSOP state")
        loader(vector)


def _loader_tensors(loader) -> tuple[torch.Tensor, torch.Tensor]:
    xs, ys = [], []
    for batch in loader:
        if len(batch) < 2:
            raise ValueError("Stage-A tournament requires (x, y) batches")
        xs.append(batch[0].detach().cpu())
        ys.append(batch[1].detach().cpu())
    if not xs:
        raise ValueError("Stage-A tournament received an empty loader")
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def _loader_tensor_input_dim(loader) -> Optional[int]:
    """Mirror ``train_initial_model``'s TensorDataset input-width inference."""
    dataset = getattr(loader, "dataset", None)
    while dataset is not None and not hasattr(dataset, "tensors"):
        parent = getattr(dataset, "dataset", None)
        if parent is None or parent is dataset:
            dataset = None
            break
        dataset = parent
    tensors = getattr(dataset, "tensors", ()) if dataset is not None else ()
    x_tensor = tensors[0] if tensors else None
    if torch.is_tensor(x_tensor) and x_tensor.ndim >= 2:
        return int(x_tensor.shape[-1])
    return None


class _OpaqueTensorDataset(Dataset):
    """Tensor dataset that deliberately does not advertise a ``tensors`` field.

    ``train_initial_model`` uses that field to switch a one-leaf AST to its
    transparent-identity optimizer contract.  A spawned fit must mirror the
    source loader's contract rather than change it merely because its portable
    rows were reconstructed from tensors.
    """

    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self._x = x
        self._y = y

    def __len__(self) -> int:
        return int(self._x.shape[0])

    def __getitem__(self, index):
        return self._x[index], self._y[index]


def _tensor_loader(
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    *,
    expose_tensors: bool = True,
) -> DataLoader:
    dataset = TensorDataset(x, y) if expose_tensors else _OpaqueTensorDataset(x, y)
    return DataLoader(
        dataset,
        batch_size=max(1, min(int(batch_size), int(x.shape[0]))),
        shuffle=False,
        drop_last=False,
    )


def _quiet_fit_worker(payload: dict[str, Any]) -> StageAFitLane:
    started = time.perf_counter()
    lane = StageAFitLane(
        slice_id=int(payload["slice_id"]), status="error",
        train_start=int(payload["train_start"]), train_stop=int(payload["train_stop"]),
    )
    try:
        model = _rebuild_model(payload["model_recipe"])
        # Preserve the master's initial-fit provider contract.  Normal SR
        # PhysDataset loaders do not expose ``.tensors``; rebuilding their rows
        # as a TensorDataset used to make train_initial_model declare the input
        # width only in spawned lanes, activating a different AST optimizer
        # path.  Explicitly reproduce the master's declaration decision, then
        # keep the portable datasets opaque to a second inference pass.
        source_input_dim = payload.get("train_tensor_input_dim")
        if bool(payload.get("initial_fit", False)) and source_input_dim is not None:
            declare_input_dim = getattr(model, "declare_global_input_dim", None)
            if callable(declare_input_dim):
                declare_input_dim(int(source_input_dim))
        train_loader = _tensor_loader(
            payload["train_x"],
            payload["train_y"],
            payload["batch_size"],
            expose_tensors=False,
        )
        fit_val_loader = _tensor_loader(
            payload["fit_val_x"],
            payload["fit_val_y"],
            payload["batch_size"],
            expose_tensors=False,
        )
        hp = payload["lm_hp"]
        hp.log_file = None
        hp.log_to_console = False
        with open(os.devnull, "w") as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
            common = dict(
                epochs=payload["epochs"],
                LM_strategy=payload["LM_strategy"],
                nval_patience=payload["nval_patience"],
                loss_target=payload["loss_target"],
                epochs_min=payload["epochs_min"],
                chisq_tol=payload["chisq_tol"],
                device=torch.device("cpu"),
                epochs_awful_check=payload["epochs_awful_check"],
                awful_threshold=payload["awful_threshold"],
                log_file=None,
                log_to_console=False,
                log_level=payload["log_level"],
                lm_verbose=False,
                lm_hp=hp,
            )
            if bool(payload.get("initial_fit", False)):
                val_loss, train_loss, best_params, opt = train_initial_model(
                    model, train_loader, fit_val_loader, **common
                )
                accepted = math.isfinite(float(val_loss)) and math.isfinite(
                    float(train_loss)
                )
            else:
                accepted, val_loss, train_loss, best_params, opt = train_candidate_model(
                    model,
                    train_loader,
                    fit_val_loader,
                    accept_threshold=payload["accept_threshold"],
                    **common,
                )
        if best_params is not None:
            opt._update_param_groups(best_params)
        lane.status = "success"
        lane.accepted = bool(accepted)
        lane.fit_val_loss = float(val_loss)
        lane.train_loss = float(train_loss)
        lane.best_params = None if best_params is None else best_params.detach().cpu()
        # state_dict and best_params are diagnostic/caller compatibility data:
        # segmented LM changes transient per-block *_fit registrations, and
        # the optimizer vector alone omits fixed-piece storage. The complete
        # full_parameters TSOP snapshot is authoritative for portability.
        lane.state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        lane.full_parameters, lane.parameter_layout_fingerprint = (
            _full_parameter_snapshot(model)
        )
        model.eval()
        with torch.no_grad():
            lane.comparison_predictions = (
                model(payload["comparison_x"]).reshape(-1).detach().cpu().tolist()
            )
        lane.canonical_fingerprint = getattr(
            opt, "_sr_canonical_state_fingerprint", None
        )
        lane.canonical_init_applied = bool(
            getattr(opt, "_sr_canonical_init_applied", False)
        )
    except Exception as exc:  # worker failures are deliberately fail-soft
        lane.error = f"{type(exc).__name__}: {exc}"
    lane.elapsed_seconds = time.perf_counter() - started
    return lane


def _row_losses(model, loader, device, y_op_inv=None) -> tuple[list[float], float]:
    parts: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for x, target in loader:
            pred = model(x.to(device)).reshape(-1)
            target = target.to(device).reshape(-1)
            if y_op_inv is not None:
                pred = y_op_inv(pred).reshape(-1)
                target = y_op_inv(target).reshape(-1)
            loss = (pred - target).double().square().detach().cpu().numpy()
            loss[~np.isfinite(loss)] = np.nan
            parts.append(loss)
    rows = np.concatenate(parts) if parts else np.empty(0)
    finite = np.isfinite(rows)
    frac = float(np.mean(finite)) if rows.size else 0.0
    return rows.tolist(), frac


def _prediction_row_losses(
    predictions: Sequence[float], targets: torch.Tensor, y_op_inv=None
) -> tuple[list[float], float]:
    pred = torch.as_tensor(predictions, dtype=targets.dtype).reshape(-1)
    target = targets.reshape(-1)
    if pred.numel() != target.numel():
        raise ValueError(
            f"worker returned {pred.numel()} predictions for {target.numel()} rows"
        )
    if y_op_inv is not None:
        pred = y_op_inv(pred).reshape(-1)
        target = y_op_inv(target).reshape(-1)
    rows = (pred - target).double().square().detach().cpu().numpy()
    rows[~np.isfinite(rows)] = np.nan
    finite = np.isfinite(rows)
    return rows.tolist(), float(np.mean(finite)) if rows.size else 0.0


def _verify_worker_lane_predictions(
    model: torch.nn.Module,
    lane: StageAFitLane,
    comparison_x: torch.Tensor,
) -> None:
    if lane.comparison_predictions is None:
        raise ValueError("selected worker returned incomplete portable state")
    model.eval()
    with torch.no_grad():
        adopted = model(comparison_x).reshape(-1).detach().cpu()
    expected = torch.as_tensor(
        lane.comparison_predictions, dtype=adopted.dtype
    ).reshape(-1)
    torch.testing.assert_close(adopted, expected, rtol=1.0e-11, atol=1.0e-12)


def choose_stageA_fit_lane(
    lanes: Sequence[StageAFitLane],
    *,
    master_slice: int,
    alpha: float,
    min_rel_improvement: float,
    noise_floor: float,
    target_scale: float,
    unit_keys: Sequence[int],
    seed: int,
) -> tuple[StageAFitLane, dict[str, Any]]:
    """Choose the valid identical-model fit with the lowest common-row MSE."""
    master = next(lane for lane in lanes if int(lane.slice_id) == int(master_slice))
    valid = [
        lane
        for lane in lanes
        if lane.status == "success" and lane.accepted and lane.row_losses is not None
        and lane.canonical_init_applied and bool(lane.canonical_fingerprint)
        and math.isfinite(float(lane.comparison_mse))
    ]
    summary: dict[str, Any] = {
        "decision": "keep_master",
        "selected_slice": int(master_slice),
        "selection_rule": "best_valid_common_comparison_mse",
    }
    challengers = [lane for lane in valid if int(lane.slice_id) != int(master_slice)]
    if master not in valid:
        if challengers:
            selected = min(challengers, key=lambda lane: (lane.comparison_mse, lane.slice_id))
            summary.update({"decision": "replace_invalid_master", "selected_slice": selected.slice_id})
            return selected, summary
        summary["reason"] = "no valid fit lane"
        return master, summary
    if not challengers:
        summary["reason"] = "no valid challengers"
        return master, summary

    selected = min(
        valid,
        key=lambda lane: (
            float(lane.comparison_mse),
            0 if lane is master else 1,
            int(lane.slice_id),
        ),
    )
    if selected is master:
        summary["reason"] = "master has the lowest valid common-comparison MSE"
        return master, summary
    summary.update({"decision": "replace_master", "selected_slice": int(selected.slice_id)})
    return selected, summary


def _parse_slices(value: Any) -> list[int]:
    if isinstance(value, str):
        raw = value.replace(",", " ").split()
    else:
        raw = list(value or [])
    out: list[int] = []
    for item in raw:
        sid = int(item)
        if sid >= 0 and sid not in out:
            out.append(sid)
    return out


def validate_stageA_fit_slice_firewall(
    slices: Any, *, witness_start: int, witness_count: int
) -> list[int]:
    parsed = _parse_slices(slices)
    if int(witness_count) > 0 and any(sid >= int(witness_start) for sid in parsed):
        raise ValueError(
            "Stage-A fit slices must precede --coe_start_slice so fit rows "
            "cannot overlap CoE witness slices"
        )
    return parsed


def _fit_options(fit_kwargs: dict[str, Any]) -> dict[str, Any]:
    required = (
        "epochs", "LM_strategy", "nval_patience", "loss_target", "epochs_min",
        "chisq_tol",
    )
    missing = [name for name in required if name not in fit_kwargs]
    if missing:
        raise ValueError(f"Stage-A tournament missing fit options: {', '.join(missing)}")
    options = {name: fit_kwargs[name] for name in required}
    options.update(
        epochs_awful_check=fit_kwargs.get("epochs_awful_check"),
        awful_threshold=fit_kwargs.get("awful_threshold"),
        log_level=fit_kwargs.get("log_level"),
    )
    return options


def _terminate_pool(pool: ProcessPoolExecutor) -> None:
    processes = list(getattr(pool, "_processes", {}).values())
    pool.shutdown(wait=False, cancel_futures=True)
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=5.0)


def _apply_lane_train_loss_cap(
    lanes: Sequence[StageAFitLane], max_lane_train_loss: Optional[float]
) -> None:
    if max_lane_train_loss is None:
        return
    train_cap = float(max_lane_train_loss)
    for lane in lanes:
        if lane.status == "success" and float(lane.train_loss) > train_cap:
            lane.accepted = False
            lane.error = (
                f"training loss {float(lane.train_loss):.6g} exceeds "
                f"lane validity cap {train_cap:.6g}"
            )


def fit_stageA_candidate_with_tournament(
    model,
    train_dl,
    val_dl,
    *,
    y_op=None,
    y_op_inv=None,
    lm_hp,
    device,
    accept_threshold: float,
    extra_train_factories=None,
    max_lane_train_loss: Optional[float] = None,
    _initial_fit: bool = False,
    **fit_kwargs,
):
    """Drop-in Stage-A candidate fit with parallel slice-trained challengers."""
    active = bool(getattr(lm_hp, "coe_stageA_fit_tournament", False))
    mode = str(getattr(lm_hp, "coe_mode", "off") or "off")
    def direct_fit():
        if _initial_fit:
            val_loss, train_loss, params, opt = train_initial_model(
                model, train_dl, val_dl, lm_hp=lm_hp, device=device, **fit_kwargs
            )
            accepted = math.isfinite(float(val_loss)) and math.isfinite(
                float(train_loss)
            )
            return accepted, val_loss, train_loss, params, opt
        return train_candidate_model(
            model, train_dl, val_dl, accept_threshold=accept_threshold,
            extra_train_factories=extra_train_factories, lm_hp=lm_hp,
            device=device, **fit_kwargs
        )

    if not active or mode not in {"committee_gated", "reservoir_discovery"} or extra_train_factories:
        return direct_fit()
    if not bool(getattr(lm_hp, "canonical_init", False)):
        # The CLI rejects a tournament configured without canonical init. This
        # branch is for the established one-shot internal random/no-evidence
        # recovery fit, which temporarily disables canonical init at runtime.
        return direct_fit()
    if torch.device(device).type != "cpu" or not getattr(lm_hp, "coe_filepath", None):
        return direct_fit()
    # A tournament is useful only when every lane can obtain the requested
    # data-dependent canonical state. Unsupported composite providers keep the
    # pre-existing single master fit instead of running same-state lanes.
    if _sr_canonical_init_provider(model) is None:
        return direct_fit()

    master_slice = int(getattr(lm_hp, "coe_reference_slice", 0) or 0)
    total_fit_parallelism = max(1, int(getattr(lm_hp, "coe_scout_parallelism", 1) or 1))
    slices = [sid for sid in _parse_slices(getattr(lm_hp, "coe_stageA_fit_slices", ())) if sid != master_slice]
    slices = slices[: max(0, total_fit_parallelism - 1)]
    if not slices:
        return direct_fit()
    train_x, train_y = _loader_tensors(train_dl)
    val_x, val_y = _loader_tensors(val_dl)
    train_tensor_input_dim = _loader_tensor_input_dim(train_dl)
    if int(val_x.shape[0]) < 4:
        return direct_fit()
    frac = float(getattr(lm_hp, "coe_stageA_fit_comparison_fraction", 0.5) or 0.5)
    n_compare = max(2, min(int(val_x.shape[0]) - 1, int(round(frac * int(val_x.shape[0])))))
    split = int(val_x.shape[0]) - n_compare
    fit_val_x, fit_val_y = val_x[:split], val_y[:split]
    compare_loader = _tensor_loader(val_x[split:], val_y[split:], int(getattr(train_dl, "batch_size", 1) or 1))

    template = copy.deepcopy(model).cpu()
    template_fp = _state_fingerprint(template)
    _template_parameters, template_layout_fp = _full_parameter_snapshot(template)
    try:
        model_recipe = _make_model_recipe(template)
    except Exception as exc:
        print(
            "[CoE StageA fit tournament] model recipe unavailable; "
            f"using master fit ({type(exc).__name__}: {exc})"
        )
        return direct_fit()
    from .coe_committee import _load_dataset_arrays

    x_all, y_all, _ = _load_dataset_arrays(str(lm_hp.coe_filepath))
    ntrain = int(getattr(lm_hp, "coe_ndata_train", train_x.shape[0]) or train_x.shape[0])
    nval = int(getattr(lm_hp, "coe_ndata_val", val_x.shape[0]) or val_x.shape[0])
    worker_hp = copy.deepcopy(lm_hp)
    worker_options = _fit_options(fit_kwargs)
    payloads = []
    layout_errors: list[StageAFitLane] = []
    for sid in slices:
        start = sid * (ntrain + nval)
        stop = start + ntrain
        if stop > int(y_all.shape[0]):
            layout_errors.append(
                StageAFitLane(
                    int(sid), "invalid_layout", train_start=start, train_stop=stop,
                    error=f"training rows exceed available search rows ({int(y_all.shape[0])})",
                )
            )
            continue
        y_slice = np.asarray(y_all[start:stop], dtype=np.float64).reshape(-1, 1)
        if y_op is not None:
            y_slice = np.asarray(y_op(y_slice), dtype=np.float64).reshape(-1, 1)
        payloads.append({
            "slice_id": sid, "model_recipe": model_recipe,
            "train_start": start, "train_stop": stop,
            "train_x": torch.as_tensor(np.array(x_all[start:stop], copy=True), dtype=train_x.dtype),
            "train_y": torch.as_tensor(np.array(y_slice, copy=True), dtype=train_y.dtype),
            "fit_val_x": fit_val_x, "fit_val_y": fit_val_y,
            "comparison_x": val_x[split:],
            "train_tensor_input_dim": train_tensor_input_dim,
            "batch_size": int(getattr(train_dl, "batch_size", 1) or 1), "lm_hp": worker_hp,
            "accept_threshold": float(accept_threshold),
            "initial_fit": bool(_initial_fit),
            **worker_options,
        })

    max_workers = min(len(payloads), max(1, total_fit_parallelism - 1))
    pool = None
    futures = {}
    launch_error = None
    master_started = time.perf_counter()
    try:
        if payloads:
            pool = ProcessPoolExecutor(
                max_workers=max_workers,
                mp_context=mp.get_context("spawn"),
                initializer=_thread_cap_initializer,
            )
            futures = {pool.submit(_quiet_fit_worker, payload): payload["slice_id"] for payload in payloads}
    except Exception as exc:
        launch_error = f"{type(exc).__name__}: {exc}"
        if pool is not None:
            _terminate_pool(pool)
        pool = None
        futures = {}

    fit_kind = "initial_teacher" if _initial_fit else "stageA_candidate"
    print(
        "[CoE StageA fit tournament] "
        f"starting kind={fit_kind} master_slice={master_slice} "
        f"spawned_lanes={len(futures)} total_fit_parallelism={total_fit_parallelism}"
    )

    master_error: Optional[Exception] = None
    master_opt = None
    best_params = None
    fit_loss = float("inf")
    train_loss = float("inf")
    accepted = False
    try:
        master_val_dl = _tensor_loader(
            fit_val_x, fit_val_y, int(getattr(val_dl, "batch_size", 1) or 1)
        )
        if _initial_fit:
            val_loss, train_loss, best_params, master_opt = train_initial_model(
                model, train_dl, master_val_dl, lm_hp=lm_hp, device=device,
                **fit_kwargs,
            )
            master_result = (
                math.isfinite(float(val_loss)) and math.isfinite(float(train_loss)),
                val_loss,
                train_loss,
                best_params,
                master_opt,
            )
        else:
            master_result = train_candidate_model(
                model,
                train_dl,
                master_val_dl,
                accept_threshold=accept_threshold,
                extra_train_factories=None,
                lm_hp=lm_hp,
                device=device,
                **fit_kwargs,
            )
        accepted, fit_loss, train_loss, best_params, master_opt = master_result
    except Exception as exc:
        master_error = exc
    if best_params is not None and master_opt is not None:
        master_opt._update_param_groups(best_params)
    master_start = master_slice * (ntrain + nval)
    if master_error is None:
        master_full_parameters, master_layout_fp = _full_parameter_snapshot(model)
        master_lane = StageAFitLane(master_slice, "success", bool(accepted), float(fit_loss), float(train_loss), train_start=master_start, train_stop=master_start + ntrain, canonical_init_applied=bool(getattr(master_opt, "_sr_canonical_init_applied", False)), canonical_fingerprint=getattr(master_opt, "_sr_canonical_state_fingerprint", None), elapsed_seconds=time.perf_counter() - master_started, state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()}, best_params=None if best_params is None else best_params.detach().cpu(), full_parameters=master_full_parameters, parameter_layout_fingerprint=master_layout_fp)
    else:
        master_lane = StageAFitLane(
            master_slice,
            "error",
            accepted=False,
            train_start=master_start,
            train_stop=master_start + ntrain,
            elapsed_seconds=time.perf_counter() - master_started,
            error=f"{type(master_error).__name__}: {master_error}",
        )
    lanes = [master_lane]
    lanes.extend(layout_errors)
    if launch_error is not None:
        lanes.extend(
            StageAFitLane(int(payload["slice_id"]), "error", error=f"worker launch failed: {launch_error}")
            for payload in payloads
        )
    if pool is not None:
        try:
            timeout = float(getattr(lm_hp, "coe_scout_timeout_seconds", 0.0) or 0.0)
            iterator = as_completed(futures, timeout=timeout if timeout > 0.0 else None)
            for future in iterator:
                try:
                    lanes.append(future.result())
                except Exception as exc:
                    lanes.append(StageAFitLane(int(futures[future]), "error", error=f"{type(exc).__name__}: {exc}"))
        except TimeoutError:
            completed = {lane.slice_id for lane in lanes}
            for future, sid in futures.items():
                if sid not in completed and not future.done():
                    lanes.append(StageAFitLane(int(sid), "timeout", error="fit worker timeout"))
            _terminate_pool(pool)
            pool = None
        finally:
            if pool is not None:
                pool.shutdown(wait=True, cancel_futures=True)

    _apply_lane_train_loss_cap(lanes, max_lane_train_loss)

    min_valid = float(getattr(lm_hp, "coe_min_valid_fraction", 0.8) or 0.8)
    for lane in lanes:
        if lane.status != "success":
            continue
        if (
            lane.full_parameters is None
            or lane.parameter_layout_fingerprint != template_layout_fp
        ):
            lane.status = "incompatible"
            lane.accepted = False
            lane.error = (
                "full TSOP parameter layout differs from frozen template "
                f"({lane.parameter_layout_fingerprint} != {template_layout_fp})"
            )
            continue
        try:
            if lane.slice_id == master_slice:
                lane.row_losses, lane.finite_fraction = _row_losses(
                    model, compare_loader, torch.device("cpu"), y_op_inv
                )
            else:
                if lane.comparison_predictions is None:
                    raise ValueError("worker returned no comparison predictions")
                lane.row_losses, lane.finite_fraction = _prediction_row_losses(
                    lane.comparison_predictions, val_y[split:], y_op_inv
                )
            finite = np.asarray(lane.row_losses, dtype=float)
            lane.comparison_mse = (
                float(np.nanmean(finite))
                if lane.finite_fraction >= min_valid
                else float("inf")
            )
        except Exception as exc:
            lane.status = "comparison_error"
            lane.error = f"{type(exc).__name__}: {exc}"
            lane.row_losses = None
            lane.comparison_mse = float("inf")

    comparison_setup_error = None
    target_scale = 1.0
    try:
        target_rows = []
        for _x, target in compare_loader:
            target = target.reshape(-1)
            if y_op_inv is not None:
                target = y_op_inv(target).reshape(-1)
            target_rows.append(target.double().numpy())
        target_scale = (
            float(np.mean(np.concatenate(target_rows) ** 2))
            if target_rows
            else 1.0
        )
    except Exception as exc:
        comparison_setup_error = f"{type(exc).__name__}: {exc}"
    unit_start = master_slice * (ntrain + nval) + ntrain + split
    master_lane = lanes[0]
    fatal_master_error = None
    fatal_master_cause = None
    if comparison_setup_error is not None:
        selected = master_lane
        decision = {
            "decision": "keep_master",
            "selected_slice": master_slice,
            "reason": f"common comparison failed soft: {comparison_setup_error}",
        }
        if master_error is not None:
            fatal_master_error = master_error
    elif master_error is not None:
        try:
            selected, decision = choose_stageA_fit_lane(
                lanes, master_slice=master_slice,
                alpha=float(getattr(lm_hp, "coe_stageA_fit_alpha", 0.05) or 0.05),
                min_rel_improvement=float(getattr(lm_hp, "coe_stageA_fit_min_rel_improvement", 0.01) or 0.01),
                noise_floor=float(getattr(lm_hp, "coe_noise_floor_raw", 0.0) or 0.0),
                target_scale=target_scale,
                unit_keys=np.arange(unit_start, unit_start + n_compare),
                seed=int(getattr(lm_hp, "coe_maxt_seed", 0) or 0) + 7919,
            )
        except Exception as exc:
            selected = master_lane
            decision = {
                "decision": "keep_master",
                "selected_slice": master_slice,
                "reason": f"invalid-master fallback failed: {type(exc).__name__}: {exc}",
            }
        if selected is master_lane:
            fatal_master_error = master_error
    elif master_lane.row_losses is None:
        selected = master_lane
        decision = {
            "decision": "keep_master",
            "selected_slice": master_slice,
            "reason": f"common comparison failed soft: {master_lane.error or 'master comparison unavailable'}",
        }
    else:
        try:
            selected, decision = choose_stageA_fit_lane(
                lanes, master_slice=master_slice,
                alpha=float(getattr(lm_hp, "coe_stageA_fit_alpha", 0.05) or 0.05),
                min_rel_improvement=float(getattr(lm_hp, "coe_stageA_fit_min_rel_improvement", 0.01) or 0.01),
                noise_floor=float(getattr(lm_hp, "coe_noise_floor_raw", 0.0) or 0.0),
                target_scale=target_scale,
                unit_keys=np.arange(unit_start, unit_start + n_compare),
                seed=int(getattr(lm_hp, "coe_maxt_seed", 0) or 0) + 7919,
            )
        except Exception as exc:
            selected = master_lane
            decision = {
                "decision": "keep_master", "selected_slice": master_slice,
                "reason": f"fit selection failed soft: {type(exc).__name__}: {exc}",
            }
    decision.update({
        "enabled": True,
        "fit_kind": fit_kind,
        "template_fingerprint": template_fp,
        "total_fit_parallelism": int(total_fit_parallelism),
        "fit_validation_rows": [
            master_slice * (ntrain + nval) + ntrain,
            master_slice * (ntrain + nval) + ntrain + split,
        ],
        "comparison_rows": [unit_start, unit_start + n_compare],
        "lanes": [lane.summary() for lane in sorted(lanes, key=lambda row: row.slice_id)],
    })
    selected_opt = master_opt
    if selected.slice_id != master_slice and selected.full_parameters is not None:
        master_full_parameters = master_lane.full_parameters
        try:
            if master_opt is not None:
                _load_full_parameter_snapshot(model, selected.full_parameters)
                _verify_worker_lane_predictions(model, selected, val_x[split:])
            else:
                recovered_model = _rebuild_model(model_recipe)
                _load_full_parameter_snapshot(
                    recovered_model, selected.full_parameters
                )
                _verify_worker_lane_predictions(
                    recovered_model, selected, val_x[split:]
                )
                adopted = copy.deepcopy(recovered_model.__dict__)
                model.__dict__.clear()
                model.__dict__.update(adopted)
        except Exception as exc:
            if master_error is not None:
                fatal_master_error = master_error
                fatal_master_cause = exc
            elif master_full_parameters is not None:
                _load_full_parameter_snapshot(model, master_full_parameters)
            decision["decision"] = "keep_master"
            decision["selected_slice"] = int(master_slice)
            decision["reason"] = (
                "selected worker state could not be adopted: "
                f"{type(exc).__name__}: {exc}"
            )
            selected = master_lane
        else:
            selected_opt = _AppliedStateOptimizer()
    records = list(getattr(lm_hp, "coe_stageA_fit_tournament_records", []) or [])
    records.append(decision)
    lm_hp.coe_stageA_fit_tournament_records = records
    if fatal_master_error is not None:
        if fatal_master_cause is not None:
            raise fatal_master_error from fatal_master_cause
        raise fatal_master_error
    print(
        "[CoE StageA fit tournament] "
        f"kind={fit_kind} lanes={len(lanes)} decision={decision['decision']} "
        f"selected_slice={selected.slice_id} "
        f"master_mse={lanes[0].comparison_mse:.4e} selected_mse={selected.comparison_mse:.4e}"
    )
    return (
        bool(selected.accepted), float(selected.fit_val_loss), float(selected.train_loss),
        selected.best_params, selected_opt,
    )


def fit_initial_model_with_tournament(
    model,
    train_dl,
    val_dl,
    *,
    y_op=None,
    y_op_inv=None,
    lm_hp,
    device,
    **fit_kwargs,
):
    """Initial-teacher counterpart returning the ordinary four-value API."""
    _accepted, val_loss, train_loss, params, opt = fit_stageA_candidate_with_tournament(
        model,
        train_dl,
        val_dl,
        y_op=y_op,
        y_op_inv=y_op_inv,
        lm_hp=lm_hp,
        device=device,
        accept_threshold=float("inf"),
        _initial_fit=True,
        **fit_kwargs,
    )
    return val_loss, train_loss, params, opt


__all__ = [
    "StageAFitLane",
    "choose_stageA_fit_lane",
    "fit_stageA_candidate_with_tournament",
    "fit_initial_model_with_tournament",
    "validate_stageA_fit_slice_firewall",
]
