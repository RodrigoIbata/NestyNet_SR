# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Execution helpers for CoE witness evaluations.

The first implementation is intentionally behavior-preserving: a serial
executor that gives Stage-A and Stage-B committee code one common scheduling
boundary.  Later subprocess backends should keep the same row contract and
leave gate/vote logic in the reference process.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Sequence


_WORKER_THREADPOOL_LIMITER: Any = None


def _witness_job_family(jobs: Sequence["CoEWitnessJob"], *, fallback: str = "witness") -> str:
    if not jobs:
        return str(fallback)
    try:
        job_id = str(jobs[0].job_id)
        return job_id.split(":", 1)[0] or str(fallback)
    except Exception:
        return str(fallback)


def _log_witness_execution(
    *,
    family: str,
    jobs: Sequence["CoEWitnessJob"],
    executor: "CoEWitnessExecutor",
    backend: str,
    workers: int,
    note: Optional[str] = None,
) -> None:
    """Emit one stable, grep-friendly line per CoE witness batch."""

    try:
        slice_count = len({int(job.slice_id) for job in jobs})
    except Exception:
        slice_count = 0
    msg = (
        f"[CoE witnesses] {family}: backend={backend}, "
        f"requested_parallelism={int(getattr(executor, 'parallelism', 1) or 1)}, "
        f"workers={max(1, int(workers or 1))}, jobs={len(jobs)}, slices={slice_count}"
    )
    if note:
        msg += f", note={note}"
    print(msg)


@dataclass(frozen=True)
class CoEWitnessJob:
    """A single witness task submitted by a Stage-A/Stage-B gate."""

    job_id: str
    slice_id: int
    payload: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


class CoEWitnessExecutor:
    """Run CoE witness jobs through a stable scheduling API.

    ``parallelism`` is stored for reporting and for portable helper backends.
    The generic ``run`` method is intentionally serial because most existing
    call sites close over live model/search state.  Portable helpers may use
    ``parallelism`` when their payloads are process-safe.
    """

    def __init__(self, *, parallelism: int = 1, backend: str = "serial") -> None:
        try:
            self.parallelism = max(1, int(parallelism))
        except Exception:
            self.parallelism = 1
        self.backend = str(backend or "serial")
        if self.backend != "serial":
            raise ValueError(f"unsupported CoE witness executor backend: {self.backend}")

    @classmethod
    def from_config(cls, owner: Any, *, attr: str = "coe_witness_parallelism") -> "CoEWitnessExecutor":
        return cls(parallelism=getattr(owner, attr, 1) or 1, backend="serial")

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "parallelism": int(self.parallelism),
        }

    def run(
        self,
        jobs: Iterable[CoEWitnessJob],
        worker: Callable[[CoEWitnessJob], dict[str, Any]],
        *,
        stop_after: Optional[Callable[[list[dict[str, Any]]], bool]] = None,
    ) -> list[dict[str, Any]]:
        jobs = list(jobs)
        _log_witness_execution(
            family=_witness_job_family(jobs),
            jobs=jobs,
            executor=self,
            backend="serial",
            workers=1,
            note="generic_serial",
        )
        rows: list[dict[str, Any]] = []
        for job in jobs:
            row = worker(job)
            if not isinstance(row, dict):
                row = {
                    "status": "error",
                    "slice_id": int(job.slice_id),
                    "error": f"witness worker returned {type(row).__name__}, expected dict",
                }
            row.setdefault("slice_id", int(job.slice_id))
            row.setdefault("job_id", str(job.job_id))
            row.setdefault("executor_backend", self.backend)
            rows.append(row)
            if stop_after is not None and bool(stop_after(rows)):
                break
        return rows


def _normalize_witness_row(row: Any, job: CoEWitnessJob, *, backend: str) -> dict[str, Any]:
    if not isinstance(row, dict):
        row = {
            "status": "error",
            "slice_id": int(job.slice_id),
            "error": f"witness worker returned {type(row).__name__}, expected dict",
        }
    row.setdefault("slice_id", int(job.slice_id))
    row.setdefault("job_id", str(job.job_id))
    row.setdefault("executor_backend", str(backend))
    return row


def summarize_witness_errors(rows: Sequence[dict[str, Any]], *, limit: int = 3) -> str:
    """Return a compact summary of witness worker errors for logs."""

    parts: list[str] = []
    for row in rows:
        if not isinstance(row, dict) or row.get("status") != "error":
            continue
        sid = row.get("slice_id", "?")
        err = str(row.get("error") or "unknown error")
        parts.append(f"slice {sid}: {err}")
        if len(parts) >= max(1, int(limit)):
            break
    if not parts:
        return "no worker error details"
    error_count = sum(
        1
        for row in rows
        if isinstance(row, dict) and row.get("status") == "error"
    )
    more = error_count - len(parts)
    suffix = f"; ... {more} more" if more > 0 else ""
    return "; ".join(parts) + suffix


def coe_witness_execution_metadata(
    executor: CoEWitnessExecutor,
    rows: Optional[Sequence[dict[str, Any]]] = None,
    *,
    parallel_disabled_reason: Optional[str] = None,
) -> dict[str, Any]:
    meta = executor.metadata()
    backends = []
    for row in list(rows or ()):
        backend = row.get("executor_backend")
        if backend is not None and str(backend) not in backends:
            backends.append(str(backend))
    if backends:
        meta["effective_backend"] = backends[0] if len(backends) == 1 else backends
    if parallel_disabled_reason and int(getattr(executor, "parallelism", 1) or 1) > 1:
        meta["parallel_disabled_reason"] = str(parallel_disabled_reason)
    return meta


def run_threaded_witnesses(
    jobs: Sequence[CoEWitnessJob],
    worker: Callable[[CoEWitnessJob], dict[str, Any]],
    *,
    executor: CoEWitnessExecutor,
    stop_after: Optional[Callable[[list[dict[str, Any]]], bool]] = None,
) -> list[dict[str, Any]]:
    """Run side-effect-free live-model witness jobs in a thread pool.

    This helper is for read-only evaluation workers that close over live model
    objects. It deliberately runs in batches so output ordering is stable and
    early-stop checks remain deterministic up to one batch of over-evaluation.
    Mutating refit gates should keep using ``CoEWitnessExecutor.run``.
    """

    jobs = list(jobs)
    if int(executor.parallelism) <= 1 or len(jobs) <= 1:
        return executor.run(jobs, worker, stop_after=stop_after)

    rows: list[dict[str, Any]] = []
    max_workers = max(1, int(executor.parallelism))
    _log_witness_execution(
        family=_witness_job_family(jobs),
        jobs=jobs,
        executor=executor,
        backend="thread",
        workers=max_workers,
        note="read_only_live_model",
    )
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for start in range(0, len(jobs), max_workers):
            batch = jobs[start : start + max_workers]
            futures = [pool.submit(worker, job) for job in batch]
            for job, fut in zip(batch, futures):
                try:
                    row = fut.result()
                except Exception as exc:
                    row = {
                        "status": "error",
                        "slice_id": int(job.slice_id),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                rows.append(_normalize_witness_row(row, job, backend="thread"))
            if stop_after is not None and bool(stop_after(rows)):
                break
    return rows


def coe_witness_jobs_from_specs(
    specs: Iterable[Any],
    *,
    prefix: str,
    payload: Any = None,
) -> list[CoEWitnessJob]:
    jobs: list[CoEWitnessJob] = []
    for spec in specs:
        slice_id = int(getattr(spec, "slice_id"))
        jobs.append(
            CoEWitnessJob(
                job_id=f"{prefix}:{slice_id}",
                slice_id=slice_id,
                payload=spec if payload is None else payload,
                metadata={
                    "train_rows": [
                        int(getattr(spec, "train_start")),
                        int(getattr(spec, "train_stop")),
                    ],
                    "val_rows": [
                        int(getattr(spec, "val_start")),
                        int(getattr(spec, "val_stop")),
                    ],
                },
            )
        )
    return jobs


def coe_pair_vote(delta: float, tolerance: float) -> str:
    if float(delta) < -float(tolerance):
        return "win"
    if float(delta) > float(tolerance):
        return "loss"
    return "tie"


def coe_stageB_refit_ast_to_payload(node: Any) -> dict[str, Any]:
    """Convert an SR AST node into a plain recursive payload.

    Same-history refit workers must not receive a live ``StageBContext``.  This
    helper keeps the structural part portable; fitted leaf modules are handled
    separately through the reuse maps.
    """

    from nestynet_sr.sr_core.bridges import (
        AbsNode,
        AcosNode,
        AddNode,
        ArgNode,
        AsinNode,
        AtanNode,
        AtomNode,
        ConjNode,
        ConstNode,
        CosNode,
        ExpNode,
        ImagNode,
        LogNode,
        MulNode,
        PowNode,
        RealNode,
        SinNode,
    )

    def _value_to_payload(value: Any) -> Any:
        if isinstance(value, complex):
            return {"__complex__": [float(value.real), float(value.imag)]}
        if isinstance(value, tuple):
            return {"__tuple__": [_value_to_payload(v) for v in value]}
        if isinstance(value, list):
            return [_value_to_payload(v) for v in value]
        if isinstance(value, dict):
            return {str(k): _value_to_payload(v) for k, v in value.items()}
        if isinstance(value, (
            AtomNode,
            AddNode,
            MulNode,
            PowNode,
            LogNode,
            ExpNode,
            SinNode,
            CosNode,
            AsinNode,
            AcosNode,
            AtanNode,
            ConjNode,
            RealNode,
            ImagNode,
            AbsNode,
            ArgNode,
            ConstNode,
        )):
            return {"__ast__": coe_stageB_refit_ast_to_payload(value)}
        return value

    if isinstance(node, AtomNode):
        return {
            "type": "atom",
            "kind": str(node.kind),
            "var_idxs": [int(v) for v in node.var_idxs],
            "kwargs": _value_to_payload(dict(getattr(node, "kwargs", {}) or {})),
            "tag": None if getattr(node, "tag", None) is None else str(node.tag),
            "inputs": None
            if getattr(node, "inputs", None) is None
            else [coe_stageB_refit_ast_to_payload(v) for v in node.inputs],
            "scope": str(getattr(node, "scope", "experiment") or "experiment"),
        }
    if isinstance(node, AddNode):
        return {
            "type": "add",
            "left": coe_stageB_refit_ast_to_payload(node.left),
            "right": coe_stageB_refit_ast_to_payload(node.right),
        }
    if isinstance(node, MulNode):
        return {
            "type": "mul",
            "left": coe_stageB_refit_ast_to_payload(node.left),
            "right": coe_stageB_refit_ast_to_payload(node.right),
        }
    if isinstance(node, PowNode):
        return {
            "type": "pow",
            "base": coe_stageB_refit_ast_to_payload(node.base),
            "exponent": _value_to_payload(node.exponent),
        }
    unary_types = {
        LogNode: "log",
        ExpNode: "exp",
        SinNode: "sin",
        CosNode: "cos",
        AsinNode: "asin",
        AcosNode: "acos",
        AtanNode: "atan",
        ConjNode: "conj",
        RealNode: "real",
        ImagNode: "imag",
        AbsNode: "abs",
        ArgNode: "arg",
    }
    for cls, kind in unary_types.items():
        if isinstance(node, cls):
            return {"type": kind, "arg": coe_stageB_refit_ast_to_payload(node.arg)}
    if isinstance(node, ConstNode):
        return {"type": "const", "value": _value_to_payload(node.value)}
    raise TypeError(f"unsupported AST node for CoE refit payload: {type(node).__name__}")


def coe_stageB_refit_ast_from_payload(payload: Any) -> Any:
    """Rebuild an SR AST from ``coe_stageB_refit_ast_to_payload`` output."""

    from nestynet_sr.sr_core.bridges import (
        AbsNode,
        AcosNode,
        AddNode,
        ArgNode,
        AsinNode,
        AtanNode,
        AtomNode,
        ConjNode,
        ConstNode,
        CosNode,
        ExpNode,
        ImagNode,
        LogNode,
        MulNode,
        PowNode,
        RealNode,
        SinNode,
    )

    def _value_from_payload(value: Any) -> Any:
        if isinstance(value, dict):
            if "__complex__" in value:
                real, imag = value["__complex__"]
                return complex(float(real), float(imag))
            if "__tuple__" in value:
                return tuple(_value_from_payload(v) for v in value["__tuple__"])
            if "__ast__" in value:
                return coe_stageB_refit_ast_from_payload(value["__ast__"])
            return {k: _value_from_payload(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_value_from_payload(v) for v in value]
        return value

    row = dict(payload)
    typ = str(row.get("type", ""))
    if typ == "atom":
        inputs_raw = row.get("inputs", None)
        inputs = (
            None
            if inputs_raw is None
            else tuple(coe_stageB_refit_ast_from_payload(v) for v in inputs_raw)
        )
        node = AtomNode(
            kind=str(row.get("kind", "")),
            var_idxs=tuple(int(v) for v in row.get("var_idxs", ())),
            kwargs=dict(_value_from_payload(row.get("kwargs", {})) or {}),
            tag=row.get("tag", None),
            inputs=inputs,
        )
        node.scope = str(row.get("scope", "experiment") or "experiment")
        return node
    if typ == "add":
        return AddNode(
            coe_stageB_refit_ast_from_payload(row["left"]),
            coe_stageB_refit_ast_from_payload(row["right"]),
        )
    if typ == "mul":
        return MulNode(
            coe_stageB_refit_ast_from_payload(row["left"]),
            coe_stageB_refit_ast_from_payload(row["right"]),
        )
    if typ == "pow":
        return PowNode(
            coe_stageB_refit_ast_from_payload(row["base"]),
            _value_from_payload(row["exponent"]),
        )
    unary_types = {
        "log": LogNode,
        "exp": ExpNode,
        "sin": SinNode,
        "cos": CosNode,
        "asin": AsinNode,
        "acos": AcosNode,
        "atan": AtanNode,
        "conj": ConjNode,
        "real": RealNode,
        "imag": ImagNode,
        "abs": AbsNode,
        "arg": ArgNode,
    }
    if typ in unary_types:
        return unary_types[typ](coe_stageB_refit_ast_from_payload(row["arg"]))
    if typ == "const":
        return ConstNode(_value_from_payload(row.get("value", 0.0)))
    raise ValueError(f"unsupported CoE refit AST payload type: {typ!r}")


def _thread_cap_initializer() -> None:
    global _WORKER_THREADPOOL_LIMITER
    try:
        worker_threads = max(1, int(os.environ.get("COE_WORKER_THREADS", "1")))
    except (TypeError, ValueError):
        worker_threads = 1
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        # Assignment is intentional: setdefault would retain the primary
        # process's larger thread budget in forked/spawned witnesses.
        os.environ[key] = str(worker_threads)
    try:
        import torch

        torch.set_num_threads(worker_threads)
        try:
            torch.set_num_interop_threads(worker_threads)
        except RuntimeError:
            # PyTorch only permits setting inter-op threads once and before
            # parallel work starts. Intra-op is the important worker cap.
            pass
    except Exception:
        pass
    if _WORKER_THREADPOOL_LIMITER is None:
        try:
            from threadpoolctl import threadpool_limits

            # Environment variables are too late for BLAS/OpenMP libraries already
            # initialized before a Linux fork. Keep the controller alive for the
            # lifetime of the witness process so their live pools remain capped.
            _WORKER_THREADPOOL_LIMITER = threadpool_limits(limits=worker_threads)
        except Exception:
            _WORKER_THREADPOOL_LIMITER = None
    _set_torch_mp_sharing_strategy()


def _set_torch_mp_sharing_strategy() -> None:
    """Use a PyTorch multiprocessing strategy that is friendlier to low ulimits.

    The default Linux ``file_descriptor`` strategy can consume one descriptor per
    tensor storage when pickled model payloads are sent to many witness workers.
    Some cluster nodes run with a low ``ulimit -n``; ``file_system`` avoids the
    fd-backed handoff and lets the process backend remain usable there.
    """

    strategy = os.environ.get("COE_TORCH_MP_SHARING_STRATEGY", "file_system")
    if not strategy:
        return
    try:
        import torch.multiprocessing as torch_mp

        current = torch_mp.get_sharing_strategy()
        if current != strategy:
            torch_mp.set_sharing_strategy(strategy)
    except Exception:
        # Best-effort only. If the runtime does not support the strategy, the
        # worker code still falls back to serial refit on process errors.
        return


def _spec_payload(spec: Any) -> dict[str, int]:
    if hasattr(spec, "to_dict"):
        row = dict(spec.to_dict())
    else:
        row = dict(spec)
    return {
        "slice_id": int(row["slice_id"]),
        "train_start": int(row["train_start"]),
        "train_stop": int(row["train_stop"]),
        "val_start": int(row["val_start"]),
        "val_stop": int(row["val_stop"]),
    }


def _artifact_payload(artifact: Any) -> dict[str, Any]:
    if hasattr(artifact, "to_dict"):
        return dict(artifact.to_dict())
    return dict(artifact)


def _dtype_from_name(name: Any):
    import torch

    text = str(name or "float64").replace("torch.", "")
    return {
        "float64": torch.float64,
        "double": torch.float64,
        "float32": torch.float32,
        "float": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(text, torch.float64)


def stageB_refit_row_losses(
    model: Any,
    val_loader: Any,
    device: Any,
    *,
    y_op_inv: Any = None,
) -> list[float]:
    """Per-row squared validation errors for one refitted model, NaN where
    non-finite.

    The comparison space mirrors the refit gate's scalar comparison: raw-y
    (both prediction and target inverse-transformed) when ``y_op_inv`` is
    given, model-output space otherwise.  Rows from a single refit share that
    refit's fit noise, so the paired max-T observer keys their bootstrap
    multipliers by slice (block bootstrap) rather than by row.
    """
    import numpy as _np
    import torch

    model.eval()
    parts: list[Any] = []
    with torch.no_grad():
        for batch in val_loader:
            x, target = batch[0].to(device), batch[1].to(device)
            pred = model(x)
            pred = pred[:, 0] if pred.dim() == 2 and pred.shape[1] == 1 else pred.view(-1)
            target = (
                target[:, 0]
                if target.dim() == 2 and target.shape[1] == 1
                else target.view(-1)
            )
            if y_op_inv is not None:
                pred = y_op_inv(pred).view(-1)
                target = y_op_inv(target).view(-1)
            finite = (torch.isfinite(pred) & torch.isfinite(target)).cpu().numpy()
            diff = (pred - target).double().cpu().numpy()
            parts.append(_np.where(finite, diff * diff, _np.nan))
    if not parts:
        return []
    return [float(v) for v in _np.concatenate(parts)]


def _stageB_refit_pair_worker(payload: dict[str, Any]) -> dict[str, Any]:
    spec_row = _spec_payload(payload["spec"])
    row = {
        "method": "refit_compare",
        "refit_tier": str(payload.get("refit_tier", "tier0")),
        "slice_id": int(spec_row["slice_id"]),
        "train_rows": [int(spec_row["train_start"]), int(spec_row["train_stop"])],
        "val_rows": [int(spec_row["val_start"]), int(spec_row["val_stop"])],
        "epochs": int(payload.get("epochs", 1) or 1),
        "status": "error",
    }
    try:
        import math

        import numpy as _np
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        from nestynet_sr.sr_search.coe_committee import _load_dataset_arrays
        from nestynet_sr.sr_search.stageB.evaluation import (
            _eval_original_y_mse_with_inverse,
            _eval_val_mse,
        )
        from nestynet_sr.sr_search.stageB.fitting import _fit_candidate_root
        from nestynet_sr.sr_search.y_transforms import resolve_y_transform_name

        X_all, y_all, _cols = _load_dataset_arrays(str(payload["filepath"]))
        dtype = _dtype_from_name(payload.get("dtype", "float64"))
        device = torch.device(str(payload.get("device", "cpu") or "cpu"))
        if device.type != "cpu" and bool(payload.get("force_cpu", True)):
            device = torch.device("cpu")
        y_name = str(payload.get("y_transform_name", "identity") or "identity")
        y_op = None
        y_op_inv = None
        if y_name and y_name != "identity":
            y_op, y_op_inv, _ = resolve_y_transform_name(y_name)
            if y_op is None or y_op_inv is None:
                raise ValueError(f"y-transform {y_name!r} is not replayable")

        xcoords = payload.get("xcoords", None)
        xcoords_active = bool(payload.get("xcoords_active", False) or xcoords is not None)
        if xcoords_active and xcoords is None:
            raise ValueError("active x-coordinate transform has no replay object")

        def _slice_loader(start: int, stop: int):
            if start < 0 or stop <= start or stop > int(y_all.shape[0]):
                raise ValueError(
                    f"slice rows [{start}, {stop}) outside dataset with {int(y_all.shape[0])} rows"
                )
            x_slice = _np.array(X_all[start:stop], dtype=_np.float64, copy=True)
            if xcoords_active:
                x_slice = _np.array(
                    xcoords.apply_np(x_slice),
                    dtype=_np.float64,
                    copy=True,
                )
                if not _np.all(_np.isfinite(x_slice)):
                    raise ValueError(
                        f"non-finite x-coordinate transform values on rows [{start}, {stop})"
                    )
            y_slice = _np.array(y_all[start:stop], dtype=_np.float64, copy=True).reshape(-1, 1)
            if y_op is not None:
                y_slice = _np.array(y_op(y_slice), dtype=_np.float64, copy=True).reshape(-1, 1)
                if not _np.all(_np.isfinite(y_slice)):
                    raise ValueError(f"non-finite y-transform target values on rows [{start}, {stop})")
            xb = torch.as_tensor(x_slice, dtype=dtype)
            yb = torch.as_tensor(y_slice, dtype=dtype)
            batch_size = int(payload.get("batch_size", 0) or xb.shape[0])
            batch_size = max(1, min(batch_size, int(xb.shape[0])))
            return DataLoader(
                TensorDataset(xb, yb),
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
            )

        train_loader = _slice_loader(int(spec_row["train_start"]), int(spec_row["train_stop"]))
        val_loader = _slice_loader(int(spec_row["val_start"]), int(spec_row["val_stop"]))
        lm_hp = payload["lm_hp"]
        parent_root = coe_stageB_refit_ast_from_payload(payload["incumbent_root"])
        candidate_root = coe_stageB_refit_ast_from_payload(payload["candidate_root"])

        seed = payload.get("seed", None)
        if seed is not None:
            try:
                torch.manual_seed(int(seed))
                _np.random.seed(int(seed) % (2**32 - 1))
            except Exception:
                pass
        incumbent_state = _fit_candidate_root(
            root=parent_root,
            reuse=dict(payload.get("incumbent_reuse", {}) or {}),
            train_loader=train_loader,
            val_loader=val_loader,
            lm_hp=lm_hp,
            device=device,
            dtype=dtype,
            epochs_stageB=int(payload.get("epochs", 1) or 1),
            loss_scale=float(payload.get("loss_scale", 1.0) or 1.0),
            trig_by_axis=payload.get("trig_by_axis", None),
            custom_init_fn=None,
            fresh_nn_factory=None,
            atom_factory=payload.get("atom_factory", None),
        )
        if seed is not None:
            try:
                torch.manual_seed(int(seed))
                _np.random.seed(int(seed) % (2**32 - 1))
            except Exception:
                pass
        candidate_state = _fit_candidate_root(
            root=candidate_root,
            reuse=dict(payload.get("candidate_reuse", {}) or {}),
            train_loader=train_loader,
            val_loader=val_loader,
            lm_hp=lm_hp,
            device=device,
            dtype=dtype,
            epochs_stageB=int(payload.get("epochs", 1) or 1),
            loss_scale=float(payload.get("loss_scale", 1.0) or 1.0),
            trig_by_axis=payload.get("trig_by_axis", None),
            custom_init_fn=None,
            fresh_nn_factory=None,
            atom_factory=payload.get("atom_factory", None),
        )
        fit_link_active = bool(getattr(lm_hp, "fit_y_link", None))
        y_transform_active = y_op_inv is not None
        compare_original_y = bool(fit_link_active or y_transform_active)
        inc_compare_loss = float(incumbent_state.val_loss)
        cand_compare_loss = float(candidate_state.val_loss)
        inc_raw_y_mse = float("nan")
        cand_raw_y_mse = float("nan")
        comparison_space = "fit_space"
        if compare_original_y:
            if y_transform_active:
                inc_raw_y_mse = float(
                    _eval_original_y_mse_with_inverse(
                        incumbent_state.model,
                        val_loader,
                        device,
                        y_op_inv,
                    )
                )
                cand_raw_y_mse = float(
                    _eval_original_y_mse_with_inverse(
                        candidate_state.model,
                        val_loader,
                        device,
                        y_op_inv,
                    )
                )
            else:
                inc_raw_y_mse = float(_eval_val_mse(incumbent_state.model, val_loader, device))
                cand_raw_y_mse = float(_eval_val_mse(candidate_state.model, val_loader, device))
            if not (math.isfinite(inc_raw_y_mse) and math.isfinite(cand_raw_y_mse)):
                raise ValueError("non-finite raw-y MSE under transformed committee refit")
            inc_compare_loss = inc_raw_y_mse
            cand_compare_loss = cand_raw_y_mse
            comparison_space = "raw_y"

        row.update(
            {
                "status": "success",
                "n_train": int(spec_row["train_stop"] - spec_row["train_start"]),
                "n_val": int(spec_row["val_stop"] - spec_row["val_start"]),
                "incumbent_val_loss": float(incumbent_state.val_loss),
                "candidate_val_loss": float(candidate_state.val_loss),
                "incumbent_compare_loss": float(inc_compare_loss),
                "candidate_compare_loss": float(cand_compare_loss),
                "comparison_space": str(comparison_space),
                "x_coordinate_space": "internal_x" if xcoords_active else "raw_x",
                "x_transform_active": bool(xcoords_active),
                "fit_y_link": str(getattr(lm_hp, "fit_y_link", None))
                if fit_link_active
                else None,
                "y_transform_active": bool(y_transform_active),
            }
        )
        if compare_original_y:
            row.update(
                {
                    "incumbent_raw_y_mse": float(inc_raw_y_mse),
                    "candidate_raw_y_mse": float(cand_raw_y_mse),
                    "incumbent_fit_loss": float(incumbent_state.val_loss),
                    "candidate_fit_loss": float(candidate_state.val_loss),
                }
            )
        if bool(payload.get("return_row_losses", False)):
            row_inv = y_op_inv if y_transform_active else None
            row["incumbent_row_losses"] = stageB_refit_row_losses(
                incumbent_state.model, val_loader, device, y_op_inv=row_inv
            )
            row["candidate_row_losses"] = stageB_refit_row_losses(
                candidate_state.model, val_loader, device, y_op_inv=row_inv
            )
    except Exception as exc:
        row["error"] = (
            f"slice {int(spec_row.get('slice_id', -1))} "
            f"{row.get('refit_tier', 'tier0')} refit failed: "
            f"{type(exc).__name__}: {exc}"
        )
    return row


def _fixed_expression_pair_worker(payload: dict[str, Any]) -> dict[str, Any]:
    spec_row = _spec_payload(payload["spec"])
    row = {
        "method": "fixed_expression_compare",
        "slice_id": int(spec_row["slice_id"]),
        "train_rows": [int(spec_row["train_start"]), int(spec_row["train_stop"])],
        "val_rows": [int(spec_row["val_start"]), int(spec_row["val_stop"])],
        "status": "error",
    }
    try:
        from nestynet_sr.sr_search.coe_committee import (
            CandidateArtifact,
            SliceSpec,
            evaluate_candidate_on_slice,
        )

        spec = SliceSpec(**spec_row)
        inc_art = CandidateArtifact(**dict(payload["incumbent"]))
        cand_art = CandidateArtifact(**dict(payload["candidate"]))
        filepath = str(payload["filepath"])
        min_valid_fraction = float(payload.get("min_valid_fraction", 0.80) or 0.80)
        return_row_losses = bool(payload.get("return_row_losses", False))
        inc_res = evaluate_candidate_on_slice(
            inc_art,
            filepath=filepath,
            spec=spec,
            min_valid_fraction=min_valid_fraction,
            return_row_losses=return_row_losses,
        )
        cand_res = evaluate_candidate_on_slice(
            cand_art,
            filepath=filepath,
            spec=spec,
            min_valid_fraction=min_valid_fraction,
            return_row_losses=return_row_losses,
        )
        inc_row = inc_res.to_dict()
        cand_row = cand_res.to_dict()
        row.update(
            {
                "status": (
                    "success"
                    if inc_res.status == "success" and cand_res.status == "success"
                    else "error"
                ),
                "incumbent_result": inc_row,
                "candidate_result": cand_row,
                "results": [inc_row, cand_row],
            }
        )
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def _fixed_expression_candidate_worker(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        from nestynet_sr.sr_search.coe_committee import (
            CandidateArtifact,
            SliceSpec,
            evaluate_candidate_on_slice,
        )

        spec = SliceSpec(**_spec_payload(payload["spec"]))
        cand = CandidateArtifact(**dict(payload["candidate"]))
        res = evaluate_candidate_on_slice(
            cand,
            filepath=str(payload["filepath"]),
            spec=spec,
            min_valid_fraction=float(payload.get("min_valid_fraction", 0.80) or 0.80),
            return_row_losses=bool(payload.get("return_row_losses", False)),
        )
        return res.to_dict()
    except Exception as exc:
        try:
            spec_row = _spec_payload(payload.get("spec", {}))
            slice_id = int(spec_row.get("slice_id", -1))
        except Exception:
            slice_id = -1
        cand = dict(payload.get("candidate", {}) or {})
        return {
            "candidate_id": str(cand.get("candidate_id", "")),
            "slice_id": int(slice_id),
            "val_mse": float("inf"),
            "val_mse_se": float("inf"),
            "frac_valid": 0.0,
            "n_val": 0,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }


def run_fixed_expression_candidate_witnesses(
    *,
    specs: Sequence[Any],
    candidates: Sequence[Any],
    filepath: str,
    min_valid_fraction: float,
    executor: CoEWitnessExecutor,
    prefix: str,
    stop_after: Optional[Callable[[list[dict[str, Any]]], bool]] = None,
    return_row_losses: bool = False,
) -> list[dict[str, Any]]:
    """Evaluate fixed-expression candidates on witness slices.

    Output rows are ``SliceFitResult.to_dict()`` payloads plus executor
    metadata.  The result order is stable: slice order outer, candidate order
    inner, matching the historical final committee audit loop.

    ``return_row_losses=True`` adds per-row squared errors to each result so
    the paired-row committee inference can treat rows as the statistical units
    and slices as pure compute partitions.
    """

    jobs: list[CoEWitnessJob] = []
    for spec in specs:
        spec_payload = _spec_payload(spec)
        for cand in candidates:
            cand_payload = _artifact_payload(cand)
            slice_id = int(spec_payload["slice_id"])
            cand_id = str(cand_payload.get("candidate_id", "candidate"))
            jobs.append(
                CoEWitnessJob(
                    job_id=f"{prefix}:{slice_id}:{cand_id}",
                    slice_id=slice_id,
                    payload={
                        "spec": spec_payload,
                        "candidate": cand_payload,
                        "filepath": str(filepath),
                        "min_valid_fraction": float(min_valid_fraction),
                        "return_row_losses": bool(return_row_losses),
                    },
                )
            )

    if int(executor.parallelism) <= 1:
        return executor.run(
            jobs,
            lambda job: _fixed_expression_candidate_worker(job.payload),
            stop_after=stop_after,
        )

    def _serial_fallback(reason: str) -> list[dict[str, Any]]:
        rows_i = executor.run(
            jobs,
            lambda job: _fixed_expression_candidate_worker(job.payload),
            stop_after=stop_after,
        )
        for row_i in rows_i:
            row_i["executor_fallback_reason"] = str(reason)
        return rows_i

    rows: list[dict[str, Any]] = []
    max_workers = max(1, int(executor.parallelism))
    _log_witness_execution(
        family=_witness_job_family(jobs),
        jobs=jobs,
        executor=executor,
        backend="process",
        workers=max_workers,
        note="fixed_expression_candidates",
    )
    try:
        _set_torch_mp_sharing_strategy()
        with ProcessPoolExecutor(max_workers=max_workers, initializer=_thread_cap_initializer) as pool:
            for start in range(0, len(jobs), max_workers):
                batch = jobs[start : start + max_workers]
                futures = [pool.submit(_fixed_expression_candidate_worker, job.payload) for job in batch]
                for job, fut in zip(batch, futures):
                    try:
                        row = fut.result()
                    except Exception as exc:
                        row = {
                            "candidate_id": "",
                            "slice_id": int(job.slice_id),
                            "val_mse": float("inf"),
                            "val_mse_se": float("inf"),
                            "frac_valid": 0.0,
                            "n_val": 0,
                            "status": "error",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    rows.append(_normalize_witness_row(row, job, backend="process"))
                if stop_after is not None and bool(stop_after(rows)):
                    break
    except Exception as exc:
        return _serial_fallback(f"{type(exc).__name__}: {exc}")
    return rows


def run_stageB_refit_pair_witness_preflight(
    *,
    payload: dict[str, Any],
    prefix: str,
) -> dict[str, Any]:
    """Run one portable Stage-B refit witness in-process before multiprocessing."""

    spec_payload = _spec_payload(payload["spec"])
    slice_id = int(spec_payload["slice_id"])
    job = CoEWitnessJob(
        job_id=f"{prefix}:{slice_id}:preflight",
        slice_id=slice_id,
        payload=dict(payload),
    )
    row = _stageB_refit_pair_worker(job.payload)
    return _normalize_witness_row(row, job, backend="portable_serial")


def run_fixed_expression_pair_witnesses(
    *,
    specs: Sequence[Any],
    incumbent: Any,
    candidate: Any,
    filepath: str,
    min_valid_fraction: float,
    executor: CoEWitnessExecutor,
    prefix: str,
    stop_after: Optional[Callable[[list[dict[str, Any]]], bool]] = None,
    return_row_losses: bool = False,
) -> list[dict[str, Any]]:
    """Evaluate fixed-expression incumbent/candidate pairs on witness slices.

    This helper is process-safe: payloads are plain dicts and workers import the
    fixed-expression evaluator in the child process.  Early-stop checks happen
    after each completed batch, so a parallel run may evaluate up to
    ``parallelism - 1`` extra slices compared with strict serial execution.
    """

    jobs: list[CoEWitnessJob] = []
    inc_payload = _artifact_payload(incumbent)
    cand_payload = _artifact_payload(candidate)
    for spec in specs:
        spec_payload = _spec_payload(spec)
        jobs.append(
            CoEWitnessJob(
                job_id=f"{prefix}:{int(spec_payload['slice_id'])}",
                slice_id=int(spec_payload["slice_id"]),
                payload={
                    "spec": spec_payload,
                    "incumbent": inc_payload,
                    "candidate": cand_payload,
                    "filepath": str(filepath),
                    "min_valid_fraction": float(min_valid_fraction),
                    "return_row_losses": bool(return_row_losses),
                },
            )
        )

    if int(executor.parallelism) <= 1:
        return executor.run(jobs, lambda job: _fixed_expression_pair_worker(job.payload), stop_after=stop_after)

    def _serial_fallback(reason: str) -> list[dict[str, Any]]:
        rows_i = executor.run(
            jobs,
            lambda job: _fixed_expression_pair_worker(job.payload),
            stop_after=stop_after,
        )
        for row_i in rows_i:
            row_i["executor_fallback_reason"] = str(reason)
        return rows_i

    rows: list[dict[str, Any]] = []
    max_workers = max(1, int(executor.parallelism))
    _log_witness_execution(
        family=_witness_job_family(jobs),
        jobs=jobs,
        executor=executor,
        backend="process",
        workers=max_workers,
        note="fixed_expression_pair",
    )
    try:
        _set_torch_mp_sharing_strategy()
        with ProcessPoolExecutor(max_workers=max_workers, initializer=_thread_cap_initializer) as pool:
            for start in range(0, len(jobs), max_workers):
                batch = jobs[start : start + max_workers]
                futures = [pool.submit(_fixed_expression_pair_worker, job.payload) for job in batch]
                for job, fut in zip(batch, futures):
                    try:
                        row = fut.result()
                    except Exception as exc:
                        row = {
                            "status": "error",
                            "slice_id": int(job.slice_id),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    if not isinstance(row, dict):
                        row = {
                            "status": "error",
                            "slice_id": int(job.slice_id),
                            "error": (
                                "fixed-expression worker returned "
                                f"{type(row).__name__}, expected dict"
                            ),
                        }
                    row.setdefault("slice_id", int(job.slice_id))
                    row.setdefault("job_id", str(job.job_id))
                    row.setdefault("executor_backend", "process")
                    rows.append(row)
                if stop_after is not None and bool(stop_after(rows)):
                    break
    except Exception as exc:
        return _serial_fallback(f"{type(exc).__name__}: {exc}")
    return rows


def run_stageB_refit_pair_witnesses(
    *,
    payloads: Sequence[dict[str, Any]],
    executor: CoEWitnessExecutor,
    prefix: str,
    stop_after: Optional[Callable[[list[dict[str, Any]]], bool]] = None,
) -> list[dict[str, Any]]:
    """Run portable Stage-B same-history refit witness payloads.

    This is the multiprocessing-ready boundary for the expensive CoE refit
    gate.  Payloads are intentionally explicit: workers rebuild the two
    candidate models from AST payloads and reuse maps, train them on one slice,
    and return row evidence only.
    """

    jobs: list[CoEWitnessJob] = []
    for payload in payloads:
        spec_payload = _spec_payload(payload["spec"])
        slice_id = int(spec_payload["slice_id"])
        jobs.append(
            CoEWitnessJob(
                job_id=f"{prefix}:{slice_id}",
                slice_id=slice_id,
                payload=dict(payload),
            )
        )

    if int(executor.parallelism) <= 1:
        return executor.run(
            jobs,
            lambda job: _stageB_refit_pair_worker(job.payload),
            stop_after=stop_after,
        )

    def _serial_fallback(reason: str) -> list[dict[str, Any]]:
        rows_i = executor.run(
            jobs,
            lambda job: _stageB_refit_pair_worker(job.payload),
            stop_after=stop_after,
        )
        for row_i in rows_i:
            row_i["executor_fallback_reason"] = str(reason)
        return rows_i

    rows: list[dict[str, Any]] = []
    max_workers = max(1, int(executor.parallelism))
    _log_witness_execution(
        family=_witness_job_family(jobs),
        jobs=jobs,
        executor=executor,
        backend="process",
        workers=max_workers,
        note="stageB_same_history_refit",
    )
    try:
        _set_torch_mp_sharing_strategy()
        with ProcessPoolExecutor(max_workers=max_workers, initializer=_thread_cap_initializer) as pool:
            for start in range(0, len(jobs), max_workers):
                batch = jobs[start : start + max_workers]
                futures = [pool.submit(_stageB_refit_pair_worker, job.payload) for job in batch]
                for job, fut in zip(batch, futures):
                    try:
                        row = fut.result()
                    except Exception as exc:
                        row = {
                            "status": "error",
                            "slice_id": int(job.slice_id),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    rows.append(_normalize_witness_row(row, job, backend="process"))
                if stop_after is not None and bool(stop_after(rows)):
                    break
    except Exception as exc:
        return _serial_fallback(f"{type(exc).__name__}: {exc}")
    return rows
