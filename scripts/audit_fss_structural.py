# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Structural-recovery audit of the factorized-search oracle benchmark.

Applies the SAME criterion as the noisy full-pipeline audit (see
``_structural_verdict.py``): a case counts as solved when the recovered
expression is algebraically identical to the target up to the values of its
fitted constants, decided by refitting those constants on the canonical
noiseless data, re-snapping them with the pipeline's own polisher, and
requiring the noiseless-fit floor.  No symbolic-equivalence judgment and no
hand-chosen MSE gate is involved.

The oracle runs record only the skeleton and the mapping FAMILY, not the
fitted mapping coefficients, so the candidate is first reconstructed: parse
the recorded skeleton, refit the outer mapping on a dense sample of the
exact oracle target over the equation's own box, and convert the result to
an expression string.  That string is then handed to the shared verdict.

Every eligible case in the run is audited under this single criterion.

    python3 scripts/audit_fss_structural.py \
        [--run-dir results/factorized_search_aif_rerun20260811_gs_audit]
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
import sympy as sp
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _structural_verdict import (  # noqa: E402
    DEFAULT_TOL,
    find_noiseless_csv,
    load_problem_csv,
    structural_verdict,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from nestynet_sr.sr_search.factorized_search.aif_closure_benchmark import (  # noqa: E402
    parse_equations_txt,
)
from nestynet_sr.sr_search.factorized_search.expr_ast import eval_node  # noqa: E402
from nestynet_sr.sr_search.factorized_search.expr_mapping import (  # noqa: E402
    eval_mapping,
    fit_best,
)

# ---------------------------------------------------------------- parser

_UNARY = ("sin", "cos", "exp", "log", "sqrt", "sqr", "asin", "acos")


def parse_expr(s: str):
    """Invert expr_ast.node_str (fully parenthesised infix grammar)."""
    pos = 0

    def peek():
        return s[pos] if pos < len(s) else ""

    def parse():
        nonlocal pos
        if s.startswith("(", pos):
            pos += 1
            if (s.startswith("-", pos)
                    and pos + 1 < len(s)
                    and (s[pos + 1].isdigit() or s[pos + 1] == ".")):
                left = parse()  # negative numeric constant as left operand
            elif s.startswith("-", pos):
                pos += 1
                inner = parse()
                assert s[pos] == ")"
                pos += 1
                return ("neg", inner)
            else:
                left = parse()
            if s[pos] == ")":  # parenthesised single value, e.g. "(-2)"
                pos += 1
                return left
            op = {"+": "add", "-": "sub", "*": "mul", "/": "div"}[s[pos]]
            pos += 1
            right = parse()
            assert s[pos] == ")", (s, pos)
            pos += 1
            return (op, left, right)
        for name in _UNARY:
            if s.startswith(name + "(", pos):
                pos += len(name) + 1
                inner = parse()
                assert s[pos] == ")"
                pos += 1
                return (name, inner)
        if s.startswith("x", pos):
            pos += 1
            j = pos
            while j < len(s) and s[j].isdigit():
                j += 1
            idx = int(s[pos:j])
            pos = j
            return ("var", idx)
        # numeric constant (as printed by %g)
        j = pos
        while j < len(s) and (s[j].isdigit() or s[j] in ".eE+-"):
            # stop '+'/'-' from eating binary operators: only allow after e/E
            if s[j] in "+-" and j > pos and s[j - 1] not in "eE":
                break
            j += 1
        val = float(s[pos:j])
        pos = j
        return ("const", val)

    node = parse()
    assert pos == len(s), (s, pos)
    return node


def ast_to_sympy(node, xs):
    op = node[0]
    if op == "var":
        return xs[node[1]]
    if op == "const":
        return sp.Float(node[1])
    if op == "neg":
        return -ast_to_sympy(node[1], xs)
    if op in ("add", "sub", "mul", "div"):
        a, b = ast_to_sympy(node[1], xs), ast_to_sympy(node[2], xs)
        return {"add": a + b, "sub": a - b, "mul": a * b, "div": a / b}[op]
    if op == "sqr":
        return ast_to_sympy(node[1], xs) ** 2
    if op in ("sin", "cos", "exp", "log", "sqrt", "asin", "acos"):
        return getattr(sp, op)(ast_to_sympy(node[1], xs))
    raise ValueError(op)


def mapping_to_sympy(mapping, t):
    kind = mapping["kind"]
    if kind == "basis_state_native":
        return t
    if kind == "poly":
        z = (t - mapping["mu"]) / mapping["std"]
        out = sp.Integer(0)
        for k in range(len(mapping["coeffs"]) - 1, -1, -1):
            out = float(mapping["coeffs"][k]) + z * out
        return out
    if kind == "power":
        sgn_f = float(mapping.get("sgn_f", 1.0))
        sgn_y = float(mapping.get("sgn_y", 1.0))
        return sgn_y * sp.exp(float(mapping["log_a"])
                              + float(mapping["b"]) * sp.log(sgn_f * t))
    if kind == "pade":
        z = (t - mapping["mu"]) / mapping["std"]
        num = sum(float(c) * z ** k for k, c in enumerate(mapping["numer"]))
        den = sum(float(c) * z ** k for k, c in enumerate(mapping["denom"]))
        return num / den
    if kind == "sine":
        z = (t - mapping["mu"]) / mapping["std"]
        w = float(mapping["omega"])
        return (float(mapping["A"]) * sp.sin(w * z)
                + float(mapping["B"]) * sp.cos(w * z) + float(mapping["c"]))
    if kind == "exp":
        z = (t - mapping["mu"]) / mapping["std"]
        return (float(mapping["a"]) * sp.exp(float(mapping["b"]) * z)
                + float(mapping["c"]))
    raise ValueError(kind)


def snap_constants(expr, rel_tol=1e-4):
    """Replace Float atoms by nearby exact values when within rel_tol."""
    subs = {}
    for atom in expr.atoms(sp.Float):
        f = float(atom)
        tol = abs(f) * rel_tol + 1e-12
        for cand in (
            sp.nsimplify(atom, tolerance=tol, rational=True),
            sp.nsimplify(atom, rational=False,
                         constants=(sp.pi, sp.E, sp.sqrt(2), sp.sqrt(3)),
                         tolerance=tol),
        ):
            if cand != atom and abs(float(cand) - f) <= tol:
                subs[atom] = cand
                break
    return expr.xreplace(subs) if subs else expr


# ---------------------------------------------------------------- audit


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path,
                    default=REPO_ROOT / "results" / "factorized_search_aif_rerun20260811_gs_audit")
    ap.add_argument("--equations", type=Path,
                    default=REPO_ROOT / "data" / "equations.txt")
    ap.add_argument("--n-fit", type=int, default=4096,
                    help="Dense oracle samples used to reconstruct the mapping")
    ap.add_argument("--noiseless-data", type=Path,
                    default=REPO_ROOT.parent / "SRBench_0.000" / "data",
                    help="Directory of canonical noiseless pb*_data.csv files")
    ap.add_argument("--tol", type=float, default=DEFAULT_TOL)
    ap.add_argument("--only", type=str, default=None, help="Comma-separated ids")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    specs = {s["id"]: s for s in parse_equations_txt(args.equations)}
    torch.set_default_dtype(torch.float64)
    rng = np.random.default_rng(args.seed)

    def num(v):
        return float(v) if v is not None else math.inf

    rows = {}
    for f in sorted(glob.glob(str(args.run_dir / "feynman_*.json"))):
        j = json.load(open(f))
        for r in j.get("results", []):
            if r.get("status") != "skipped":
                rows[r["id"]] = r
    def _pid(key: str) -> str:
        """'feynman_042' -> '042' (the id shared with the noiseless CSVs)."""
        return str(key).rsplit("_", 1)[-1]

    wanted = None if not args.only else {t.strip().rsplit("_", 1)[-1]
                                         for t in args.only.split(",")}
    if wanted is not None:
        rows = {k: r for k, r in rows.items() if _pid(k) in wanted}
    solved = rows
    print(f"auditing all {len(solved)} cases from {args.run_dir.name}\n")

    classes = {}
    details = {}
    for k in sorted(solved):
        r = solved[k]
        spec = specs[k]
        nv = len(spec["variables"])
        lo = np.array([v["bounds"][0] for v in spec["variables"]])
        hi = np.array([v["bounds"][1] for v in spec["variables"]])
        xs = sp.symbols(f"x0:{nv}")
        target = sp.sympify(r["target"], locals={f"x{i}": xs[i] for i in range(nv)})
        tf = sp.lambdify(xs, target, "numpy")

        try:
            ast = parse_expr(r["expr"])
        except Exception as exc:
            classes[k] = "parse_error"
            details[k] = {"error": str(exc)}
            continue

        # dense in-box refit of the mapping
        X = rng.uniform(lo, hi, size=(args.n_fit, nv))
        y = np.asarray(tf(*[X[:, i] for i in range(nv)]), dtype=float)
        ok = np.isfinite(y)
        Xt = torch.tensor(X[ok])
        yt = torch.tensor(y[ok]).unsqueeze(-1)
        sk = eval_node(ast, Xt)
        if not torch.isfinite(sk).all():
            classes[k] = "skeleton_domain_error"
            details[k] = {}
            continue
        poly_deg = int((r.get("resolved_config") or {}).get("poly_degree", 4) or 4)
        fit = fit_best(sk, yt, poly_deg)
        if fit is None:
            classes[k] = "refit_failed"
            details[k] = {}
            continue
        _fit_mse, mapping = fit
        pred = eval_mapping(sk, mapping).reshape(-1)
        refit_mse = float(((pred - yt.reshape(-1)) ** 2).mean())
        rec = num(r.get("final_validated_mse"))
        faithful = refit_mse <= max(rec * 10.0, 1e-10)

        det = {"mapping_kind": mapping.get("kind"), "refit_mse": refit_mse,
               "recorded_dense_mse": rec, "refit_faithful": faithful}

        # Reconstruct a closed-form string for the refitted candidate and
        # hand it to the shared verdict used by the noisy-benchmark audit.
        try:
            t_sym = sp.Symbol("t")
            sk_sym = ast_to_sympy(ast, xs)
            cand = mapping_to_sympy(mapping, t_sym).subs(t_sym, sk_sym)
            # numeric self-check of the sympy port before trusting the string
            cf = sp.lambdify(xs, cand, "numpy")
            Xc = X[ok][:64]
            port = np.asarray(cf(*[Xc[:, i] for i in range(nv)]), dtype=float)
            ref = pred[:64].numpy()
            scale = float(np.abs(ref).max()) + 1e-12
            if not np.all(np.abs(port - ref) <= 1e-8 * scale):
                classes[k] = "port_check_failed"
                details[k] = det
                continue
            expr_str = sp.sstr(cand)
        except Exception as exc:
            classes[k] = "reconstruction_failed"
            det["reconstruction_error"] = str(exc)[:120]
            details[k] = det
            continue
        det["candidate_expr"] = expr_str[:400]

        csv_path = find_noiseless_csv(args.noiseless_data, _pid(k))
        if csv_path is None:
            classes[k] = "missing_noiseless_data"
            details[k] = det
            continue
        try:
            Xn, yn, names = load_problem_csv(csv_path)
            is_struct, vdet = structural_verdict(
                expr_str, Xn, yn, variable_names=names, tol=args.tol
            )
        except Exception as exc:
            classes[k] = "verdict_error"
            det["verdict_error"] = str(exc)[:120]
            details[k] = det
            continue
        det.update(vdet)
        classes[k] = "structural" if is_struct else "none"
        details[k] = det
        print(f"  {k}: {classes[k]}"
              + (f" (rel={vdet['polish_rel_rmse']:.2e})"
                 if vdet.get("polish_rel_rmse") is not None else ""), flush=True)

    from collections import Counter
    counts = Counter(classes.values())
    n_struct = counts.get("structural", 0)
    print("\nclassification counts:", dict(counts))
    print(f"structural recoveries: {n_struct}/{len(classes)}")

    out = args.run_dir / "structural_audit.json"
    json.dump({"criterion": "refit + pipeline snap arsenal on canonical noiseless "
                            f"data; rel RMSE <= {args.tol:g} (shared with the noisy audit)",
               "tol": args.tol,
               "n_cases": len(classes),
               "structural_recoveries": n_struct,
               "classes": classes, "details": details,
               "counts": dict(counts)}, open(out, "w"), indent=1, default=str)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
