# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""
Leaf manipulation utilities for Stage B.

This module provides functions for manipulating analytic leaf modules:
- Finding and unwrapping leaf modules from adaptors
- Copying weights between compatible leaves
- Initializing leaves from reuse maps
- Setting specific polynomial coefficients
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Set, Tuple

import torch
import torch.nn as nn

from nestynet_sr.sr_core.bridges import AtomNode, Node
from nestynet_sr.sr_core.separability_math import _sample_plane_for_pair

from .atom_mapping import _collect_all_atoms, build_atom_to_leaf_map


def _fit_poly_1d_trapped(x: torch.Tensor, y: torch.Tensor, deg: int = 3):
    x = x.view(-1)
    y = y.view(-1)
    mask = torch.isfinite(x) & torch.isfinite(y)
    if mask.sum() < deg + 1:
        return None
    x = x[mask]
    y = y[mask]
    X = torch.stack([x**d for d in range(deg + 1)], dim=-1)
    sol = torch.linalg.lstsq(X, y.unsqueeze(-1)).solution.squeeze(-1)
    return sol


def _find_first_submodule(
    root: nn.Module, pred: Callable[[nn.Module], bool]
) -> Optional[nn.Module]:
    """
    Find the first submodule (including self) satisfying `pred`.
    This is used to unwrap thin adaptors (e.g. AutogradAdaptor) and
    operate on the underlying analytic leaf module.
    """
    try:
        for m in root.modules():
            try:
                if pred(m):
                    return m
            except Exception:
                continue
    except Exception:
        return None
    return None


def _poly_like_core(leaf: nn.Module) -> nn.Module:
    """
    Return the actual poly-like leaf module inside wrappers.
    We identify it by the presence of a Tensor-like `.exps` attribute.
    """
    m = _find_first_submodule(leaf, lambda s: torch.is_tensor(getattr(s, "exps", None)))
    return m if m is not None else leaf


def _module_path(root: nn.Module, target: nn.Module) -> str:
    """
    Return the 'named_modules' path of `target` inside `root`.
    Useful for wrapper adaptors (e.g. AutogradAdaptor.model).
    """
    try:
        for name, m in root.named_modules():
            if m is target:
                return name if name else "<self>"
    except Exception:
        pass
    return "<unknown>"


def _debug_report_leaf_cores(
    root: Node,
    model: nn.Module,
    *,
    kinds: Optional[Set[str]] = None,
    only_tagged: bool = True,
    max_exps_preview: int = 6,
    raise_on_missing: bool = False,
    header: Optional[str] = None,
):
    """
    Debug helper: for each AtomNode (optionally filtered by kind/tag),
    print which module `_poly_like_core()` selected and whether we can
    see `.exps` and a writable coefficient Parameter.

    This is designed for wrapper leaves like AutogradAdaptor(PolyLeaf),
    where the real parameters/exps live on `leaf.model`.
    """
    if header:
        print(f"[Stage B dbg] {header}")

    try:
        atom_to_leaf = build_atom_to_leaf_map(root, model)
    except Exception as e:
        msg = f"[Stage B dbg] build_atom_to_leaf_map failed: {e}"
        if raise_on_missing:
            raise RuntimeError(msg)
        print(msg)
        return

    # Default: focus on analytic leaves that *should* have coefficients.
    if kinds is None:
        kinds = {"poly", "exp_poly", "polylog", "exp_ratpoly", "power"}

    n_total = 0
    n_missing_leaf = 0
    n_missing_exps = 0
    n_missing_coeff = 0

    for a in _collect_all_atoms(root):
        if not isinstance(a, AtomNode):
            continue
        kind = str(getattr(a, "kind", "")).lower()
        if kinds is not None and kind not in kinds:
            continue
        tag = getattr(a, "tag", None)
        if only_tagged and (tag is None or str(tag) == ""):
            continue

        n_total += 1
        leaf = atom_to_leaf.get(id(a), None)
        if leaf is None:
            n_missing_leaf += 1
            msg = f"[Stage B dbg] kind={kind} vars={tuple(int(j) for j in a.var_idxs)} tag={tag}: MISSING leaf module"
            if raise_on_missing:
                raise RuntimeError(msg)
            print(msg)
            continue

        core = _poly_like_core(leaf)
        core_path = _module_path(leaf, core)

        exps = getattr(core, "exps", None)
        exps_shape = tuple(exps.shape) if torch.is_tensor(exps) else None
        if exps_shape is None and kind in {"poly", "exp_poly", "polylog", "exp_ratpoly"}:
            n_missing_exps += 1

        coeff = _leaf_coeff_param(core)
        coeff_shape = (
            tuple(coeff.shape)
            if isinstance(coeff, torch.nn.Parameter)
            else (tuple(coeff.shape) if torch.is_tensor(coeff) else None)
        )
        if coeff is None and kind in {"poly", "exp_poly", "polylog", "exp_ratpoly"}:
            n_missing_coeff += 1

        # Preview a few monomial exponents for sanity (avoid huge prints).
        preview = None
        if torch.is_tensor(exps) and exps.ndim == 2:
            try:
                ex = exps.detach().cpu()
                m = min(int(max_exps_preview), int(ex.shape[0]))
                preview = [tuple(int(v) for v in ex[i].tolist()) for i in range(m)]
            except Exception:
                preview = None

        print(
            "[Stage B dbg] "
            f"kind={kind} vars={tuple(int(j) for j in a.var_idxs)} tag={tag} | "
            f"leaf={type(leaf).__name__} -> core={type(core).__name__}@{core_path} | "
            f"coeff_shape={coeff_shape} exps_shape={exps_shape} exps_preview={preview}"
        )

        if raise_on_missing:
            if kind in {"poly", "exp_poly", "polylog", "exp_ratpoly"}:
                if coeff is None:
                    raise RuntimeError(
                        f"[Stage B dbg] Missing coeff Parameter for tag={tag} (leaf={type(leaf).__name__}, core={type(core).__name__})"
                    )
                if not torch.is_tensor(exps):
                    raise RuntimeError(
                        f"[Stage B dbg] Missing exps Tensor for tag={tag} (leaf={type(leaf).__name__}, core={type(core).__name__})"
                    )

    print(
        "[Stage B dbg] summary: "
        f"n={n_total}, missing_leaf={n_missing_leaf}, missing_exps={n_missing_exps}, missing_coeff={n_missing_coeff}"
    )


def _get_leaf_coefficients(leaf: nn.Module) -> torch.nn.Parameter | None:
    """
    Get the primary coefficient parameter from a leaf module using explicit type-based dispatch.

    Strategy:
    1. If .exps exists, pick param whose length matches exps.shape[0]
    2. Else, use explicit attribute mapping per leaf type
    3. Fallback to heuristic search only if above fails

    This replaces the fragile _leaf_first_param fallback with robust type detection.

    Args:
        leaf: Leaf module (may be wrapped in AutogradAdaptor)

    Returns:
        Primary coefficient parameter, or None if not found
    """
    # Unwrap if needed (AutogradAdaptor.model, etc.)
    # First check for .model attribute (AutogradAdaptor pattern)
    core = getattr(leaf, "model", None)
    if core is None:
        core = leaf

    # Strategy 1: If .exps exists, find parameter matching exps.shape[0]
    exps = getattr(core, "exps", None)
    if torch.is_tensor(exps) and exps.ndim >= 1:
        expected_len = int(exps.shape[0])
        for m in [core, leaf]:  # Check core first, then wrapper
            for name, param in m.named_parameters(recurse=False):
                if isinstance(param, torch.nn.Parameter):
                    if param.numel() == expected_len or param.shape[0] == expected_len:
                        return param

    # Strategy 2: Explicit attribute mapping per leaf type
    # First try direct .coeffs attribute (PolyLeaf, ExpPolyLeaf, PolyLogLeaf, etc.)
    for m in [core, leaf]:
        c = getattr(m, "coeffs", None)
        if isinstance(c, torch.nn.Parameter):
            return c

    # For rational poly leaves, try .coeffs_num (numerator coefficients)
    for m in [core, leaf]:
        c = getattr(m, "coeffs_num", None)
        if isinstance(c, torch.nn.Parameter):
            return c

    # Type-based explicit mapping for non-polynomial leaves
    core_type_name = type(core).__name__

    # SinLinearLeaf: return amp (scalar amplitude parameter)
    if core_type_name == "SinLinearLeaf":
        amp = getattr(core, "amp", None)
        if isinstance(amp, torch.nn.Parameter):
            return amp
        # Fallback to weight if amp not found
        weight = getattr(core, "weight", None)
        if isinstance(weight, torch.nn.Parameter):
            return weight

    # TanhLinearLeaf: return amp (scalar amplitude parameter)
    elif core_type_name == "TanhLinearLeaf":
        amp = getattr(core, "amp", None)
        if isinstance(amp, torch.nn.Parameter):
            return amp
        weight = getattr(core, "weight", None)
        if isinstance(weight, torch.nn.Parameter):
            return weight

    # PowerLeaf: return amp (scalar amplitude parameter)
    elif core_type_name == "PowerLeaf":
        amp = getattr(core, "amp", None)
        if isinstance(amp, torch.nn.Parameter):
            return amp
        # Fallback to exponent if amp not found
        exponent = getattr(core, "exponent", None)
        if isinstance(exponent, torch.nn.Parameter):
            return exponent

    # PlanckLeaf: return log_amp (primary scale parameter)
    elif core_type_name in ("PlanckLeaf", "PlanckFullLeaf"):
        log_amp = getattr(core, "log_amp", None)
        if isinstance(log_amp, torch.nn.Parameter):
            return log_amp
        # Fallback to p (power parameter)
        p = getattr(core, "p", None)
        if isinstance(p, torch.nn.Parameter):
            return p

    # Expm1Leaf: return log_amp (primary scale parameter)
    elif core_type_name == "Expm1Leaf":
        log_amp = getattr(core, "log_amp", None)
        if isinstance(log_amp, torch.nn.Parameter):
            return log_amp
        # Fallback to log_a
        log_a = getattr(core, "log_a", None)
        if isinstance(log_a, torch.nn.Parameter):
            return log_a

    # Strategy 3: Fallback heuristic (only if above strategies failed)
    # Try to find any parameter from the core or leaf
    for m in [core, leaf]:
        ps = [p for p in m.parameters(recurse=False)]
        if ps:
            # If single param, return it
            if len(ps) == 1:
                return ps[0]
            # If multiple, prefer smallest (likely to be coefficient vector)
            ps_sorted = sorted(ps, key=lambda p: p.numel())
            return ps_sorted[0]

    # Final fallback: recurse=True search (for deeply wrapped modules)
    ps = [p for p in leaf.parameters(recurse=True)]
    if ps:
        if len(ps) == 1:
            return ps[0]
        ps_sorted = sorted(ps, key=lambda p: p.numel())
        return ps_sorted[0]

    return None



def _leaf_coeff_param(leaf: nn.Module):
    """
    Get the primary coefficient parameter from a leaf module.

    Uses _get_leaf_coefficients() for robust type-based dispatch.
    Kept for backwards compatibility with existing code.
    """
    return _get_leaf_coefficients(leaf)


def _copy_compatible_weights(
    leaf_new: nn.Module,
    leaf_old: nn.Module,
) -> bool:
    """
    Copy weights from leaf_old to leaf_new if they are compatible.

    Compatibility means:
    - Same leaf type
    - Same parameter shapes
    - Same exponent structure (for polynomial leaves)

    This enables warm-starting Stage B models from Stage A leaves or
    from previous Stage B candidates.

    Args:
        leaf_new: Target leaf module (may be wrapped in AutogradAdaptor)
        leaf_old: Source leaf module (may be wrapped)

    Returns:
        True if weights were copied successfully, False if incompatible

    Examples:
        >>> old_leaf = PolyLeaf(n_in=1, degree=2)
        >>> new_leaf = PolyLeaf(n_in=1, degree=2)
        >>> # ... train old_leaf ...
        >>> success = _copy_compatible_weights(new_leaf, old_leaf)
        >>> # new_leaf.coeffs now has old_leaf's trained weights
    """
    # Import leaf types
    from nestynet_sr.sr_core.atoms import (
        Expm1Leaf,
        ExpPolyLeaf,
        ExpRationalPolyLeaf,
        PolyLogLeaf,
        PlanckFullLeaf,
        PlanckLeaf,
        PolyLeaf,
        PowerLeaf,
        RationalPolyLeaf,
        RExpPolyLeaf,
        SinLinearLeaf,
        TanhLinearLeaf,
    )

    # Unwrap if needed (AutogradAdaptor.model, etc.)
    core_new = getattr(leaf_new, "model", leaf_new)
    core_old = getattr(leaf_old, "model", leaf_old)

    # Must be same type
    if type(core_new) is not type(core_old):
        return False

    try:
        # PolyLeaf: Check exps and coeffs match
        if isinstance(core_new, PolyLeaf):
            if core_new.coeffs.shape != core_old.coeffs.shape:
                return False
            if hasattr(core_new, "exps") and hasattr(core_old, "exps"):
                if core_new.exps.shape != core_old.exps.shape:
                    return False
                if not torch.equal(core_new.exps.detach().cpu(), core_old.exps.detach().cpu()):
                    return False
            with torch.no_grad():
                core_new.coeffs.copy_(
                    core_old.coeffs.to(device=core_new.coeffs.device, dtype=core_new.coeffs.dtype)
                )
            return True

        # PolyLogLeaf: Check exps and coeffs match
        if isinstance(core_new, PolyLogLeaf):
            if core_new.coeffs.shape != core_old.coeffs.shape:
                return False
            if hasattr(core_new, "exps") and hasattr(core_old, "exps"):
                if core_new.exps.shape != core_old.exps.shape:
                    return False
                if not torch.equal(core_new.exps.detach().cpu(), core_old.exps.detach().cpu()):
                    return False
            with torch.no_grad():
                core_new.coeffs.copy_(
                    core_old.coeffs.to(device=core_new.coeffs.device, dtype=core_new.coeffs.dtype)
                )
            return True

        # RationalPolyLeaf: Check both numerator and denominator
        if isinstance(core_new, RationalPolyLeaf):
            if (
                core_new.coeffs_num.shape != core_old.coeffs_num.shape
                or core_new.coeffs_den.shape != core_old.coeffs_den.shape
            ):
                return False
            if hasattr(core_new, "exps_num") and hasattr(core_old, "exps_num"):
                if core_new.exps_num.shape != core_old.exps_num.shape:
                    return False
                if not torch.equal(
                    core_new.exps_num.detach().cpu(), core_old.exps_num.detach().cpu()
                ):
                    return False
            if hasattr(core_new, "exps_den") and hasattr(core_old, "exps_den"):
                if core_new.exps_den.shape != core_old.exps_den.shape:
                    return False
                if not torch.equal(
                    core_new.exps_den.detach().cpu(), core_old.exps_den.detach().cpu()
                ):
                    return False
            with torch.no_grad():
                core_new.coeffs_num.copy_(
                    core_old.coeffs_num.to(
                        device=core_new.coeffs_num.device, dtype=core_new.coeffs_num.dtype
                    )
                )
                core_new.coeffs_den.copy_(
                    core_old.coeffs_den.to(
                        device=core_new.coeffs_den.device, dtype=core_new.coeffs_den.dtype
                    )
                )
            return True

        # ExpPolyLeaf: Check exps and coeffs match
        if isinstance(core_new, ExpPolyLeaf):
            if core_new.coeffs.shape != core_old.coeffs.shape:
                return False
            if hasattr(core_new, "exps") and hasattr(core_old, "exps"):
                if core_new.exps.shape != core_old.exps.shape:
                    return False
                if not torch.equal(core_new.exps.detach().cpu(), core_old.exps.detach().cpu()):
                    return False
            with torch.no_grad():
                core_new.coeffs.copy_(
                    core_old.coeffs.to(device=core_new.coeffs.device, dtype=core_new.coeffs.dtype)
                )
            return True



        # RExpPolyLeaf: Check exps and coeffs match
        if isinstance(core_new, RExpPolyLeaf):
            if core_new.coeffs.shape != core_old.coeffs.shape:
                return False
            if hasattr(core_new, "exps") and hasattr(core_old, "exps"):
                if core_new.exps.shape != core_old.exps.shape:
                    return False
                if not torch.equal(core_new.exps.detach().cpu(), core_old.exps.detach().cpu()):
                    return False
            with torch.no_grad():
                core_new.coeffs.copy_(
                    core_old.coeffs.to(device=core_new.coeffs.device, dtype=core_new.coeffs.dtype)
                )
            return True
        # ExpRationalPolyLeaf: Check both numerator and denominator
        if isinstance(core_new, ExpRationalPolyLeaf):
            if (
                core_new.coeffs_num.shape != core_old.coeffs_num.shape
                or core_new.coeffs_den.shape != core_old.coeffs_den.shape
            ):
                return False
            if hasattr(core_new, "exps_num") and hasattr(core_old, "exps_num"):
                if core_new.exps_num.shape != core_old.exps_num.shape:
                    return False
                if not torch.equal(
                    core_new.exps_num.detach().cpu(), core_old.exps_num.detach().cpu()
                ):
                    return False
            if hasattr(core_new, "exps_den") and hasattr(core_old, "exps_den"):
                if core_new.exps_den.shape != core_old.exps_den.shape:
                    return False
                if not torch.equal(
                    core_new.exps_den.detach().cpu(), core_old.exps_den.detach().cpu()
                ):
                    return False
            with torch.no_grad():
                core_new.coeffs_num.copy_(
                    core_old.coeffs_num.to(
                        device=core_new.coeffs_num.device, dtype=core_new.coeffs_num.dtype
                    )
                )
                core_new.coeffs_den.copy_(
                    core_old.coeffs_den.to(
                        device=core_new.coeffs_den.device, dtype=core_new.coeffs_den.dtype
                    )
                )
            return True

        # SinLinearLeaf: Copy weight, bias, amp
        if isinstance(core_new, SinLinearLeaf):
            if core_new.weight.shape != core_old.weight.shape:
                return False
            with torch.no_grad():
                core_new.weight.copy_(
                    core_old.weight.to(device=core_new.weight.device, dtype=core_new.weight.dtype)
                )
                core_new.bias.copy_(
                    core_old.bias.to(device=core_new.bias.device, dtype=core_new.bias.dtype)
                )
                core_new.amp.copy_(
                    core_old.amp.to(device=core_new.amp.device, dtype=core_new.amp.dtype)
                )
            return True

        # TanhLinearLeaf: Copy weight, bias, amp
        if isinstance(core_new, TanhLinearLeaf):
            if core_new.weight.shape != core_old.weight.shape:
                return False
            with torch.no_grad():
                core_new.weight.copy_(
                    core_old.weight.to(device=core_new.weight.device, dtype=core_new.weight.dtype)
                )
                core_new.bias.copy_(
                    core_old.bias.to(device=core_new.bias.device, dtype=core_new.bias.dtype)
                )
                core_new.amp.copy_(
                    core_old.amp.to(device=core_new.amp.device, dtype=core_new.amp.dtype)
                )
            return True

        # PowerLeaf: Copy exponent and amp
        if isinstance(core_new, PowerLeaf):
            with torch.no_grad():
                core_new.exponent.copy_(
                    core_old.exponent.to(
                        device=core_new.exponent.device, dtype=core_new.exponent.dtype
                    )
                )
                core_new.amp.copy_(
                    core_old.amp.to(device=core_new.amp.device, dtype=core_new.amp.dtype)
                )
            return True

        # PlanckLeaf: Copy compatible parameters.  The reduced Planck leaf has
        # fixed structural p and no b; PlanckFullLeaf keeps the legacy full set.
        if isinstance(core_new, PlanckLeaf):
            with torch.no_grad():
                core_new.log_amp.copy_(
                    core_old.log_amp.to(
                        device=core_new.log_amp.device, dtype=core_new.log_amp.dtype
                    )
                )
                core_new.p.copy_(core_old.p.to(device=core_new.p.device, dtype=core_new.p.dtype))
                core_new.log_a.copy_(
                    core_old.log_a.to(device=core_new.log_a.device, dtype=core_new.log_a.dtype)
                )
                if hasattr(core_new, "b") and hasattr(core_old, "b"):
                    core_new.b.copy_(core_old.b.to(device=core_new.b.device, dtype=core_new.b.dtype))
            return True

        if isinstance(core_new, PlanckFullLeaf):
            with torch.no_grad():
                core_new.log_amp.copy_(
                    core_old.log_amp.to(
                        device=core_new.log_amp.device, dtype=core_new.log_amp.dtype
                    )
                )
                core_new.p.copy_(core_old.p.to(device=core_new.p.device, dtype=core_new.p.dtype))
                core_new.log_a.copy_(
                    core_old.log_a.to(device=core_new.log_a.device, dtype=core_new.log_a.dtype)
                )
                if hasattr(core_old, "b"):
                    core_new.b.copy_(core_old.b.to(device=core_new.b.device, dtype=core_new.b.dtype))
                else:
                    core_new.b.zero_()
            return True

        # Expm1Leaf: Copy all parameters
        if isinstance(core_new, Expm1Leaf):
            with torch.no_grad():
                core_new.log_amp.copy_(
                    core_old.log_amp.to(
                        device=core_new.log_amp.device, dtype=core_new.log_amp.dtype
                    )
                )
                core_new.log_a.copy_(
                    core_old.log_a.to(device=core_new.log_a.device, dtype=core_new.log_a.dtype)
                )
                if hasattr(core_old, "b"):
                    core_new.b.copy_(core_old.b.to(device=core_new.b.device, dtype=core_new.b.dtype))
                else:
                    core_new.b.zero_()
            return True

        # Unknown type
        return False

    except Exception:
        # If any error occurs during copying (shape mismatch, etc.), return False
        return False


def _initialise_analytic_leaves_from_reuse(
    root: Node,
    atom_to_leaf: Dict[int, nn.Module],
    reuse: Dict[str, nn.Module],
    verbose: bool = True,
) -> int:
    """
    Initialize analytic leaves from reuse map by copying compatible weights.

    This is a key warm-starting mechanism: when a new Stage B candidate is built,
    we can copy weights from previous candidates (or Stage A NN leaves converted
    to analytic) if the structure is compatible.

    Args:
        root: AST root node
        atom_to_leaf: Mapping from id(atom) to leaf module (from build_composite_from_ast)
        reuse: Mapping from tag to existing leaf module
        verbose: Print messages about initialization

    Returns:
        Number of leaves successfully initialized from reuse

    Examples:
        >>> # Build first model
        >>> root1 = AddNode(
        ...     AtomNode("poly", (0,), {"degree": 2}, tag="p0"),
        ...     AtomNode("poly", (1,), {"degree": 2}, tag="p1"),
        ... )
        >>> model1, atom_map1 = build_composite_from_ast(root1, return_atom_map=True)
        >>> # ... train model1 ...
        >>>
        >>> # Build second model with same structure, warm-start from first
        >>> root2 = AddNode(
        ...     AtomNode("poly", (0,), {"degree": 2}, tag="p0"),  # Same tag!
        ...     AtomNode("poly", (1,), {"degree": 2}, tag="p1"),
        ... )
        >>> reuse = {"p0": atom_map1[id(root1.left)], "p1": atom_map1[id(root1.right)]}
        >>> model2, atom_map2 = build_composite_from_ast(root2, reuse=reuse, return_atom_map=True)
        >>> # Now initialize model2 from model1's trained weights
        >>> n_init = _initialise_analytic_leaves_from_reuse(root2, atom_map2, reuse)
        >>> # model2 starts with model1's weights (warm start)
    """
    # Import Node types

    # Collect all atoms from AST
    atoms = _collect_all_atoms(root)

    n_initialized = 0

    for atom in atoms:
        # Skip if atom has no tag or tag not in reuse
        if atom.tag is None or atom.tag not in reuse:
            continue

        # Get the new leaf and the reuse (old) leaf
        leaf_new = atom_to_leaf.get(id(atom))
        leaf_old = reuse[atom.tag]

        if leaf_new is None or leaf_old is None:
            continue

        # Try to copy compatible weights
        success = _copy_compatible_weights(leaf_new, leaf_old)

        if success:
            n_initialized += 1
            if verbose:
                core_new = getattr(leaf_new, "model", leaf_new)
                print(
                    f"[Stage B] Initialized {type(core_new).__name__} from reuse "
                    f"(tag={atom.tag}, vars={tuple(int(v) for v in atom.var_idxs)})"
                )

    return n_initialized


def _poly_zero_and_set(leaf: nn.Module, exp_to_val: Dict[Tuple[int, ...], float]):
    """
    Zero all coefficients in a poly-like leaf, then set specific monomials.

    Works if the leaf exposes:
      - .exps : [n_terms, n_vars] integer tensor
      - some coefficient parameter (prefer .coeffs, else first param)
    """
    # If leaf is wrapped (AutogradAdaptor), operate on the underlying poly module.
    core = _poly_like_core(leaf)
    coeff = _leaf_coeff_param(core)
    if coeff is None:
        # last resort: try the wrapper itself
        coeff = _leaf_coeff_param(leaf)
    if coeff is None:
        raise RuntimeError(f"Leaf {type(leaf)} has no parameters to set.")

    exps = getattr(core, "exps", None)

    with torch.no_grad():
        coeff.zero_()
        if exps is None:
            # Last-resort fallback: if user’s PolyLeaf doesn’t expose exps,
            # assume constant is index 0 and just write those indices in order.
            # (Better than silently doing nothing.)
            for k, v in exp_to_val.items():
                if k == (0,) or k == (0, 0):
                    coeff.view(-1)[0] = float(v)
            return

        exps = exps.to(device=coeff.device)
        for k, v in exp_to_val.items():
            kk = torch.tensor(k, device=exps.device, dtype=exps.dtype)
            if kk.ndim == 1:
                kk = kk.view(1, -1)
            idx = (exps == kk).all(dim=1).nonzero(as_tuple=False)
            if idx.numel() == 0:
                # Check if this is RPolyLeaf and the monomial is the fixed leading term
                from nestynet_sr.sr_core.atoms import RPolyLeaf
                if isinstance(core, RPolyLeaf):
                    lead_exp = getattr(core, "lead_exp", None)
                    if lead_exp is not None:
                        kk_cpu = torch.tensor(k, dtype=lead_exp.dtype, device=lead_exp.device)
                        if kk_cpu.shape == lead_exp.shape and torch.equal(kk_cpu, lead_exp):
                            # This is the fixed leading monomial (coefficient = 1.0)
                            # Skip setting it - it can't be changed
                            print(f"[Stage B] Note: skipping fixed leading monomial {k} in RPolyLeaf (coeff=1.0)")
                            continue
                raise RuntimeError(f"Monomial {k} not found in leaf.exps for {type(core)}")
            i = int(idx[0, 0])
            if coeff.dim() == 1:
                coeff[i] = float(v)
            else:
                # If coeff is [n_terms, out_dim], set all outputs (typically out_dim=1)
                coeff[i, :] = float(v)


def _set_constant_leaf_value(leaf: nn.Module, value: float) -> bool:
    """Set a scalar value on a constant-like leaf.

    Supports:
      - FreeConst/Scale-style leaves exposing ``.value``
      - Constant polynomial-style leaves exposing ``.coeffs`` with ``n_in == 0``

    Returns True when a compatible constant parameter was found and written.
    """
    core = getattr(leaf, "base_model", getattr(leaf, "model", leaf))
    core = getattr(core, "core", core)
    v = float(value)

    with torch.no_grad():
        value_param = getattr(core, "value", None)
        if torch.is_tensor(value_param):
            value_param.fill_(v)
            return True

        coeffs = getattr(core, "coeffs", None)
        if torch.is_tensor(coeffs):
            n_in = getattr(core, "n_in", None)
            if n_in is not None:
                try:
                    if int(n_in) != 0:
                        return False
                except Exception:
                    return False
            elif coeffs.numel() != 1:
                # If arity is unknown, only treat a scalar coeff vector as constant-like.
                return False

            coeffs.zero_()
            if coeffs.numel() > 0:
                coeffs.view(-1)[0] = v
                return True

    return False


def _compute_trapped_factorization(
    model: nn.Module,
    datagen,
    trapped_idx: int,
    leaky_idx: int,
    device: torch.device,
    dtype: torch.dtype,
    candidate_P: str = "product",
    n_points: int = 4096,
    min_points: int = 400,
):
    T = int(trapped_idx)
    L = int(leaky_idx)
    X_plane = _sample_plane_for_pair(datagen, i=T, j=L, n_points=n_points)
    X_plane = X_plane.to(device=device, dtype=dtype)
    with torch.no_grad():
        y = model.forward(X_plane).view(-1)
        g = model.grad(X_plane).view(-1, X_plane.shape[1])
    dyT = g[:, T]
    y_abs = y.abs()
    m_y = torch.isfinite(y_abs)
    if not m_y.any():
        return None
    y_scale = y_abs[m_y].median().clamp_min(1e-12)
    eps_y = 1e-6 * y_scale + 1e-12
    sign_y = torch.sign(y)
    sign_y = torch.where(sign_y == 0, torch.ones_like(sign_y), sign_y)
    y_safe = sign_y * y_abs.clamp_min(eps_y)
    Lvals = torch.log(y_abs.clamp_min(eps_y))
    xT = X_plane[:, T]
    xL = X_plane[:, L]
    if candidate_P == "product":
        P = xT * xL
        dP = xL
    elif candidate_P == "sum":
        P = xT + xL
        dP = torch.ones_like(xT)
    else:
        raise ValueError(f"Unknown candidate_P {candidate_P}")
    g_log = dyT / y_safe
    mask = torch.isfinite(P) & torch.isfinite(dP) & torch.isfinite(g_log)
    mask &= dP.abs() > 1e-8
    mask &= y_abs > eps_y
    if not mask.any():
        return None
    P = P[mask]
    dP = dP[mask]
    g_log = g_log[mask]
    Lvals = Lvals[mask]
    xL = xL[mask]
    if P.numel() < min_points:
        return None
    Q = g_log / dP
    maskQ = torch.isfinite(Q)
    if not maskQ.any():
        return None
    P = P[maskQ]
    Q = Q[maskQ]
    Lvals = Lvals[maskQ]
    xL = xL[maskQ]
    if P.numel() < min_points:
        return None
    N = P.numel()
    order = torch.argsort(P)
    P_s = P[order]
    Q_s = Q[order]
    dP_s = P_s[1:] - P_s[:-1]
    sign = torch.sign(dP_s)
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    dP_s_clamped = torch.where(dP_s.abs() < 1e-6, sign * 1e-6, dP_s)
    H_mid = 0.5 * (Q_s[1:] + Q_s[:-1])
    dlogB = H_mid * dP_s_clamped
    logB_s = torch.zeros_like(P_s)
    logB_s[1:] = torch.cumsum(dlogB, dim=0)
    logB_s = logB_s - logB_s.median()
    inv = torch.empty_like(order)
    inv[order] = torch.arange(N, device=order.device)
    logB = logB_s[inv]
    logA = Lvals - logB
    return P, xL, logB, logA
