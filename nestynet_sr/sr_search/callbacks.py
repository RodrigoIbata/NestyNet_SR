# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""
Callback system for symbolic regression progress tracking.

Provides hooks for GUI and logging to monitor SR execution in real-time.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Terminal colors for logging
GREEN = "\033[32m"
RESET = "\033[0m"


@dataclass
class SRState:
    """Comprehensive state snapshot of symbolic regression execution."""

    # General
    stage: str = "init"  # "init", "stageA", "stageB", "complete"
    start_time: float = field(default_factory=time.time)
    elapsed_time: float = 0.0

    # Stage A state
    current_ytransform: Optional[str] = None
    ytransform_index: int = 0
    total_ytransforms: int = 0
    stageA_iteration: int = 0
    stageA_trial: int = 0
    dual_layer: bool = False
    num_segments: int = 0
    num_leaves: int = 0
    num_params: int = 0

    # Current AST
    current_ast: Any = None  # Node object
    current_ast_str: str = ""

    # Losses
    current_loss: float = float("inf")
    current_val_loss: float = float("inf")
    best_val_loss: float = float("inf")
    loss_target: float = 1e-7
    loss_acceptable: float = 1e-3

    # Separability
    separability_checks: List[Dict[str, Any]] = field(default_factory=list)
    separability_found: bool = False
    rest_add: Optional[List[int]] = None
    rest_mult: Optional[List[int]] = None

    # Features discovered
    features: Dict[str, Any] = field(default_factory=dict)

    # Stage B state
    stageB_outer_iter: int = 0
    stageB_max_outer: int = 4
    stageB_pattern: Optional[str] = None
    stageB_candidates: List[Dict[str, Any]] = field(default_factory=list)

    # Training state
    lm_epoch: int = 0
    lm_max_epochs: int = 0
    lm_halt_reason: Optional[str] = None

    # History
    loss_history: List[Dict[str, float]] = field(default_factory=list)
    ytransform_results: List[Dict[str, Any]] = field(default_factory=list)

    # Final results
    final_expression_phi: Optional[str] = None
    final_expression_y: Optional[str] = None


class SRCallback:
    """Base class for symbolic regression callbacks."""

    def on_start(self, state: SRState, **kwargs):
        """Called at the beginning of SR execution."""
        pass

    def on_ytransform_start(
        self, state: SRState, ytransform_name: str, index: int, total: int, **kwargs
    ):
        """Called when starting a new y-transform trial."""
        pass

    def on_ytransform_end(
        self, state: SRState, ytransform_name: str, success: bool, val_loss: float, **kwargs
    ):
        """Called when finishing a y-transform trial."""
        pass

    def on_stageA_iteration_start(self, state: SRState, iteration: int, **kwargs):
        """Called at the start of each Stage A iteration."""
        pass

    def on_stageA_trial_start(
        self, state: SRState, trial: int, dual_layer: bool, num_segments: int, **kwargs
    ):
        """Called when starting a new model trial."""
        pass

    def on_ast_update(
        self, state: SRState, ast: Any, ast_str: str, num_leaves: int, num_params: int, **kwargs
    ):
        """Called when AST is updated."""
        pass

    def on_features_discovered(self, state: SRState, features: Dict[str, Any], **kwargs):
        """Called when features are discovered."""
        pass

    def on_separability_check(
        self, state: SRState, atom_info: Dict[str, Any], result: Optional[Dict[str, Any]], **kwargs
    ):
        """Called after checking separability on an atom."""
        pass

    def on_candidate_accept(
        self, state: SRState, candidate_info: Dict[str, Any], reason: str, **kwargs
    ):
        """Called when a candidate is accepted."""
        pass

    def on_candidate_reject(
        self, state: SRState, candidate_info: Dict[str, Any], reason: str, **kwargs
    ):
        """Called when a candidate is rejected."""
        pass

    def on_lm_epoch(self, state: SRState, epoch: int, loss: float, val_loss: float, **kwargs):
        """Called after each LM epoch."""
        pass

    def on_lm_complete(
        self, state: SRState, reason: str, final_loss: float, final_val_loss: float, **kwargs
    ):
        """Called when LM training completes."""
        pass

    def on_stageA_complete(self, state: SRState, success: bool, final_ast: Any, **kwargs):
        """Called when Stage A completes."""
        pass

    def on_stageB_start(
        self, state: SRState, initial_params: int, initial_val_loss: float, **kwargs
    ):
        """Called when Stage B starts."""
        pass

    def on_stageB_outer_start(self, state: SRState, outer_iter: int, max_outer: int, **kwargs):
        """Called at the start of each Stage B outer iteration."""
        pass

    def on_stageB_pattern_start(
        self, state: SRState, pattern: str, target_info: Dict[str, Any], **kwargs
    ):
        """Called when trying a new Stage B pattern."""
        pass

    def on_stageB_candidate(
        self, state: SRState, pattern: str, accepted: bool, candidate_info: Dict[str, Any], **kwargs
    ):
        """Called after evaluating a Stage B candidate."""
        pass

    def on_stageB_complete(self, state: SRState, final_ast: Any, final_val_loss: float, **kwargs):
        """Called when Stage B completes."""
        pass

    def on_complete(self, state: SRState, **kwargs):
        """Called when entire SR process completes."""
        pass

    def on_error(self, state: SRState, error: Exception, **kwargs):
        """Called when an error occurs."""
        pass


class CallbackList:
    """Container for multiple callbacks."""

    def __init__(self, callbacks: Optional[List[SRCallback]] = None):
        self.callbacks = callbacks or []

    def add(self, callback: SRCallback):
        """Add a callback to the list."""
        self.callbacks.append(callback)

    def __getattr__(self, name: str):
        """Forward method calls to all callbacks."""
        if name.startswith("on_"):

            def method(*args, **kwargs):
                for callback in self.callbacks:
                    getattr(callback, name)(*args, **kwargs)

            return method
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


class PrintCallback(SRCallback):
    """Simple callback that prints progress to stdout."""

    def __init__(self, verbose: int = 1):
        self.verbose = verbose

    def on_ytransform_start(self, state: SRState, ytransform_name: str, **kwargs):
        if self.verbose >= 1:
            print(f"\n{'=' * 60}")
            print(
                f"Y-Transform: {ytransform_name} ({state.ytransform_index + 1}/{state.total_ytransforms})"
            )
            print(f"{'=' * 60}")

    def on_stageA_iteration_start(self, state: SRState, iteration: int, **kwargs):
        if self.verbose >= 1:
            print(f"\n--- Stage A Iteration {iteration} ---")

    def on_candidate_accept(
        self, state: SRState, candidate_info: Dict[str, Any], reason: str, **kwargs
    ):
        if self.verbose >= 1:
            print(
                f"{GREEN}✓ ACCEPTED{RESET}: {reason} - Loss: {candidate_info.get('val_loss', 'N/A'):.2e}"
            )

    def on_candidate_reject(
        self, state: SRState, candidate_info: Dict[str, Any], reason: str, **kwargs
    ):
        if self.verbose >= 2:
            print(f"✗ REJECTED: {reason} - Loss: {candidate_info.get('val_loss', 'N/A'):.2e}")

    def on_stageB_outer_start(self, state: SRState, outer_iter: int, max_outer: int, **kwargs):
        if self.verbose >= 1:
            print(f"\n{'=' * 60}")
            print(f"Stage B Outer Iteration {outer_iter}/{max_outer}")
            print(f"{'=' * 60}")

    def on_complete(self, state: SRState, **kwargs):
        if self.verbose >= 1:
            print(f"\n{'=' * 60}")
            print("SYMBOLIC REGRESSION COMPLETE")
            print(f"Total time: {state.elapsed_time:.1f}s")
            print(f"Final loss: {state.best_val_loss:.2e}")
            print(f"{'=' * 60}")


class FileLogCallback(SRCallback):
    """Callback that logs progress to a file."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.file = open(filepath, "w")

    def _log(self, message: str):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.file.write(f"[{timestamp}] {message}\n")
        self.file.flush()

    def on_start(self, state: SRState, **kwargs):
        self._log("Symbolic regression started")

    def on_ytransform_start(self, state: SRState, ytransform_name: str, **kwargs):
        self._log(f"Y-transform: {ytransform_name}")

    def on_candidate_accept(
        self, state: SRState, candidate_info: Dict[str, Any], reason: str, **kwargs
    ):
        self._log(f"Accepted candidate: {reason}")

    def on_complete(self, state: SRState, **kwargs):
        self._log(f"Symbolic regression complete - {state.elapsed_time:.1f}s")
        self.file.close()

    def on_error(self, state: SRState, error: Exception, **kwargs):
        self._log(f"ERROR: {error}")
        self.file.close()
