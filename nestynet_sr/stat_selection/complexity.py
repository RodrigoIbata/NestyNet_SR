# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Auditable multi-objective model complexity.

The search engines are free to use scalar scores to allocate computation.  The
statistical audit should not silently inherit those scalarisations.  A
``ComplexityVector`` therefore stores named, minimised components and supplies
only the Pareto partial order needed by the certification layer.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence


@dataclass(frozen=True)
class ComplexityVector:
    """Named model-complexity objectives, all interpreted as "smaller is better".

    Candidate vectors in one audit must contain the same component names.  The
    names are sorted on construction, making serialisation and archive hashes
    independent of dictionary insertion order.
    """

    components: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        normalised: list[tuple[str, float]] = []
        seen: set[str] = set()
        for raw_name, raw_value in self.components:
            name = str(raw_name).strip()
            if not name:
                raise ValueError("complexity component names must be non-empty")
            if name in seen:
                raise ValueError(f"duplicate complexity component {name!r}")
            try:
                value = float(raw_value)
            except Exception as exc:
                raise TypeError(f"complexity component {name!r} is not numeric") from exc
            if not math.isfinite(value):
                raise ValueError(f"complexity component {name!r} must be finite")
            if value < 0.0:
                raise ValueError(f"complexity component {name!r} must be non-negative")
            seen.add(name)
            normalised.append((name, value))
        if not normalised:
            raise ValueError("a complexity vector must contain at least one component")
        normalised.sort(key=lambda item: item[0])
        object.__setattr__(self, "components", tuple(normalised))

    @classmethod
    def from_mapping(cls, components: Mapping[str, float]) -> "ComplexityVector":
        return cls(tuple((str(name), float(value)) for name, value in components.items()))

    @classmethod
    def scalar(cls, value: float, *, name: str = "description_length") -> "ComplexityVector":
        return cls(((str(name), float(value)),))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.components)

    @property
    def values(self) -> tuple[float, ...]:
        return tuple(value for _, value in self.components)

    def as_dict(self) -> dict[str, float]:
        return {name: float(value) for name, value in self.components}

    def _check_compatible(self, other: "ComplexityVector") -> None:
        if not isinstance(other, ComplexityVector):
            raise TypeError("complexity comparisons require ComplexityVector operands")
        if self.names != other.names:
            raise ValueError(
                "complexity vectors use different objectives: "
                f"{self.names!r} != {other.names!r}"
            )

    def no_worse_than(self, other: "ComplexityVector", *, atol: float = 0.0) -> bool:
        """Return whether every component is no larger than ``other``."""

        self._check_compatible(other)
        tol = _nonnegative_tolerance(atol)
        return all(a <= b + tol for a, b in zip(self.values, other.values))

    def strictly_better_than(self, other: "ComplexityVector", *, atol: float = 0.0) -> bool:
        """Return whether this vector is Pareto-better in at least one component."""

        self._check_compatible(other)
        tol = _nonnegative_tolerance(atol)
        return self.no_worse_than(other, atol=tol) and any(
            a < b - tol for a, b in zip(self.values, other.values)
        )

    def equivalent_to(self, other: "ComplexityVector", *, atol: float = 0.0) -> bool:
        self._check_compatible(other)
        tol = _nonnegative_tolerance(atol)
        return all(abs(a - b) <= tol for a, b in zip(self.values, other.values))


def validate_complexity_collection(
    complexities: Sequence[ComplexityVector],
) -> tuple[ComplexityVector, ...]:
    """Validate that a candidate collection uses one declared objective set."""

    out = tuple(complexities)
    if not out:
        raise ValueError("at least one complexity vector is required")
    names = out[0].names
    for i, complexity in enumerate(out):
        if not isinstance(complexity, ComplexityVector):
            raise TypeError(f"complexity {i} is not a ComplexityVector")
        if complexity.names != names:
            raise ValueError(
                f"complexity {i} uses objectives {complexity.names!r}; expected {names!r}"
            )
    return out


def _nonnegative_tolerance(value: float) -> float:
    try:
        out = float(value)
    except Exception as exc:
        raise TypeError("tolerance must be numeric") from exc
    if not math.isfinite(out) or out < 0.0:
        raise ValueError("tolerance must be finite and non-negative")
    return out
