# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Policy and steering modules for factorized symbolic search."""

from __future__ import annotations

from .features import (
    ParentStateFeatures,
    RepairControllerFeatureRecord,
    build_controller_state_record,
    coerce_repair_feature_record,
    coerce_repair_feature_row,
)


def __getattr__(name: str):
    if name in {"choose_parent", "choose_parent_repair_aware"}:
        from .parent_selection import choose_parent, choose_parent_repair_aware

        return {
            "choose_parent": choose_parent,
            "choose_parent_repair_aware": choose_parent_repair_aware,
        }[name]
    raise AttributeError(name)

__all__ = [
    "ParentStateFeatures",
    "RepairControllerFeatureRecord",
    "build_controller_state_record",
    "choose_parent",
    "choose_parent_repair_aware",
    "coerce_repair_feature_record",
    "coerce_repair_feature_row",
]
