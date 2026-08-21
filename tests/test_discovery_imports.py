# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import importlib
import sys


def test_nestynet_sr_discovery_is_lazy():
    sys.modules.pop("nestynet_sr.discovery", None)
    sys.modules.pop("nestynet_sr", None)

    package = importlib.import_module("nestynet_sr")

    assert "discovery" not in package.__dict__

    discovery_mod = package.discovery

    assert discovery_mod.__name__ == "nestynet_sr.discovery"
    assert package.discovery is discovery_mod
