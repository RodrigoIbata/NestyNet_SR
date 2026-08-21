# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from nestynet_sr.sr_search.search import _stageA_split_group_record_payload


def test_stagea_split_group_record_payload_preserves_compound_tokens():
    assert _stageA_split_group_record_payload([0, "z", "z0", "z2", 3]) == [
        0,
        "z",
        "z0",
        "z2",
        3,
    ]


def test_stagea_split_group_record_payload_keeps_integer_axes_json_safe():
    assert _stageA_split_group_record_payload(["0", 1, "2"]) == [0, 1, 2]
