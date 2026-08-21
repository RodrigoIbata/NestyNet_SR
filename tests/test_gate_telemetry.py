# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Gate-margin telemetry collector semantics."""

import json
import math

from nestynet_sr.sr_search.gate_telemetry import (
    drain,
    record_gate,
    reset,
    snapshot,
    summarize,
)


def setup_function(_fn):
    reset()


def test_record_and_drain_roundtrip():
    record_gate("ruleA", "rel_rms", 0.03, 0.05, accepted=True)
    record_gate("ruleA", "rel_rms", 0.08, 0.05, accepted=False)
    assert len(snapshot()) == 2
    records = drain()
    assert len(records) == 2
    assert snapshot() == []  # drained

    r0 = records[0]
    assert r0["rule"] == "ruleA"
    assert r0["accepted"] is True
    assert math.isclose(r0["margin_ratio"], 0.6)
    assert math.isclose(records[1]["margin_ratio"], 1.6)


def test_degenerate_thresholds_yield_no_margin():
    record_gate("r", "g", 0.5, float("inf"), accepted=True)   # disabled gate
    record_gate("r", "g", float("nan"), 0.05, accepted=False)  # NaN statistic
    record_gate("r", "g", 0.5, 0.0, accepted=False)            # zero threshold
    records = drain()
    assert [r["margin_ratio"] for r in records] == [None, None, None]
    # NaN value must serialize as null, not NaN
    assert records[1]["value"] is None
    json.dumps(records)  # must be JSON-safe


def test_record_never_raises_on_junk():
    record_gate("r", "g", object(), "not-a-number", accepted=True,
                context={"weird": object()})
    records = drain()
    assert len(records) == 1
    json.dumps(records)


def test_summarize_flip_risk_band():
    record_gate("r", "g", 0.04, 0.05, accepted=True)    # margin 0.8  -> in band
    record_gate("r", "g", 0.075, 0.05, accepted=False)  # margin 1.5  -> in band (edge)
    record_gate("r", "g", 0.005, 0.05, accepted=True)   # margin 0.1  -> safe
    record_gate("r", "g", 0.5, 0.05, accepted=False)    # margin 10   -> safe
    summary = summarize(drain(), band=1.5)
    assert summary["n_records"] == 4
    assert summary["n_flip_risk"] == 2
    # sorted by closeness to the threshold: 0.8 (|ln|=0.22) before 1.5 (|ln|=0.41)
    margins = [r["margin_ratio"] for r in summary["flip_risk"]]
    assert margins == sorted(margins, key=lambda m: abs(math.log(m)))
    json.dumps(summary)
