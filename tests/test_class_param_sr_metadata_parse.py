# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

import pytest

from nestynet_sr.run_SR import (
    _metadata_linked_invariants,
    _normalize_class_param_sr_metadata,
)


def test_param_sr_metadata_parse_columnwise():
    rows = _normalize_class_param_sr_metadata(
        '{"temp":[1,2,3],"pressure":[10.0,20.0,30.0]}',
        dataset_paths=["a.csv", "b.csv", "c.csv"],
    )
    assert rows == [
        {"temp": 1.0, "pressure": 10.0},
        {"temp": 2.0, "pressure": 20.0},
        {"temp": 3.0, "pressure": 30.0},
    ]


def test_param_sr_metadata_parse_rowwise():
    rows = _normalize_class_param_sr_metadata(
        '[{"temp":1.0},{"temp":2.0},{"temp":3.0}]',
        dataset_paths=["a.csv", "b.csv", "c.csv"],
    )
    assert rows == [{"temp": 1.0}, {"temp": 2.0}, {"temp": 3.0}]


def test_param_sr_metadata_parse_dataset_keyed_by_stem():
    rows = _normalize_class_param_sr_metadata(
        '{"quad_1":{"temp":1.0},"quad_2":{"temp":2.0},"quad_3":{"temp":3.0}}',
        dataset_paths=[
            "examples/classSR/data/quad_1.csv",
            "examples/classSR/data/quad_2.csv",
            "examples/classSR/data/quad_3.csv",
        ],
    )
    assert rows == [{"temp": 1.0}, {"temp": 2.0}, {"temp": 3.0}]


def test_param_sr_metadata_parse_bad_length_raises():
    with pytest.raises(ValueError):
        _normalize_class_param_sr_metadata(
            '{"temp":[1.0,2.0]}',
            dataset_paths=["a.csv", "b.csv", "c.csv"],
        )


def test_metadata_linked_invariants_filter():
    invs = [
        {"expr": "k#p0/meta:temp", "score": 0.1, "cv": 0.2},
        {"expr": "a#p0*b#p0", "score": 0.3, "cv": 0.4},
        {"expr": "meta:pressure/meta:temp", "score": 0.05, "cv": 0.1},
    ]
    out = _metadata_linked_invariants(invs)
    assert [d["expr"] for d in out] == ["k#p0/meta:temp", "meta:pressure/meta:temp"]
