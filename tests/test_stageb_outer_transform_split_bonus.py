# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

from nestynet_sr.sr_search.stageB.rules import RuleOuterTransformSplitNN


def test_outer_transform_split_bonus_matrix():
    is_bonus = RuleOuterTransformSplitNN._is_separability_like_outer_split

    assert is_bonus("identity", "add") is True
    assert is_bonus("identity", "mul") is True
    assert is_bonus("log", "add") is True
    assert is_bonus("log", "mul") is False
    assert is_bonus("sqrt", "add") is False
    assert is_bonus("sqrt", "mul") is True
