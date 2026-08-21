# SPDX-License-Identifier: MPL-2.0

from nestynet_sr.sr_core.problem_identity import canonical_problem_id


def test_canonical_problem_id_strips_only_owned_terminal_stat_views():
    original = "pb079_II_35_18_data"
    search = f"{original}.stat-search-n80000.df9f46dd4c58.csv"
    audit = f"{original}.stat-audit-n80000-100000.df9f46dd4c58.csv"
    repeated = f"{search[:-4]}.stat-audit-n64000-80000.0123456789ab.csv"

    assert canonical_problem_id(original) == original
    assert canonical_problem_id(search) == original
    assert canonical_problem_id(audit) == original
    assert canonical_problem_id(repeated) == original


def test_canonical_problem_id_preserves_unowned_and_distinct_stems():
    assert canonical_problem_id("pb079.stat-search-n80000.not-a-digest.csv") == (
        "pb079.stat-search-n80000.not-a-digest"
    )
    assert canonical_problem_id("pb079_II_35_18_data") != canonical_problem_id(
        "pb119_Klein-Nishina (13_132 Schwarz)_data"
    )
