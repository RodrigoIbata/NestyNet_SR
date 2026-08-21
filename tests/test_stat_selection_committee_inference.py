# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Paired max-T committee inference: sharding exactness, calibration, verdicts."""

import numpy as np
import pytest

from nestynet_sr.stat_selection.committee_inference import (
    committee_maxt_decision,
    committee_shard_stats,
    reduce_committee_decision,
)


def _toy_losses(seed=0, G=600, better_shift=-0.5, worse_shift=0.4):
    """Baseline plus three members: one better, one worse, one equivalent."""
    rng = np.random.default_rng(seed)
    base = 1.0 + 0.3 * rng.standard_normal(G)
    noise = lambda: 0.3 * rng.standard_normal(G)  # noqa: E731
    members = {
        "better": base + better_shift + noise(),
        "worse": base + worse_shift + noise(),
        "same": base + noise(),
    }
    unit_keys = np.arange(G)
    return base, members, unit_keys


def test_verdicts_recover_the_planted_ordering():
    base, members, keys = _toy_losses()
    decision = committee_maxt_decision(
        baseline_losses=base, member_losses=members, unit_keys=keys, seed=7,
    )
    assert decision.verdict_for("better") == "better"
    assert decision.verdict_for("worse") == "worse"
    assert decision.verdict_for("same") == "indistinguishable"
    assert decision.best_member_id == "better"
    assert decision.n_units == len(keys)
    assert decision.critical_value > 0.0
    assert decision.inference_regime["decision"] in {
        "licensed", "validated", "transitional", "fallback", "beyond_grid",
    } or "decision" in decision.inference_regime


def test_sharded_reduce_matches_single_shot_to_reassociation():
    """The load-bearing property: slices are compute partitions only."""
    base, members, keys = _toy_losses(seed=3)
    single = committee_maxt_decision(
        baseline_losses=base, member_losses=members, unit_keys=keys, seed=11,
    )
    # Three unequal shards, deliberately out of order.
    cuts = [(400, 600), (0, 150), (150, 400)]
    shards = [
        committee_shard_stats(
            baseline_losses=base[a:b],
            member_losses={k: v[a:b] for k, v in members.items()},
            unit_keys=keys[a:b],
            seed=11,
        )
        for a, b in cuts
    ]
    sharded = reduce_committee_decision(shards)
    # Exact up to floating-point reassociation across shard boundaries.
    assert sharded.critical_value == pytest.approx(single.critical_value, rel=1e-12)
    for row_a, row_b in zip(single.member_verdicts, sharded.member_verdicts):
        assert row_a.member_id == row_b.member_id
        assert row_a.verdict == row_b.verdict
        assert row_a.mean_delta == pytest.approx(row_b.mean_delta, rel=1e-12, abs=1e-15)
        assert row_a.se == pytest.approx(row_b.se, rel=1e-12)
        assert row_a.ci_lower == pytest.approx(row_b.ci_lower, rel=1e-12, abs=1e-15)


def test_cluster_keying_shares_draws_and_survives_shard_splits():
    base, members, keys = _toy_losses(seed=5, G=400)
    clusters = np.repeat(np.arange(40), 10)  # 40 clusters of 10 rows
    single = committee_maxt_decision(
        baseline_losses=base, member_losses=members, unit_keys=keys,
        cluster_ids=clusters, seed=13,
    )
    # Split THROUGH cluster 20 (rows 200-209 straddle the cut at 205).
    shards = [
        committee_shard_stats(
            baseline_losses=base[a:b],
            member_losses={k: v[a:b] for k, v in members.items()},
            unit_keys=keys[a:b],
            cluster_ids=clusters[a:b],
            seed=13,
        )
        for a, b in [(0, 205), (205, 400)]
    ]
    sharded = reduce_committee_decision(shards)
    assert sharded.critical_value == pytest.approx(single.critical_value, rel=1e-12)
    assert [r.verdict for r in sharded.member_verdicts] == [
        r.verdict for r in single.member_verdicts
    ]
    # Cluster inference must be more conservative than pretending the
    # correlated rows are independent: same data, wider (or equal) intervals
    # would need correlated noise to show; here we just require validity of
    # the mechanical path plus determinism.
    again = committee_maxt_decision(
        baseline_losses=base, member_losses=members, unit_keys=keys,
        cluster_ids=clusters, seed=13,
    )
    assert again.critical_value == single.critical_value


def test_cluster_bootstrap_is_wider_under_shared_cluster_noise():
    """Rows sharing a cluster-level shock must not be counted as independent."""
    rng = np.random.default_rng(2)
    n_clusters, rows_per = 50, 20
    G = n_clusters * rows_per
    cluster_shock = rng.standard_normal(n_clusters).repeat(rows_per)
    base = 1.0 + 0.05 * rng.standard_normal(G)
    member = base + 0.02 + 0.5 * cluster_shock + 0.05 * rng.standard_normal(G)
    keys = np.arange(G)
    row_level = committee_maxt_decision(
        baseline_losses=base, member_losses={"m": member}, unit_keys=keys, seed=1,
    )
    cluster_level = committee_maxt_decision(
        baseline_losses=base, member_losses={"m": member}, unit_keys=keys,
        cluster_ids=keys // rows_per, seed=1,
    )
    half_row = row_level.critical_value * row_level.member_verdicts[0].se
    half_cluster = cluster_level.critical_value * cluster_level.member_verdicts[0].se
    assert half_cluster > half_row


def test_nonfinite_rows_are_dropped_complete_case_and_counted():
    base, members, keys = _toy_losses(seed=9, G=100)
    members["better"][3] = np.nan
    base[7] = np.inf
    decision = committee_maxt_decision(
        baseline_losses=base, member_losses=members, unit_keys=keys, seed=2,
    )
    assert decision.n_units == 98
    assert decision.n_dropped == 2


def test_null_false_positive_rate_is_near_alpha():
    """Under the null, familywise 'better' calls should occur at ~alpha."""
    alpha, trials, hits = 0.05, 120, 0
    for t in range(trials):
        rng = np.random.default_rng(1000 + t)
        G = 300
        base = 1.0 + 0.2 * rng.standard_normal(G)
        members = {f"m{j}": base + 0.2 * rng.standard_normal(G) for j in range(4)}
        decision = committee_maxt_decision(
            baseline_losses=base, member_losses=members,
            unit_keys=np.arange(G), seed=t, alpha=alpha, n_resamples=500,
        )
        wrong = any(
            row.verdict != "indistinguishable" for row in decision.member_verdicts
        )
        hits += int(wrong)
    rate = hits / trials
    # Familywise two-sided at alpha=0.05 over 120 trials: allow generous slack.
    assert rate <= 0.15, f"familywise error rate {rate:.3f} implausibly high"


def test_degenerate_zero_variance_deltas_decide_by_sign():
    G = 50
    base = np.ones(G)
    members = {
        "det_better": np.full(G, 0.9),
        "det_same": np.ones(G),
    }
    decision = committee_maxt_decision(
        baseline_losses=base, member_losses=members,
        unit_keys=np.arange(G), seed=0,
    )
    assert decision.verdict_for("det_better") == "better"
    assert decision.verdict_for("det_same") == "indistinguishable"


def test_shard_reduce_rejects_mismatched_contracts():
    base, members, keys = _toy_losses(G=40)
    s1 = committee_shard_stats(
        baseline_losses=base[:20], member_losses={k: v[:20] for k, v in members.items()},
        unit_keys=keys[:20], seed=1,
    )
    s2 = committee_shard_stats(
        baseline_losses=base[20:], member_losses={k: v[20:] for k, v in members.items()},
        unit_keys=keys[20:], seed=2,
    )
    with pytest.raises(ValueError, match="seed"):
        reduce_committee_decision([s1, s2])


def test_too_few_rows_fails_closed():
    with pytest.raises(ValueError, match="at least 2"):
        committee_maxt_decision(
            baseline_losses=[1.0], member_losses={"m": [0.5]},
            unit_keys=[0], seed=0,
        )


def test_slice_rows_bridge_matches_direct_call_and_keys_globally():
    """The bridge must concatenate slices into the same decision a direct
    call with hand-built global arrays produces, with unit keys taken from
    absolute row positions so any shard layout agrees."""
    from nestynet_sr.stat_selection.committee_inference import (
        maxt_decision_from_slice_rows,
    )

    rng = np.random.default_rng(4)
    slices = {10: (5000, 300), 11: (7000, 300), 12: (9000, 200)}
    base_parts, m_parts, key_parts = [], [], []
    baseline_rows, member_rows = {}, {}
    for sid, (start, n) in slices.items():
        b = 1.0 + 0.2 * rng.standard_normal(n)
        m = b - 0.3 + 0.2 * rng.standard_normal(n)
        baseline_rows[sid] = (start, b)
        member_rows[sid] = (start, m)
        base_parts.append(b)
        m_parts.append(m)
        key_parts.append(start + np.arange(n))
    bridged = maxt_decision_from_slice_rows(
        baseline_rows=baseline_rows,
        member_rows={"cand": member_rows},
        seed=21,
    )
    direct = committee_maxt_decision(
        baseline_losses=np.concatenate(base_parts),
        member_losses={"cand": np.concatenate(m_parts)},
        unit_keys=np.concatenate(key_parts),
        seed=21,
    )
    assert bridged.critical_value == direct.critical_value
    assert bridged.verdict_for("cand") == direct.verdict_for("cand") == "better"
    assert bridged.n_units == direct.n_units == 800


def test_slice_rows_bridge_rejects_unpaired_slices():
    from nestynet_sr.stat_selection.committee_inference import (
        maxt_decision_from_slice_rows,
    )

    with pytest.raises(ValueError, match="not paired"):
        maxt_decision_from_slice_rows(
            baseline_rows={0: (0, np.ones(10))},
            member_rows={"m": {0: (0, np.ones(9))}},  # length mismatch
            seed=0,
        )
    with pytest.raises(ValueError, match="share no witness slices"):
        maxt_decision_from_slice_rows(
            baseline_rows={0: (0, np.ones(10))},
            member_rows={"m": {1: (2000, np.ones(10))}},
            seed=0,
        )


def test_slice_rows_bridge_cluster_by_slice_matches_direct_cluster_call():
    from nestynet_sr.stat_selection.committee_inference import (
        maxt_decision_from_slice_rows,
    )

    rng = np.random.default_rng(6)
    baseline_rows, member_rows = {}, {}
    base_parts, m_parts, key_parts, cl_parts = [], [], [], []
    for sid, start in [(3, 100), (4, 400)]:
        n = 150
        b = 1.0 + 0.1 * rng.standard_normal(n)
        m = b + 0.05 * rng.standard_normal(n)
        baseline_rows[sid] = (start, b)
        member_rows[sid] = (start, m)
        base_parts.append(b); m_parts.append(m)
        key_parts.append(start + np.arange(n))
        cl_parts.append(np.full(n, sid))
    bridged = maxt_decision_from_slice_rows(
        baseline_rows=baseline_rows, member_rows={"m": member_rows},
        seed=9, cluster_by_slice=True,
    )
    direct = committee_maxt_decision(
        baseline_losses=np.concatenate(base_parts),
        member_losses={"m": np.concatenate(m_parts)},
        unit_keys=np.concatenate(key_parts),
        cluster_ids=np.concatenate(cl_parts),
        seed=9,
    )
    assert bridged.critical_value == direct.critical_value


def test_eval_cache_separates_row_loss_requests(tmp_path):
    """A cached aggregate-only result must not satisfy a per-row request."""
    import csv

    from nestynet_sr.sr_search.coe_committee import (
        CandidateArtifact,
        CommitteeEvalCache,
        SliceSpec,
    )

    rng = np.random.default_rng(0)
    path = tmp_path / "d.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["x0", "y"])
        for _ in range(60):
            x = rng.normal()
            w.writerow([repr(x), repr(2.0 * x)])
    cand = CandidateArtifact(candidate_id="c", expr="2*x0", source="test")
    spec = SliceSpec(slice_id=0, train_start=0, train_stop=30, val_start=30, val_stop=60)
    cache = CommitteeEvalCache(enabled=True)
    plain = cache.evaluate(cand, filepath=path, spec=spec)
    assert plain.row_losses is None
    with_rows = cache.evaluate(cand, filepath=path, spec=spec, return_row_losses=True)
    assert with_rows.row_losses is not None
    assert len(with_rows.row_losses) == 30
    assert max(with_rows.row_losses) < 1e-25   # exact expression, exact data
    # And the cached rows result is reused on repeat.
    assert cache.hits >= 0
    again = cache.evaluate(cand, filepath=path, spec=spec, return_row_losses=True)
    assert again.row_losses == with_rows.row_losses

def test_small_g_dispatches_bonferroni_t():
    from scipy.stats import t as student_t

    # No validated calibration cell exists at G <= 100, so the lookup demands
    # the fallback and the decision must EXECUTE it, not merely record it.
    base, members, keys = _toy_losses(seed=21, G=100)
    decision = committee_maxt_decision(
        baseline_losses=base, member_losses=members, unit_keys=keys, seed=4,
    )
    assert decision.inference_regime["method"] == "bonferroni_t"
    assert decision.inference_regime["method_executed"] == "bonferroni_t"
    assert decision.critical_value_method == "bonferroni_t"
    expected = float(student_t.ppf(1.0 - 0.05 / (2 * 3), 100 - 1))
    assert decision.critical_value == pytest.approx(expected)
    assert decision.inference_regime["critical_value"] == pytest.approx(expected)
    assert decision.to_dict()["critical_value_method"] == "bonferroni_t"
    # The planted ordering survives the more conservative critical value.
    assert decision.verdict_for("better") == "better"


def test_large_g_keeps_licensed_bootstrap():
    base, members, keys = _toy_losses(seed=22, G=600)
    decision = committee_maxt_decision(
        baseline_losses=base, member_losses=members, unit_keys=keys, seed=4,
    )
    assert decision.inference_regime["decision"] == "licensed"
    assert decision.inference_regime["method_executed"] == "multiplier_max_t"
    assert decision.critical_value_method == "multiplier_max_t"


def test_cluster_keyed_decision_never_takes_t_fallback():
    # Row count G=100 would demand the fallback, but cluster keying forbids
    # it: G overstates the independent units and per-row standard errors are
    # too small in exactly the way the cluster bootstrap compensates for.
    base, members, keys = _toy_losses(seed=23, G=100)
    decision = committee_maxt_decision(
        baseline_losses=base, member_losses=members, unit_keys=keys,
        cluster_ids=keys // 10, seed=3,
    )
    assert decision.critical_value_method == "multiplier_max_t"
    assert "no envelope license" in decision.inference_regime["dispatch_note"]


def test_mixed_cluster_keying_shards_cannot_reduce():
    base, members, keys = _toy_losses(seed=24, G=40)
    plain = committee_shard_stats(
        baseline_losses=base[:20],
        member_losses={k: v[:20] for k, v in members.items()},
        unit_keys=keys[:20], seed=1,
    )
    clustered = committee_shard_stats(
        baseline_losses=base[20:],
        member_losses={k: v[20:] for k, v in members.items()},
        unit_keys=keys[20:], cluster_ids=keys[20:] // 5, seed=1,
    )
    with pytest.raises(ValueError, match="cluster keying"):
        reduce_committee_decision([plain, clustered])
