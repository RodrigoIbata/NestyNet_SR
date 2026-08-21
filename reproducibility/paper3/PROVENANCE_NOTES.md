# Paper III benchmark provenance notes

The two benchmark tables are backed by capsules of per-problem evidence
rather than by the campaign workspaces that produced them. These notes
record what each cell is, how "solved" is decided, and which claims the
evidence does and does not support.

## What a capsule is

One capsule per table cell: every `pb*.report.json`, the `summary.csv`
carrying the recovered expressions and the campaign's own exact-recovery
verdicts, a `campaign_manifest.json`, and the cell's `structural_audit.json`.
Models, checkpoints and logs are excluded; a 120-problem cell is a few
megabytes instead of gigabytes. Capsules are collected with
`scripts/collect_srbench_capsule.py`, which also works on a committee
workspace through `--results-dir results_CoE`.

## Experimental arms, read from the runs

Directory names are not evidence, so the collector summarizes each
campaign's actual configuration into `campaign_manifest.json:arm_signature`,
counting every field across the campaign's own reports so that variation
between problems stays visible. The three arms are distinguished by what
holds selection authority, not by data volume:

| arm | data | selection authority | committee |
|---|---|---|---|
| `ndata2k` | 2000 train / 2000 validation | heuristic acceptance | none |
| `stat` | 2000 / 2000 | statistical selection | none |
| `coe_pareto` | 2000 / 2000 | statistical selection | enabled, not authoritative |

In the committee arm the reports state the relationship explicitly: the
committee is retained as a proposal and diagnostic layer and does not select
the reported structure. The sealed audit partition used for the statistical
decisions is 20000 rows, drawn from the full dataset and therefore much
larger than the training split; the training budget and the audit budget are
independent quantities.

## What counts as solved

Both tables use one criterion, implemented once in
`scripts/_structural_verdict.py` and imported by both auditors so they cannot
drift apart. A recovered expression counts as a structural recovery when it
is algebraically identical to the target up to the values of its fitted
constants:

1. refit every free numeric constant on the canonical noiseless data for the
   same problem, including constants inside transcendentals such as a
   frequency or phase within a sine, which a coefficient-only refit leaves
   untouched; exponents are held fixed because they are structure rather
   than calibration,
2. re-snap the coefficients with the pipeline's own polisher at its shipped
   defaults,
3. accept when the polished expression predicts that data at the
   noiseless-fit floor, a relative RMSE of 1e-10.

No symbolic-equivalence judgment enters the verdict, and no threshold is
chosen by hand: 1e-10 is anchored to the measured floor of the noiseless
campaign, whose worst exact fit is pb116 at 2.4e-13.

## Revisions

The committee campaigns ran on an HPC cluster at NestyNet-SR revision
`44520a2` with a locally modified tree, uniformly across all problems. The
laptop campaigns span several revisions, recorded per problem in each
capsule manifest. Audits and verdicts were computed afterwards from the
capsules and are therefore independent of the campaign revisions.

## Scope of the claims

Case-level outcomes are knife-edges at fixed code: identical-code reruns can
flip individual problems, and the allocation of a run to one processor model
rather than another is enough to change a search trajectory. The quantities
claimed are therefore the aggregate per-cell counts, reproducible at the
level of a few cases rather than case by case. Capsules record the wall
time of every run; the noiseless campaign's distribution is stored with the
frozen expectations, and those timings come from a campaign that ran eight
problems concurrently at one thread each, so they include some contention.

Wall-clock budgets were finite. A cell may declare problems under
`not_completed` in `expected_results.json`: those runs exceeded the budget
and never returned, they count as non-solves in the totals, and they are
recorded separately from problems that ran and failed, because those are
different facts.
