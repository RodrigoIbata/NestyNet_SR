# Reproducible cheap-to-CoE campaign escalation

`scripts/run_allstages_escalating.sh` runs an AI Feynman campaign in two
resumable phases. It first runs the ordinary configuration, builds a
deterministic decision manifest, and sends only completed problems without an
eligible symbolic selection to the committee of experts (CoE).

The escalation decision is truth-blind. It never reads `truth_eval`, nested
truth canaries, algebraic equivalence to the benchmark answer, or the answer
formula in `equations.txt`. It uses only:

- process completion from each per-problem suite summary;
- the pipeline's final symbolic selection and applied/eligibility state;
- coefficient-metadata coverage for symbols in that expression;
- Stage-C and final-selection unit certificates; and
- the explicit final-polish no-safe-replacement marker.

This separates search control from post-run benchmark scoring. A process crash
is retried in the same phase and can never trigger CoE.

## Run a new campaign

In a workspace created by `scripts/setup_paper3_reproduction.py`:

```bash
cd SRBench_0.001
JOBS=1 ./scripts/run_allstages_escalating.sh
```

The cheap outputs remain in `results/`, CoE outputs go to `results_CoE/`, and
the controller writes `coe_escalation_manifest.json` in the campaign root.
Set `IDS`, `START_ID`, `END_ID`, `SKIP_IDS`, and `PROBLEMS_IGNORE` exactly as
for the ordinary campaign runner.

For an older workspace that does not yet contain the wrapper, invoke it from
the source checkout:

```bash
CAMPAIGN_ROOT=/path/to/SRBench_0.001 \
JOBS=1 \
/path/to/NestyNet_SR/scripts/run_allstages_escalating.sh
```

Rerun the same command after interruption. Eligible completed work is skipped;
only pending or retryable work is launched.

## Escalate an already completed cheap campaign

Build and inspect the queue without launching anything expensive:

```bash
CAMPAIGN_ROOT=/path/to/SRBench_0.001_test \
RUN_CHEAP=0 \
RUN_COE=0 \
/path/to/NestyNet_SR/scripts/run_allstages_escalating.sh
```

List the queued IDs explicitly:

```bash
python /path/to/NestyNet_SR/scripts/build_coe_escalation_manifest.py list \
  --manifest /path/to/SRBench_0.001_test/coe_escalation_manifest.json \
  --action run_coe
```

After reviewing the queue, run it with:

```bash
CAMPAIGN_ROOT=/path/to/SRBench_0.001_test \
RUN_CHEAP=0 \
JOBS=1 \
/path/to/NestyNet_SR/scripts/run_allstages_escalating.sh
```

The default expensive phase is `COE_MODE=reservoir_discovery`, with the cheap
results directory supplied as `COE_RESERVOIR_PATHS`. Override
`ESCALATION_COE_MODE` or any ordinary `COE_*` setting when a campaign protocol
requires a different committee configuration.

## Manifest states

Each active problem has exactly one action and a stable reason code:

- `pending`: no completed cheap summary exists;
- `retry_cheap`: the cheap process, summary, or report failed;
- `run_coe`: the cheap process completed but no eligible symbolic selection
  exists;
- `retry_coe`: an attempted CoE process, summary, or report failed;
- `skip`: an eligible cheap or CoE selection already exists; or
- `terminal_failure`: CoE completed without an eligible selection.

For identical summary/report artifacts, requested IDs, and exclusions, the
manifest is byte-identical. It contains no timestamp or absolute result
directory, and its decision digests cover only the truth-blind projection.
New reports also persist a `campaign_outcome` block so the exact settled reason
travels with the report.
