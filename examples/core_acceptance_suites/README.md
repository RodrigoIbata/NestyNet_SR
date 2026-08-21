# Core Acceptance Suites

These manifests drive [`nestynet_sr/run_core_acceptance_suite.py`](../../nestynet_sr/run_core_acceptance_suite.py).

They are meant to protect the frozen SR/DE core with explicit mathematical
checks, rather than just subprocess exit codes.

## Available manifests

- `frozen_core_fast.json`
  - Default fast frozen-core gate.
  - Combines a small set of existing SR component tests, the controller harness smoke run, one exact AI Feynman SR recovery case, and one first-class DE recovery case.
- `frozen_core_smoke.json`
  - Small verified smoke gate.
  - Covers one exact AI Feynman SR case without factorized symbolic search and one first-class DE case through `run_SR.py`.

## Typical workflow

Run the frozen suite:

```bash
python nestynet_sr/run_core_acceptance_suite.py \
  --suite_manifest examples/core_acceptance_suites/frozen_core_fast.json
```

Bless a trusted run as the baseline:

```bash
python nestynet_sr/run_core_acceptance_suite.py \
  --suite_manifest examples/core_acceptance_suites/frozen_core_fast.json \
  --bless_baseline results/core_acceptance_baselines
```

That writes a JSON file such as:

```text
results/core_acceptance_baselines/frozen_core_fast.baseline.json
```

Compare a later run against that blessed baseline:

```bash
python nestynet_sr/run_core_acceptance_suite.py \
  --suite_manifest examples/core_acceptance_suites/frozen_core_fast.json \
  --baseline results/core_acceptance_baselines/frozen_core_fast.baseline.json \
  --fail_on_regression
```

## Why both thresholds and baselines?

- The manifest expectations answer: "does this case still meet the absolute bar?"
- The blessed baseline answers: "did this change get materially worse than the last trusted run?"

For frozen exact-recovery SR cases, the absolute checks are the primary guard.
The baseline comparison is more useful for runtime drift and softer DE metrics.
