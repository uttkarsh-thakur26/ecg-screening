#!/usr/bin/env bash
# The one-time evaluation of the frozen release model on fold 10 (reports/TEST_PLAN.md); writes reports/metrics.json.
# It refuses to run twice; see the module docstring for the rehearsal on fold 9 and the bug-fix rerun.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.evaluation.evaluate_test "$@"
