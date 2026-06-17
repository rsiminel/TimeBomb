"""Calibration tests: the belief's stated probabilities match empirical frequencies.

A Bayesian posterior under a correctly-specified model is *calibrated* — among players the
model calls `P(bad)=p`, about a fraction `p` really are bad — so these are correctness /
regression guards built on the now-complete `General.py`. They are also what the
calibration work surfaced: two simulator/model mismatches that made the *simulation*
diverge from the assumed model (the inference itself was already oracle-correct) —

  1. `DistributeWires` dealt wires uniformly over *players*, not over *slots* (they differ
     only because the bomb hand has one fewer slot), and
  2. `PlayAuto` folded the bomb-detonating cut into the role belief as a "no-bomb"
     observation,

both fixed in `General.py`. Removing them dropped the `P(bad)` ECE from ~0.026 to ~0.01
(see `Calibration.py`).

Run:  .venv/bin/python -m pytest tests/test_calibration.py -q
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "timebomb"))

import Calibration as cal


def test_reliability_metric_sanity():
    """The ECE metric itself: ~0 for a perfectly calibrated synthetic forecaster, large
    for an over-confident one (guards the test's own measuring stick)."""
    rng = np.random.default_rng(0)
    # well-calibrated: outcome ~ Bernoulli(prediction)
    p = rng.uniform(0.0, 1.0, 40000)
    y = (rng.uniform(0.0, 1.0, 40000) < p).astype(float)
    _, ece = cal.reliability(p, y)
    assert ece < 0.02, f"calibrated forecaster scored ECE={ece:.4f}"
    # over-confident: always predicts 0.9 but the truth is 0.3
    p2 = np.full(40000, 0.9)
    y2 = (rng.uniform(0.0, 1.0, 40000) < 0.3).astype(float)
    _, ece2 = cal.reliability(p2, y2)
    assert ece2 > 0.5, f"over-confident forecaster scored only ECE={ece2:.4f}"


def test_pbad_is_calibrated():
    """Over full simulated games, P(bad) matches the empirical bad-rate. Calibrated ECE
    here is ~0.009 (seed 0, 500 games, 5 bins); the threshold guards against gross
    miscalibration regressions."""
    pred, out = cal.collect_pbad(num_players=5, num_games=500, seed=0)
    rows, ece = cal.reliability(pred, out, n_bins=5)
    assert ece < 0.03, f"P(bad) miscalibrated: ECE={ece:.4f}\n{rows}"


def test_pbomb_is_calibrated():
    """Over round-starts, declaration-time P(bomb) matches the empirical bomb-rate.
    Calibrated ECE ~0.008 (seed 0, 3000 rounds). The bomb prior is the piece the ADR-0007
    lie-count factor corrected, so this is a direct guard on that fix."""
    pred, out = cal.collect_pbomb(num_players=5, num_rounds=3000, num_bad=2, seed=0)
    rows, ece = cal.reliability(pred, out, n_bins=10)
    assert ece < 0.02, f"P(bomb) miscalibrated: ECE={ece:.4f}\n{rows}"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
