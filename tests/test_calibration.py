"""Calibration tests: each module's stated probabilities match empirical frequencies.

A Bayesian posterior under a correctly-specified model is *calibrated* — among players the
model calls `P(bad)=p`, about a fraction `p` really are bad — so these are correctness /
regression guards across every variant and `General`. They are also the detector that
surfaced a class of simulator/model mismatch (the inference itself is oracle-correct, but
the `PlayAuto` *generators* had drifted from the model):

  * wires dealt uniformly over *players* instead of *slots* (`DistributeWires`-style loops),
    which diverge whenever capacities differ — and even for equal capacity (N=2,H=2,A=2:
    `(1/4,1/2,1/4)` vs the hypergeometric `(1/6,2/3,1/6)`);
  * the no-bomb variants generating the bad guy's lie as `randint(0, min(H, A))` — capped at
    the active-wire total — instead of uniform over `{0..H}`;
  * `General.PlayAuto` folding the bomb-detonating cut into the role belief as a "no-bomb"
    observation.

All fixed; this suite locks the fixes in. Run:
  .venv/bin/python -m pytest tests/test_calibration.py -q
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "timebomb"))

import Calibration as cal
import OneBadGuyNoBomb as v1
import TwoBadGuysNoBomb as v2
import OneBadGuyOneBomb as v3
import TwoBadGuysOneBomb as v4
import General as gen


def test_reliability_metric_sanity():
    """The ECE metric itself: ~0 for a perfectly calibrated synthetic forecaster, large for
    an over-confident one (guards the test's own measuring stick)."""
    rng = np.random.default_rng(0)
    p = rng.uniform(0.0, 1.0, 40000)
    y = (rng.uniform(0.0, 1.0, 40000) < p).astype(float)  # outcome ~ Bernoulli(prediction)
    _, ece = cal.reliability(p, y)
    assert ece < 0.02, f"calibrated forecaster scored ECE={ece:.4f}"
    p2 = np.full(40000, 0.9)
    y2 = (rng.uniform(0.0, 1.0, 40000) < 0.3).astype(float)  # claims 0.9, truth 0.3
    _, ece2 = cal.reliability(p2, y2)
    assert ece2 > 0.5, f"over-confident forecaster scored only ECE={ece2:.4f}"


# (module, PlayAuto, num_players, P(bad) ECE threshold)
PBAD_CASES = [
    ("OneBadGuyNoBomb", v1.PlayAuto, 4, 0.025),
    ("TwoBadGuysNoBomb", v2.PlayAuto, 6, 0.025),
    ("OneBadGuyOneBomb", v3.PlayAuto, 4, 0.035),
    ("TwoBadGuysOneBomb", v4.PlayAuto, 6, 0.035),
    ("General(N=5)", gen.PlayAuto, 5, 0.030),
]

# (module, ProbDeclaration, bomb-marginal extractor, num_players, num_bad)
PBOMB_CASES = [
    ("OneBadGuyOneBomb", lambda d, h, a: v3.ProbDeclaration(d, h, a),
     lambda m: v3.DeMatrix(m)[1], 4, 1),
    ("TwoBadGuysOneBomb", lambda d, h, a: v4.ProbDeclaration(d, h, a),
     lambda m: v4.DeTensor(m)[1], 6, 2),
    ("General(N=5)", lambda d, h, a: gen.ProbDeclaration(d, h, a, 2, 1),
     lambda m: gen.Separate(m, 2, 1)[1], 5, 2),
]


def test_pbad_is_calibrated():
    """Every PlayAuto's final P(bad) matches the empirical bad-rate (calibrated ECE here is
    ~0.002-0.02 by variant; thresholds guard against gross miscalibration)."""
    for name, play, n, thresh in PBAD_CASES:
        pred, out = cal.collect_pbad(play, n, 400, seed=0)
        rows, ece = cal.reliability(pred, out, n_bins=5)
        assert ece < thresh, f"{name}: P(bad) ECE={ece:.4f} >= {thresh}\n{rows}"


def test_pbomb_is_calibrated():
    """Declaration-time P(bomb) matches the empirical bomb-rate — the piece ADR 0007's
    lie-count factor corrected (calibrated ECE ~0.003)."""
    for name, pdecl, bmarg, n, num_bad in PBOMB_CASES:
        pred, out = cal.collect_pbomb(pdecl, bmarg, n, num_bad, 3000, seed=0)
        rows, ece = cal.reliability(pred, out, n_bins=10)
        assert ece < 0.015, f"{name}: P(bomb) ECE={ece:.4f} >= 0.015\n{rows}"


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
