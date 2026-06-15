"""Tests for the OneBadGuyNoBomb variant (1 bad guy, no bomb).

Run with pytest:   .venv/bin/python -m pytest test_OneBadGuyNoBomb.py -q
Or standalone:     .venv/bin/python test_OneBadGuyNoBomb.py

The reference implementations below (hypergeometric PMF, brute-force posterior and
expected-wire calculations) are written independently of the module under test --
using math.comb rather than UsefulFunctions -- so a shared bug cannot hide behind
a matching test.
"""
import math
from random import Random

import numpy as np

import OneBadGuyNoBomb as ob

TOL = 1e-9


# --- independent reference implementations -------------------------------------

def hypergeom_pmf(found, hand_size, wires, revealed):
    """P(find `found` wires in `revealed` draws from a `hand_size`-card hand that
    holds `wires` wires) -- the standard hypergeometric PMF, via math.comb."""
    found, hand_size = int(round(found)), int(round(hand_size))
    wires, revealed = int(round(wires)), int(round(revealed))
    if not (0 <= wires <= hand_size and 0 <= revealed <= hand_size):
        return 0.0
    denom = math.comb(hand_size, revealed)
    if denom == 0:
        return 0.0
    return math.comb(wires, found) * math.comb(hand_size - wires, revealed - found) / denom


def bad_wire_count(decls, active_wires, found, bad):
    """Wires the bad guy must hold under hypothesis `bad`, given truthful goods."""
    return active_wires + np.sum(found) - np.sum(decls) + decls[bad]


def brute_probcut(decls, prior, revealed, found, hand_size, active_wires):
    """Exact posterior over who-is-bad, recomputed independently of the module."""
    n = len(decls)
    post = np.zeros(n)
    for bad in range(n):
        wires = decls.astype(float).copy()
        wires[bad] = bad_wire_count(decls, active_wires, found, bad)
        like = 1.0
        for p in range(n):
            like *= hypergeom_pmf(found[p], hand_size, wires[p], revealed[p])
        post[bad] = prior[bad] * like
    s = post.sum()
    return prior.astype(float).copy() if s == 0 else post / s


def brute_pwire(decls, probs, revealed, found, hand_size, active_wires):
    """Expected wire probability per player, recomputed independently."""
    n = len(decls)
    pw = np.zeros(n)
    for i in range(n):
        remaining = hand_size - revealed[i]
        if remaining <= 0:
            continue
        expected = 0.0
        for bad in range(n):
            wires_i = bad_wire_count(decls, active_wires, found, bad) if bad == i else decls[i]
            wires_i -= found[i]
            if 0 <= wires_i <= remaining:
                expected += probs[bad] * wires_i
        pw[i] = expected / remaining
    return pw


# --- helpers -------------------------------------------------------------------

def is_distribution(v, total=1.0):
    return (not np.any(np.isnan(v)) and np.all(v >= -TOL)
            and np.all(v <= 1 + TOL) and abs(v.sum() - total) < 1e-6)


def random_consistent_state(rng):
    """A state drawn from an actual play-out with a known hidden bad guy, so all
    counts are mutually consistent (no impossible observations).

    Returns ``(decls, revealed, found, hand_size, active_now, total_active)`` where
    ``total_active`` is the round's wire count at declaration time (the argument
    ``ProbDeclaration`` expects) and ``active_now`` is what remains after the cuts
    so far (the argument ``ProbCut`` / ``P_wire`` expect)."""
    n = rng.randint(4, 6)
    hand_size = rng.randint(2, 5)
    bad = rng.randrange(n)
    total_active = rng.randint(0, n)
    # Deal `total_active` wires across hands, at most hand_size each.
    wires = np.zeros(n, dtype=int)
    given = 0
    while given < total_active:
        c = rng.randrange(n)
        if wires[c] < hand_size:
            wires[c] += 1
            given += 1
    # Declarations: good guys truthful, bad guy lies uniformly.
    decls = wires.astype(float).copy()
    decls[bad] = rng.randint(0, min(hand_size, total_active))
    # Cut some cards honestly from the hidden hands.
    revealed = np.zeros(n, dtype=int)
    found = np.zeros(n, dtype=int)
    active_wires = total_active
    for _ in range(rng.randint(0, n)):
        c = rng.randrange(n)
        if revealed[c] >= hand_size:
            continue
        # Draw a card: probability it is a wire is (remaining wires)/(remaining cards)
        rem_cards = hand_size - revealed[c]
        rem_wires = wires[c] - found[c]
        if rng.randint(1, rem_cards) <= rem_wires:
            found[c] += 1
            active_wires -= 1
        revealed[c] += 1
    return decls, revealed, found, hand_size, active_wires, total_active


# --- ProbDeclaration -----------------------------------------------------------

def test_probdeclaration_uniform_when_no_excess():
    decls = np.array([2., 1., 1., 0.])  # sum 4 == active_wires
    p = ob.ProbDeclaration(decls, hand_size=5, active_wires=4)
    assert np.allclose(p, 0.25)


def test_probdeclaration_known_values_positive_excess():
    # sum(decls)=4, active=2 -> excess=2. weights C(decls[i],2): [3,0,0] -> [1,0,0]
    p = ob.ProbDeclaration(np.array([3., 1., 0.]), hand_size=5, active_wires=2)
    assert np.allclose(p, [1., 0., 0.])


def test_probdeclaration_known_values_negative_excess():
    # sum=2, active=4 -> excess=-2. weights C(hand_size-decls[i],2): [1,1,3] -> [.2,.2,.6]
    p = ob.ProbDeclaration(np.array([1., 1., 0.]), hand_size=3, active_wires=4)
    assert np.allclose(p, [0.2, 0.2, 0.6])


def test_probdeclaration_invariants_sweep():
    rng = Random(1)
    for _ in range(4000):
        n = rng.randint(3, 6)
        hand_size = rng.randint(2, 5)
        active = rng.randint(0, n)
        decls = np.array([float(rng.randint(0, hand_size)) for _ in range(n)])
        p = ob.ProbDeclaration(decls, hand_size, active)
        # Either a proper distribution or all-zeros (impossible declarations).
        assert not np.any(np.isnan(p))
        assert np.all(p >= -TOL) and np.all(p <= 1 + TOL)
        assert abs(p.sum() - 1) < 1e-6 or abs(p.sum()) < 1e-6


# --- ProbCut -------------------------------------------------------------------

def test_probcut_matches_bruteforce_sweep():
    rng = Random(2)
    max_diff = 0.0
    for _ in range(5000):
        decls, revealed, found, hand_size, active, total = random_consistent_state(rng)
        prior = ob.ProbDeclaration(decls, hand_size, total)
        if prior.sum() == 0:
            continue
        got = ob.ProbCut(decls, prior, revealed, found, hand_size, active)
        ref = brute_probcut(decls, prior, revealed, found, hand_size, active)
        max_diff = max(max_diff, np.max(np.abs(got - ref)))
    assert max_diff < 1e-9, f"max diff vs brute force = {max_diff}"


def test_probcut_preserves_distribution_sweep():
    rng = Random(3)
    for _ in range(5000):
        decls, revealed, found, hand_size, active, total = random_consistent_state(rng)
        prior = ob.ProbDeclaration(decls, hand_size, total)
        if prior.sum() == 0:
            continue
        post = ob.ProbCut(decls, prior, revealed, found, hand_size, active)
        assert is_distribution(post)


def test_probcut_handles_impossible_observation_without_nan():
    # The edge case that produced all-NaN in the removed legacy ProbCut:
    # player 2 declared both their cards as wires, yet a cut found none.
    decls = np.array([1., 1., 2., 0.])
    prior = ob.ProbDeclaration(decls, hand_size=2, active_wires=4)
    post = ob.ProbCut(decls, prior, np.array([0, 0, 1, 0]), np.zeros(4, int),
                      hand_size=2, active_wires=4)
    assert not np.any(np.isnan(post))
    assert is_distribution(post)


def test_probcut_certainty_is_absorbing():
    decls = np.array([2., 1., 0.])
    prior = np.array([0., 1., 0.])  # player 1 already known to be bad
    post = ob.ProbCut(decls, prior, np.array([1, 1, 0]), np.array([0, 1, 0]),
                      hand_size=3, active_wires=2)
    assert np.allclose(post, [0., 1., 0.])


# --- P_wire --------------------------------------------------------------------

def test_pwire_uses_remaining_cards_not_full_hand():
    # Player 0 is certainly good (probs[0]=0), declared 3, found 1 of 2 revealed.
    # Correct: (3-1)/(5-2) = 2/3. The old (buggy) denominator H=5 gave 2/5.
    decls = np.array([3., 2., 2., 1.])
    probs = np.array([0., 0., 0., 1.])
    pw = ob.P_wire(decls, probs, np.array([2, 0, 0, 0]), np.array([1, 0, 0, 0]),
                   hand_size=5, active_wires=7)
    assert abs(pw[0] - 2.0 / 3.0) < TOL
    assert abs(pw[0] - 2.0 / 5.0) > 0.1  # decisively not the old behaviour


def test_pwire_zero_when_hand_fully_revealed():
    decls = np.array([2., 2., 1., 0.])
    probs = np.array([0.25, 0.25, 0.25, 0.25])
    pw = ob.P_wire(decls, probs, np.array([5, 0, 0, 0]), np.array([2, 0, 0, 0]),
                   hand_size=5, active_wires=3)
    assert pw[0] == 0.0


def test_pwire_matches_bruteforce_sweep():
    rng = Random(4)
    max_diff = 0.0
    for _ in range(5000):
        decls, revealed, found, hand_size, active, total = random_consistent_state(rng)
        prior = ob.ProbDeclaration(decls, hand_size, total)
        if prior.sum() == 0:
            continue
        probs = ob.ProbCut(decls, prior, revealed, found, hand_size, active)
        got = ob.P_wire(decls, probs, revealed, found, hand_size, active)
        ref = brute_pwire(decls, probs, revealed, found, hand_size, active)
        max_diff = max(max_diff, np.max(np.abs(got - ref)))
        assert np.all(got >= -TOL) and np.all(got <= 1 + TOL)
    assert max_diff < 1e-9, f"max diff vs brute force = {max_diff}"


# --- CombineProbs --------------------------------------------------------------

def test_combineprobs_single_round_is_identity():
    p = np.array([0.2, 0.3, 0.5])
    assert np.allclose(ob.CombineProbs([p]), p)


def test_combineprobs_product_then_normalise():
    a = np.array([0.2, 0.8])
    b = np.array([0.5, 0.5])
    # product [0.1, 0.4] -> normalised [0.2, 0.8]
    assert np.allclose(ob.CombineProbs([a, b]), [0.2, 0.8])


def test_combineprobs_zero_is_absorbing():
    a = np.array([0.0, 1.0])
    b = np.array([0.5, 0.5])
    assert np.allclose(ob.CombineProbs([a, b]), [0.0, 1.0])


# --- entropy -------------------------------------------------------------------

def test_entropy_uniform_and_certain():
    assert abs(ob.H(np.full(4, 0.25)) - 2.0) < 1e-9   # log2(4) = 2 bits
    assert ob.H(np.array([0., 1., 0., 0.])) == 0.0


# --- ProbSus -------------------------------------------------------------------

def test_probsus_updates_in_place_without_nameerror(monkeypatch=None):
    # Feed the three input() prompts: suspect name, P(behaviour|good), P(.|bad).
    answers = iter(["Bob", "30", "60"])
    import builtins
    original = builtins.input
    builtins.input = lambda *a, **k: next(answers)
    try:
        players = ["Alice", "Bob", "Clara", "Darryl"]
        probs = np.array([0.25, 0.25, 0.25, 0.25])
        ob.ProbSus(players, probs)
    finally:
        builtins.input = original
    # Suspect (Bob): 0.25*60 / (0.25*60 + 0.75*30) = 0.4; others rescaled to 0.2 each.
    assert np.allclose(probs, [0.2, 0.4, 0.2, 0.2])


# --- standalone runner (no pytest required) ------------------------------------

if __name__ == "__main__":
    import sys
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
