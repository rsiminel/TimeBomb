"""Tests for the TwoBadGuysNoBomb variant (2 bad guys, no bomb).

Run with pytest:   .venv/bin/python -m pytest tests/test_TwoBadGuysNoBomb.py -q
Or standalone:     .venv/bin/python tests/test_TwoBadGuysNoBomb.py

The belief state is a lower-triangular matrix ``probs[i][j]`` (i > j) holding
P(players i and j are the bad pair). The reference implementations below are written
independently of the module under test -- the bad pair's wire split is summed
explicitly with multivariate-hypergeometric weights C(H,wi)·C(H,wj)/C(2H,bg) (the
docs/model.md §3.4.1 placement law), structurally different from the module's
collapsed closed form -- so a shared bug cannot hide behind a matching test.
"""
import itertools
import math
import random
import sys
from pathlib import Path
from random import Random

import numpy as np

# Make the timebomb/ source root importable when run standalone (pytest uses the
# repo-root conftest.py for the same effect).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "timebomb"))

import TwoBadGuysNoBomb as tb

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


def pairs(n):
    """The (i, j) index pairs of the lower-triangular belief matrix, i > j."""
    return [(i, j) for i in range(n) for j in range(i)]


def group_wires(decls, found, active_wires, i, j):
    """Wires the bad pair {i, j} must hold given truthful good guys."""
    good_decls = np.sum(decls) - decls[i] - decls[j]
    return active_wires + np.sum(found) - good_decls


def generative_declaration_prior(decls, hand_size, active_wires):
    """Posterior P(pair {i,j} is bad | decls) under the generative model
    (model.md §2 lie model + §3.3), enumerated independently of the module:

      * the bad pair is chosen uniformly over the C(N,2) pairs;
      * the A wires are dealt across the N hands ~ multivariate hypergeometric, so a
        true wire vector w (0<=w[g]<=H, sum==A) has weight Π_g C(H, w[g]);
      * good guys (g not in {i,j}) declare truthfully -> w[g] == decls[g];
      * both bad guys declare uniformly over {0..H} -> constant factor, cancels.

    Enumerates every wire vector via itertools.product and sums raw deal weights --
    no closed form, no division -- so it shares no algebra with ProbDeclaration.
    Falls back to the uniform-over-pairs matrix on impossible declarations (A3).
    """
    decls = [int(round(d)) for d in decls]
    n, H, A = len(decls), int(hand_size), int(active_wires)
    post = np.zeros((n, n))
    for i, j in pairs(n):
        for w in itertools.product(range(H + 1), repeat=n):
            if sum(w) != A:
                continue
            if any(w[g] != decls[g] for g in range(n) if g != i and g != j):
                continue
            weight = 1
            for g in range(n):
                weight *= math.comb(H, w[g])
            post[i][j] += weight
    s = post.sum()
    if s == 0:
        out = np.zeros((n, n))
        for i, j in pairs(n):
            out[i][j] = 1.0
        return out / out.sum()
    return post / s


def L_bad_pair_ref(decls, revealed, found, hand_size, active_wires, i, j):
    """Likelihood of the cut observation in bad hands i, j, summed independently
    over every wire split (wi, wj) of the group total, weighted by the §3.4.1
    placement law C(H,wi)·C(H,wj)/C(2H,bg) and the per-hand hypergeometric draws."""
    H = int(hand_size)
    bg = int(round(group_wires(decls, found, active_wires, i, j)))
    if not (0 <= bg <= 2 * H):
        return 0.0
    denom = math.comb(2 * H, bg)
    total = 0.0
    for wi in range(bg + 1):
        wj = bg - wi
        place = math.comb(H, wi) * math.comb(H, wj)  # 0 when wi or wj > H
        if place == 0:
            continue
        obs = (hypergeom_pmf(found[i], H, wi, revealed[i])
               * hypergeom_pmf(found[j], H, wj, revealed[j]))
        total += place * obs
    return total / denom


def brute_probcut(decls, prior, revealed, found, hand_size, active_wires):
    """Exact posterior over which pair is bad, recomputed independently."""
    n = len(decls)
    post = np.zeros((n, n))
    for i, j in pairs(n):
        like = L_bad_pair_ref(decls, revealed, found, hand_size, active_wires, i, j)
        for g in range(n):
            if g != i and g != j:
                like *= hypergeom_pmf(found[g], hand_size, decls[g], revealed[g])
        post[i][j] = prior[i][j] * like
    s = post.sum()
    return prior.astype(float).copy() if s == 0 else post / s


def marginal_from_pairs(probs):
    """P(player i is bad) = sum over pairs containing i, recomputed independently."""
    n = probs.shape[0]
    lin = np.zeros(n)
    for i, j in pairs(n):
        lin[i] += probs[i][j]
        lin[j] += probs[i][j]
    return lin


def brute_pwire(decls, probs, revealed, found, hand_size, active_wires):
    """Expected wire probability per player, recomputed independently: average over
    pairs of the expected hidden wires in each hand under the §3.4.1 split posterior."""
    n = len(decls)
    H = int(hand_size)
    pw = np.zeros(n)
    # Bad-pair branch: for each pair, the posterior over splits given the observation.
    for i, j in pairs(n):
        bg = int(round(group_wires(decls, found, active_wires, i, j)))
        wsum_i = wsum_j = norm = 0.0
        for wi in range(max(bg, 0) + 1):
            wj = bg - wi
            if wj < 0:
                continue
            place = math.comb(H, wi) * math.comb(H, wj)
            if place == 0:
                continue
            wt = place * (hypergeom_pmf(found[i], H, wi, revealed[i])
                          * hypergeom_pmf(found[j], H, wj, revealed[j]))
            wsum_i += wt * (wi - found[i])
            wsum_j += wt * (wj - found[j])
            norm += wt
        if norm > 0:
            rem_i, rem_j = H - revealed[i], H - revealed[j]
            if rem_i > 0:
                pw[i] += probs[i][j] * (wsum_i / norm) / rem_i
            if rem_j > 0:
                pw[j] += probs[i][j] * (wsum_j / norm) / rem_j
    # Good-guy branch: i holds exactly decls[i], gated for feasibility.
    lin = marginal_from_pairs(probs)
    for i in range(n):
        rem_i = H - revealed[i]
        if rem_i <= 0:
            continue
        good_wires = decls[i] - found[i]
        if 0 <= good_wires <= rem_i:
            pw[i] += (1 - lin[i]) * good_wires / rem_i
    return pw


# --- helpers -------------------------------------------------------------------

def is_pair_distribution(m, total=1.0):
    """A valid lower-triangular pair distribution: no NaN, entries in [0,1], upper
    triangle (incl. diagonal) zero, lower triangle sums to `total`."""
    n = m.shape[0]
    if np.any(np.isnan(m)) or np.any(m < -TOL) or np.any(m > 1 + TOL):
        return False
    for i in range(n):
        for j in range(i, n):
            if abs(m[i][j]) > TOL:  # diagonal and upper triangle must be empty
                return False
    return abs(m.sum() - total) < 1e-6


def random_consistent_state(rng):
    """A state drawn from an actual play-out with two known hidden bad guys, so all
    counts are mutually consistent (no impossible observations).

    Returns ``(decls, revealed, found, hand_size, active_now, total_active)`` -- as in
    the OneBadGuyNoBomb suite, ``total_active`` is the declaration-time wire count and
    ``active_now`` is what remains after the cuts so far."""
    n = rng.randint(4, 5)
    hand_size = rng.randint(2, 4)
    bad = rng.sample(range(n), 2)
    total_active = rng.randint(0, n)
    wires = np.zeros(n, dtype=int)
    given = 0
    while given < total_active:
        c = rng.randrange(n)
        if wires[c] < hand_size:
            wires[c] += 1
            given += 1
    decls = wires.astype(float).copy()
    for b in bad:
        decls[b] = rng.randint(0, min(hand_size, total_active))
    revealed = np.zeros(n, dtype=int)
    found = np.zeros(n, dtype=int)
    active_wires = total_active
    for _ in range(rng.randint(0, n)):
        c = rng.randrange(n)
        if revealed[c] >= hand_size:
            continue
        rem_cards = hand_size - revealed[c]
        rem_wires = wires[c] - found[c]
        if rng.randint(1, rem_cards) <= rem_wires:
            found[c] += 1
            active_wires -= 1
        revealed[c] += 1
    return decls, revealed, found, hand_size, active_wires, total_active


# --- ProbDeclaration -----------------------------------------------------------

def test_probdeclaration_matches_generative_oracle():
    rng = Random(1)
    max_diff = 0.0
    for _ in range(1500):
        n = rng.randint(4, 5)
        hand_size = rng.randint(2, 3)
        active = rng.randint(0, n * hand_size)
        decls = np.array([float(rng.randint(0, hand_size)) for _ in range(n)])
        got = tb.ProbDeclaration(decls, hand_size, active)
        ref = generative_declaration_prior(decls, hand_size, active)
        max_diff = max(max_diff, np.max(np.abs(got - ref)))
    assert max_diff < 1e-9, f"max diff vs generative oracle = {max_diff}"


def test_probdeclaration_is_valid_pair_distribution():
    rng = Random(2)
    for _ in range(500):
        n = rng.randint(4, 5)
        hand_size = rng.randint(2, 3)
        active = rng.randint(0, n)
        decls = np.array([float(rng.randint(0, hand_size)) for _ in range(n)])
        m = tb.ProbDeclaration(decls, hand_size, active)
        assert is_pair_distribution(m)


def test_probdeclaration_degeneracy_falls_back_to_uniform_pairs():
    # Impossible declarations (no pair can absorb the excess): fall back to uniform
    # over the C(4,2)=6 pairs, never an all-zeros matrix.
    decls = np.array([0., 0., 0., 0.])  # 0 declared but 8 active wires demanded
    m = tb.ProbDeclaration(decls, hand_size=3, active_wires=8)
    assert is_pair_distribution(m)
    for i, j in pairs(4):
        assert abs(m[i][j] - 1 / 6) < 1e-9


# --- ProbCut -------------------------------------------------------------------

def test_probcut_matches_bruteforce_sweep():
    rng = Random(3)
    max_diff = 0.0
    for _ in range(4000):
        decls, revealed, found, hand_size, active, total = random_consistent_state(rng)
        prior = tb.ProbDeclaration(decls, hand_size, total)
        if prior.sum() == 0:
            continue
        got = tb.ProbCut(decls, prior, revealed, found, hand_size, active)
        ref = brute_probcut(decls, prior, revealed, found, hand_size, active)
        max_diff = max(max_diff, np.max(np.abs(got - ref)))
    assert max_diff < 1e-9, f"max diff vs brute force = {max_diff}"


def test_probcut_preserves_distribution_sweep():
    rng = Random(4)
    for _ in range(4000):
        decls, revealed, found, hand_size, active, total = random_consistent_state(rng)
        prior = tb.ProbDeclaration(decls, hand_size, total)
        if prior.sum() == 0:
            continue
        post = tb.ProbCut(decls, prior, revealed, found, hand_size, active)
        assert is_pair_distribution(post)


def test_probcut_handles_impossible_observation_without_nan():
    # Player 2 declared both cards as wires, yet a cut found none -> impossible under
    # any pair containing player 2; must not produce NaN.
    decls = np.array([1., 1., 2., 0.])
    prior = tb.ProbDeclaration(decls, hand_size=2, active_wires=4)
    post = tb.ProbCut(decls, prior, np.array([0, 0, 1, 0]), np.zeros(4, int),
                      hand_size=2, active_wires=4)
    assert not np.any(np.isnan(post))
    assert is_pair_distribution(post)


def test_probcut_certainty_is_absorbing():
    decls = np.array([2., 1., 0., 1.])
    prior = np.zeros((4, 4))
    prior[1][0] = 1.0  # players 0 and 1 already known to be the bad pair
    post = tb.ProbCut(decls, prior, np.array([1, 1, 0, 0]), np.array([0, 1, 0, 0]),
                      hand_size=3, active_wires=3)
    assert abs(post[1][0] - 1.0) < TOL
    assert abs(post.sum() - 1.0) < TOL


# --- P_wire --------------------------------------------------------------------

def test_pwire_matches_bruteforce_sweep():
    rng = Random(5)
    max_diff = 0.0
    for _ in range(4000):
        decls, revealed, found, hand_size, active, total = random_consistent_state(rng)
        prior = tb.ProbDeclaration(decls, hand_size, total)
        if prior.sum() == 0:
            continue
        probs = tb.ProbCut(decls, prior, revealed, found, hand_size, active)
        got = tb.P_wire(decls, probs, revealed, found, hand_size, active)
        ref = brute_pwire(decls, probs, revealed, found, hand_size, active)
        max_diff = max(max_diff, np.max(np.abs(got - ref)))
        assert np.all(got >= -TOL) and np.all(got <= 1 + TOL)
    assert max_diff < 1e-9, f"max diff vs brute force = {max_diff}"


def test_pwire_never_negative_when_found_exceeds_declared():
    # The known ungated-good-branch bug: a "good" player whose found wires exceed
    # their declaration must contribute 0, not a negative probability.
    decls = np.array([2., 1., 1., 0.])
    probs = tb.ProbDeclaration(decls, hand_size=4, active_wires=4)
    # Player 0 declared 2 but 3 of 3 revealed cards were wires (so as a good guy
    # they are infeasible); ensure no negative leaks through.
    pw = tb.P_wire(decls, probs, np.array([3, 0, 0, 0]), np.array([3, 0, 0, 0]),
                   hand_size=4, active_wires=4)
    assert np.all(pw >= -TOL)


# --- DeMatrix ------------------------------------------------------------------

def test_dematrix_marginalises_pairs():
    m = np.zeros((3, 3))
    m[1][0] = 0.5  # pair {0,1}
    m[2][0] = 0.3  # pair {0,2}
    m[2][1] = 0.2  # pair {1,2}
    lin = tb.DeMatrix(m)
    # P(0 bad)=0.5+0.3, P(1 bad)=0.5+0.2, P(2 bad)=0.3+0.2
    assert np.allclose(lin, [0.8, 0.7, 0.5])


# --- end-to-end accuracy (does the model beat random?) -------------------------

def test_inference_beats_random_chance():
    """A model can be arithmetically correct yet uninformative. This runs full
    simulated games (declarations -> cuts -> combine -> marginalise) and checks the
    final belief actually concentrates on the true bad guys far above the random
    baseline. Under no information every player's P(bad) marginal averages 2/N
    (= 1/3 for N=6); the oracle tests cannot catch a regression that destroys this
    edge, so it is guarded separately here."""
    random.seed(12345)  # PlayAuto draws from the global RNG; seed for determinism
    N, K = 6, 400
    bad_mass = good_mass = 0.0
    top2_hits = 0
    for _ in range(K):
        _, marg, roles = tb.PlayAuto(num_players=N, initial_hand_size=5, verbosity=0)
        bad_idx = set(np.where(roles == 1)[0])
        for i in range(N):
            if i in bad_idx:
                bad_mass += marg[i]
            else:
                good_mass += marg[i]
        top2_hits += len(set(np.argsort(marg)[-2:]) & bad_idx)
    p_bad_on_bad = bad_mass / (2 * K)
    p_bad_on_good = good_mass / ((N - 2) * K)
    top2_precision = top2_hits / (2 * K)
    baseline = 2 / N
    # Generous margins: observed values are ~0.94 / ~0.03 / ~0.95, baseline 0.333.
    assert p_bad_on_bad > 0.6, f"P(bad|true bad)={p_bad_on_bad:.3f} not above baseline {baseline:.3f}"
    assert p_bad_on_good < 0.15, f"P(bad|true good)={p_bad_on_good:.3f} not below baseline {baseline:.3f}"
    assert top2_precision > 0.6, f"top-2 precision={top2_precision:.3f} not above baseline {baseline:.3f}"


# --- standalone runner (no pytest required) ------------------------------------

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
