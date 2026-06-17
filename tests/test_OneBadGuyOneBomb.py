"""Tests for the OneBadGuyOneBomb variant (1 bad guy, 1 bomb).

Run with pytest:   .venv/bin/python -m pytest tests/test_OneBadGuyOneBomb.py -q
Or standalone:     .venv/bin/python tests/test_OneBadGuyOneBomb.py

The belief state is the full N x N matrix ``probs[b][h]`` = P(player b is bad and
player h holds the bomb), with the diagonal b == h allowed (the bad guy may be dealt
his own bomb). The reference implementations below are written independently of the
module under test (model.md §3.2–§3.4, the uniform-lie bomb model):

  * a config is a pair (b, h); every hand except b and h declares truthfully;
  * the bomb occupies one card slot, so the bomb hand has H-1 wire-able slots;
  * deals place the bomb in hand h and the A wires uniformly among the remaining
    slots -> a wire vector w has weight Π_g C(slots_g, w[g]),  slots_g = H-1 if g==h;
  * cuts condition on "no bomb drawn yet": the bomb hand contributes a must-not-draw
    hypergeometric C(w_h,found)·C(H-1-w_h, rev-found)/C(H, rev).

Every reference sums the wire split explicitly via math.comb / itertools.product --
no closed form, no Vandermonde collapse -- so it shares no algebra with the module
and a common bug cannot hide behind a matching assertion.
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

import OneBadGuyOneBomb as ob

TOL = 1e-9
comb = math.comb


# --- independent reference implementations -------------------------------------

def hypergeom_pmf(found, hand_size, wires, revealed):
    """P(find `found` wires in `revealed` draws from a `hand_size`-card hand holding
    `wires` wires and no bomb) -- the standard hypergeometric PMF, via math.comb."""
    found, hand_size = int(round(found)), int(round(hand_size))
    wires, revealed = int(round(wires)), int(round(revealed))
    if not (0 <= wires <= hand_size and 0 <= revealed <= hand_size):
        return 0.0
    denom = comb(hand_size, revealed)
    if denom == 0:
        return 0.0
    return comb(wires, found) * comb(hand_size - wires, revealed - found) / denom


def bomb_pmf(found, hand_size, wires, revealed):
    """P(find `found` wires AND draw no bomb in `revealed` draws from the bomb hand),
    given the hand holds `wires` wires, one bomb, and H-1-wires blanks (model.md
    §3.2 must-not-draw term):  C(w,found)·C(H-1-w, rev-found)/C(H, rev)."""
    found, hand_size = int(round(found)), int(round(hand_size))
    wires, revealed = int(round(wires)), int(round(revealed))
    if not (0 <= wires <= hand_size - 1 and 0 <= revealed <= hand_size):
        return 0.0
    denom = comb(hand_size, revealed)
    if denom == 0:
        return 0.0
    return comb(wires, found) * comb(hand_size - 1 - wires, revealed - found) / denom


def configs(n):
    """Every (bad, bomb) configuration: the full N x N grid, diagonal included."""
    return [(b, h) for b in range(n) for h in range(n)]


def free_wire_total(decls, found, active_wires, b, h):
    """Wires the non-truthful hands (b, and h when distinct) must hold given the
    truthful good guys: A + Σfound - Σ_{j truthful} decls[j]."""
    truthful = sum(decls[j] for j in range(len(decls)) if j != b and j != h)
    return active_wires + np.sum(found) - truthful


def generative_declaration_prior(decls, hand_size, active_wires):
    """Posterior P(config (b,h) | decls) under the generative model (model.md §2 lie
    model + §3.3), enumerated independently of the module:

      * the config (b, h) is chosen uniformly over the N^2 pairs;
      * the bomb sits in hand h (one slot), the A wires are dealt among the remaining
        slots ~ multivariate hypergeometric, so a wire vector w has weight
        Π_g C(slots_g, w[g]) with slots_g = H-1 if g==h else H;
      * truthful hands (g not in {b,h}) declare exactly w[g] == decls[g];
      * the liars (b, and h if distinct) declare uniformly -> constant factor, cancels.

    Enumerates every wire vector via itertools.product -- no closed form, no division
    -- so it shares no algebra with ProbDeclaration. Falls back to the uniform N x N
    matrix on impossible declarations (degeneracy, §3.3)."""
    decls = [int(round(d)) for d in decls]
    n, H, A = len(decls), int(hand_size), int(active_wires)
    post = np.zeros((n, n))
    for b, h in configs(n):
        for w in itertools.product(range(H + 1), repeat=n):
            if sum(w) != A:
                continue
            if any(w[g] != decls[g] for g in range(n) if g != b and g != h):
                continue
            slots = [H - 1 if g == h else H for g in range(n)]
            if any(w[g] > slots[g] for g in range(n)):
                continue
            weight = 1
            for g in range(n):
                weight *= comb(slots[g], w[g])
            post[b][h] += weight
    s = post.sum()
    if s == 0:
        return np.full((n, n), 1.0 / (n * n))
    return post / s


def cut_likelihood_ref(decls, revealed, found, hand_size, active_wires, b, h):
    """Likelihood of the cut observation under config (b, h), summed independently
    over every wire split of the free hands (model.md §3.4). Truthful hands use the
    plain hypergeometric; the bomb hand uses the must-not-draw term; the split is
    weighted by the §3.4.1 placement law over the 2H-1 (or H-1) free slots."""
    n, H = len(decls), int(hand_size)
    A = int(active_wires)
    # Truthful hands: pinned to their declarations, no bomb.
    Ltruth = 1.0
    for j in range(n):
        if j != b and j != h:
            Ltruth *= hypergeom_pmf(found[j], H, decls[j], revealed[j])
    t_free = int(round(free_wire_total(decls, found, active_wires, b, h)))
    if b == h:
        # One free hand, holding the bomb: the split is deterministic (w_h = t_free).
        if not (0 <= t_free <= H - 1):
            return 0.0
        return Ltruth * bomb_pmf(found[b], H, t_free, revealed[b])
    # Two free hands: bad hand b (H slots) and bomb hand h (H-1 slots).
    if not (0 <= t_free <= 2 * H - 1):
        return 0.0
    denom = comb(2 * H - 1, t_free)
    if denom == 0:
        return 0.0
    s = 0.0
    for wb in range(t_free + 1):
        wh = t_free - wb
        place = comb(H, wb) * comb(H - 1, wh)  # 0 when wb > H or wh > H-1
        if place == 0:
            continue
        obs = (hypergeom_pmf(found[b], H, wb, revealed[b])
               * bomb_pmf(found[h], H, wh, revealed[h]))
        s += place * obs
    return Ltruth * s / denom


def brute_probcut(decls, prior, revealed, found, hand_size, active_wires):
    """Exact posterior over (bad, bomb) configs, recomputed independently."""
    n = len(decls)
    post = np.zeros((n, n))
    for b, h in configs(n):
        post[b][h] = prior[b][h] * cut_likelihood_ref(
            decls, revealed, found, hand_size, active_wires, b, h)
    s = post.sum()
    return np.asarray(prior, dtype=float).copy() if s == 0 else post / s


def split_posterior_means(decls, revealed, found, hand_size, active_wires, b, h):
    """E[remaining wires] in the free hand(s) of config (b,h) given the observation,
    under the split posterior. Returns a dict {hand_index: expected_remaining_wires}."""
    n, H = len(decls), int(hand_size)
    t_free = int(round(free_wire_total(decls, found, active_wires, b, h)))
    if b == h:
        if not (0 <= t_free <= H - 1):
            return {}
        return {b: t_free - found[b]}
    if not (0 <= t_free <= 2 * H - 1):
        return {}
    sb = sh = norm = 0.0
    for wb in range(t_free + 1):
        wh = t_free - wb
        place = comb(H, wb) * comb(H - 1, wh)
        if place == 0:
            continue
        wt = place * (hypergeom_pmf(found[b], H, wb, revealed[b])
                      * bomb_pmf(found[h], H, wh, revealed[h]))
        sb += wt * (wb - found[b])
        sh += wt * (wh - found[h])
        norm += wt
    if norm <= 0:
        return {}
    return {b: sb / norm, h: sh / norm}


def brute_pwire(decls, probs, revealed, found, hand_size, active_wires):
    """Expected wire probability per player, recomputed independently: for every
    config, classify each hand as truthful / bad / bomb, take its expected remaining
    wires over remaining cards (H - revealed, which still counts the bomb card), gate
    for feasibility, and average over the config posterior."""
    n, H = len(decls), int(hand_size)
    pw = np.zeros(n)
    for b, h in configs(n):
        p = probs[b][h]
        if p == 0:
            continue
        means = split_posterior_means(decls, revealed, found, hand_size, active_wires, b, h)
        for i in range(n):
            cards_left = H - revealed[i]
            if cards_left <= 0:
                continue
            if i != b and i != h:
                rem = decls[i] - found[i]  # truthful: pinned to declaration
            else:
                if i not in means:
                    continue
                rem = means[i]
            if 0 <= rem <= cards_left:
                pw[i] += p * rem / cards_left
    return pw


# --- helpers -------------------------------------------------------------------

def is_matrix_distribution(m, total=1.0):
    """A valid N x N config distribution: no NaN, entries in [0,1], sums to `total`."""
    if np.any(np.isnan(m)) or np.any(m < -TOL) or np.any(m > 1 + TOL):
        return False
    return abs(m.sum() - total) < 1e-6


def random_consistent_state(rng):
    """A state drawn from an actual play-out with a known hidden bad guy and a known
    hidden bomb holder, conditioned on no bomb being cut (live inference always
    assumes "no bomb yet", §3.4). All counts are mutually consistent.

    Returns ``(decls, revealed, found, hand_size, active_now, total_active)`` --
    ``total_active`` is the declaration-time wire count, ``active_now`` what remains
    after the cuts so far."""
    n = rng.randint(3, 4)
    hand_size = rng.randint(2, 4)
    bad = rng.randrange(n)
    bomb = rng.randrange(n)
    # Deal the A wires among the non-bomb slots (bomb hand has H-1 wire-able slots).
    capacity = [hand_size - (1 if g == bomb else 0) for g in range(n)]
    total_active = rng.randint(0, sum(capacity))
    wires = np.zeros(n, dtype=int)
    given = 0
    while given < total_active:
        c = rng.randrange(n)
        if wires[c] < capacity[c]:
            wires[c] += 1
            given += 1
    # Declarations: truthful iff good and bomb-free; everyone else uniform on {0..H}.
    decls = wires.astype(float).copy()
    for i in range(n):
        if i == bad or i == bomb:
            decls[i] = rng.randint(0, hand_size)
    # Cuts: draw only from non-bomb cards (condition on no bomb cut yet).
    revealed = np.zeros(n, dtype=int)
    found = np.zeros(n, dtype=int)
    active_wires = total_active
    for _ in range(rng.randint(0, n)):
        c = rng.randrange(n)
        # non-bomb cards still face down in hand c
        bomb_here = 1 if c == bomb else 0
        nonbomb_left = (hand_size - revealed[c]) - bomb_here
        if nonbomb_left <= 0:
            continue
        rem_wires = wires[c] - found[c]
        if rng.randint(1, nonbomb_left) <= rem_wires:
            found[c] += 1
            active_wires -= 1
        revealed[c] += 1
    return decls, revealed, found, hand_size, active_wires, total_active


# --- ProbDeclaration -----------------------------------------------------------

def test_probdeclaration_matches_generative_oracle():
    rng = Random(1)
    max_diff = 0.0
    for _ in range(800):
        n = rng.randint(3, 4)
        hand_size = rng.randint(2, 3)
        active = rng.randint(0, n * hand_size - 1)
        decls = np.array([float(rng.randint(0, hand_size)) for _ in range(n)])
        got = ob.ProbDeclaration(decls, hand_size, active)
        ref = generative_declaration_prior(decls, hand_size, active)
        max_diff = max(max_diff, np.max(np.abs(got - ref)))
    assert max_diff < 1e-9, f"max diff vs generative oracle = {max_diff}"


def test_probdeclaration_is_valid_distribution():
    rng = Random(2)
    for _ in range(400):
        n = rng.randint(3, 4)
        hand_size = rng.randint(2, 3)
        active = rng.randint(0, n)
        decls = np.array([float(rng.randint(0, hand_size)) for _ in range(n)])
        m = ob.ProbDeclaration(decls, hand_size, active)
        assert is_matrix_distribution(m)


def test_probdeclaration_degeneracy_falls_back_to_uniform():
    # Impossible declarations: 0 declared but many active wires demanded -> uniform.
    decls = np.array([0., 0., 0.])
    m = ob.ProbDeclaration(decls, hand_size=3, active_wires=9)
    assert is_matrix_distribution(m)
    assert np.allclose(m, 1.0 / 9)


# --- ProbCut -------------------------------------------------------------------

def test_probcut_matches_bruteforce_sweep():
    rng = Random(3)
    max_diff = 0.0
    for _ in range(3000):
        decls, revealed, found, hand_size, active, total = random_consistent_state(rng)
        prior = ob.ProbDeclaration(decls, hand_size, total)
        if prior.sum() == 0:
            continue
        got = ob.ProbCut(decls, prior, revealed, found, hand_size, active)
        ref = brute_probcut(decls, prior, revealed, found, hand_size, active)
        max_diff = max(max_diff, np.max(np.abs(got - ref)))
    assert max_diff < 1e-9, f"max diff vs brute force = {max_diff}"


def test_probcut_preserves_distribution_sweep():
    rng = Random(4)
    for _ in range(3000):
        decls, revealed, found, hand_size, active, total = random_consistent_state(rng)
        prior = ob.ProbDeclaration(decls, hand_size, total)
        if prior.sum() == 0:
            continue
        post = ob.ProbCut(decls, prior, revealed, found, hand_size, active)
        assert is_matrix_distribution(post)


def test_probcut_handles_impossible_observation_without_nan():
    decls = np.array([1., 1., 2., 0.])
    prior = ob.ProbDeclaration(decls, hand_size=2, active_wires=4)
    post = ob.ProbCut(decls, prior, np.array([0, 0, 1, 0]), np.zeros(4, int),
                      hand_size=2, active_wires=4)
    assert not np.any(np.isnan(post))
    assert is_matrix_distribution(post)


# --- P_wire --------------------------------------------------------------------

def test_pwire_matches_bruteforce_sweep():
    rng = Random(5)
    max_diff = 0.0
    for _ in range(3000):
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


# --- DeMatrix ------------------------------------------------------------------

def test_dematrix_marginalises_rows_and_cols():
    m = np.zeros((3, 3))
    m[0][1] = 0.5  # bad 0, bomb 1
    m[2][2] = 0.3  # bad 2, bomb 2
    m[1][0] = 0.2  # bad 1, bomb 0
    p_bad, p_bomb = ob.DeMatrix(m)
    assert np.allclose(p_bad, [0.5, 0.2, 0.3])   # row sums
    assert np.allclose(p_bomb, [0.2, 0.5, 0.3])  # col sums


# --- end-to-end accuracy (does the model beat random?) -------------------------

def test_inference_beats_random_chance():
    """A model can be arithmetically correct yet uninformative. This runs full
    simulated games and checks the final belief concentrates on the true bad guy far
    above the random baseline (1/N). The bomb is per-round noise -- a good guy holding
    it is forced to lie and looks bad -- so the edge is weaker than the no-bomb
    variants, but must still clear the baseline by a wide margin."""
    random.seed(12345)  # PlayAuto draws from the global RNG; seed for determinism
    N, K = 6, 400
    bad_mass = good_mass = 0.0
    top1_hits = 0
    for _ in range(K):
        _, marg, roles = ob.PlayAuto(num_players=N, initial_hand_size=5, verbosity=0)
        bad_idx = int(np.where(roles == 1)[0][0])
        for i in range(N):
            if i == bad_idx:
                bad_mass += marg[i]
            else:
                good_mass += marg[i]
        if int(np.argmax(marg)) == bad_idx:
            top1_hits += 1
    p_bad_on_bad = bad_mass / K
    p_bad_on_good = good_mass / ((N - 1) * K)
    top1 = top1_hits / K
    baseline = 1 / N
    assert p_bad_on_bad > 0.4, f"P(bad|true bad)={p_bad_on_bad:.3f} not above baseline {baseline:.3f}"
    assert p_bad_on_good < 0.18, f"P(bad|true good)={p_bad_on_good:.3f} not below baseline {baseline:.3f}"
    assert top1 > 0.4, f"top-1 accuracy={top1:.3f} not above baseline {baseline:.3f}"


def test_bomb_inference_beats_random_chance():
    """The bomb is this variant's whole reason to exist, and its column marginal is a
    distinct readout the P(bad) test cannot vouch for -- a bug that scrambled the bomb
    axis (e.g. a transposed config matrix) would leave P(bad) intact yet destroy
    P(bomb). Build rounds directly under the uniform-lie model and check the column
    marginal concentrates on the true bomb holder above the 1/N baseline. P(bomb) is
    per-round (§3.5), so this works one round at a time and never combines."""
    rng = Random(999)
    N, K = 6, 3000
    mass_on_true = 0.0
    top1 = 0
    for _ in range(K):
        H, active = 5, N
        bomb = rng.randrange(N)
        capacity = [H - (1 if g == bomb else 0) for g in range(N)]
        wires = np.zeros(N, dtype=int)
        given = 0
        while given < active:
            c = rng.randrange(N)
            if wires[c] < capacity[c]:
                wires[c] += 1
                given += 1
        bad = rng.randrange(N)
        decls = wires.astype(float).copy()
        for i in range(N):
            if i == bad or i == bomb:
                decls[i] = rng.randint(0, H)
        probs = ob.ProbDeclaration(decls, H, active)
        _, p_bomb = ob.DeMatrix(probs)
        mass_on_true += p_bomb[bomb]
        if int(np.argmax(p_bomb)) == bomb:
            top1 += 1
    p_bomb_on_true = mass_on_true / K
    baseline = 1 / N
    # Observed ~0.30 / ~0.42 vs baseline 0.167; generous margins.
    assert p_bomb_on_true > 0.22, f"P(bomb|true holder)={p_bomb_on_true:.3f} not above baseline {baseline:.3f}"
    assert top1 / K > 0.3, f"top-1 bomb accuracy={top1 / K:.3f} not above baseline {baseline:.3f}"


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
