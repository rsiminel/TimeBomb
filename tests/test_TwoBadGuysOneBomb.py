"""Tests for the TwoBadGuysOneBomb variant (B=2, M=1).

Run with pytest:   .venv/bin/python -m pytest tests/test_TwoBadGuysOneBomb.py -q
Or standalone:     .venv/bin/python tests/test_TwoBadGuysOneBomb.py

This variant combines the B>1 pair structure (model.md §3.4.1) with the bomb
sub-model (§3.2–§3.4). A configuration is the triple ``(b1, b2, h)``: ``{b1, b2}`` is the
unordered bad pair (stored lower-triangular, ``b1 > b2``) and ``h`` holds the bomb,
with ``h`` allowed to coincide with a bad guy. The belief state is the N x N x N
tensor ``probs[b1][b2][h]`` (non-zero only for ``b1 > b2``).

The reference implementations below are written independently of the module under
test, straight from the generative model:

  * the bomb sits in hand ``h`` (one slot), so that hand has H-1 wire-able slots;
  * the A wires are placed uniformly among the remaining slots (multivariate
    hypergeometric);
  * a hand is truthful iff good *and* bomb-free, i.e. every hand except the free set
    ``{b1, b2, h}``; truthful hands declare exactly their wires, the liars declare
    uniformly (a constant factor that cancels);
  * cuts condition on "no bomb drawn yet": the bomb hand contributes the must-not-draw
    hypergeometric ``C(w,found)·C(H-1-w, rev-found)/C(H, rev)``.

Every reference sums the wire split over the *free hands* explicitly via math.comb /
itertools.product -- no closed form, no Vandermonde collapse, no L_bad_pair reuse --
so it shares no algebra with the module and a common bug cannot hide behind a matching
assertion.
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

import TwoBadGuysOneBomb as tb

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
    """Every (b1, b2, h) configuration: the bad pair b1 > b2 and any bomb holder h."""
    return [(b1, b2, h) for b1 in range(n) for b2 in range(b1) for h in range(n)]


def free_hands(b1, b2, h):
    """The non-truthful hands of a config: the bad pair plus the bomb holder, deduped
    (h may coincide with a bad guy). Sorted for a deterministic split enumeration."""
    return sorted({b1, b2, h})


def slots_of(g, h, hand_size):
    """Wire-able slots in hand g: H-1 if it holds the bomb (g == h), else H."""
    return hand_size - 1 if g == h else hand_size


def free_splits(free, t_free, h, hand_size):
    """Every wire vector over the free hands summing to t_free, each entry within its
    hand's wire-able slot count. Yields dicts {hand: wires}."""
    H = int(hand_size)
    for combo in itertools.product(range(H + 1), repeat=len(free)):
        if sum(combo) != t_free:
            continue
        if any(combo[k] > slots_of(free[k], h, H) for k in range(len(free))):
            continue
        yield dict(zip(free, combo))


def generative_declaration_prior(decls, hand_size, active_wires):
    """Posterior P(config (b1,b2,h) | decls) under the generative model (model.md §2
    lie model + §3.3), enumerated independently of the module:

      * the config is chosen uniformly over the C(N,2)·N triples;
      * the bomb sits in hand h (one slot), the A wires are dealt among the remaining
        slots ~ multivariate hypergeometric, so a wire vector w has weight
        Π_g C(slots_g, w[g]) with slots_g = H-1 if g==h else H;
      * truthful hands (g not in {b1,b2,h}) declare exactly w[g] == decls[g];
      * each free hand is a uniform liar, contributing a factor (H+1)^{-|F|} for the
        |F| free hands -- this does NOT cancel (ADR 0007): |F|=2 when the bomb sits on
        a bad guy, |F|=3 when on a good guy, so it down-weights bomb-on-good configs.

    Enumerates every wire vector via itertools.product -- no closed form, no division
    -- so it shares no algebra with ProbDeclaration. Falls back to the uniform tensor
    on impossible declarations (degeneracy, §3.3)."""
    decls = [int(round(d)) for d in decls]
    n, H, A = len(decls), int(hand_size), int(active_wires)
    post = np.zeros((n, n, n))
    for b1, b2, h in configs(n):
        free = {b1, b2, h}
        lie_factor = (H + 1) ** (-len(free))  # uniform-lie declaration probability
        for w in itertools.product(range(H + 1), repeat=n):
            if sum(w) != A:
                continue
            if any(w[g] != decls[g] for g in range(n) if g not in free):
                continue
            slots = [slots_of(g, h, H) for g in range(n)]
            if any(w[g] > slots[g] for g in range(n)):
                continue
            weight = lie_factor
            for g in range(n):
                weight *= comb(slots[g], w[g])
            post[b1][b2][h] += weight
    s = post.sum()
    if s == 0:
        ref = np.zeros((n, n, n))
        for b1, b2, h in configs(n):
            ref[b1][b2][h] = 1.0
        return ref / ref.sum()
    return post / s


def free_wire_total(decls, found, active_wires, free):
    """Wires the free hands must hold given the truthful good guys:
    A + Σfound - Σ_{j truthful} decls[j]."""
    truthful = sum(int(round(decls[j])) for j in range(len(decls)) if j not in free)
    return int(active_wires) + int(np.sum(found)) - truthful


def cut_likelihood_ref(decls, revealed, found, hand_size, active_wires, b1, b2, h):
    """Likelihood of the cut observation under config (b1,b2,h), summed independently
    over every wire split of the free hands (model.md §3.4). Truthful hands use the
    plain hypergeometric; the bomb hand uses the must-not-draw term; the split is
    weighted by the §3.4.1 placement law over the free hands' non-bomb slots."""
    n, H = len(decls), int(hand_size)
    free = free_hands(b1, b2, h)
    # Truthful hands: pinned to their declarations, no bomb.
    Ltruth = 1.0
    for j in range(n):
        if j not in free:
            Ltruth *= hypergeom_pmf(found[j], H, decls[j], revealed[j])
    t_free = free_wire_total(decls, found, active_wires, set(free))
    free_slots = sum(slots_of(g, h, H) for g in free)
    if not (0 <= t_free <= free_slots):
        return 0.0
    denom = comb(free_slots, t_free)
    if denom == 0:
        return 0.0
    s = 0.0
    for split in free_splits(free, t_free, h, H):
        place = 1
        obs = 1.0
        for g, wg in split.items():
            place *= comb(slots_of(g, h, H), wg)
            if g == h:
                obs *= bomb_pmf(found[g], H, wg, revealed[g])
            else:
                obs *= hypergeom_pmf(found[g], H, wg, revealed[g])
        s += place * obs
    return Ltruth * s / denom


def brute_probcut(decls, prior, revealed, found, hand_size, active_wires):
    """Exact posterior over (b1,b2,h) configs, recomputed independently."""
    n = len(decls)
    post = np.zeros((n, n, n))
    for b1, b2, h in configs(n):
        post[b1][b2][h] = prior[b1][b2][h] * cut_likelihood_ref(
            decls, revealed, found, hand_size, active_wires, b1, b2, h)
    s = post.sum()
    return np.asarray(prior, dtype=float).copy() if s == 0 else post / s


def split_posterior_means(decls, revealed, found, hand_size, active_wires, b1, b2, h):
    """E[remaining wires] in the free hands of config (b1,b2,h) given the observation,
    under the split posterior. Returns a dict {hand_index: expected_remaining_wires}."""
    H = int(hand_size)
    free = free_hands(b1, b2, h)
    t_free = free_wire_total(decls, found, active_wires, set(free))
    free_slots = sum(slots_of(g, h, H) for g in free)
    if not (0 <= t_free <= free_slots):
        return {}
    acc = {g: 0.0 for g in free}
    norm = 0.0
    for split in free_splits(free, t_free, h, H):
        place = 1
        obs = 1.0
        for g, wg in split.items():
            place *= comb(slots_of(g, h, H), wg)
            if g == h:
                obs *= bomb_pmf(found[g], H, wg, revealed[g])
            else:
                obs *= hypergeom_pmf(found[g], H, wg, revealed[g])
        wt = place * obs
        for g, wg in split.items():
            acc[g] += wt * (wg - found[g])
        norm += wt
    if norm <= 0:
        return {}
    return {g: acc[g] / norm for g in free}


def brute_pwire(decls, probs, revealed, found, hand_size, active_wires):
    """Expected wire probability per player, recomputed independently: for every
    config, classify each hand as truthful / free, take its expected remaining wires
    over remaining cards (H - revealed, which still counts the bomb card), gate for
    feasibility, and average over the config posterior."""
    n, H = len(decls), int(hand_size)
    pw = np.zeros(n)
    for b1, b2, h in configs(n):
        p = probs[b1][b2][h]
        if p == 0:
            continue
        free = set(free_hands(b1, b2, h))
        means = split_posterior_means(decls, revealed, found, hand_size, active_wires, b1, b2, h)
        for i in range(n):
            cards_left = H - revealed[i]
            if cards_left <= 0:
                continue
            if i not in free:
                rem = decls[i] - found[i]  # truthful: pinned to declaration
            else:
                if i not in means:
                    continue
                rem = means[i]
            if 0 <= rem <= cards_left:
                pw[i] += p * rem / cards_left
    return pw


# --- helpers -------------------------------------------------------------------

def is_tensor_distribution(m, total=1.0):
    """A valid N x N x N config distribution: no NaN, entries in [0,1], sums to
    `total`, and zero off the b1 > b2 lower triangle."""
    m = np.asarray(m)
    if np.any(np.isnan(m)) or np.any(m < -TOL) or np.any(m > 1 + TOL):
        return False
    n = m.shape[0]
    for b1 in range(n):
        for b2 in range(b1, n):  # upper triangle + diagonal must stay empty
            if np.any(np.abs(m[b1][b2]) > TOL):
                return False
    return abs(m.sum() - total) < 1e-6


def random_consistent_state(rng):
    """A state drawn from an actual play-out with a known bad pair and a known bomb
    holder, conditioned on no bomb being cut (live inference always assumes "no bomb
    yet", §3.4). All counts are mutually consistent.

    Returns ``(decls, revealed, found, hand_size, active_now, total_active)`` --
    ``total_active`` is the declaration-time wire count, ``active_now`` what remains
    after the cuts so far."""
    n = rng.randint(3, 4)
    hand_size = rng.randint(2, 4)
    bad1, bad2 = rng.sample(range(n), 2)
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
        if i in (bad1, bad2) or i == bomb:
            decls[i] = rng.randint(0, hand_size)
    # Cuts: draw only from non-bomb cards (condition on no bomb cut yet).
    revealed = np.zeros(n, dtype=int)
    found = np.zeros(n, dtype=int)
    active_wires = total_active
    for _ in range(rng.randint(0, n)):
        c = rng.randrange(n)
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
    for _ in range(600):
        n = rng.randint(3, 4)
        hand_size = rng.randint(2, 3)
        active = rng.randint(0, n * hand_size - 1)
        decls = np.array([float(rng.randint(0, hand_size)) for _ in range(n)])
        got = tb.ProbDeclaration(decls, hand_size, active)
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
        m = tb.ProbDeclaration(decls, hand_size, active)
        assert is_tensor_distribution(m)


def test_probdeclaration_degeneracy_falls_back_to_uniform():
    # Impossible declarations: 0 declared but many active wires demanded -> uniform
    # over the C(N,2)·N valid (b1>b2, any h) cells.
    decls = np.array([0., 0., 0.])
    m = tb.ProbDeclaration(decls, hand_size=3, active_wires=9)
    assert is_tensor_distribution(m)
    n = 3
    valid_cells = (n * (n - 1) // 2) * n
    for b1 in range(n):
        for b2 in range(b1):
            for h in range(n):
                assert abs(m[b1][b2][h] - 1.0 / valid_cells) < TOL


# --- ProbCut -------------------------------------------------------------------

def test_probcut_matches_bruteforce_sweep():
    rng = Random(3)
    max_diff = 0.0
    for _ in range(2500):
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
    for _ in range(2500):
        decls, revealed, found, hand_size, active, total = random_consistent_state(rng)
        prior = tb.ProbDeclaration(decls, hand_size, total)
        if prior.sum() == 0:
            continue
        post = tb.ProbCut(decls, prior, revealed, found, hand_size, active)
        assert is_tensor_distribution(post)


def test_probcut_handles_impossible_observation_without_nan():
    decls = np.array([1., 1., 2., 0.])
    prior = tb.ProbDeclaration(decls, hand_size=2, active_wires=4)
    post = tb.ProbCut(decls, prior, np.array([0, 0, 1, 0]), np.zeros(4, int),
                      hand_size=2, active_wires=4)
    assert not np.any(np.isnan(post))
    assert is_tensor_distribution(post)


# --- P_wire --------------------------------------------------------------------

def test_pwire_matches_bruteforce_sweep():
    rng = Random(5)
    max_diff = 0.0
    for _ in range(2500):
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


# --- DeTensor / DeMatrix -------------------------------------------------------

def test_detensor_marginalises_pair_and_bomb():
    n = 3
    m = np.zeros((n, n, n))
    m[1][0][2] = 0.5  # bad pair {1,0}, bomb 2
    m[2][0][0] = 0.3  # bad pair {2,0}, bomb 0
    m[2][1][1] = 0.2  # bad pair {2,1}, bomb 1
    pair, bomb = tb.DeTensor(m)
    assert abs(pair[1][0] - 0.5) < TOL
    assert abs(pair[2][0] - 0.3) < TOL
    assert abs(pair[2][1] - 0.2) < TOL
    assert np.allclose(bomb, [0.3, 0.2, 0.5])  # bomb column sums


def test_dematrix_marginalises_pair_to_players():
    n = 3
    pair = np.zeros((n, n))
    pair[1][0] = 0.5  # players 0 and 1
    pair[2][0] = 0.3  # players 0 and 2
    pair[2][1] = 0.2  # players 1 and 2
    line = tb.DeMatrix(pair)
    assert np.allclose(line, [0.8, 0.7, 0.5])  # each player's P(bad); sums to 2


# --- CombineProbs --------------------------------------------------------------

def test_combineprobs_multiplies_and_renormalises():
    n = 3
    a = tb.ProbDeclaration(np.array([1., 1., 1.]), 3, 3)
    pair_a = tb.DeTensor(a)[0]
    combined = tb.CombineProbs([pair_a, pair_a])
    manual = pair_a * pair_a
    manual /= manual.sum()
    assert np.allclose(combined, manual)
    assert abs(combined.sum() - 1.0) < 1e-6


# --- end-to-end accuracy (does the model beat random?) -------------------------

def test_inference_beats_random_chance():
    """A model can be arithmetically correct yet uninformative. This runs full
    simulated games and checks the final belief concentrates on the true bad pair far
    above the random baseline (2/N per player). The bomb is per-round noise -- a good
    guy holding it is forced to lie and looks bad -- so the edge is weaker than the
    no-bomb pair variant, but must still clear the baseline by a wide margin."""
    random.seed(12345)  # PlayAuto draws from the global RNG; seed for determinism
    N, K = 6, 400
    bad_mass = good_mass = 0.0
    for _ in range(K):
        _, marg, roles = tb.PlayAuto(num_players=N, initial_hand_size=5, verbosity=0)
        bad_idx = set(int(i) for i in np.where(roles == 1)[0])
        for i in range(N):
            if i in bad_idx:
                bad_mass += marg[i]
            else:
                good_mass += marg[i]
    p_bad_on_bad = bad_mass / (2 * K)
    p_bad_on_good = good_mass / ((N - 2) * K)
    baseline = 2 / N
    assert p_bad_on_bad > 0.55, f"P(bad|true bad)={p_bad_on_bad:.3f} not above baseline {baseline:.3f}"
    assert p_bad_on_good < baseline, f"P(bad|true good)={p_bad_on_good:.3f} not below baseline {baseline:.3f}"


def test_bomb_inference_beats_random_chance():
    """The bomb column marginal is a distinct readout the P(bad) test cannot vouch for
    -- a bug that scrambled the bomb axis would leave P(bad) intact yet destroy
    P(bomb). Build rounds directly under the uniform-lie model and check the column
    marginal concentrates on the true bomb holder above the 1/N baseline. P(bomb) is
    per-round (§3.5), so this works one round at a time and never combines."""
    rng = Random(999)
    N, K = 6, 2000
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
        bad1, bad2 = rng.sample(range(N), 2)
        decls = wires.astype(float).copy()
        for i in range(N):
            if i in (bad1, bad2) or i == bomb:
                decls[i] = rng.randint(0, H)
        probs = tb.ProbDeclaration(decls, H, active)
        _, p_bomb = tb.DeTensor(probs)
        mass_on_true += p_bomb[bomb]
        if int(np.argmax(p_bomb)) == bomb:
            top1 += 1
    p_bomb_on_true = mass_on_true / K
    baseline = 1 / N
    assert p_bomb_on_true > 0.22, f"P(bomb|true holder)={p_bomb_on_true:.3f} not above baseline {baseline:.3f}"
    assert top1 / K > 0.3, f"top-1 bomb accuracy={top1 / K:.3f} not above baseline {baseline:.3f}"


# --- four-stat cut panel: independent oracle -----------------------------------

def brute_role_entropy(probs):
    """Shannon entropy (bits) of the role posterior P(bad pair), recomputed straight
    from the config tensor: marginalise the bomb axis to the pair distribution (sums
    to 1) and sum -p*log2(p). Independent of EntropyBad / DeTensor."""
    n = probs.shape[0]
    pair = np.zeros((n, n))
    for b1, b2, h in configs(n):
        pair[b1][b2] += probs[b1][b2][h]
    ent = 0.0
    for b1 in range(n):
        for b2 in range(b1):
            p = pair[b1][b2]
            if p > 0:
                ent -= p * math.log2(p)
    return ent


def brute_expected_post_entropy_1ply(decls, probs, revealed, found, hand_size, active_wires):
    """The stat-3 quantity (1-ply expected post-cut role entropy) recomputed entirely
    from the independent oracle: oracle P_wire for the wire/dud weights, oracle
    ProbCut for each successor belief, oracle role entropy for the value. A wire drops
    active_wires by one (and increments found); a dud leaves them. Shares no algebra
    with the module's NextHBad."""
    n, H = len(decls), int(hand_size)
    pw = brute_pwire(decls, probs, revealed, found, hand_size, active_wires)
    out = np.full(n, np.nan)
    for i in range(n):
        if revealed[i] >= H:
            continue
        e = np.zeros(n, dtype=int)
        e[i] = 1
        h_wire = h_dud = 0.0
        if pw[i] > 1e-9 and active_wires > 0:
            post_w = brute_probcut(decls, probs, revealed + e, found + e, H, active_wires - 1)
            h_wire = brute_role_entropy(post_w)
        if pw[i] < 1 - 1e-9:
            post_d = brute_probcut(decls, probs, revealed + e, found, H, active_wires)
            h_dud = brute_role_entropy(post_d)
        out[i] = pw[i] * h_wire + (1 - pw[i]) * h_dud
    return out


# --- four-stat cut panel: tests ------------------------------------------------

def test_entropybad_known_values():
    n = 3
    certain = np.zeros((n, n, n))
    certain[1][0][2] = 1.0  # one pair, bomb somewhere -> no role uncertainty
    assert abs(tb.EntropyBad(certain)) < TOL
    uniform = np.zeros((n, n, n))
    for b1, b2 in [(1, 0), (2, 0), (2, 1)]:
        uniform[b1][b2][0] = 1.0 / 3  # uniform over the 3 pairs
    assert abs(tb.EntropyBad(uniform) - math.log2(3)) < 1e-9


def test_nexthbad_matches_oracle_sweep():
    """Stat 3 against a fully independent oracle (no module call inside the reference):
    a model can update beliefs correctly yet the information lookahead can still be
    wrong, so this pins NextHBad to the generative definition."""
    rng = Random(7)
    max_diff = 0.0
    for _ in range(600):
        decls, revealed, found, hand_size, active, total = random_consistent_state(rng)
        prior = tb.ProbDeclaration(decls, hand_size, total)
        if prior.sum() == 0:
            continue
        probs = tb.ProbCut(decls, prior, revealed, found, hand_size, active)
        got = tb.NextHBad(decls, probs, revealed, found, hand_size, active)
        ref = brute_expected_post_entropy_1ply(decls, probs, revealed, found, hand_size, active)
        for i in range(len(decls)):
            if np.isnan(got[i]) and np.isnan(ref[i]):
                continue
            assert not (np.isnan(got[i]) or np.isnan(ref[i])), "nan mismatch"
            max_diff = max(max_diff, abs(got[i] - ref[i]))
    assert max_diff < 1e-9, f"max diff vs oracle = {max_diff}"


def test_cutpanel_invariants_and_assembly_sweep():
    """The panel's entries are in range (probabilities in [0,1], entropies in
    [0, log2(C(N,2))]), non-cuttable rows are all-nan, and the assembled columns equal
    the standalone stat functions."""
    rng = Random(8)
    for _ in range(400):
        decls, revealed, found, hand_size, active, total = random_consistent_state(rng)
        prior = tb.ProbDeclaration(decls, hand_size, total)
        if prior.sum() == 0:
            continue
        probs = tb.ProbCut(decls, prior, revealed, found, hand_size, active)
        n = len(decls)
        num_pairs = n * (n - 1) // 2
        max_ent = math.log2(num_pairs) if num_pairs > 1 else 0.0
        panel = tb.CutPanel(decls, probs, revealed, found, hand_size, active)
        ps_ref = tb.P_wire(decls, probs, revealed, found, hand_size, active)
        _, pb_ref = tb.DeTensor(probs)
        for i in range(n):
            if revealed[i] >= hand_size:
                assert np.all(np.isnan(panel[i]))
                continue
            ps, pb, dh, rh = panel[i]
            assert -TOL <= ps <= 1 + TOL
            assert -TOL <= pb <= 1 + TOL
            assert -TOL <= dh <= max_ent + 1e-6
            assert -TOL <= rh <= max_ent + 1e-6
            assert abs(ps - ps_ref[i]) < TOL  # column 0 == P_wire
            assert abs(pb - pb_ref[i]) < TOL  # column 1 == DeTensor bomb marginal


def test_roundhorizon_depth1_equals_nexthbad():
    """Stat 4 at lookahead depth 1 is exactly stat 3: a single opening cut expanded
    over its two outcomes, with no further continuation."""
    rng = Random(9)
    for _ in range(300):
        decls, revealed, found, hand_size, active, total = random_consistent_state(rng)
        prior = tb.ProbDeclaration(decls, hand_size, total)
        if prior.sum() == 0:
            continue
        probs = tb.ProbCut(decls, prior, revealed, found, hand_size, active)
        nh = tb.NextHBad(decls, probs, revealed, found, hand_size, active)
        rh = tb.RoundHorizonH(decls, probs, revealed, found, hand_size, active, max_depth=1)
        for i in range(len(decls)):
            if np.isnan(nh[i]) and np.isnan(rh[i]):
                continue
            assert abs(nh[i] - rh[i]) < 1e-12


# --- CombineProbs robustness (eps-floor + log-space, ADR 0005) -----------------

def test_combineprobs_matches_exact_product_when_positive():
    """On strictly-positive per-round vectors (no hard zeros) the robust CombineProbs
    is numerically identical to the plain elementwise-product-and-renormalise."""
    rng = Random(11)
    for _ in range(100):
        n = rng.randint(3, 6)
        mats = []
        ref = np.ones((n, n))
        for _ in range(rng.randint(1, 8)):
            m = np.zeros((n, n))
            for b1 in range(n):
                for b2 in range(b1):
                    m[b1][b2] = rng.uniform(0.05, 1.0)  # strictly positive
            m /= m.sum()
            mats.append(m)
            ref = ref * m
        ref = np.tril(ref, -1)
        ref /= ref.sum()
        got = tb.CombineProbs(mats)
        assert np.allclose(got, ref, atol=1e-7), np.max(np.abs(got - ref))


def test_combineprobs_eps_floor_revives_zeroed_pair():
    """A pair a single round calls impossible (hard 0) is not permanently eliminated:
    later rounds favouring it can revive it to the top suspect. The old bare product
    would pin it at 0 forever."""
    n = 3
    r1 = np.zeros((n, n))
    r1[1][0] = 0.5
    r1[2][0] = 0.5            # pair (2,1) declared impossible this round
    r2 = np.zeros((n, n))
    r2[2][1] = 1.0           # later rounds point hard at (2,1)
    r3 = np.zeros((n, n))
    r3[2][1] = 1.0
    got = tb.CombineProbs([r1, r2, r3])
    assert got[2][1] > 0.0                              # revived from a hard zero
    assert np.unravel_index(np.argmax(got), got.shape) == (2, 1)
    assert abs(got.sum() - 1.0) < 1e-9


def test_combineprobs_logspace_no_collapse():
    """Many peaked rounds underflow the raw product to all-zeros (old code then dumps
    to uniform, discarding the evidence); log-space stays exact and concentrated."""
    n = 4
    base = np.zeros((n, n))
    for b1 in range(n):
        for b2 in range(b1):
            base[b1][b2] = 0.1
    base[3][0] = 0.5         # dominant pair; max entry 0.5 underflows at this depth
    base = np.tril(base, -1)
    base /= base.sum()
    top = np.unravel_index(np.argmax(base), base.shape)
    got = tb.CombineProbs([base] * 1100)
    assert not np.any(np.isnan(got))
    assert got[top] > 0.999                             # concentrated, not collapsed
    assert abs(got.sum() - 1.0) < 1e-9


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
