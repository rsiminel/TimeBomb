"""Tests for the General variant (arbitrary B bad guys, M in {0,1} bomb).

Run with pytest:   .venv/bin/python -m pytest tests/test_General.py -q
Or standalone:     .venv/bin/python tests/test_General.py

`General` writes the uniform-lie model (model.md §3.1–§3.5) once over the general
configuration space; every hardcoded variant is the projection at a fixed
``(num_bad, num_bom)``. The references below are written independently of the module,
straight from the generative model, and — crucially — include the **lie-count factor**
``(H+1)^{-|F|}`` (ADR 0007) that the earlier variant oracles dropped:

  * a config is a bad set ``S`` (size num_bad) and, for M=1, a bomb holder ``h``;
  * the bomb sits in hand h (one slot); the A wires are placed uniformly among the
    remaining slots (multivariate hypergeometric, weight Π_g C(slots_g, w[g]));
  * a hand is truthful iff good and bomb-free; each of the |F| free hands is a uniform
    liar, contributing a factor (H+1)^{-|F|};
  * cuts condition on "no bomb drawn yet": the bomb hand uses the must-not-draw term.

Every reference enumerates the free-hand wire split explicitly (no closed form, no
Vandermonde collapse), so it shares no algebra with the module.
"""
import itertools
import math
import random
import sys
from pathlib import Path
from random import Random

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "timebomb"))

import General as gen

TOL = 1e-9
comb = math.comb
product = itertools.product
combinations = itertools.combinations

# the (num_bad, num_bom) projections to exercise
CASES = [(1, 0), (2, 0), (1, 1), (2, 1)]


# --- independent reference implementations -------------------------------------

def hypergeom_pmf(found, hand_size, wires, revealed):
    """Standard hypergeometric PMF via math.comb (a bomb-free hand)."""
    found, hand_size = int(round(found)), int(round(hand_size))
    wires, revealed = int(round(wires)), int(round(revealed))
    if not (0 <= wires <= hand_size and 0 <= revealed <= hand_size):
        return 0.0
    denom = comb(hand_size, revealed)
    if denom == 0:
        return 0.0
    return comb(wires, found) * comb(hand_size - wires, revealed - found) / denom


def bomb_pmf(found, hand_size, wires, revealed):
    """Must-not-draw term (model.md §3.2): find `found` wires AND no bomb in `revealed`
    draws from a hand of `wires` wires, one bomb, H-1-wires blanks."""
    found, hand_size = int(round(found)), int(round(hand_size))
    wires, revealed = int(round(wires)), int(round(revealed))
    if not (0 <= wires <= hand_size - 1 and 0 <= revealed <= hand_size):
        return 0.0
    denom = comb(hand_size, revealed)
    if denom == 0:
        return 0.0
    return comb(wires, found) * comb(hand_size - 1 - wires, revealed - found) / denom


def configs(n, num_bad, num_bom):
    """Every (bad_set, bom_set) configuration as an index tuple bad + bom."""
    return [bad + bom
            for bad in combinations(range(n), num_bad)
            for bom in combinations(range(n), num_bom)]


def split_indices(idx, num_bad):
    """Recover (bad_set, bom_set) from a config index tuple."""
    return idx[:num_bad], idx[num_bad:]


def slots_of(g, bom_set, hand_size):
    return hand_size - 1 if g in bom_set else hand_size


def free_splits(free, t_free, slots):
    """Every wire vector over the free hands summing to t_free, each within its slots."""
    free = list(free)
    for combo in product(*[range(slots[g] + 1) for g in free]):
        if sum(combo) == t_free:
            yield dict(zip(free, combo))


def generative_declaration_prior(decls, hand_size, active_wires, num_bad, num_bom):
    """Posterior P(config | decls) under the generative model (model.md §2/§3.3, incl.
    the (H+1)^{-|F|} lie factor of ADR 0007), enumerated over wire vectors. Falls back
    to uniform over valid configs on impossible declarations (degeneracy)."""
    decls = [int(round(d)) for d in decls]
    n, H, A = len(decls), int(hand_size), int(active_wires)
    shape = tuple([n] * (num_bad + num_bom))
    post = np.zeros(shape)
    for bad in combinations(range(n), num_bad):
        for bom in combinations(range(n), num_bom):
            free = set(bad) | set(bom)
            lie = (H + 1.0) ** (-len(free))
            slots = [slots_of(g, bom, H) for g in range(n)]
            tot = 0.0
            for w in product(range(H + 1), repeat=n):
                if sum(w) != A:
                    continue
                if any(w[g] != decls[g] for g in range(n) if g not in free):
                    continue
                if any(w[g] > slots[g] for g in range(n)):
                    continue
                weight = lie
                for g in range(n):
                    weight *= comb(slots[g], w[g])
                tot += weight
            post[bad + bom] += tot
    s = post.sum()
    if s == 0:
        for idx in configs(n, num_bad, num_bom):
            post[idx] = 1.0
        return post / post.sum()
    return post / s


def cut_likelihood_ref(decls, revealed, found, hand_size, active_wires, bad, bom):
    """Cut likelihood under config (bad, bom), free-hand split summed explicitly."""
    n, H = len(decls), int(hand_size)
    free = sorted(set(bad) | set(bom))
    bomb = bom[0] if bom else None
    Ltruth = 1.0
    for j in range(n):
        if j not in free:
            Ltruth *= hypergeom_pmf(found[j], H, decls[j], revealed[j])
    truthful = sum(int(round(decls[j])) for j in range(n) if j not in free)
    t_free = int(active_wires) + int(np.sum(found)) - truthful
    slots = {g: slots_of(g, bom, H) for g in free}
    free_slots = sum(slots.values())
    if not (0 <= t_free <= free_slots):
        return 0.0
    denom = comb(free_slots, t_free)
    if denom == 0:
        return 0.0
    s = 0.0
    for split in free_splits(free, t_free, slots):
        place = 1
        obs = 1.0
        for g in free:
            wg = split[g]
            place *= comb(slots[g], wg)
            obs *= (bomb_pmf(found[g], H, wg, revealed[g]) if g == bomb
                    else hypergeom_pmf(found[g], H, wg, revealed[g]))
        s += place * obs
    return Ltruth * s / denom


def brute_probcut(decls, prior, revealed, found, hand_size, active_wires, num_bad, num_bom):
    n = len(decls)
    post = np.zeros_like(np.asarray(prior, dtype=float))
    for idx in configs(n, num_bad, num_bom):
        bad, bom = split_indices(idx, num_bad)
        post[idx] = prior[idx] * cut_likelihood_ref(
            decls, revealed, found, hand_size, active_wires, bad, bom)
    s = post.sum()
    return np.asarray(prior, dtype=float).copy() if s == 0 else post / s


def split_posterior_means(decls, revealed, found, hand_size, active_wires, bad, bom):
    """E[remaining wires] in each free hand given the observation, split posterior."""
    H = int(hand_size)
    free = sorted(set(bad) | set(bom))
    bomb = bom[0] if bom else None
    truthful = sum(int(round(decls[j])) for j in range(len(decls)) if j not in free)
    t_free = int(active_wires) + int(np.sum(found)) - truthful
    slots = {g: slots_of(g, bom, H) for g in free}
    free_slots = sum(slots.values())
    if not (0 <= t_free <= free_slots):
        return {}
    acc = {g: 0.0 for g in free}
    norm = 0.0
    for split in free_splits(free, t_free, slots):
        place = 1
        obs = 1.0
        for g in free:
            wg = split[g]
            place *= comb(slots[g], wg)
            obs *= (bomb_pmf(found[g], H, wg, revealed[g]) if g == bomb
                    else hypergeom_pmf(found[g], H, wg, revealed[g]))
        wt = place * obs
        for g in free:
            acc[g] += wt * (split[g] - found[g])
        norm += wt
    if norm <= 0:
        return {}
    return {g: acc[g] / norm for g in free}


def brute_pwire(decls, probs, revealed, found, hand_size, active_wires, num_bad, num_bom):
    n, H = len(decls), int(hand_size)
    pw = np.zeros(n)
    for idx in configs(n, num_bad, num_bom):
        p = probs[idx]
        if p == 0:
            continue
        bad, bom = split_indices(idx, num_bad)
        free = set(bad) | set(bom)
        means = split_posterior_means(decls, revealed, found, hand_size, active_wires, bad, bom)
        for i in range(n):
            cards_left = H - revealed[i]
            if cards_left <= 0:
                continue
            if i not in free:
                rem = decls[i] - found[i]
            else:
                if i not in means:
                    continue
                rem = means[i]
            if 0 <= rem <= cards_left:
                pw[i] += p * rem / cards_left
    return pw


# --- helpers -------------------------------------------------------------------

def is_config_distribution(m, num_bad, num_bom, total=1.0):
    """Valid config tensor: no NaN, entries in [0,1], sums to `total`, and zero off the
    sorted-index cells."""
    m = np.asarray(m)
    if np.any(np.isnan(m)) or np.any(m < -TOL) or np.any(m > 1 + TOL):
        return False
    n = m.shape[0]
    valid = set(configs(n, num_bad, num_bom))
    for idx in itertools.product(range(n), repeat=num_bad + num_bom):
        if idx not in valid and abs(m[idx]) > TOL:
            return False
    return abs(m.sum() - total) < 1e-6


def random_consistent_state(rng, num_bad, num_bom):
    """A state from an actual play-out with known roles + bomb, conditioned on no bomb
    cut (live inference always assumes "no bomb yet"). Returns
    (decls, revealed, found, hand_size, active_now, total_active)."""
    n = rng.randint(num_bad + 1, num_bad + 3)
    hand_size = rng.randint(2, 4)
    bad_set = rng.sample(range(n), num_bad)
    bomb = rng.randrange(n) if num_bom else None
    capacity = [hand_size - (1 if g == bomb else 0) for g in range(n)]
    total_active = rng.randint(0, sum(capacity))
    wires = np.zeros(n, dtype=int)
    given = 0
    while given < total_active:
        c = rng.randrange(n)
        if wires[c] < capacity[c]:
            wires[c] += 1
            given += 1
    decls = wires.astype(float).copy()
    for i in range(n):
        if i in bad_set or i == bomb:
            decls[i] = rng.randint(0, hand_size)
    revealed = np.zeros(n, dtype=int)
    found = np.zeros(n, dtype=int)
    active_wires = total_active
    for _ in range(rng.randint(0, n)):
        c = rng.randrange(n)
        bomb_here = 1 if c == bomb else 0
        nonbomb_left = (hand_size - revealed[c]) - bomb_here
        if nonbomb_left <= 0:
            continue
        if rng.randint(1, nonbomb_left) <= wires[c] - found[c]:
            found[c] += 1
            active_wires -= 1
        revealed[c] += 1
    return decls, revealed, found, hand_size, active_wires, total_active


# --- ProbDeclaration -----------------------------------------------------------

def test_probdeclaration_matches_generative_oracle():
    rng = Random(1)
    for num_bad, num_bom in CASES:
        max_diff = 0.0
        for _ in range(150):
            n = rng.randint(num_bad + 1, num_bad + 2)
            hand_size = rng.randint(2, 3)
            active = rng.randint(0, n * hand_size - 1)
            decls = np.array([float(rng.randint(0, hand_size)) for _ in range(n)])
            got = gen.ProbDeclaration(decls, hand_size, active, num_bad, num_bom)
            ref = generative_declaration_prior(decls, hand_size, active, num_bad, num_bom)
            max_diff = max(max_diff, np.max(np.abs(got - ref)))
        assert max_diff < 1e-9, f"(B={num_bad},M={num_bom}) max diff vs oracle = {max_diff}"


def test_probdeclaration_is_valid_distribution():
    rng = Random(2)
    for num_bad, num_bom in CASES:
        for _ in range(100):
            n = rng.randint(num_bad + 1, num_bad + 2)
            hand_size = rng.randint(2, 3)
            active = rng.randint(0, n)
            decls = np.array([float(rng.randint(0, hand_size)) for _ in range(n)])
            m = gen.ProbDeclaration(decls, hand_size, active, num_bad, num_bom)
            assert is_config_distribution(m, num_bad, num_bom)


def test_probdeclaration_degeneracy_falls_back_to_uniform():
    # impossible: declare 0 wires but demand many active -> uniform over valid configs
    for num_bad, num_bom in CASES:
        decls = np.array([0., 0., 0., 0.])
        m = gen.ProbDeclaration(decls, hand_size=3, active_wires=12,
                                num_bad=num_bad, num_bom=num_bom)
        assert is_config_distribution(m, num_bad, num_bom)
        cells = configs(4, num_bad, num_bom)
        for idx in cells:
            assert abs(m[idx] - 1.0 / len(cells)) < TOL


# --- ProbCut -------------------------------------------------------------------

def test_probcut_matches_bruteforce_sweep():
    rng = Random(3)
    for num_bad, num_bom in CASES:
        max_diff = 0.0
        for _ in range(500):
            decls, revealed, found, hand_size, active, total = random_consistent_state(rng, num_bad, num_bom)
            prior = gen.ProbDeclaration(decls, hand_size, total, num_bad, num_bom)
            if prior.sum() == 0:
                continue
            got = gen.ProbCut(decls, prior, revealed, found, hand_size, active, num_bad, num_bom)
            ref = brute_probcut(decls, prior, revealed, found, hand_size, active, num_bad, num_bom)
            max_diff = max(max_diff, np.max(np.abs(got - ref)))
        assert max_diff < 1e-9, f"(B={num_bad},M={num_bom}) max diff vs brute = {max_diff}"


def test_probcut_preserves_distribution_sweep():
    rng = Random(4)
    for num_bad, num_bom in CASES:
        for _ in range(500):
            decls, revealed, found, hand_size, active, total = random_consistent_state(rng, num_bad, num_bom)
            prior = gen.ProbDeclaration(decls, hand_size, total, num_bad, num_bom)
            if prior.sum() == 0:
                continue
            post = gen.ProbCut(decls, prior, revealed, found, hand_size, active, num_bad, num_bom)
            assert is_config_distribution(post, num_bad, num_bom)


# --- P_wire --------------------------------------------------------------------

def test_pwire_matches_bruteforce_sweep():
    rng = Random(5)
    for num_bad, num_bom in CASES:
        max_diff = 0.0
        for _ in range(500):
            decls, revealed, found, hand_size, active, total = random_consistent_state(rng, num_bad, num_bom)
            prior = gen.ProbDeclaration(decls, hand_size, total, num_bad, num_bom)
            if prior.sum() == 0:
                continue
            probs = gen.ProbCut(decls, prior, revealed, found, hand_size, active, num_bad, num_bom)
            got = gen.P_wire(decls, probs, revealed, found, hand_size, active, num_bad, num_bom)
            ref = brute_pwire(decls, probs, revealed, found, hand_size, active, num_bad, num_bom)
            max_diff = max(max_diff, np.max(np.abs(got - ref)))
            assert np.all(got >= -TOL) and np.all(got <= 1 + TOL)
        assert max_diff < 1e-9, f"(B={num_bad},M={num_bom}) max diff vs brute = {max_diff}"


# --- standalone runner ---------------------------------------------------------

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
