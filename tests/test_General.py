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
from baseline_stats import exceeds_baseline, gap_is_positive

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


# --- cut panel + CombineProbs: independent references --------------------------

def brute_role_entropy(probs, num_bad, num_bom):
    """Entropy (bits) of the role posterior P(bad set), recomputed independently."""
    n = probs.shape[0]
    prob_bad = np.zeros([n] * num_bad)
    for idx in configs(n, num_bad, num_bom):
        bad, _ = split_indices(idx, num_bad)
        prob_bad[bad] += probs[idx]
    ent = 0.0
    for bad in combinations(range(n), num_bad):
        p = prob_bad[bad]
        if p > 0:
            ent -= p * math.log2(p)
    return ent


def brute_expected_post_entropy_1ply(decls, probs, revealed, found, hand_size,
                                     active_wires, num_bad, num_bom):
    """Stat-3 quantity recomputed entirely from the oracle (P_wire + ProbCut +
    role entropy); a wire drops active_wires by one, a dud leaves it."""
    n, H = len(decls), int(hand_size)
    pw = brute_pwire(decls, probs, revealed, found, hand_size, active_wires, num_bad, num_bom)
    out = np.full(n, np.nan)
    for i in range(n):
        if revealed[i] >= H:
            continue
        e = np.zeros(n, dtype=int)
        e[i] = 1
        h_wire = h_dud = 0.0
        if pw[i] > 1e-9 and active_wires > 0:
            post_w = brute_probcut(decls, probs, revealed + e, found + e, H, active_wires - 1, num_bad, num_bom)
            h_wire = brute_role_entropy(post_w, num_bad, num_bom)
        if pw[i] < 1 - 1e-9:
            post_d = brute_probcut(decls, probs, revealed + e, found, H, active_wires, num_bad, num_bom)
            h_dud = brute_role_entropy(post_d, num_bad, num_bom)
        out[i] = pw[i] * h_wire + (1 - pw[i]) * h_dud
    return out


def test_entropybad_known_values():
    for num_bad, num_bom in CASES:
        n = num_bad + 2
        sets = list(combinations(range(n), num_bad))
        # certain on one bad set -> entropy 0
        certain = np.zeros([n] * (num_bad + num_bom))
        idx = configs(n, num_bad, num_bom)[0]
        certain[idx] = 1.0
        assert abs(gen.EntropyBad(certain, num_bad, num_bom)) < TOL
        # uniform over all bad sets (bomb fixed if any) -> log2(#sets)
        uni = np.zeros([n] * (num_bad + num_bom))
        for bad in sets:
            bom = (0,) if num_bom else ()
            uni[bad + bom] = 1.0 / len(sets)
        assert abs(gen.EntropyBad(uni, num_bad, num_bom) - math.log2(len(sets))) < 1e-9


def test_nexthbad_matches_oracle_sweep():
    rng = Random(7)
    for num_bad, num_bom in CASES:
        max_diff = 0.0
        for _ in range(150):
            decls, revealed, found, hand_size, active, total = random_consistent_state(rng, num_bad, num_bom)
            prior = gen.ProbDeclaration(decls, hand_size, total, num_bad, num_bom)
            if prior.sum() == 0:
                continue
            probs = gen.ProbCut(decls, prior, revealed, found, hand_size, active, num_bad, num_bom)
            got = gen.NextHBad(decls, probs, revealed, found, hand_size, active, num_bad, num_bom)
            ref = brute_expected_post_entropy_1ply(decls, probs, revealed, found, hand_size, active, num_bad, num_bom)
            for i in range(len(decls)):
                if np.isnan(got[i]) and np.isnan(ref[i]):
                    continue
                assert not (np.isnan(got[i]) or np.isnan(ref[i]))
                max_diff = max(max_diff, abs(got[i] - ref[i]))
        assert max_diff < 1e-9, f"(B={num_bad},M={num_bom}) NextHBad vs oracle = {max_diff}"


def test_cutpanel_invariants_and_assembly_sweep():
    rng = Random(8)
    for num_bad, num_bom in CASES:
        for _ in range(80):
            decls, revealed, found, hand_size, active, total = random_consistent_state(rng, num_bad, num_bom)
            prior = gen.ProbDeclaration(decls, hand_size, total, num_bad, num_bom)
            if prior.sum() == 0:
                continue
            probs = gen.ProbCut(decls, prior, revealed, found, hand_size, active, num_bad, num_bom)
            n = len(decls)
            num_sets = comb(n, num_bad)
            max_ent = math.log2(num_sets) if num_sets > 1 else 0.0
            # cap the round-horizon lookahead: the exact depth is O((2N)^cuts_left)
            panel = gen.CutPanel(decls, probs, revealed, found, hand_size, active,
                                 num_bad, num_bom, max_depth=2)
            ps_ref = gen.P_wire(decls, probs, revealed, found, hand_size, active, num_bad, num_bom)
            for i in range(n):
                if revealed[i] >= hand_size:
                    assert np.all(np.isnan(panel[i]))
                    continue
                psafe, pbomb, dh, rh = panel[i]
                assert -TOL <= psafe <= 1 + TOL
                assert -TOL <= pbomb <= 1 + TOL
                assert -TOL <= dh <= max_ent + 1e-6
                assert -TOL <= rh <= max_ent + 1e-6
                assert abs(psafe - ps_ref[i]) < TOL


def test_roundhorizon_depth1_equals_nexthbad():
    rng = Random(9)
    for num_bad, num_bom in CASES:
        for _ in range(80):
            decls, revealed, found, hand_size, active, total = random_consistent_state(rng, num_bad, num_bom)
            prior = gen.ProbDeclaration(decls, hand_size, total, num_bad, num_bom)
            if prior.sum() == 0:
                continue
            probs = gen.ProbCut(decls, prior, revealed, found, hand_size, active, num_bad, num_bom)
            nh = gen.NextHBad(decls, probs, revealed, found, hand_size, active, num_bad, num_bom)
            rh = gen.RoundHorizonH(decls, probs, revealed, found, hand_size, active, num_bad, num_bom, max_depth=1)
            for i in range(len(decls)):
                if np.isnan(nh[i]) and np.isnan(rh[i]):
                    continue
                assert abs(nh[i] - rh[i]) < 1e-12


def test_panel_matches_twobadguysonebomb():
    """Cross-variant: General at (num_bad,num_bom)=(2,1) must produce the same per-player
    cut panel as the dedicated TwoBadGuysOneBomb module (convention-independent)."""
    import TwoBadGuysOneBomb as b4
    rng = Random(21)
    for _ in range(40):
        decls, revealed, found, hand_size, active, total = random_consistent_state(rng, 2, 1)
        prior_g = gen.ProbDeclaration(decls, hand_size, total, 2, 1)
        prior_b = b4.ProbDeclaration(decls, hand_size, total)
        if prior_g.sum() == 0:
            continue
        probs_g = gen.ProbCut(decls, prior_g, revealed, found, hand_size, active, 2, 1)
        probs_b = b4.ProbCut(decls, prior_b, revealed, found, hand_size, active)
        pg = gen.CutPanel(decls, probs_g, revealed, found, hand_size, active, 2, 1, max_depth=2)
        pb = b4.CutPanel(decls, probs_b, revealed, found, hand_size, active, max_depth=2)
        both_nan = np.isnan(pg) & np.isnan(pb)
        assert np.all(both_nan | (np.abs(np.nan_to_num(pg) - np.nan_to_num(pb)) < 1e-9))


# --- CombineProbs robustness (eps-floor + log-space, ADR 0005) -----------------

def test_combineprobs_matches_exact_product_when_positive():
    rng = Random(11)
    for num_bad in (1, 2):
        for _ in range(60):
            n = rng.randint(num_bad + 1, num_bad + 3)
            mats, ref = [], np.ones([n] * num_bad)
            for _ in range(rng.randint(1, 6)):
                m = np.zeros([n] * num_bad)
                for bad in combinations(range(n), num_bad):
                    m[bad] = rng.uniform(0.05, 1.0)  # strictly positive
                m /= m.sum()
                mats.append(m)
                ref = ref * m
            mask = np.zeros([n] * num_bad, dtype=bool)
            for bad in combinations(range(n), num_bad):
                mask[bad] = True
            ref = np.where(mask, ref, 0.0)
            ref /= ref.sum()
            got = gen.CombineProbs(mats)
            assert np.allclose(got, ref, atol=1e-7)


def test_combineprobs_eps_floor_revives_zeroed_set():
    # General uses ascending-index bad sets (itertools.combinations, i < j).
    n = 4
    r1 = np.zeros((n, n)); r1[0][1] = 0.5; r1[2][3] = 0.5  # set (1,2) is hard-zeroed
    r2 = np.zeros((n, n)); r2[1][2] = 1.0
    r3 = np.zeros((n, n)); r3[1][2] = 1.0
    got = gen.CombineProbs([r1, r2, r3])
    assert got[1][2] > 0.0
    assert np.unravel_index(np.argmax(got), got.shape) == (1, 2)
    assert abs(got.sum() - 1.0) < 1e-9


def test_combineprobs_logspace_no_collapse():
    n = 4
    base = np.zeros((n, n))
    for bad in combinations(range(n), 2):
        base[bad] = 0.1
    base[0][3] = 0.5  # dominant set (0,3); max 0.5 underflows the raw product at depth 1100
    base /= base.sum()
    top = np.unravel_index(np.argmax(base), base.shape)
    got = gen.CombineProbs([base] * 1100)
    assert not np.any(np.isnan(got))
    assert got[top] > 0.999
    assert abs(got.sum() - 1.0) < 1e-9


# --- joint num_bad inference (ADR 0008) ----------------------------------------

def brute_joint_one_round(decls, revealed, found, hand_size, active_total, active_now,
                          num_bom, prior_b):
    """Independent re-derivation of the joint num_bad posterior after one round:
    u(S;B) = Σ_h ŵ_decl(S,h)·L_config(S,h), with ŵ_decl the lie-factored declaration
    weight and L_config from `cut_likelihood_ref` — then
    P(B) ∝ prior(B)·(1/C(N,B))·Σ_S u(S;B), P(S|B) ∝ u(S;B). Returns
    (p_bad, p_num_bad)."""
    n, H = len(decls), int(hand_size)
    decls = [int(round(d)) for d in decls]
    log_ev, p_set = {}, {}
    for B in prior_b:
        u = np.zeros([n] * B)
        for bad in combinations(range(n), B):
            tot = 0.0
            for bom in combinations(range(n), num_bom):
                free = set(bad) | set(bom)
                t_free = active_total - (sum(decls) - sum(decls[g] for g in free))
                free_slots = sum(H - (1 if g in bom else 0) for g in free)
                if not (0 <= t_free <= free_slots):
                    continue
                denom = 1.0
                for g in free:
                    denom *= comb(H, decls[g])
                if denom == 0:
                    continue
                wdecl = (H + 1.0) ** (-len(free)) * comb(free_slots, t_free) / denom
                tot += wdecl * cut_likelihood_ref(decls, revealed, found, H, active_now, bad, bom)
            u[bad] = tot
        s = u.sum()
        p_set[B] = (u / s) if s > 0 else None
        log_ev[B] = (math.log(prior_b[B]) - math.log(comb(n, B)) + math.log(s)) if s > 0 else -math.inf
    bs = list(prior_b)
    evs = np.array([log_ev[B] for B in bs])
    mm = evs[np.isfinite(evs)].max()
    ww = np.where(np.isfinite(evs), np.exp(evs - mm), 0.0)
    ww /= ww.sum()
    p_num_bad = {bs[k]: ww[k] for k in range(len(bs))}
    p_bad = np.zeros(n)
    for k, B in enumerate(bs):
        if p_set[B] is not None:
            for bad in combinations(range(n), B):
                for i in bad:
                    p_bad[i] += ww[k] * p_set[B][bad]
    return p_bad, p_num_bad


def test_joint_one_round_matches_independent_formula():
    """JointBadBelief after one RoundLogU equals an independent re-derivation of the
    ADR-0008 cross-B formula, for the player counts where num_bad is uncertain."""
    rng = Random(31)
    for n, prior_b in [(4, gen.NUM_BAD_PRIOR(4)), (7, {2: 3 / 8, 3: 5 / 8})]:
        max_diff = 0.0
        for _ in range(60):
            num_bom = 1
            # a single consistent round at this n
            hand_size = rng.randint(2, 3)
            cap_total = n * hand_size - 1
            total = rng.randint(0, min(cap_total, n + 2))
            decls, revealed, found, _, active_now, _ = _round_at(rng, n, hand_size, total, num_bom)
            log_u = {B: gen.RoundLogU(decls, revealed, found, hand_size, total, active_now, B, num_bom)
                     for B in prior_b}
            p_bad, p_nb, _ = gen.JointBadBelief(log_u, prior_b)
            rb, rnb = brute_joint_one_round(decls, revealed, found, hand_size, total, active_now, num_bom, prior_b)
            max_diff = max(max_diff, np.max(np.abs(p_bad - rb)))
            for B in prior_b:
                max_diff = max(max_diff, abs(p_nb[B] - rnb[B]))
        assert max_diff < 1e-9, f"n={n} joint vs independent formula = {max_diff}"


def _round_at(rng, n, hand_size, total_active, num_bom):
    """A single consistent round's (decls, revealed, found, hand_size, active_now,
    total) at fixed n / hand_size / total_active (helper for the joint tests)."""
    bomb = rng.randrange(n) if num_bom else None
    cap = [hand_size - (1 if g == bomb else 0) for g in range(n)]
    total_active = min(total_active, sum(cap))
    wires = np.zeros(n, dtype=int)
    given = 0
    while given < total_active:
        c = rng.randrange(n)
        if wires[c] < cap[c]:
            wires[c] += 1
            given += 1
    bad_set = rng.sample(range(n), rng.randint(1, min(3, n)))
    decls = wires.astype(float).copy()
    for i in range(n):
        if i in bad_set or i == bomb:
            decls[i] = rng.randint(0, hand_size)
    revealed = np.zeros(n, dtype=int)
    found = np.zeros(n, dtype=int)
    active = total_active
    for _ in range(rng.randint(0, n)):
        c = rng.randrange(n)
        bomb_here = 1 if c == bomb else 0
        nonbomb_left = (hand_size - revealed[c]) - bomb_here
        if nonbomb_left <= 0:
            continue
        if rng.randint(1, nonbomb_left) <= wires[c] - found[c]:
            found[c] += 1
            active -= 1
        revealed[c] += 1
    return decls, revealed, found, hand_size, active, total_active


def test_joint_within_b_matches_combineprobs():
    """Within a fixed num_bad, JointBadBelief's P(S|B) equals CombineProbs over the
    per-round normalised bad-set marginals (the two code paths must agree)."""
    rng = Random(32)
    for B in (1, 2):
        for _ in range(40):
            n = rng.randint(B + 1, B + 2)
            hand_size = rng.randint(2, 3)
            num_bom = 1
            log_u = np.zeros([n] * B)
            per_round = []
            for _ in range(rng.randint(1, 4)):
                total = rng.randint(0, n + 1)
                decls, revealed, found, _, active, _ = _round_at(rng, n, hand_size, total, num_bom)
                log_u = log_u + gen.RoundLogU(decls, revealed, found, hand_size, total, active, B, num_bom)
                prior = gen.ProbDeclaration(decls, hand_size, total, B, num_bom)
                post = gen.ProbCut(decls, prior, revealed, found, hand_size, active, B, num_bom)
                per_round.append(gen.Separate(post, B, num_bom)[0])
            _, _, p_set = gen.JointBadBelief({B: log_u}, {B: 1.0})
            combined = gen.CombineProbs(per_round)
            # both may hit the all-ruled-out fallback; compare only when both are proper
            if np.all(np.isfinite(log_u[gen._badset_mask(log_u.shape)])):
                assert np.allclose(p_set[B], combined, atol=1e-9)


def _check_joint_beats_random(N):
    """Over many games the joint belief concentrates P(num_bad) on the true count and
    P(bad) on the true bad guys. Split per player count (N=4, N=7) into its own test so
    the two heavy simulation runs schedule on separate xdist workers."""
    random.seed(100 + N)
    np.random.seed(100 + N)
    K = 250
    nbad_pg, gap_pg = [], []  # one entry per game (i.i.d.)
    for _ in range(K):
        _, p_bad, roles, p_nb = gen.PlayAuto(num_players=N, verbosity=0)
        true_b = int(roles.sum())
        nbad_pg.append(p_nb.get(true_b, 0.0))
        bad_idx = set(int(i) for i in np.where(roles == 1)[0])
        bg = float(np.mean([p_bad[i] for i in range(N) if i in bad_idx]))
        gg = float(np.mean([p_bad[i] for i in range(N) if i not in bad_idx]))
        gap_pg.append(bg - gg)
    # No-information baseline for P(num_bad): echoing the prior gives mean P(true B) =
    # Σ_B prior(B)^2 (true B ~ prior). Self-calibrating 5-sigma bars (baseline_stats):
    # the count belief beats that, and the role belief separates bad from good guys.
    no_info = sum(p * p for p in gen.NUM_BAD_PRIOR(N).values())
    ok, m, lo = exceeds_baseline(nbad_pg, no_info)
    assert ok, f"N={N} mean P(num_bad=true)={m:.3f} (5-sigma lower {lo:.3f}) <= no-info {no_info:.3f}"
    ok, gap, lo = gap_is_positive(gap_pg)
    assert ok, f"N={N} P(bad) bad-good gap={gap:.3f} (5-sigma lower {lo:.3f}) not > 0"


def test_joint_beats_random_n4():
    _check_joint_beats_random(4)


def test_joint_beats_random_n7():
    _check_joint_beats_random(7)


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
