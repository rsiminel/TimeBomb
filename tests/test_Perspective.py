"""Tests for the player-perspective belief (model.md §3.5.2).

Run with pytest:   .venv/bin/python -m pytest tests/test_Perspective.py -q
Or standalone:     .venv/bin/python tests/test_Perspective.py

The perspective model conditions the §3.3–§3.5 pipeline on a seated player's private
knowledge: their own role, and — each round — their own hand (true wire count, bomb).
The references below are written independently of the module, straight from the
generative model: enumerate every wire deal, keep those matching the truthful
declarations AND the viewer's known hand, restrict to viewer-consistent configurations,
and sum. Every reference enumerates the free-hand wire split explicitly (no closed
form), so it shares no algebra with the module.
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
from baseline_stats import gap_is_positive

TOL = 1e-9
comb = math.comb
product = itertools.product
combinations = itertools.combinations

CASES = [(1, 0), (2, 0), (1, 1), (2, 1)]


# --- independent reference implementations -------------------------------------

def hypergeom_pmf(found, hand_size, wires, revealed):
    found, hand_size = int(round(found)), int(round(hand_size))
    wires, revealed = int(round(wires)), int(round(revealed))
    if not (0 <= wires <= hand_size and 0 <= revealed <= hand_size):
        return 0.0
    denom = comb(hand_size, revealed)
    if denom == 0:
        return 0.0
    return comb(wires, found) * comb(hand_size - wires, revealed - found) / denom


def bomb_pmf(found, hand_size, wires, revealed):
    """Must-not-draw term (model.md §3.2)."""
    found, hand_size = int(round(found)), int(round(hand_size))
    wires, revealed = int(round(wires)), int(round(revealed))
    if not (0 <= wires <= hand_size - 1 and 0 <= revealed <= hand_size):
        return 0.0
    denom = comb(hand_size, revealed)
    if denom == 0:
        return 0.0
    return comb(wires, found) * comb(hand_size - 1 - wires, revealed - found) / denom


def configs(n, num_bad, num_bom):
    return [bad + bom
            for bad in combinations(range(n), num_bad)
            for bom in combinations(range(n), num_bom)]


def split_indices(idx, num_bad):
    return idx[:num_bad], idx[num_bad:]


def consistent(bad, bom, viewer):
    """Config compatible with the viewer's role and (per-round) bomb knowledge."""
    if (viewer.idx in bad) != bool(viewer.is_bad):
        return False
    return not bom or (bom[0] == viewer.idx) == bool(viewer.has_bomb)


def persp_generative_prior(decls, hand_size, active_wires, num_bad, num_bom, viewer):
    """P(config | decls, viewer's hand) by full enumeration of wire deals under the
    generative model (uniform placement over non-bomb slots, uniform lies with the
    (H+1)^{-|F|} factor over the FULL free set, viewer included when they lie)."""
    decls = [int(round(d)) for d in decls]
    n, H, A = len(decls), int(hand_size), int(active_wires)
    v = viewer.idx
    post = np.zeros(tuple([n] * (num_bad + num_bom)))
    for bad in combinations(range(n), num_bad):
        for bom in combinations(range(n), num_bom):
            if not consistent(bad, bom, viewer):
                continue
            free = set(bad) | set(bom)
            lie = (H + 1.0) ** (-len(free))
            slots = [H - 1 if g in bom else H for g in range(n)]
            tot = 0.0
            for w in product(range(H + 1), repeat=n):
                if sum(w) != A or w[v] != int(viewer.wires):
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
            bad, bom = split_indices(idx, num_bad)
            if consistent(bad, bom, viewer):
                post[idx] = 1.0
        return post / post.sum()
    return post / s


def persp_cut_likelihood_ref(decls, revealed, found, hand_size, active_wires, bad, bom,
                             viewer):
    """Cut likelihood under config (bad, bom) with the viewer's hand pinned to its true
    count; the split over the OTHER free hands summed explicitly."""
    n, H = len(decls), int(hand_size)
    v = viewer.idx
    free = sorted((set(bad) | set(bom)) - {v})
    bomb = bom[0] if bom else None
    L = (bomb_pmf(found[v], H, viewer.wires, revealed[v]) if viewer.has_bomb
         else hypergeom_pmf(found[v], H, viewer.wires, revealed[v]))
    for j in range(n):
        if j != v and j not in free:
            L *= hypergeom_pmf(found[j], H, decls[j], revealed[j])
    truthful = sum(int(round(decls[j])) for j in range(n) if j != v and j not in free)
    t_free = int(active_wires) + int(np.sum(found)) - truthful - int(viewer.wires)
    slots = {g: H - 1 if g == bomb else H for g in free}
    free_slots = sum(slots.values())
    if not (0 <= t_free <= free_slots):
        return 0.0
    denom = comb(free_slots, t_free)
    if denom == 0:
        return 0.0
    s = 0.0
    for combo in product(*[range(slots[g] + 1) for g in free]) if free else [()]:
        if sum(combo) != t_free:
            continue
        split = dict(zip(free, combo))
        place = 1
        obs = 1.0
        for g in free:
            wg = split[g]
            place *= comb(slots[g], wg)
            obs *= (bomb_pmf(found[g], H, wg, revealed[g]) if g == bomb
                    else hypergeom_pmf(found[g], H, wg, revealed[g]))
        s += place * obs
    if not free:
        s = 1.0 if t_free == 0 else 0.0
    return L * s / denom


def brute_persp_probcut(decls, prior, revealed, found, hand_size, active_wires,
                        num_bad, num_bom, viewer):
    n = len(decls)
    post = np.zeros_like(np.asarray(prior, dtype=float))
    for idx in configs(n, num_bad, num_bom):
        bad, bom = split_indices(idx, num_bad)
        if not consistent(bad, bom, viewer):
            continue
        post[idx] = prior[idx] * persp_cut_likelihood_ref(
            decls, revealed, found, hand_size, active_wires, bad, bom, viewer)
    s = post.sum()
    return np.asarray(prior, dtype=float).copy() if s == 0 else post / s


def persp_split_means(decls, revealed, found, hand_size, active_wires, bad, bom, viewer):
    """E[remaining wires] in each non-viewer free hand given the observation."""
    H = int(hand_size)
    v = viewer.idx
    free = sorted((set(bad) | set(bom)) - {v})
    bomb = bom[0] if bom else None
    truthful = sum(int(round(decls[j])) for j in range(len(decls))
                   if j != v and j not in free)
    t_free = int(active_wires) + int(np.sum(found)) - truthful - int(viewer.wires)
    slots = {g: H - 1 if g == bomb else H for g in free}
    free_slots = sum(slots.values())
    if not (0 <= t_free <= free_slots):
        return {}
    acc = {g: 0.0 for g in free}
    norm = 0.0
    for combo in product(*[range(slots[g] + 1) for g in free]) if free else [()]:
        if sum(combo) != t_free:
            continue
        split = dict(zip(free, combo))
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


def brute_persp_pwire(decls, probs, revealed, found, hand_size, active_wires,
                      num_bad, num_bom, viewer):
    n, H = len(decls), int(hand_size)
    v = viewer.idx
    pw = np.zeros(n)
    for idx in configs(n, num_bad, num_bom):
        p = probs[idx]
        if p == 0:
            continue
        bad, bom = split_indices(idx, num_bad)
        if not consistent(bad, bom, viewer):
            continue
        free = (set(bad) | set(bom)) - {v}
        means = persp_split_means(decls, revealed, found, hand_size, active_wires,
                                  bad, bom, viewer)
        for i in range(n):
            cards_left = H - revealed[i]
            if cards_left <= 0:
                continue
            if i == v:
                rem = viewer.wires - found[v]
            elif i not in free:
                rem = decls[i] - found[i]
            else:
                if i not in means:
                    continue
                rem = means[i]
            if 0 <= rem <= cards_left:
                pw[i] += p * rem / cards_left
    return pw


# --- helpers -------------------------------------------------------------------

def is_viewer_distribution(m, num_bad, num_bom, viewer):
    """Valid config tensor AND zero on every config the viewer can rule out."""
    m = np.asarray(m)
    if np.any(np.isnan(m)) or np.any(m < -TOL) or np.any(m > 1 + TOL):
        return False
    n = m.shape[0]
    for idx in configs(n, num_bad, num_bom):
        bad, bom = split_indices(idx, num_bad)
        if not consistent(bad, bom, viewer) and abs(m[idx]) > TOL:
            return False
    return abs(m.sum() - 1.0) < 1e-6


def random_state_with_truth(rng, num_bad, num_bom, max_extra=3, max_hand=4):
    """A play-out state (no bomb cut yet) that also returns the ground truth needed to
    build a genuine Viewer: (decls, revealed, found, hand_size, active_now, total,
    wires, bad_set, bomb). ``max_extra``/``max_hand`` cap the size for the oracle tests
    that enumerate every wire deal."""
    n = rng.randint(num_bad + 1, num_bad + max_extra)
    hand_size = rng.randint(2, max_hand)
    bad_set = set(rng.sample(range(n), num_bad))
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
    return decls, revealed, found, hand_size, active_wires, total_active, wires, bad_set, bomb


def viewer_at(rng, n, wires, bad_set, bomb):
    v = rng.randrange(n)
    return gen.Viewer(v, v in bad_set, int(wires[v]), bomb == v)


# --- PerspectiveProbDeclaration --------------------------------------------------

def test_persp_prior_matches_generative_oracle():
    rng = Random(41)
    for num_bad, num_bom in CASES:
        max_diff = 0.0
        for _ in range(150):
            state = random_state_with_truth(rng, num_bad, num_bom,
                                            max_extra=2, max_hand=3)
            decls, _, _, hand_size, _, total, wires, bad_set, bomb = state
            viewer = viewer_at(rng, len(decls), wires, bad_set, bomb)
            got = gen.PerspectiveProbDeclaration(decls, hand_size, total,
                                                 num_bad, num_bom, viewer)
            ref = persp_generative_prior(decls, hand_size, total, num_bad, num_bom, viewer)
            max_diff = max(max_diff, np.max(np.abs(got - ref)))
        assert max_diff < 1e-9, f"(B={num_bad},M={num_bom}) prior vs oracle = {max_diff}"


def test_persp_prior_valid_and_viewer_certain():
    """Distribution invariants, plus the viewer's own readouts are exact: P(bad=v) is
    0/1 by role and, with a bomb, P(bomb=v) is 0/1 by their hand."""
    rng = Random(42)
    for num_bad, num_bom in CASES:
        for _ in range(100):
            state = random_state_with_truth(rng, num_bad, num_bom)
            decls, _, _, hand_size, _, total, wires, bad_set, bomb = state
            viewer = viewer_at(rng, len(decls), wires, bad_set, bomb)
            m = gen.PerspectiveProbDeclaration(decls, hand_size, total,
                                               num_bad, num_bom, viewer)
            assert is_viewer_distribution(m, num_bad, num_bom, viewer)
            prob_bad, prob_bom = gen.Separate(m, num_bad, num_bom)
            p_v = gen.DeMatrix(prob_bad)[viewer.idx]
            assert abs(p_v - (1.0 if viewer.is_bad else 0.0)) < TOL
            if num_bom:
                pb_v = np.asarray(prob_bom).reshape(-1)[viewer.idx]
                assert abs(pb_v - (1.0 if viewer.has_bomb else 0.0)) < TOL


def test_persp_prior_degeneracy_uniform_over_consistent():
    # impossible: everyone declares 0 but 12 wires demanded -> uniform over the configs
    # the viewer cannot rule out
    for num_bad, num_bom in CASES:
        decls = np.array([0., 0., 0., 0.])
        for is_bad, has_bomb in [(False, False), (True, False),
                                 (False, num_bom == 1), (True, num_bom == 1)]:
            viewer = gen.Viewer(1, is_bad, 0, has_bomb)
            m = gen.PerspectiveProbDeclaration(decls, hand_size=3, active_wires=12,
                                               num_bad=num_bad, num_bom=num_bom,
                                               viewer=viewer)
            assert is_viewer_distribution(m, num_bad, num_bom, viewer)
            cells = [idx for idx in configs(4, num_bad, num_bom)
                     if consistent(*split_indices(idx, num_bad), viewer)]
            for idx in cells:
                assert abs(m[idx] - 1.0 / len(cells)) < TOL


# --- PerspectiveProbCut ----------------------------------------------------------

def test_persp_probcut_matches_bruteforce_sweep():
    rng = Random(43)
    for num_bad, num_bom in CASES:
        max_diff = 0.0
        for _ in range(400):
            state = random_state_with_truth(rng, num_bad, num_bom)
            decls, revealed, found, hand_size, active, total, wires, bad_set, bomb = state
            viewer = viewer_at(rng, len(decls), wires, bad_set, bomb)
            prior = gen.PerspectiveProbDeclaration(decls, hand_size, total,
                                                   num_bad, num_bom, viewer)
            got = gen.PerspectiveProbCut(decls, prior, revealed, found, hand_size,
                                         active, num_bad, num_bom, viewer)
            ref = brute_persp_probcut(decls, prior, revealed, found, hand_size,
                                      active, num_bad, num_bom, viewer)
            max_diff = max(max_diff, np.max(np.abs(got - ref)))
            assert is_viewer_distribution(got, num_bad, num_bom, viewer)
        assert max_diff < 1e-9, f"(B={num_bad},M={num_bom}) probcut vs brute = {max_diff}"


def test_persp_probcut_impossible_observation_keeps_prior():
    """A cut record impossible under every viewer-consistent config (the viewer holds 0
    wires but a wire was 'found' in their hand) returns the prior unchanged."""
    for num_bad, num_bom in CASES:
        decls = np.array([1., 1., 1., 1.])
        viewer = gen.Viewer(0, False, 0, False)  # knows they hold no wires...
        prior = gen.PerspectiveProbDeclaration(decls, 3, 3, num_bad, num_bom, viewer)
        revealed = np.array([1, 0, 0, 0])
        found = np.array([1, 0, 0, 0])  # ...yet a wire came out of their hand
        post = gen.PerspectiveProbCut(decls, prior, revealed, found, 3, 2,
                                      num_bad, num_bom, viewer)
        assert post is prior


# --- PerspectiveP_wire -----------------------------------------------------------

def test_persp_pwire_matches_bruteforce_sweep():
    rng = Random(44)
    for num_bad, num_bom in CASES:
        max_diff = 0.0
        for _ in range(400):
            state = random_state_with_truth(rng, num_bad, num_bom)
            decls, revealed, found, hand_size, active, total, wires, bad_set, bomb = state
            viewer = viewer_at(rng, len(decls), wires, bad_set, bomb)
            prior = gen.PerspectiveProbDeclaration(decls, hand_size, total,
                                                   num_bad, num_bom, viewer)
            probs = gen.PerspectiveProbCut(decls, prior, revealed, found, hand_size,
                                           active, num_bad, num_bom, viewer)
            got = gen.PerspectiveP_wire(decls, probs, revealed, found, hand_size,
                                        active, num_bad, num_bom, viewer)
            ref = brute_persp_pwire(decls, probs, revealed, found, hand_size,
                                    active, num_bad, num_bom, viewer)
            max_diff = max(max_diff, np.max(np.abs(got - ref)))
            assert np.all(got >= -TOL) and np.all(got <= 1 + TOL)
            # the viewer's own row is exact
            v = viewer.idx
            if revealed[v] < hand_size:
                exact = (viewer.wires - found[v]) / (hand_size - revealed[v])
                assert abs(got[v] - exact) < 1e-9
        assert max_diff < 1e-9, f"(B={num_bad},M={num_bom}) pwire vs brute = {max_diff}"


# --- reduction: a good, bomb-free viewer == masked public posterior ---------------

def test_good_viewer_reduces_to_masked_public():
    """For a good, bomb-free viewer the perspective pinning coincides with the public
    truthful pinning (decls[v] = wires[v]), so the perspective belief must equal the
    public posterior masked to the viewer-consistent configs and renormalised."""
    rng = Random(45)
    for num_bad, num_bom in CASES:
        for _ in range(200):
            state = random_state_with_truth(rng, num_bad, num_bom)
            decls, revealed, found, hand_size, active, total, wires, bad_set, bomb = state
            n = len(decls)
            good = [i for i in range(n) if i not in bad_set and i != bomb]
            if not good:
                continue
            v = rng.choice(good)
            viewer = gen.Viewer(v, False, int(wires[v]), False)
            pub = gen.ProbDeclaration(decls, hand_size, total, num_bad, num_bom)
            mask = np.zeros_like(pub)
            for idx in configs(n, num_bad, num_bom):
                if consistent(*split_indices(idx, num_bad), viewer):
                    mask[idx] = 1.0
            masked = pub * mask
            if masked.sum() == 0:
                continue
            masked = masked / masked.sum()
            persp = gen.PerspectiveProbDeclaration(decls, hand_size, total,
                                                   num_bad, num_bom, viewer)
            assert np.allclose(persp, masked, atol=1e-9)
            post_pub = gen.ProbCut(decls, masked, revealed, found, hand_size, active,
                                   num_bad, num_bom)
            post_persp = gen.PerspectiveProbCut(decls, persp, revealed, found, hand_size,
                                                active, num_bad, num_bom, viewer)
            assert np.allclose(post_persp, post_pub, atol=1e-9)


# --- joint num_bad from the viewer's seat (§3.5.1 + §3.5.2) -----------------------

def brute_persp_joint_one_round(decls, revealed, found, hand_size, active_total,
                                active_now, num_bom, prior_b, viewer):
    """Independent re-derivation of the one-round joint num_bad posterior conditioned
    on the viewer's knowledge: u(S;B) = Σ_h ŵ_decl·L over consistent (S,h) only, then
    P(B) ∝ prior(B)·(1/C(N,B))·Σ_S u(S;B) — the C(N,B) subset prior restricted to the
    consistent sets IS the viewer-role update."""
    n, H = len(decls), int(hand_size)
    decls_i = [int(round(d)) for d in decls]
    v = viewer.idx
    log_ev, p_set = {}, {}
    for B in prior_b:
        u = np.zeros([n] * B)
        for bad in combinations(range(n), B):
            tot = 0.0
            for bom in combinations(range(n), num_bom):
                if not consistent(bad, bom, viewer):
                    continue
                free = (set(bad) | set(bom)) - {v}
                truthful = sum(decls_i[j] for j in range(n) if j != v and j not in free)
                t_free = active_total - int(viewer.wires) - truthful
                free_slots = sum(H - (1 if g in bom else 0) for g in free)
                if not (0 <= t_free <= free_slots):
                    continue
                denom = 1.0
                for g in free:
                    denom *= comb(H, decls_i[g])
                if denom == 0:
                    continue
                wdecl = (H + 1.0) ** (-len(free)) * comb(free_slots, t_free) / denom
                tot += wdecl * persp_cut_likelihood_ref(decls_i, revealed, found, H,
                                                        active_now, bad, bom, viewer)
            u[bad] = tot
        s = u.sum()
        p_set[B] = (u / s) if s > 0 else None
        log_ev[B] = (math.log(prior_b[B]) - math.log(comb(n, B)) + math.log(s)) \
            if s > 0 else -math.inf
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


def test_persp_joint_one_round_matches_independent_formula():
    """PerspectiveRoundLogU -> JointBadBelief equals the independent §3.5.1+§3.5.2
    formula at N=4 (B in {1,2}) — the cross-B absolute-weight guard from a seat."""
    rng = Random(46)
    n, num_bom = 4, 1
    prior_b = gen.NUM_BAD_PRIOR(4)
    max_diff = 0.0
    for _ in range(60):
        true_b = rng.choice(list(prior_b))
        state = random_state_with_truth(rng, true_b, num_bom)
        while len(state[0]) != n:  # the joint prior is defined at N=4 exactly
            state = random_state_with_truth(rng, true_b, num_bom)
        decls, revealed, found, hand_size, active, total, wires, bad_set, bomb = state
        viewer = viewer_at(rng, n, wires, bad_set, bomb)
        # viewer's role knowledge must not contradict a candidate B outright at B=1:
        # (a bad viewer under B=1 forces S={v}; still a valid, consistent case)
        log_u = {B: gen.PerspectiveRoundLogU(decls, revealed, found, hand_size, total,
                                             active, B, num_bom, viewer)
                 for B in prior_b}
        p_bad, p_nb, _ = gen.JointBadBelief(log_u, prior_b)
        rb, rnb = brute_persp_joint_one_round(decls, revealed, found, hand_size, total,
                                              active, num_bom, prior_b, viewer)
        max_diff = max(max_diff, np.max(np.abs(p_bad - rb)))
        for B in prior_b:
            max_diff = max(max_diff, abs(p_nb[B] - rnb[B]))
        assert abs(p_bad[viewer.idx] - (1.0 if viewer.is_bad else 0.0)) < 1e-9
    assert max_diff < 1e-9, f"perspective joint vs independent formula = {max_diff}"


# --- panel from the viewer's seat --------------------------------------------------

def test_persp_cutpanel_invariants_and_self_row():
    rng = Random(47)
    for num_bad, num_bom in CASES:
        for _ in range(60):
            state = random_state_with_truth(rng, num_bad, num_bom)
            decls, revealed, found, hand_size, active, total, wires, bad_set, bomb = state
            n = len(decls)
            viewer = viewer_at(rng, n, wires, bad_set, bomb)
            prior = gen.PerspectiveProbDeclaration(decls, hand_size, total,
                                                   num_bad, num_bom, viewer)
            probs = gen.PerspectiveProbCut(decls, prior, revealed, found, hand_size,
                                           active, num_bad, num_bom, viewer)
            num_sets = comb(n, num_bad)
            max_ent = math.log2(num_sets) if num_sets > 1 else 0.0
            panel = gen.CutPanel(decls, probs, revealed, found, hand_size, active,
                                 num_bad, num_bom, max_depth=2, viewer=viewer)
            ps_ref = gen.PerspectiveP_wire(decls, probs, revealed, found, hand_size,
                                           active, num_bad, num_bom, viewer)
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
            # the viewer's own risk row is certain knowledge
            v = viewer.idx
            if num_bom and revealed[v] < hand_size:
                assert abs(panel[v][1] - (1.0 if viewer.has_bomb else 0.0)) < TOL


def test_persp_roundhorizon_depth1_equals_nexthbad():
    rng = Random(48)
    for num_bad, num_bom in CASES:
        for _ in range(60):
            state = random_state_with_truth(rng, num_bad, num_bom)
            decls, revealed, found, hand_size, active, total, wires, bad_set, bomb = state
            viewer = viewer_at(rng, len(decls), wires, bad_set, bomb)
            prior = gen.PerspectiveProbDeclaration(decls, hand_size, total,
                                                   num_bad, num_bom, viewer)
            probs = gen.PerspectiveProbCut(decls, prior, revealed, found, hand_size,
                                           active, num_bad, num_bom, viewer)
            nh = gen.NextHBad(decls, probs, revealed, found, hand_size, active,
                              num_bad, num_bom, viewer)
            rh = gen.RoundHorizonH(decls, probs, revealed, found, hand_size, active,
                                   num_bad, num_bom, max_depth=1, viewer=viewer)
            for i in range(len(decls)):
                if np.isnan(nh[i]) and np.isnan(rh[i]):
                    continue
                assert abs(nh[i] - rh[i]) < 1e-12


# --- end-to-end: the seat beats the table -----------------------------------------

def _simulate_game_both_beliefs(N=5):
    """One full game (B=2, M=1) tracked by BOTH pipelines for viewer seat 0. Returns
    (roles, p_bad_public, p_bad_perspective)."""
    roles = np.zeros(N, dtype=int)
    for i in random.sample(range(N), 2):
        roles[i] = 1
    prior_b = {2: 1.0}
    log_pub = {2: np.zeros([N] * 2)}
    log_per = {2: np.zeros([N] * 2)}
    hand_size, active = 5, N
    while hand_size > 1 and active > 0:
        total = active
        wires, bombs = gen.DistributeWires(N, hand_size, active, 1)
        decls = wires.copy()
        for i in range(N):
            if roles[i] == 1 or bombs[i] == 1:
                decls[i] = random.randint(0, hand_size)
        viewer = gen.Viewer(0, bool(roles[0]), int(wires[0]), bool(bombs[0]))
        revealed = np.zeros(N, dtype=int)
        found = np.zeros(N, dtype=int)
        bomb_cut = False
        for _ in range(N):
            cutee = random.randrange(N)
            while revealed[cutee] >= hand_size:
                cutee = random.randrange(N)
            randy = random.randint(1, hand_size - revealed[cutee])
            if bombs[cutee] == 1 and randy == hand_size - revealed[cutee]:
                bomb_cut = True  # exclude the detonating cut itself (conditions broken)
                break
            if randy <= wires[cutee] - found[cutee]:
                found[cutee] += 1
                active -= 1
            revealed[cutee] += 1
            if active <= 0:
                break
        d = decls.astype(float)
        log_pub[2] = log_pub[2] + gen.RoundLogU(d, revealed, found, hand_size, total,
                                                active, 2, 1)
        log_per[2] = log_per[2] + gen.PerspectiveRoundLogU(d, revealed, found, hand_size,
                                                           total, active, 2, 1, viewer)
        if bomb_cut:
            break
        hand_size -= 1
    p_pub, _, _ = gen.JointBadBelief(log_pub, prior_b)
    p_per, _, _ = gen.JointBadBelief(log_per, prior_b)
    return roles, p_pub, p_per


def test_perspective_beats_public_end_to_end():
    """Over many games, seat 0's perspective belief (i) is exact about seat 0 itself and
    (ii) separates the OTHER players' roles at least as well as the public belief —
    strictly better on average, since it conditions on a superset of the evidence."""
    random.seed(146)
    np.random.seed(146)
    K = 300
    persp_gaps, paired_diffs = [], []
    for _ in range(K):
        roles, p_pub, p_per = _simulate_game_both_beliefs()
        assert abs(p_per[0] - roles[0]) < 1e-6  # own role is certain knowledge
        others = range(1, len(roles))
        bad = [i for i in others if roles[i] == 1]
        good = [i for i in others if roles[i] == 0]
        gap_per = np.mean([p_per[i] for i in bad]) - np.mean([p_per[i] for i in good])
        gap_pub = np.mean([p_pub[i] for i in bad]) - np.mean([p_pub[i] for i in good])
        persp_gaps.append(gap_per)
        paired_diffs.append(gap_per - gap_pub)
    ok, m, lo = gap_is_positive(persp_gaps)
    assert ok, f"perspective P(bad) bad-good gap={m:.3f} (5-sigma lower {lo:.3f}) not > 0"
    ok, m, lo = gap_is_positive(paired_diffs)
    assert ok, f"perspective-public paired gap={m:.3f} (5-sigma lower {lo:.3f}) not > 0"


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
