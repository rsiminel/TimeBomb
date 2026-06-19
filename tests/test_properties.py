"""Property-based / fuzz tests and pinned regression fixtures for `General.py` (TODO D4).

The oracle sweeps in `test_General.py` already pin every belief function against an
independent `math.comb` brute force. This file adds the coverage a value-by-value oracle
*cannot* give:

  * **Structural properties** that must hold for any correct implementation regardless of
    the numbers -- player-relabeling **equivariance** (catches index/axis bugs), round-list
    **permutation invariance** of `CombineProbs` (catches ordering bugs), and the
    no-observation **identity** of `ProbCut`. None of these compares against a reference
    distribution, so they are a genuinely independent check.
  * **Distribution invariants at scale** -- entries in `[0,1]`, sums to 1, no `NaN` -- swept
    over `N`, hand sizes, and `num_bad` *beyond the brute force's reach* (it caps `N` small
    and `hand_size <= 3` for tractability). A bug that only bites at `N=7, hand_size=5` is
    invisible to the oracle sweeps but caught here.
  * **Pinned golden fixtures** -- a few small scenarios with their exact belief vectors
    frozen in the file, so a future refactor that silently shifts the numbers shows up as a
    diff in code review. The pinned values were generated from the *independent* brute-force
    oracle (`test_General.generative_declaration_prior` / `brute_probcut` / `brute_pwire`),
    never snapshotted from `General` itself, so they remain a correctness anchor too.

Run:   .venv/bin/python -m pytest tests/test_properties.py -q
       .venv/bin/python tests/test_properties.py        # standalone
"""
import sys
from pathlib import Path
from random import Random

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "timebomb"))

import General as gen
from test_General import configs, is_config_distribution

TOL = 1e-9
CASES = [(1, 0), (2, 0), (1, 1), (2, 1)]


# --- state generation (scales past the brute-force sweeps) ---------------------

def random_state(rng, n, hand_size, num_bad, num_bom):
    """A consistent mid-game state with known hidden roles + bomb, conditioned on "no
    bomb cut yet" (what live inference always assumes). Returns
    (decls, revealed, found, hand_size, active_now, total_active). Generalises
    `test_General.random_consistent_state` to arbitrary `n` / `hand_size`."""
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


# --- relabeling helpers --------------------------------------------------------

def relabel(arr, perm):
    """New position `i` carries old player `perm[i]`'s value."""
    return np.array([arr[perm[i]] for i in range(len(perm))])


def assert_config_equivariant(got_new, ref_old, perm, num_bad, num_bom):
    """A config belief is equivariant: the belief computed on relabeled inputs, read at
    cell `Bnew`, equals the original belief at the old cell `{perm[i] : i in Bnew}`."""
    n = got_new.shape[0]
    for idx in configs(n, num_bad, num_bom):
        bad, bom = idx[:num_bad], idx[num_bad:]
        old = tuple(sorted(perm[i] for i in bad)) + tuple(sorted(perm[i] for i in bom))
        assert abs(got_new[idx] - ref_old[old]) < TOL, (idx, old)


def assert_vector_equivariant(got_new, ref_old, perm):
    """A per-player vector is equivariant: got_new[i] == ref_old[perm[i]] (NaN-aware)."""
    for i in range(len(perm)):
        a, b = got_new[i], ref_old[perm[i]]
        if np.isnan(a) and np.isnan(b):
            continue
        assert abs(a - b) < TOL, (i, a, b)


# --- property: player-relabeling equivariance ----------------------------------

def test_relabeling_equivariance_declaration():
    rng = Random(101)
    for num_bad, num_bom in CASES:
        for _ in range(120):
            n = rng.randint(num_bad + 1, num_bad + 3)
            H = rng.randint(2, 4)
            decls, _, _, _, _, total = random_state(rng, n, H, num_bad, num_bom)
            perm = list(range(n)); rng.shuffle(perm)
            ref = gen.ProbDeclaration(decls, H, total, num_bad, num_bom)
            got = gen.ProbDeclaration(relabel(decls, perm), H, total, num_bad, num_bom)
            assert_config_equivariant(got, ref, perm, num_bad, num_bom)


def test_relabeling_equivariance_cut_and_pwire():
    rng = Random(102)
    for num_bad, num_bom in CASES:
        for _ in range(120):
            n = rng.randint(num_bad + 1, num_bad + 3)
            H = rng.randint(2, 4)
            decls, revealed, found, _, active, total = random_state(rng, n, H, num_bad, num_bom)
            prior = gen.ProbDeclaration(decls, H, total, num_bad, num_bom)
            if prior.sum() == 0:
                continue
            perm = list(range(n)); rng.shuffle(perm)
            post = gen.ProbCut(decls, prior, revealed, found, H, active, num_bad, num_bom)
            pw = gen.P_wire(decls, post, revealed, found, H, active, num_bad, num_bom)

            d2, r2, f2 = relabel(decls, perm), relabel(revealed, perm), relabel(found, perm)
            prior2 = gen.ProbDeclaration(d2, H, total, num_bad, num_bom)
            post2 = gen.ProbCut(d2, prior2, r2, f2, H, active, num_bad, num_bom)
            pw2 = gen.P_wire(d2, post2, r2, f2, H, active, num_bad, num_bom)

            assert_config_equivariant(post2, post, perm, num_bad, num_bom)
            assert_vector_equivariant(pw2, pw, perm)


def test_relabeling_equivariance_panel():
    rng = Random(103)
    for num_bad, num_bom in CASES:
        for _ in range(40):
            n = rng.randint(num_bad + 1, num_bad + 2)
            H = rng.randint(2, 3)
            decls, revealed, found, _, active, total = random_state(rng, n, H, num_bad, num_bom)
            prior = gen.ProbDeclaration(decls, H, total, num_bad, num_bom)
            if prior.sum() == 0:
                continue
            perm = list(range(n)); rng.shuffle(perm)
            post = gen.ProbCut(decls, prior, revealed, found, H, active, num_bad, num_bom)
            panel = gen.CutPanel(decls, post, revealed, found, H, active,
                                 num_bad, num_bom, max_depth=2)

            d2, r2, f2 = relabel(decls, perm), relabel(revealed, perm), relabel(found, perm)
            prior2 = gen.ProbDeclaration(d2, H, total, num_bad, num_bom)
            post2 = gen.ProbCut(d2, prior2, r2, f2, H, active, num_bad, num_bom)
            panel2 = gen.CutPanel(d2, post2, r2, f2, H, active, num_bad, num_bom, max_depth=2)
            # each panel column (psafe, pbomb, dH, rH) is an equivariant per-player vector
            for col in range(panel.shape[1]):
                assert_vector_equivariant(panel2[:, col], panel[:, col], perm)


# --- property: CombineProbs is invariant under round ordering ------------------

def test_combineprobs_permutation_invariant():
    """The cross-round posterior is an elementwise product (commutative), so combining the
    per-round factors in any order -- even through the eps-floor / log-space path -- yields
    the same belief. Catches an accidental order dependence in the accumulation."""
    rng = Random(104)
    for num_bad in (1, 2):
        for _ in range(80):
            n = rng.randint(num_bad + 1, num_bad + 3)
            mats = []
            for _ in range(rng.randint(2, 7)):
                m = np.zeros([n] * num_bad)
                for bad in configs(n, num_bad, 0):
                    m[bad] = rng.uniform(0.0, 1.0) if rng.random() < 0.85 else 0.0
                if m.sum() == 0:
                    m[configs(n, num_bad, 0)[0]] = 1.0
                m /= m.sum()
                mats.append(m)
            base = gen.CombineProbs(mats)
            shuffled = mats[:]; rng.shuffle(shuffled)
            other = gen.CombineProbs(shuffled)
            assert np.allclose(base, other, atol=1e-9), num_bad


# --- property: ProbCut with no observation leaves the prior unchanged ----------

def test_probcut_no_observation_is_identity():
    """Revealing nothing (revealed == found == 0) is a no-op: the cut posterior must equal
    the declaration prior exactly."""
    rng = Random(105)
    for num_bad, num_bom in CASES:
        for _ in range(120):
            n = rng.randint(num_bad + 1, num_bad + 3)
            H = rng.randint(2, 4)
            decls, _, _, _, _, total = random_state(rng, n, H, num_bad, num_bom)
            prior = gen.ProbDeclaration(decls, H, total, num_bad, num_bom)
            if prior.sum() == 0:
                continue
            zero = np.zeros(n, dtype=int)
            post = gen.ProbCut(decls, prior, zero, zero, H, total, num_bad, num_bom)
            assert np.allclose(post, prior, atol=TOL)


# --- property: distribution invariants beyond the brute force's reach ----------

def test_distribution_invariants_large_scale():
    """Sweep the invariants (entries in [0,1], sum to 1, no NaN; P_wire in [0,1]) over
    player counts and hand sizes the brute force is too slow to enumerate."""
    rng = Random(106)
    for num_bom in (0, 1):
        for num_bad in (1, 2, 3):
            for _ in range(120):
                n = rng.randint(num_bad + 1, 7)
                H = rng.randint(2, 5)
                decls, revealed, found, _, active, total = random_state(rng, n, H, num_bad, num_bom)
                prior = gen.ProbDeclaration(decls, H, total, num_bad, num_bom)
                assert is_config_distribution(prior, num_bad, num_bom)
                if prior.sum() == 0:
                    continue
                post = gen.ProbCut(decls, prior, revealed, found, H, active, num_bad, num_bom)
                assert is_config_distribution(post, num_bad, num_bom)
                pw = gen.P_wire(decls, post, revealed, found, H, active, num_bad, num_bom)
                assert not np.any(np.isnan(pw))
                assert np.all(pw >= -TOL) and np.all(pw <= 1 + TOL)


# --- pinned golden regression fixtures -----------------------------------------
# Values generated once from the INDEPENDENT brute-force oracle in test_General.py
# (generative_declaration_prior / brute_probcut / brute_pwire), not from General itself.
# To regenerate after an intentional model change, see the header of test_General.py.

def test_regression_fixture_symmetric_prior():
    # B=1, M=0, N=3, equal declarations, no cuts -> P(bad) uniform = 1/3 each.
    got = gen.ProbDeclaration(np.array([1., 1., 1.]), 2, 3, 1, 0)
    assert np.allclose(got, [1 / 3, 1 / 3, 1 / 3], atol=TOL)


def test_regression_fixture_b1m1_round_with_cut():
    # B=1, M=1, N=4. Declarations [0,1,2,1], H=2, 3 active wires at round start; then a
    # single cut on player 1 reveals a wire (revealed=found=[0,1,0,0], 2 active left).
    decls = np.array([0., 1., 2., 1.]); H = 2; total = 3
    revealed = np.array([0, 1, 0, 0]); found = np.array([0, 1, 0, 0]); active = 2

    prior_golden = [[0.0, 0.023255813953, 0.139534883721, 0.023255813953],
                    [0.023255813953, 0.06976744186, 0.06976744186, 0.03488372093],
                    [0.139534883721, 0.06976744186, 0.139534883721, 0.06976744186],
                    [0.023255813953, 0.03488372093, 0.06976744186, 0.06976744186]]
    post_golden = [[0.0, 0.0, 0.164383561644, 0.027397260274],
                   [0.0, 0.0, 0.109589041096, 0.027397260274],
                   [0.164383561644, 0.054794520548, 0.164383561644, 0.082191780822],
                   [0.027397260274, 0.013698630137, 0.082191780822, 0.082191780822]]
    pwire_golden = [0.082191780822, 0.054794520548, 0.479452054795, 0.41095890411]

    prior = gen.ProbDeclaration(decls, H, total, 1, 1)
    post = gen.ProbCut(decls, prior, revealed, found, H, active, 1, 1)
    pwire = gen.P_wire(decls, post, revealed, found, H, active, 1, 1)
    assert np.allclose(prior, prior_golden, atol=1e-9)
    assert np.allclose(post, post_golden, atol=1e-9)
    assert np.allclose(pwire, pwire_golden, atol=1e-9)


def test_regression_fixture_b2m0_prior():
    # B=2, M=0, N=4, declarations [2,0,1,1], 4 active wires, no cuts.
    prior_golden = [[0.0, 0.387096774194, 0.129032258065, 0.129032258065],
                    [0.0, 0.0, 0.129032258065, 0.129032258065],
                    [0.0, 0.0, 0.0, 0.096774193548],
                    [0.0, 0.0, 0.0, 0.0]]
    got = gen.ProbDeclaration(np.array([2., 0., 1., 1.]), 2, 4, 2, 0)
    assert np.allclose(got, prior_golden, atol=1e-9)


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
