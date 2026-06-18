"""Tests for the real-table consistency diagnostics (timebomb/Consistency.py, model.md §3.6).

Two layers, matching the repo convention:
- the pure validators (`valid_declaration`, `can_cut`, `prompt_int`) by truth table / injected
  input — these drive the *re-prompt* half of the hybrid policy;
- `declarations_feasible` against an **independent** `math.comb`-style feasibility brute force
  (plain config enumeration straight from model.md §3.3, not the module's `General._decl_weights`
  path), plus the impossible-cut **identity contract** every `Play` relies on: an impossible
  observation makes `ProbCut` return the *same prior object*.

Run: .venv/bin/python -m pytest tests/test_consistency.py -q   (or standalone with python).
"""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "timebomb"))

import Consistency as cons
import OneBadGuyNoBomb as v1
import TwoBadGuysNoBomb as v2
import OneBadGuyOneBomb as v3
import TwoBadGuysOneBomb as v4
import General as gen


# --- pure validators -----------------------------------------------------------

def test_valid_declaration_truth_table():
    H = 5
    assert all(cons.valid_declaration(v, H) for v in (0, 3, 5))
    assert not cons.valid_declaration(-1, H)
    assert not cons.valid_declaration(6, H)      # exceeds the hand
    assert not cons.valid_declaration(2.5, H)    # not a whole number
    assert not cons.valid_declaration("3", H)    # not an int


def test_can_cut():
    assert cons.can_cut(0, 5) and cons.can_cut(4, 5)
    assert not cons.can_cut(5, 5)                # hand fully revealed
    assert not cons.can_cut(6, 5)


def test_prompt_int_reprompts_until_valid():
    # skips a non-integer and an out-of-range value, then accepts the first valid one
    replies = iter(["abc", "9", "-1", "2"])
    got = cons.prompt_int("q", 0, 5, _input=lambda _p: next(replies))
    assert got == 2
    # accepts the boundaries
    assert cons.prompt_int("q", 0, 2, _input=lambda _p: "0") == 0
    assert cons.prompt_int("q", 1, 2, _input=lambda _p: "2") == 2


# --- declarations_feasible vs an independent brute force ------------------------

def brute_feasible(decls, hand_size, active_wires, num_bad, num_bom):
    """Independent feasibility: does *any* configuration (bad set S, bomb holder h) admit a
    non-negative, in-range wire split under model.md §3.3? Plain enumeration — shares no code
    with `General._decl_weights`. Assumes each declaration is already in `[0, H]` (the Play
    loops guarantee this via `prompt_int`), which is what `declarations_feasible` also assumes."""
    n, H, A = len(decls), hand_size, active_wires
    bom_sets = [()] if num_bom == 0 else list(combinations(range(n), num_bom))
    for bad_set in combinations(range(n), num_bad):
        for bom_set in bom_sets:
            free = set(bad_set) | set(bom_set)
            t_free = A - sum(decls[j] for j in range(n) if j not in free)
            free_slots = sum(H - (1 if g in bom_set else 0) for g in free)
            if 0 <= t_free <= free_slots:  # this config can hold the wires
                return True
    return False


def test_declarations_feasible_matches_bruteforce_sweep():
    rng = np.random.default_rng(0)
    cases = [(1, 0), (2, 0), (1, 1), (2, 1)]
    for num_bad, num_bom in cases:
        for _ in range(150):
            n = int(rng.integers(max(num_bad + num_bom, 3), 6))
            H = int(rng.integers(2, 6))
            A = int(rng.integers(0, n + 3))
            decls = rng.integers(0, H + 1, size=n)  # in [0, H], as Play guarantees
            got = cons.declarations_feasible(decls, H, A, num_bad, num_bom)
            assert got == brute_feasible(decls, H, A, num_bad, num_bom), \
                f"decls={list(decls)} H={H} A={A} ({num_bad},{num_bom}): {got}"


def test_declarations_feasible_pinned_cases():
    # consistent declarations (sum == active wires) are feasible
    assert cons.declarations_feasible(np.array([1, 1, 1, 1]), 5, 4, 1, 0)
    # a lone bad guy cannot hold 4 wires in a 2-card hand when all goods truthfully say 0
    assert not cons.declarations_feasible(np.array([0, 0, 0, 0]), 2, 4, 1, 0)
    # more active wires than non-bomb slots is impossible for anyone (N*H - 1 = 11 < 12)
    assert not cons.declarations_feasible(np.array([0, 0, 0, 0]), 3, 12, 1, 1)


# --- impossible-cut identity contract (the signal every Play uses) -------------

def test_probcut_returns_prior_object_on_impossible_observation():
    """Four hands each declare 2 wires (= both of their 2 cards) yet each shows an inactive on
    its first cut — that needs four liars, more than any model has (≤3 bad + bomb), so the
    observation is impossible under *every* config. Each module's `ProbCut` must then return the
    *same prior object* (the `marginal == 0` path); the interactive `Play` loops detect the
    impossible cut by exactly this identity (`probs is probabilities`)."""
    decls = np.array([2., 2., 2., 2., 0.])
    revealed, found = np.array([1, 1, 1, 1, 0]), np.zeros(5, dtype=int)
    H, A = 2, 4

    for mod in (v1, v2, v3, v4):
        prior = mod.ProbDeclaration(decls, H, A)
        assert mod.ProbCut(decls, prior, revealed, found, H, A) is prior
    for num_bad, num_bom in [(1, 0), (2, 0), (1, 1), (2, 1)]:
        pg = gen.ProbDeclaration(decls, H, A, num_bad, num_bom)
        assert gen.ProbCut(decls, pg, revealed, found, H, A, num_bad, num_bom) is pg


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
