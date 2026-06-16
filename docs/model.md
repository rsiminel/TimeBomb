# Time Bomb Assistant — Model & Mathematics

The stable reference for the probability model shared by every variant. For the
plan and per-variant status see [roadmap.md](roadmap.md); for open work items see
[../TODO.md](../TODO.md); for orientation see the [README](../README.md).

## 1. Goal

Given only the *public* information at the table — how many wires each player
**declares**, and what each **cut** reveals — maintain, for every player, a
calibrated estimate of:

- **P(bad)** — probability the player is a Terrorist ("bad guy" / "evil"),
- **P(bomb)** — probability the player holds the Bomb,
- **P(wire)** — probability that cutting one of their face-down cards reveals an
  active (Safe) wire,

and use these to recommend **which wire to cut next**.

## 2. Game model (the rules the math depends on)

A game is played over several **rounds**. Notation used throughout the docs and
the code:

| Symbol         | Code name      | Meaning                                                             |
| -------------- | -------------- | ------------------------------------------------------------------- |
| `N`            | `num_players`  | Number of players.                                                  |
| `H`            | `hand_size`    | Cards in each player's hand **this round** (starts at 5, −1/round). |
| `A`            | `active_wires` | Safe wires still face-down across all hands.                        |
| `B`            | `num_bad`      | Number of bad guys (Terrorists). Known from player count.           |
| `M`            | `num_bom`      | Number of bombs in play (0 or 1 in the current variants).           |
| `decls[i]`     | `declarations` | Wires player `i` *claims* to hold this round.                       |
| `wires[i]`     | `wires`        | Wires player `i` *actually* holds (hidden; sim-only).               |
| `revealed[i]`  | `revealed`     | Cards already cut from player `i`'s hand this round.                |
| `found[i]`     | `found`        | Active wires found in player `i`'s hand this round.                 |

**Round flow.** Each round: wires are dealt; every player declares a wire count;
then `N` cuts happen (each cut flips one face-down card of some player), updating
`revealed`/`found`. After `N` cuts the round ends and `H` decreases by 1.

**Win/lose.** Good guys win when all `A` active wires are cut. Bad guys win if the
Bomb is cut, or if time runs out (`H` reaches 1 with wires uncut).

**Behavioral assumptions (the lie model).** These define the likelihoods:

- A good guy **without** the bomb declares **truthfully**: `decls[i] = wires[i]`.
- A good guy **with** the bomb **under-declares** (`decls[i] ≤ wires[i]`) to hide it.
- A bad guy **without** the bomb **declares uniformly at random** over `{0,…,H}`,
  independent of his true wire count. (Firm modelling choice; §3.3 derives the prior
  it implies. A richer *strategic* lie model is a possible far-future refinement —
  see roadmap.md.)
- A bad guy **with** the bomb **over-declares** (`decls[i] ≥ wires[i]`).

(For the `OneBadGuyNoBomb` variant only the first and third bullets apply.)

## 3. Mathematical models

### 3.1 State

The belief state is a probability array over **configurations** — an assignment of
which players are bad and which hold bombs. For `B` bad guys and `M` bombs it is a
`(B+M)`-dimensional array indexed `probs[bad_set + bomb_set]`. The simplest variant
(`OneBadGuyNoBomb`, `B=1, M=0`) collapses this to a length-`N` vector
`probs[i] = P(player i is the bad guy)`, normalized to sum to 1.

### 3.2 Hypergeometric likelihood

The atom shared by every computation. A hand of `n` cards contains `m` wires; `k`
cards are cut and `p` wires are found. The probability of that observation is the
hypergeometric PMF:

```
Lklhd(n, m, k, p) = C(m, p) · C(n − m, k − p) / C(n, k)
```

(`UsefulFunctions.Lklhd` writes this in the equivalent permutation form.)

### 3.3 Prior from declarations — `ProbDeclaration`

Convert the round's declarations into a prior over configurations by Bayes on the
**configuration space**, exactly parallel to `ProbCut` (§3.4). Marking which players
are bad pins every *good* hand to its declared wire count (good guys declare
truthfully), which forces the bad side's true wire total. The prior for a
configuration is then proportional to the multivariate-hypergeometric probability of
the unique deal it implies (every assignment of the `A` wires to the `N·H` card-slots
equally likely); the bad guys' declarations contribute only a constant factor under
the **uniform-lie** model (§2 — the bad guy declares uniformly over `{0,…,H}`
independent of his true count), which cancels in normalisation.

**Single bad guy (`B = 1`).** Let `excess = Σ decls − A`. Marking player `i` as bad
forces his true count to `t_i = decls[i] − excess`, giving the closed form

```
P(bad = i | decls)  ∝  C(H, t_i) / C(H, decls[i]),     t_i = decls[i] − excess
```

with `C(H, t) = 0` for `t < 0` or `t > H` (zeroing any candidate who cannot alone
account for the excess). When `excess = 0` every weight is `1` → uniform prior,
recovering the intuition that fully consistent declarations carry no information.

**Multiple bad guys (`B > 1`).** Marking the bad set `S` pins the good hands but
leaves the group's wire total `bg_wires = Σ_{b∈S} decls[b] − excess` to be split
among the `B·H` bad card-slots by the same uniform-placement model as §3.4.1.
Summing over splits via Vandermonde's identity collapses to a closed form:

```
P(bad set S | decls)  ∝  C(B·H,  Σ_{b∈S} decls[b] − excess)  /  Π_{b∈S} C(H, decls[b])
```

which reduces exactly to the `B = 1` formula when `|S| = 1`. Note that for `B > 1`,
`excess = 0` does **not** give a uniform prior — two lies can cancel in aggregate
while some bad sets remain more deal-plausible than others — so there is **no
`excess = 0` special case**; the closed form is used unconditionally.

**Degeneracy.** If every configuration has zero weight (declarations impossible under
the model — unreachable with consistent game data) fall back to the uniform prior,
never an unnormalisable all-zeros vector. This matches `ProbCut`'s "return the prior
on zero marginal" convention (§3.4).

### 3.4 Bayesian update from a cut — `ProbCut`

After each reveal, update the posterior by Bayes' rule on the **configuration
space** (which players are bad). The hypotheses are mutually exclusive and
exhaustive. Conditioned on a configuration `c`, every hand has a known wire count
— the bad candidate holds its *implied* count `bad_wires = A + Σfound − Σdecls +
decls[i]`, every truthful good guy holds exactly `decls[good]` — so the joint
observation factorises into independent per-hand hypergeometric draws:

```
L(c) = Lklhd(H, bad_wires, revealed[bad], found[bad])
       · Π_{good ∉ c} Lklhd(H, decls[good], revealed[good], found[good])

posterior(c) = prior(c) · L(c) / Σ_c′ prior(c′) · L(c′)
```

This factorisation is **exact given truthful good guys**: conditioning on a
configuration pins every good hand's wire count to `decls[good]` and the bad hand's
to its implied count, and given fixed per-hand counts the cut draws are independent
across hands. Variants where a good guy may hold the bomb and under-declare break
that pinning — there `L(c)` must marginalise over the good's hidden wire count
(deferred; §3.7, item 6).

**Why the joint form is canonical.** For `OneBadGuyNoBomb` the configuration space
is "which single player is bad", so `L` reduces to a length-`N` vector. This joint
form (not a per-player binary update) is the canonical implementation because:

- It is the literal statement of Bayes' theorem on the mutually-exclusive
  hypotheses, so its correctness is self-evident and easy to test.
- A common factor `Π_all Lklhd(good)` cancels in normalisation, making it
  equivalent to the intuitive `posterior[k] ∝ prior[k] · L_bad(k)/L_good(k)`.
- It **degrades gracefully**: a configuration whose observation is impossible gets
  zero posterior; the only failure mode (every hypothesis impossible,
  `marginal = 0`) is explicitly guarded by returning the prior.
- It **generalises** to multiple bad guys / bombs: configurations become
  `combinations` of players rather than single indices (see §3.4.1 for the extra
  wire-split modelling that `B > 1` requires).

A per-player binary update divides by `prior[i]·peh + (1−prior[i])·penh`, which is
`0` when an observation is impossible under *both* of a player's hypotheses,
producing `NaN`. (Minimal repro: `decls=[1,1,2,0]`, `H=2`, cut one of player 2's
two cards.) The joint form returns the correct distribution there.

### 3.4.1 Multiple bad guys: the wire-split model (B > 1)

For `B = 1` the single bad guy's wire count is fully determined:
`bad_wires = A + Σfound − Σ_good decls`. For `B > 1` only the bad *group*'s **total**

```
bg_wires = A + Σfound − Σ_good decls[good]
```

is determined; how it splits between the `B` bad hands is not, and the model must
say how.

**The model — uniform placement.** The `bg_wires` wires are placed uniformly at
random among the bad group's `B·H` combined card-slots (every arrangement equally
likely). This is the same assumption already used within each individual hand, and
it matches a shuffled deal: the randomness is over which *cards* are wires, not over
which bad guy is "assigned" each wire.

Under uniform placement the per-hand wire counts follow a multivariate
hypergeometric, and because the revealed cards are an exchangeable subset, the bad
group's contribution to the cut-likelihood **collapses to a single closed form** —
no summation over splits:

```
L_bad(bad set) = [ Π_{b ∈ bad set} C(revealed[b], found[b]) ]
                 · C(B·H − Σ_b revealed[b],  bg_wires − Σ_b found[b])
                 / C(B·H, bg_wires)
```

This reduces **exactly** to the single-hand `Lklhd(H, bad_wires, revealed, found)`
when `B = 1`, so it is the honest generalisation of §3.4. The full configuration
likelihood multiplies `L_bad` by the good guys' independent per-hand
hypergeometrics, and Bayes normalisation proceeds as in §3.4.

> **The current code uses a worse model — to be replaced.**
> `TwoBadGuysNoBomb.ProbCut` (and `General.py`) instead weight a split `k` of
> `bg_wires` between two bad hands by `C(bg_wires, k)`, normalised by
> `2^bg_wires` — a **Binomial(bg_wires, ½)** split, i.e. each wire flips an
> independent fair coin. That treats wires as capacity-free distinguishable objects
> and is only a large-`H` approximation to the correct hypergeometric split.
> Concretely, for `bg_wires = 2, H = 2` it gives split probabilities `(¼, ½, ¼)`
> instead of the correct `(⅙, ⅔, ⅙)`, over-crediting one bad guy hoarding the
> wires. Replace it with the closed form above (TODO.md, TwoBadGuysNoBomb Step 1).

### 3.5 Expected wire probability — `P_wire`

For ranking cut targets: the probability that cutting one of player `i`'s
unrevealed cards yields an active wire. Mix over the belief that `i` is good vs.
bad, dividing remaining wires by remaining face-down cards:

```
P_wire[i] = ( P(i good)·(decls[i] − found[i]) + P(i bad)·bad_wires_remaining[i] )
            / (H − revealed[i])
```

Two correctness requirements, both load-bearing:

- The denominator is `H − revealed[i]` (remaining face-down cards), **not** the
  full hand `H` — a card already revealed can no longer be cut.
- Each branch contributes only when its implied remaining-wire count is feasible,
  in `[0, H − revealed[i]]`. An impossible hypothesis (e.g. a "good" player who has
  already had more wires found than they declared) contributes nothing rather than
  a negative count.

For `B = 1`, `bad_wires_remaining[i]` is the fixed implied count minus `found[i]`.
For `B > 1` it is **not** a single number per hand: it is the marginal expected
remaining wires in bad hand `i` under the uniform-placement (multivariate-
hypergeometric) model of §3.4.1, averaged over configurations. `P_wire` must use
that marginal, not a per-hand fixed count. (The model is fixed by §3.4.1; computing
this marginal is an implementation task — TODO.md Axis A2.)

### 3.6 Decision strategies

Two interchangeable policies for choosing the next cut:

- **Max-score / max-safe** — cut the player maximizing expected points, driven by
  `P_wire` and (when bombs exist) `1 − P(bomb)`. See `General.CutMaxScore`.
- **Min-entropy lookahead** — cut to minimize expected Shannon entropy of the
  posterior (`H`, `NextH`, `H_Min`); an information-greedy strategy. (Here `H(·)` is
  Shannon entropy — the code reuses the name `H`, which §2's table also uses for the
  hand size; they are unrelated.)

### 3.7 Open modelling gaps

The declaration prior (§3.3), the `B > 1` wire-split (§3.4.1), the `B > 1` `P_wire`
marginal (§3.5), and the degeneracy convention are now **firm modelling choices**;
what remains for them is implementation and brute-force validation (see
[roadmap.md](roadmap.md) Cross-cutting foundations and [../TODO.md](../TODO.md) Axis
A). The one sub-model still unspecified:

- **Bomb likelihoods** — the declaration likelihood for bomb-holders
  (good-with-bomb under-declares, bad-with-bomb over-declares), the cut likelihood,
  and the `P(bomb)` readout (§1) — deferred until the `*OneBomb` variants.
