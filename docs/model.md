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
- A bad guy **without** the bomb **lies uniformly at random**.
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

Convert the round's declarations into a prior over configurations. Let the total
over-declaration be `excess = Σ decls − A`. If `excess = 0`, declarations are
consistent with everyone telling the truth, so no information is extractable →
uniform prior. Otherwise the liar(s) must account for `excess`, and each candidate
is weighted by the number of card-arrangements consistent with that lie:

- `excess > 0` (liar padded their count): weight `C(decls[i], excess)`.
- `excess < 0` (liar hid wires): weight `C(H − decls[i], −excess)`.

Normalize the weights to a probability distribution (or all-zeros if the
declarations are impossible under the model). The general version (`General.py`,
`web/app.py`) enumerates configurations over `combinations` and applies the
per-role lie model from §2.

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
- It **generalises** to multiple bad guys / bombs, matching `General.py` and
  `web/app.py` (configurations become `combinations`, not single indices).

A per-player binary update divides by `prior[i]·peh + (1−prior[i])·penh`, which is
`0` when an observation is impossible under *both* of a player's hypotheses,
producing `NaN`. (Minimal repro: `decls=[1,1,2,0]`, `H=2`, cut one of player 2's
two cards.) The joint form returns the correct distribution there.

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

### 3.6 Decision strategies

Two interchangeable policies for choosing the next cut:

- **Max-score / max-safe** — cut the player maximizing expected points, driven by
  `P_wire` and (when bombs exist) `1 − P(bomb)`. See `General.CutMaxScore`.
- **Min-entropy lookahead** — cut to minimize expected Shannon entropy of the
  posterior (`H`, `NextH`, `H_Min`); an information-greedy strategy.
