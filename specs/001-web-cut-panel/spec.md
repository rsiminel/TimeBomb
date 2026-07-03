# Feature Specification: Time Bomb Web Cut Panel

**Feature Branch**: `001-web-cut-panel`

**Created**: 2026-07-03

**Status**: Draft

**Input**: User description: "A browser assistant for playing Time Bomb with friends. The player sets up the game (number of players, names, number of bad guys, whether the bomb is in play), then each round enters what is publicly visible at the table: every player's declared wire count, and the result of each cut (safe wire, bomb, or nothing). After every entry the page shows an updated cut panel: for each player, the probability they are a bad guy, the probability they hold the bomb, the probability the next wire cut from their hand is a safe wire, and the information value of cutting them — the four-stat CutPanel computed by the timebomb/General.py solver. The panel presents these calibrated quantities only; it never tells the player whom to cut. The existing web/ folder (a small Python backend serving a static page) is being reworked to this; its old in-page math is discarded in favour of calling the solver."

## Clarifications

### Session 2026-07-03

- Q: At setup, how should the number of bad guys be specified? → A: Official
  uncertainty — setup offers the official role-deal for the player count, which can
  leave the true bad-guy count uncertain; the solver's joint inference handles it. A
  fixed count remains available as a manual override.
- Q: Should the assistant use the user's own private knowledge (their hand and role),
  or only public table information? → A: Public info only — the panel is computed
  from declarations and cut results everyone can see; own-hand conditioning is out of
  scope (it would need new solver interfaces).
- Q: Where will the assistant actually run when used at a table? → A: Local machine —
  the user starts it themselves and opens it in a browser (possibly from a phone on
  the same Wi-Fi). No accounts, no public hosting; security and multi-user isolation
  are out of scope.
- Q: How far back should undo reach? → A: Full history — undo steps back through
  every entry of the game, one at a time.
- Q: When computing the information-value stat would be too slow, what should the
  panel do? → A: Depth-capped, always shown — the stat is always displayed, computed
  at whatever lookahead depth fits the 2-second budget, and labeled as approximate
  when capped.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Live-game assistance (Priority: P1)

A player sitting at a real Time Bomb table opens the assistant, sets up the game
(number of players, their names, number of bad guys, whether the bomb is in play), and
as the round unfolds enters what everyone can publicly see: each player's declared wire
count at the start of the round, then the result of each cut (safe wire, bomb, or
nothing). After every entry the page shows the updated cut panel — for each player,
the probability they are a bad guy, the probability they hold the bomb, the probability
that a cut from their hand reveals a safe wire, and the information value of cutting
them. The panel shows these quantities only; it never marks, ranks, or recommends a
player to cut.

**Why this priority**: This is the product. Without setup → declarations → cuts →
updated panel, nothing else has value.

**Independent Test**: Set up a 5-player game, enter five declarations and a few cut
results, and confirm the four-stat panel appears and changes after every entry.

**Acceptance Scenarios**:

1. **Given** the assistant is open, **When** the user sets up a game (players, names,
   bad-guy count, bomb in play or not), **Then** the round-1 entry screen appears with
   every named player listed.
2. **Given** a set-up game, **When** the user enters all players' declarations,
   **Then** the cut panel appears showing the four quantities for every player.
3. **Given** declarations are in, **When** the user records a cut (who was cut, and
   whether it revealed a safe wire, the bomb, or nothing), **Then** the panel updates
   to reflect the new information.
4. **Given** any game state, **When** the panel is displayed, **Then** no element
   singles out a recommended cut — the four quantities are presented symmetrically for
   all players.

---

### User Story 2 - Multi-round play (Priority: P2)

The game continues past round one: when the round's cuts are done, the assistant
advances to the next round — hands shrink, cards are re-dealt at the table, and the
players declare again. Evidence accumulates: what the panel shows in round 3 reflects
declarations and cuts from rounds 1 and 2 as well. The assistant also recognises the
end of the game (all safe wires found, the bomb cut, or the final round exhausted) and
says which side won.

**Why this priority**: A single round is enough to demo the panel, but a real game is
four rounds; cross-round accumulation of evidence is the assistant's core advantage
over mental arithmetic.

**Independent Test**: Play two scripted rounds and confirm the round-2 panel differs
from what the same round-2 entries would produce alone; finish a game and confirm the
win banner.

**Acceptance Scenarios**:

1. **Given** a round in which the table's cuts for the round have all been entered,
   **When** the user advances to the next round, **Then** the assistant prompts for
   fresh declarations with the correct smaller hand size.
2. **Given** entries from earlier rounds, **When** a new round's panel is shown,
   **Then** it reflects the accumulated evidence of all rounds so far.
3. **Given** a cut that reveals the bomb (or the last safe wire, or the end of the
   final round), **When** it is entered, **Then** the assistant announces the game
   result and stops asking for input.

---

### User Story 3 - Table-mistake tolerance (Priority: P3)

Real tables are messy: the user mistypes a declaration, records the wrong cut, or the
players say something impossible (declarations that can't all be true, a cut result
that contradicts what was declared). The assistant never gets stuck: out-of-range
entries are rejected with a clear message, the most recent entry can be undone, and
impossible-but-well-formed information produces a visible warning while the assistant
continues with its best fallback interpretation.

**Why this priority**: Protects the P1/P2 experience during live play, but the happy
path works without it.

**Independent Test**: Enter an out-of-range declaration, an impossible declaration set,
and an accidental cut; confirm the rejection message, the warning, and that undo
restores the previous panel.

**Acceptance Scenarios**:

1. **Given** a declaration prompt for a hand of H cards, **When** the user enters a
   count below 0 or above H, **Then** the entry is rejected with a message and the
   user can re-enter it.
2. **Given** a set of declarations that cannot all be true under the game's rules,
   **When** the last one is entered, **Then** a warning is shown and the panel still
   renders using the assistant's fallback interpretation.
3. **Given** any sequence of entries, **When** the user chooses undo repeatedly,
   **Then** each undo steps the game state and panel back exactly one entry, all the
   way to the start of the game if desired.

---

### Edge Cases

- A cut result that is impossible given the declarations (e.g. more safe wires found in
  a hand than were declared): warn, keep the previous beliefs, allow undo.
- A player's hand runs out of cards mid-round: they cannot be selected for further cuts.
- The page is reloaded mid-game (accidental refresh at the table): the in-progress game
  is restored.
- Setup values outside the supported game (fewer than 4 or more than 8 players,
  bad-guy counts the rules don't allow): rejected at setup with a message.
- Repeated warnings in one round: the impossibility warning appears once per round, not
  on every subsequent entry.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Users MUST be able to set up a game by choosing the number of players,
  entering player names, and choosing whether the bomb is in play; the setup MUST
  default to the official role-deal for the chosen player count — including deals
  where the true bad-guy count is uncertain (e.g. "2 or 3"), which the panel MUST
  reflect faithfully — with a manual override to a fixed bad-guy count.
- **FR-002**: Each round, users MUST be able to enter every player's declared wire
  count; entries outside the valid range for the current hand size MUST be rejected
  with a clear message.
- **FR-003**: Users MUST be able to record each cut as it happens: which player was
  cut and whether it revealed a safe wire, the bomb, or nothing.
- **FR-004**: After every completed entry (a full set of declarations, or a cut), the
  page MUST display the updated four-stat cut panel for every player: probability of
  being a bad guy, probability of holding the bomb, probability that a cut from their
  hand reveals a safe wire, and the information value of cutting them. The
  information value is always shown, computed at whatever precision fits the panel's
  latency budget (SC-002) and visibly labeled as approximate when reduced precision
  was used.
- **FR-005**: All four displayed quantities MUST come from the project's validated
  solver; the web layer MUST NOT compute, approximate, or adjust any probability
  itself (Constitution I).
- **FR-006**: The panel MUST present quantities only — no recommendation, ranking,
  highlighting, or any other element that singles out a player to cut
  (Constitution II).
- **FR-007**: The assistant MUST track the official round structure automatically from
  the setup: hand sizes per round, cuts per round, round advancement, and re-declaration
  at the start of each round.
- **FR-008**: The panel shown at any point MUST reflect all evidence entered so far in
  the game, across all completed and current rounds.
- **FR-009**: The assistant MUST detect game end — all safe wires found, the bomb cut,
  or the final round exhausted — announce the winning side, and stop accepting entries.
- **FR-010**: When entries are well-formed but jointly impossible under the game's
  rules, the assistant MUST show a warning (at most once per round), continue with its
  best fallback interpretation, and never block or crash.
- **FR-011**: Users MUST be able to undo entries one at a time, stepping back through
  the game's full entry history; each undo restores the exact prior game state and
  panel.
- **FR-012**: An in-progress game MUST survive a page reload on the same device; users
  MUST be able to abandon it and start a new game at any time.

### Key Entities

- **Game**: one play-through — the player list, the possible bad-guy counts (a single
  number, or the official uncertain deal for the player count), bomb flag, current
  round, and accumulated evidence; ends with a winning side.
- **Player**: a named seat at the table; has a per-round declared wire count and a
  shrinking hand; the subject of one panel row.
- **Round**: one deal — a hand size, a full set of declarations, and the cuts made
  before the round ends.
- **Cut**: one recorded event — the player cut and its outcome (safe wire, bomb,
  nothing).
- **Cut panel**: the four quantities per player, recomputed after every entry.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A first-time user can go from opening the page to seeing their first cut
  panel in under 2 minutes for a 5-player game.
- **SC-002**: After any single entry, the updated panel is visible in under 2 seconds
  for games up to 8 players.
- **SC-003**: For any scripted game, every displayed quantity equals the solver's
  output for the same game state (exact agreement, verified over full scripted games).
- **SC-004**: 100% of malformed or impossible entries in a test script leave the
  assistant usable — a message or warning is shown and play can continue or be undone.
- **SC-005**: A full 4-round game can be completed without leaving or reloading the
  page, and a mid-game reload restores the game in 100% of test runs.

## Assumptions

- One assistant per table: a single user on a single device enters what they see; no
  accounts, no multi-device synchronisation.
- The assistant is started by the user on their own machine and used from a browser on
  the same device or a phone on the same network; public hosting, authentication, and
  multi-user isolation are out of scope.
- The game follows the official Time Bomb structure (4–8 players; four rounds with
  hands of 5, 4, 3, 2 cards; cuts per round equal to the number of players; safe wires
  to find equal to the number of players), while setup still lets the user choose the
  bad-guy count and bomb presence the table actually uses.
- The assistant is used on a phone or laptop at a live table, so the panel must be
  readable on a small screen.
- The assistant works from public table information only — declarations and cut
  results. The user's own hand and role are never entered or used; every player at the
  table could legitimately look at the same panel.
- Persistence beyond the current game (history of past games, statistics) is out of
  scope.
- The existing `web/` folder is reworked in place; its previous in-page math is
  discarded entirely (Constitution I), and no change is made to the solver or its
  tests (Constitution III).
