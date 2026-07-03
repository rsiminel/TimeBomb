# Feature Specification: Hosted Local Time Bomb Game

**Feature Branch**: `002-host-local-game`

**Created**: 2026-07-03

**Status**: Draft

**Input**: User description: "Version 2 of the TimeBomb web app: natively host the Time Bomb game with a tabletop feel, playable against AI opponents and/or local human friends, alongside the untouched v1 assistant. A new home page at / offers two doors: the v1 assistant, moved as-is to /assistant (its only change is its URL), and the hosted game at /play. The hosted game: at setup the user picks the seat mix (human hotseat seats and AI seats), player names, official role-deal for the player count (manual override available), optional random seed for a reproducible deal, and whether the public-info stats panel is allowed this game (opt-in per game; when a per-player private panel becomes possible via future solver interfaces, it will be a separate per-game opt-in). Play is hotseat on one device: pass-the-device privacy screens (tap to reveal your hand and role, tap to hide before passing), but the architecture is LAN-shaped — game state and per-player hidden-information views are constructed server-side from day one so a future online/LAN update is a transport change, not a rewrite; secrets are never shipped to the browser and merely hidden. The rules engine (dealing, declaration collection, cut resolution, win judgment, per-player views) is the referee promoted out of sim/engine.py into a shared top-level package that both sim/ and web/ import; the web layer presents state and forwards intents only, and every probability shown still comes from timebomb/General.py (constitution v2.0.0). AI opponents are pluggable behind the existing agent interface: solver-driven bots ship in this version (using General.py posteriors from their private view plus a bluffing/cut policy), with LLM agents (sim/ arena style) as an optional mode for those with an API key. Table talk is structured claims only: humans and AIs can emit formal claims from a fixed menu (e.g. trust/distrust/accusation-style claims) — no free-text chat. During play, a toggleable, collapsible side window shows the public-info four-stat CutPanel (when the game opted in). Quality of life: a hosted game can be saved and resumed later on the same machine; when a game ends, all roles and hands are revealed and players can step back through the event history to see who lied when (post-game replay); seeded games give reproducible deals. Visuals: styled-DOM tabletop — player seats around a table, card backs, flip/cut transitions — vanilla JS, no framework, no build step. Out of scope for this version: online/remote play, accounts, free-text chat, per-player private stats panels (needs new solver interfaces), and any modification to the v1 assistant's behavior."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Solo game against AI opponents (Priority: P1)

A user opens the game, sets up a table where they take one seat and AI opponents fill
the rest (names, official role-deal for the player count, optional seed), and plays a
complete game of Time Bomb: cards are dealt, every player declares a wire count each
round, players cut wires from each other's hands, structured claims fly across the
table, rounds advance with shrinking hands, and the game announces which side won. The
AI opponents play plausibly — declaring, claiming, bluffing when bad, and choosing cuts
— without the user ever seeing information they shouldn't.

**Why this priority**: This is the product — a complete, natively hosted game. One
human plus AI seats exercises setup, dealing, every phase of play, AI decision-making,
hidden-information delivery, and win judgment, with no second human required.

**Independent Test**: Set up a 5-seat game with 1 human and 4 AI seats, play it to
completion, and confirm every phase occurs, only legal moves are possible, the user
never sees another seat's hand or role, and the announced winner matches the rules.

**Acceptance Scenarios**:

1. **Given** the game page is open, **When** the user configures seats (mix of one
   human and several AIs), names, role-deal, and optional seed, **Then** a game starts
   with cards dealt and the round-1 declaration phase begins.
2. **Given** a started game, **When** the declaration phase runs, **Then** every AI
   seat produces a declared wire count without user intervention and the human is
   prompted for theirs.
3. **Given** declarations are in, **When** it is the human's turn to cut, **Then** the
   table shows the legal target hands, the human picks a card, and the result (safe
   wire, dud, or bomb) is revealed to everyone with a card-flip.
4. **Given** it is an AI seat's turn to cut, **When** the AI decides, **Then** its cut
   resolves within the latency budget and the result is revealed the same way.
5. **Given** a cut ends the game (bomb found, all safe wires found, or final round
   exhausted), **When** the game ends, **Then** the winning side is announced and all
   roles are revealed.
6. **Given** the same setup and the same seed, **When** two games are started,
   **Then** the deals are identical.

---

### User Story 2 - Hotseat game with friends (Priority: P2)

A group of friends around one device plays Time Bomb with any mix of human and AI
seats. When a human needs to see private information (their role and hand), the device
is passed to them: a privacy screen names the player, they tap to reveal, act, and tap
to hide before passing the device on. No player can see another player's secrets, on
screen or by any other means.

**Why this priority**: Local multiplayer is the second pillar of v2 ("play against AIs
or real friends locally"). It builds directly on US1's game loop, adding only the
pass-the-device privacy flow.

**Independent Test**: Set up a 4-seat game with 3 humans and 1 AI; verify each human
only ever sees their own role/hand behind an explicit reveal tap, that the shared table
view between turns shows public information only, and that the game plays to
completion.

**Acceptance Scenarios**:

1. **Given** a game with multiple human seats, **When** a human's private action is
   needed (viewing their hand, declaring), **Then** the screen first shows a
   pass-the-device prompt naming that player with nothing private visible.
2. **Given** the pass-the-device prompt, **When** that player taps to reveal, **Then**
   their role and current hand appear, and a hide control returns to the neutral
   shared view before the device is passed.
3. **Given** any moment of the game, **When** the shared table view is showing,
   **Then** it contains only public information (declarations, claims, cut results,
   counts — never any player's role or uncut cards).
4. **Given** a human seat's turn to cut, **When** they act, **Then** they choose among
   legal targets exactly as in US1.

---

### User Story 3 - Home page with two doors (Priority: P3)

A user arriving at the site sees a home page offering two clearly-labeled entries: the
v1 assistant (for helping with a physical game) and the hosted game. The assistant
behaves exactly as it does today; only its address changes.

**Why this priority**: It's the requested front door and protects the v1 investment,
but it is small and the game itself (US1/US2) carries the value.

**Independent Test**: Visit the root address and confirm both entries exist; follow
the assistant entry and run an existing v1 flow end to end unchanged; follow the game
entry and reach game setup.

**Acceptance Scenarios**:

1. **Given** the app is running, **When** the user visits the root address, **Then**
   a home page presents entries to the assistant and to the hosted game.
2. **Given** the home page, **When** the user opens the assistant, **Then** the v1
   assistant loads at its new address and behaves identically to v1 (setup,
   declarations, cuts, panel, undo).
3. **Given** the home page, **When** the user opens the game, **Then** game setup
   (US1) is reached.

---

### User Story 4 - Opt-in stats side panel (Priority: P4)

At setup, the table decides whether this game allows the public-info stats panel. When
allowed, any time during play a collapsible side window can be toggled open to show the
four-stat cut panel (per player: probability bad, probability holds bomb, probability a
cut is a safe wire, information value of cutting) computed from public information
only — the same calibrated numbers the v1 assistant shows. When not allowed, the panel
is absent entirely.

**Why this priority**: The "assistant at the table" is the project's signature, but
the game is fully playable without it, and some groups will consider it cheating —
hence opt-in and lower priority.

**Independent Test**: Start one game with the panel allowed and one without; verify
the toggle exists (collapsed by default), opens/collapses mid-game, shows the four
stats per player matching the solver's output for the same public events, and is
completely absent in the second game.

**Acceptance Scenarios**:

1. **Given** game setup, **When** the user leaves the panel opt-in unchecked (the
   default), **Then** no panel control appears anywhere during play.
2. **Given** a game with the panel allowed, **When** a player toggles the side window
   open, **Then** the four-stat panel appears for all seats, computed from public
   information only, and updates as declarations and cuts are recorded.
3. **Given** the open panel, **When** a player collapses it, **Then** play continues
   unobstructed and it can be reopened later.
4. **Given** any panel content, **When** it is displayed, **Then** it never singles
   out a recommended cut and never reflects any seat's hidden information or any AI's
   intentions.

---

### User Story 5 - Post-game reveal and replay (Priority: P5)

When a game ends, everything hidden becomes public: every seat's role and remaining
hand are revealed, and the players can step backwards and forwards through the game's
full event history — declarations, claims, and cuts — now annotated with the revealed
truth, to see who lied and when.

**Why this priority**: The post-mortem is half the fun of social deduction and makes
the hosted game better than the physical one, but it needs a finished game first.

**Independent Test**: Play any game to completion, confirm the reveal shows all roles
and hands, then step back through the history and confirm each event is shown with
truth annotations (e.g. a declaration marked as a lie against the actual dealt hand).

**Acceptance Scenarios**:

1. **Given** a game that just ended, **When** the end screen appears, **Then** every
   seat's role and remaining cards are revealed alongside the winner announcement.
2. **Given** the end screen, **When** the user enters replay, **Then** they can step
   through every recorded event of the game in order, in both directions.
3. **Given** a replayed declaration, **When** it is displayed, **Then** it is
   annotated against that seat's actual hand at the time (truthful or a lie, and by
   how much).

---

### User Story 6 - Save and resume (Priority: P6)

A game in progress can be saved and resumed later on the same machine — same seats,
same hands, same round, same pending turn — even after the app has been stopped and
restarted. A game interrupted by a browser refresh simply resumes where it was.

**Why this priority**: Real games get interrupted; but this is pure quality of life on
top of a working game.

**Independent Test**: Play into round 2, save, stop and restart the app, resume the
save, and confirm the game state (round, hands, declarations, whose turn) is identical
and play continues to a normal finish. Separately, refresh the browser mid-game and
confirm the game continues without loss.

**Acceptance Scenarios**:

1. **Given** a game in progress, **When** the user saves it, **Then** a named save
   exists that survives stopping and restarting the app.
2. **Given** a saved game, **When** the user resumes it from the home or game page,
   **Then** play continues from the exact saved position and all hidden information is
   intact and still hidden.
3. **Given** a game in progress, **When** the browser is refreshed or the page is
   reopened, **Then** the current game is re-entered at the current position (with
   privacy screens re-locked).

---

### User Story 7 - LLM opponents (optional mode) (Priority: P7)

A user who has an API key can fill AI seats with LLM-driven agents (the same kind that
play in the simulation arena) instead of the built-in solver bots, for more human-like
table behavior. The game indicates when an LLM seat is thinking. Without a key, solver
bots are always available and the game is fully playable offline.

**Why this priority**: Pluggability is an architectural requirement from day one, but
the LLM mode itself is optional flavor with external dependencies (key, network,
cost) — last in line.

**Independent Test**: With a key configured, set up a game with at least one LLM seat
and one solver-bot seat and play a round; both seat types declare, claim, and cut
through the same interface. Without a key, verify LLM seats cannot be selected and the
rest of setup is unaffected.

**Acceptance Scenarios**:

1. **Given** an API key is configured, **When** the user assigns an LLM agent to a
   seat, **Then** that seat plays all phases through the same agent interface as
   solver bots.
2. **Given** no API key, **When** the user reaches seat setup, **Then** LLM agents are
   unavailable (with a brief explanation) and solver bots remain selectable.
3. **Given** an LLM seat is deciding, **When** the wait exceeds a moment, **Then** the
   table shows a thinking indicator for that seat.
4. **Given** an LLM seat fails to respond, **When** retries are exhausted, **Then**
   the game offers to retry again or hand that seat to a solver bot — it never hangs
   or crashes the game.

---

### Edge Cases

- The very first cut of the game reveals the bomb → immediate end, bad guys win,
  reveal and replay still work over the short history.
- The final round's last cut exhausts the round without finding all safe wires → bad
  guys win; the judgment matches the shared rules engine, not a UI reimplementation.
- Manual role-deal override that is impossible for the seat count (e.g. more bad guys
  than the deal supports) → setup rejects it with a clear message before dealing.
- A human seat is asked to declare a wire count outside what their hand could support →
  input is bounded to declarable values (declarations may be lies, but must be within
  the legal declaration range).
- Refresh or navigation during a privacy reveal → the private view is re-locked behind
  the pass-the-device screen; secrets never persist on a shared surface.
- Resuming a save produced by an older, incompatible version of the app → a clear
  error; the save is not corrupted or silently misread.
- The same seed with a different seat count or role-deal → no promise of similarity;
  reproducibility is defined as identical setup + identical seed ⇒ identical deal.
- An AI seat (solver bot) faces a state where its policy has no strictly-best move →
  it still acts within the latency budget with a legal move.
- The stats panel is opted in but the solver cannot finish the information-value stat
  within its budget → the panel behaves as v1 does (depth-capped, labeled approximate).
- Two browser tabs open on the same game → both show consistent state; actions from a
  stale tab are rejected or reconciled, never double-applied.

## Requirements *(mandatory)*

### Functional Requirements

**Home & access**

- **FR-001**: The app MUST serve a home page at the root address with two entries: the
  v1 assistant and the hosted game.
- **FR-002**: The v1 assistant MUST remain available with behavior identical to v1;
  only its address changes. All existing v1 acceptance behavior (setup, declarations,
  cuts, panel, undo, dark mode) continues to hold.

**Game setup**

- **FR-003**: Users MUST be able to configure a game: number of seats (the player
  counts the official game supports), each seat as human or AI, and a display name per
  seat.
- **FR-004**: Setup MUST offer the official role-deal for the chosen seat count as the
  default (including its inherent uncertainty in the number of bad guys), with a
  manual override; impossible overrides are rejected at setup.
- **FR-005**: Setup MUST accept an optional seed; identical setup plus identical seed
  MUST produce an identical deal.
- **FR-006**: Setup MUST include a per-game opt-in (default off) for the public-info
  stats panel. The design MUST accommodate a second, separate opt-in for a per-player
  private panel in a future version without rework of the setup contract.

**Rules & hidden information**

- **FR-007**: All game rules — dealing, declaration collection, claim recording, cut
  legality and resolution, round advancement, and win judgment — MUST be adjudicated
  by the single shared rules engine promoted from the simulation referee, used by both
  the simulation arena and the web app. The web layer presents state and forwards
  player intents only.
- **FR-008**: Per-seat hidden information (role, uncut hand, bomb possession) MUST be
  compartmentalized server-side: a browser is only ever sent the view it is entitled
  to at that moment. Secrets MUST NOT be delivered to the client and merely hidden by
  the interface.
- **FR-009**: Human seats' private information MUST sit behind a pass-the-device
  privacy screen: the screen names the player, reveals role and hand only on an
  explicit tap, and offers an explicit hide before the device is passed on. Leaving or
  refreshing the page re-locks all private views.

**Play**

- **FR-010**: Each round, every seat MUST declare a wire count (AIs via their policy,
  humans via bounded input within the legal declaration range), and declarations MUST
  be public to all seats once made.
- **FR-011**: Players (human and AI) MUST be able to emit structured claims from a
  fixed menu (directed trust/distrust and accusation-style claims); claims are public,
  attributed, timestamped in the event history, and have no mechanical effect on the
  rules. Free-text chat is excluded.
- **FR-012**: On their turn, the cutter MUST choose among legal target hands only; the
  cut's result (safe wire, dud, or bomb) MUST be revealed to all seats, and the turn
  MUST pass according to the official rules.
- **FR-013**: Rounds MUST advance per the official rules (hands shrink and are
  re-dealt) and the game MUST end with the correct winner announced in exactly the
  cases the rules define (all safe wires found; bomb cut; final round exhausted).
- **FR-014**: The table MUST be presented as a tabletop: seats arranged around a
  table, face-down cards, and reveal transitions for cuts — while remaining usable on
  a phone-sized screen.

**AI opponents**

- **FR-015**: AI seats MUST be pluggable behind one agent interface covering all
  decisions (declare, claim, cut). Solver-driven bots MUST ship in this version and
  work fully offline.
- **FR-016**: Solver bots MUST base decisions only on information their seat is
  entitled to (their private view plus public state), MUST make only legal moves, and
  MUST be capable of deception when dealt a bad-guy role (e.g. lying in declarations).
- **FR-017**: LLM-driven agents MUST be selectable as an optional per-seat mode when
  the user has configured an API key; without a key the option is unavailable and the
  rest of the game is unaffected. A failed LLM decision MUST degrade gracefully
  (retry, then offer to substitute a solver bot).
- **FR-018**: AI decisions MUST respect a latency budget: solver-bot actions feel
  immediate; any AI wait beyond a moment shows a per-seat thinking indicator.

**Stats panel**

- **FR-019**: In a game that opted in, a collapsible side window MUST be toggleable at
  any time during play, showing the public-info four-stat cut panel per seat
  (probability bad, probability holds the bomb, probability a cut reveals a safe wire,
  information value of cutting), computed by the solver from public events only and
  numerically identical to what the v1 assistant would show for the same public
  events.
- **FR-020**: In a game that did not opt in, no panel or panel control appears.
- **FR-021**: The panel MUST never recommend a cut, and MUST never reflect hidden
  information or any AI seat's intentions.

**Persistence & post-game**

- **FR-022**: The full game history MUST be recorded as an ordered event log
  sufficient to reconstruct the game (setup, deal, declarations, claims, cuts,
  round advances, judgment).
- **FR-023**: A game in progress MUST be saveable and resumable on the same machine,
  surviving app restarts, with hidden information intact and still hidden; an
  incompatible save is rejected with a clear error.
- **FR-024**: A browser refresh or page reopen mid-game MUST return to the current
  game position without loss (privacy screens re-locked).
- **FR-025**: When a game ends, all roles and remaining hands MUST be revealed, and a
  replay MUST let users step through the event history in both directions with truth
  annotations (e.g. declarations marked truthful or lies against the actual dealt
  hand).

### Key Entities

- **Hosted Game**: One configured match — seats, role-deal choice, seed, panel
  opt-in(s), current phase, and its event log. Exists server-side; the browser holds
  no authoritative state.
- **Seat**: A position at the table with a display name and an occupant type (human
  or a specific AI agent kind). Owns private state during a game: role, hand, bomb
  possession.
- **Agent**: The decision-maker occupying a non-human seat, behind one interface
  (declare / claim / cut). Kinds: solver bot (built-in, offline) and LLM agent
  (optional, key-required).
- **Player View**: What one seat is entitled to see at a moment: the public state for
  everyone, plus that seat's own private state. The only shape in which game state
  reaches a browser.
- **Declaration**: A seat's public wire-count statement for a round; may be a lie;
  bounded by the legal declaration range.
- **Claim**: A structured, attributed public statement from the fixed menu (directed
  trust/distrust/accusation); recorded in the event log; no mechanical effect.
- **Cut**: One cutter, one target seat, one revealed result (safe wire, dud, bomb);
  advances the turn and possibly ends the game.
- **Event Log**: The ordered, replayable record of everything public plus the sealed
  deal, sufficient for resume and post-game replay.
- **Saved Game**: A persisted Hosted Game (including its sealed hidden state) that can
  be listed, resumed, or rejected as incompatible.
- **Panel Snapshot**: The four-stat public-info cut panel for the current public
  state, produced by the solver; shown only in opted-in games.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A first-time user can go from the home page to a started solo game
  (1 human + AI seats) in under 90 seconds.
- **SC-002**: A full solo game at the default seat count completes in under 15 minutes
  of wall-clock time, with every solver-bot decision resolving in at most 2 seconds.
- **SC-003**: Across an entire game, no server response delivered to the browser ever
  contains another seat's role, uncut hand contents, or bomb possession — verifiable
  by inspecting all traffic in a played game (100% of responses clean).
- **SC-004**: The complete v1 assistant test suite passes unchanged (URL move aside);
  a v1 user following the v1 quickstart notices no behavioral difference.
- **SC-005**: 100 seeded self-play games (solver bots in every seat) all run to
  completion with zero illegal moves and a winner matching the shared rules engine's
  judgment; re-running any seed reproduces the identical deal and, for deterministic
  bot policies, the identical game.
- **SC-006**: Solver-bot play is better than chance: over the self-play corpus, the
  good-guys' cut choices find safe wires at a rate measurably above uniform-random
  cutting (using the repo's established beats-random methodology).
- **SC-007**: Save → restart → resume reproduces the exact game position in 100% of
  tested cases, including hidden state, across all phases (declaration, mid-cuts,
  between rounds).
- **SC-008**: In an opted-in game, the side panel's four stats are numerically
  identical to the v1 assistant's output for the same public event sequence, at every
  point of the game.
- **SC-009**: Post-game replay can step through 100% of a finished game's events with
  correct truth annotations (every lie flagged, every honest declaration unflagged).

## Assumptions

- **Official seat counts**: the game supports the official player counts (4–8, as v1
  does); the shared engine's existing defaults define hand sizes and round structure.
- **Claim menu (initial)**: directed claims "I trust [seat]", "I don't trust [seat]",
  "[seat] is lying", plus the self-claim "my declaration is honest". The exact menu is
  expected to be refined during clarification/planning; the requirement is a fixed,
  finite menu with attribution, not this exact list.
- **Claim timing**: claims are emitted from the shared table surface between actions,
  attributed to an explicitly chosen seat (hotseat has no per-seat sessions); AIs emit
  claims at their decision points.
- **One active game at a time**: the app hosts a single in-progress game per running
  instance; starting a new game requires finishing, saving, or abandoning the current
  one. (The LAN-shaped state model may later host several, but v2.0 does not.)
- **Saves live on the hosting machine** (no accounts, no sync), consistent with v1's
  local-machine deployment model; "same machine" means the machine running the app.
- **LLM agents reuse the simulation arena's agents and prompts** — per the arena's own
  rules (emergent play, no strategy steering); the web app configures and hosts them,
  it does not redesign them.
- **Bot quality bar**: solver bots must beat random (SC-006) but are not required to
  match arena-LLM or human strength in v2.0; bluffing policy specifics are a design
  decision for planning, guided by the solver's posteriors.
- **The engine promotion is in scope**: extracting the referee from the simulation
  into a shared package (with the simulation adapted to import it, its tests still
  passing) is part of this feature, per constitution v2.0.0's carve-out.
- **English-only UI**, matching v1.
- **No spectator mode**: every human present holds a seat; watching without a seat is
  out of scope.

## Out of Scope (this version)

- Online/remote/LAN play (the state model is built LAN-shaped, but no network
  multiplayer ships).
- Accounts, profiles, or cross-machine persistence.
- Free-text chat or voice.
- Per-player private stats panels (requires new solver interfaces; will be its own
  opt-in when it arrives).
- Any modification to the v1 assistant's behavior, or to the backend solver and its
  test suite.
