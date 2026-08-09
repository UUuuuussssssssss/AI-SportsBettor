# Entity Bank: Step-by-Step Outcome Flow

This document explains how the entity bank is created and how names from X and
Polymarket are connected to it. It focuses on decisions and outcomes rather
than implementation details.

## Part 1: Create the canonical NFL entity bank

### Step 1: Collect the source lists

The system gets three nflverse datasets:

- Current NFL teams
- Current-season rosters
- Complete player history
- Current coaching-staff directories from all 32 official NFL club sites

The complete history is retained for identity bookkeeping. The active entity
bank contains current teams, people on the current-season roster, and people
listed on the current official coaching staffs.

**Outcome:** We have a current NFL source of truth without losing historical
identity information.

### Step 2: Create one identity per real team or person

Each team, player, and coach receives one stable internal identity. Provider IDs
such as GSIS, ESPN, and PFR IDs and official club directory evidence are attached
as supporting identifiers rather than used as the main identity.

**Outcome:** A person can change teams or gain a new provider ID without
becoming a new person in the bank.

### Step 3: Add known names and aliases

The bank stores the canonical name and supported alternatives, such as:

- Full names
- Common source-name variations
- Team city, nickname, and abbreviation forms

Aliases help source wording such as `49ers`, `Patriots`, or `A.J. Brown` find
the correct canonical record.

**Outcome:** Different valid spellings can point to the same identity.

### Step 4: Add current roles and team relationships

Current roster data supplies factual roles and team membership. Tweets and
prediction markets cannot change these relationships because they may contain
rumors, questions, or hypothetical outcomes.

**Outcome:** The bank records what the roster source says is true, not what a
tweet or market predicts might happen.

### Step 5: Handle duplicate and conflicting source rows

Compatible rows are merged when their identifiers and identity details agree.
Conflicting rows are quarantined instead of guessed.

**Outcome:** Supported duplicates become one identity; contradictions do not
silently corrupt the bank.

## Part 2: Identify mentions in incoming content

### Step 6: Analyze an X news record

The LLM reviews the available tweet, linked article text, image evidence, and
video-frame evidence together. It identifies explicitly supported:

- People and teams
- Their role in the story
- The evidence passage containing each name
- Claims, while preserving whether they are facts, rumors, questions, jokes,
  or sarcasm

This extraction happens once. Later resolution uses the stored mentions rather
than asking another LLM to re-extract the same names.

**Outcome:** The news record produces a clean list of source-backed mentions.

### Step 7: Analyze a Polymarket event

The LLM reviews an event and its grouped markets together. For every market it
identifies:

- Market topic
- Contract type
- Named players, coaches, or teams
- The role each name has in that market

The event context helps interpret related markets consistently, while each
market keeps its own classification and entity links.

**Outcome:** Every relevant market gets market-level results with shared
event-level context.

### Step 8: Remove placeholders and generic outcomes

Values such as these are not treated as entities:

- `Player A`, `Person B`, `Coach C`, or `Team D`
- `Other` or `another player`
- Binary `Yes` and `No` outcomes

**Outcome:** Placeholder labels and contract mechanics do not pollute the
entity bank or accidentally match real teams.

## Part 3: Match each mention to the bank

### Step 9: Try a safe exact alias match

The system first checks whether the normalized source name has one unique,
safe alias match.

Example:

- `Patrick Mahomes` has one supported canonical match.
- The result is immediately marked **resolved**.

Short abbreviations receive additional safeguards so ordinary words do not
accidentally become team matches.

**Outcome:** Clear names resolve quickly without another LLM call.

### Step 10: Build a shortlist when the match is not clear

If there is no safe unique match, name similarity is used only to retrieve up
to eight plausible candidates. Similarity does not make the final decision.

The shortlist can use:

- Similar names and aliases
- Expected entity type
- Person-role hints
- Current or historical team context

**Outcome:** The next AI check receives a small, relevant option set instead of
the entire entity bank.

### Step 11: Ask the LLM to check only those candidates

The LLM receives the source context and the shortlist of up to eight allowed
entities. It may:

- Select one allowed identity
- Mark several candidates as **ambiguous**
- Mark the mention **unresolved**

It cannot choose an entity outside the shortlist or invent a canonical ID.

Example:

- `Sanders` may match several players.
- If the source does not distinguish them, the correct outcome is
  **ambiguous**, not a guess.

**Outcome:** Context resolves difficult names while the allowlist prevents
model-memory guesses.

## Part 4: Store the final outcome

### Step 12: Assign one resolution status

Each mention ends in one of these states:

- **Resolved:** Connected to one canonical entity.
- **Ambiguous:** Multiple candidates remain plausible.
- **Unresolved:** No candidate is adequately supported.
- **Provisional:** A structured Polymarket person is absent from the canonical
  bank but is suitable for temporary tracking.
- **Ignored:** The value is a placeholder or generic outcome.

X content cannot create provisional identities. An unknown name from X remains
unresolved until a trusted source adds that identity.

**Outcome:** Every mention has an explicit, reviewable state; nothing needs to
be silently guessed or discarded.

### Step 13: Preserve why the decision happened

The system retains the source mention, evidence, candidate shortlist, matching
method, confidence, model decision, and final status.

**Outcome:** A resolved or unresolved result can be explained and audited
later.

### Step 14: Revisit unresolved results when the bank changes

When nflverse changes, the canonical bank is updated. Previously unresolved,
ambiguous, or provisional mentions can be reconsidered when the available
candidates have changed.

Example:

- A draft prospect initially absent from the roster may be provisional in a
  Polymarket record.
- After nflverse adds that player, the provisional identity can merge into the
  new canonical record when the identity match is unique.

**Outcome:** Old mentions improve as the source of truth improves, without
rerunning every unchanged record.

## Final result

The completed process produces:

- A canonical bank of current NFL teams, rostered players, and coaching staffs
- Supported aliases, roles, roster relationships, and coach/team relationships
- Market-level and news-level entity mentions
- Canonical links where identity is supported
- Explicit ambiguous, unresolved, provisional, and ignored outcomes where it
  is not
- A decision history that explains how every outcome was reached
