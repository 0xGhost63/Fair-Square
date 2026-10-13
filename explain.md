# Fair Square — How It All Works (Plain English Edition)

> A complete beginner-friendly walkthrough of the codebase, from dataset to suspicion score.

---

## The Big Picture

Fair Square is a web app that looks at a Chess.com player's recent games and asks:
**"Do these games look like a human played them, or does something feel off?"**

It does not look at the player's account, their history, or any ban records.
It **only** looks at the moves themselves — and compares them to what a chess engine (Stockfish) would have played.

The full journey looks like this:

```
User types username
       ↓
App fetches their recent games from Chess.com
       ↓
Stockfish analyses every move in every game
       ↓
Features (numbers) are extracted from that analysis
       ↓
A trained ML model scores each game (0–100% suspicious)
       ↓
Scores are aggregated into one final Suspicion %
       ↓
Result is shown on screen
```

---

## Part 1 — The Dataset (Training Data)

Before the app can detect anything, a **model** needs to be trained.
To train it, we need examples of **clean games** and **cheated games**.

### Where does the data come from?

The dataset (`dataset.csv`) contains ~48,000 chess games from **Lichess** (another chess platform).
Lichess provides a cheat label for each game — specifically a string like `"01001101"` where:
- `0` = that move was played clean
- `1` = engine assistance was detected on that move

### How is "cheating" defined for training?

We calculate an **assist rate** per player per game:

```
assist_rate = (number of 1s in the cheat string) / (total moves)
```

- `assist_rate = 0.0`  → **Clean** (labeled 0)
- `assist_rate >= 0.30` → **Cheater** (labeled 1, meaning 30%+ of moves had engine help)
- `0 < assist_rate < 0.30` → Excluded from training (ambiguous grey zone)

### What games are used?

To avoid bias, games are **stratified** — meaning we carefully balance:

| Filter | Value |
|---|---|
| Rating range | 1000 – 2000 Elo |
| Clean vs clean games | ~50% of sample |
| Cheater games | ~50% of sample (split across cheat intensity tiers) |
| Total games sampled | Up to 5,000 |

The four cheat intensity tiers are: 10–30%, 30–50%, 50–70%, 70–100% assist rate.
This ensures the model sees *all kinds* of cheaters, not just blatant ones.

---

## Part 2 — Stockfish: The Chess Engine

Stockfish is an open-source chess engine — essentially the world's strongest chess "calculator."
It can look at any position and tell you: **"What are the best moves here, and how good are they?"**

### How is Stockfish used?

For every move played in a game, Stockfish is asked:
> "Before this move was played, what are the **top 3 best moves** in this position, and how good are they in centipawns?"

**Centipawn (cp):** A unit of chess advantage. 100 cp = roughly the value of one pawn.
The engine returns scores like `+45 cp` (slight edge) or `+350 cp` (winning position).

### The key settings used:

| Setting | Value | Why |
|---|---|---|
| `ENGINE_DEPTH` | 12 | How many moves ahead Stockfish thinks. Fixed at 12 for consistency |
| `MultiPV` | 3 | Stockfish returns its top 3 best moves per position |
| `Threads` | 1 | Single thread for 100% reproducible results |
| `Hash` | 32 MB | Memory used for caching positions |
| `MAX_EVAL_CP` | 600 cp | If the position is already ±600 cp (basically won/lost), skip it |
| `ONLY_MOVE_GAP_CP` | 75 cp | If top-1 move is 75+ cp better than top-2, it's the "only move" |
| `MAX_MOVE_LOSS_CP` | 300 cp | Clip centipawn loss at 300 (a blunder ceiling) |
| `SKIP_OPENING_MOVES` | 10 | Skip the first 10 moves (opening book, often memorised) |

### What does Stockfish tell us per move?

For every move (starting from move 11), Stockfish answers:

| Question | Answer stored as |
|---|---|
| What's the best score in this position? | `score_cp` |
| Is the game already decided (±600 cp)? | `is_decided` (True/False) |
| Is there only one good move here? | `is_only_move` (True/False) |
| Was the played move a simple recapture? | `is_recapture` (True/False) |
| Is this a genuinely tricky position? | `is_non_trivial` (True/False) |
| Did the player play Stockfish's #1 move? | `matches_top1` (True/False) |
| Did the player play one of Stockfish's top 3? | `matches_top3` (True/False) |
| How many centipawns did the player lose? | `loss_cp` (0–300) |

> **Key insight:** Matching the #1 engine move in a "non-trivial" position (where there are many options
> and no obvious right answer) is the most suspicious signal. Anyone can find a forced recapture —
> but consistently finding the exact best move in complex positions is very unlikely without engine help.

---

## Part 3 — Feature Extraction

After running Stockfish on all the moves, we **summarise** the move-by-move data into **7 numbers**
(features) per player per game:

| Feature | What it measures | Why it matters |
|---|---|---|
| `non_trivial_top3_match` | % of tricky moves where player chose a top-3 engine move | Cheaters pick engine moves even when the position is complex |
| `longest_top3_streak` | Longest consecutive run of top-3 engine moves | Cheaters often go on long "perfect" streaks |
| `only_move_match_rate` | % of "forced" positions where player found the only good move | Even when there's only one saving move, cheaters always find it |
| `top1_match` | % of all moves that exactly matched Stockfish's #1 choice | Overall engine agreement rate |
| `acpl` | Average Centipawn Loss — average quality drop per move | Lower = better. Cheaters have unnaturally low ACPL |
| `blunder_rate` | % of moves that lost 200+ cp | Cheaters rarely blunder |
| `elo` | The player's Elo rating | Keeps the model honest — a 2000-rated player is *expected* to play better |

### What gets filtered out?

- **Decided positions** (`|score| > 600 cp`): Ignored — in a winning position anyone plays well
- **Only-moves** (one forced reply): Ignored for `non_trivial_top3_match` — not impressive to find a forced move
- **Recaptures**: Also excluded from non-trivial stats — recapturing is obvious
- **Opening moves** (first 10): Skipped entirely — opening theory is memorised by everyone

This filtering is crucial. Without it, a clean grandmaster playing in a won position would look
suspicious. By only measuring **genuinely hard decisions**, the signal becomes much sharper.

---

## Part 4 — Training the Model

With the 7 features computed for thousands of games, we train a **machine learning classifier**.

### How it works (very simply):

Imagine plotting each game as a dot on a chart.
- Clean games cluster together (low top1_match, higher ACPL)
- Cheater games cluster together (high top1_match, very low ACPL)

The model learns where to draw the line between those clusters.

### Two models are trained and compared:

| Model | What it is |
|---|---|
| Logistic Regression | A simple, fast, linear classifier |
| HistGradientBoosting | A more powerful tree-based model |

Whichever has the **better AUC score** on the validation set wins and gets saved.

### How is the split done?

The data is split **by game**, not by row — so the same game can never appear in both training
and test sets (which would inflate the score artificially):

| Split | Size | Purpose |
|---|---|---|
| Train | 70% | The model learns from this |
| Validation | 15% | Used to pick the best model |
| Test | 15% | Final honest evaluation, never touched during training |

### The sanity check:

Before trusting any result, we run a **shuffled label test**: we randomly scramble all the labels
and train again. If the model scores ~0.50 AUC on shuffled labels, our features are real signals
(not data leakage). If it's still high, something is wrong.

The saved model file is: `models/cheat_model.joblib`

---

## Part 5 — Live Analysis (What Happens When You Type a Username)

When you type a username and click Analyse, here's what happens step by step:

### Step 1: Fetch games from Chess.com
The app calls the **public Chess.com API** (no account needed, no API key).
It fetches the most recent **rated, standard time-control** games that include clock data.
Daily games are excluded — clock patterns are not useful there.

### Step 2: Rating scaling
Chess.com ratings are slightly different from Lichess (where the model was trained).
So the app converts: `train_elo = 1.05 × chess.com_rating + 100`

This keeps the `elo` feature consistent with what the model was trained on.

### Step 3: Stockfish analyses every game
For each game, Stockfish evaluates every move from move 11 onwards (same settings as training).
The 7 features are extracted per game.

### Step 4: Model scores each game
The trained model receives those 7 numbers and outputs a probability:
> "This game looks X% like a cheated game"

This becomes the **per-game probability** (0–100%).

### Step 5: Final suspicion score
All per-game probabilities are **aggregated** into one number using a hybrid formula:

```
suspicion_score = 0.5 × mean(all game probabilities)
                + 0.5 × 85th percentile(all game probabilities)
```

**Why not just the average?**
A partial cheater (who only cheats in some games) would look fine on average — the clean games
drag the score down. The **85th percentile** catches those suspicious outlier games and makes
sure they count.

A **confidence interval** is also calculated:
```
margin = 1.96 × std_dev / sqrt(number_of_games)
```
This gives you a range like `42.3% ± 8.1%` so you know how reliable the score is.

---

## Part 6 — Verdicts

The final suspicion score maps to one of three verdicts:

| Score | Verdict |
|---|---|
| Below 35% | Low suspicion (consistent with clean play) |
| 35% – 50% | Moderate variation (inconclusive indicator) |
| Above 50% | Elevated anomaly (statistical suspicion indicator, not proof) |

> **Important:** This is a **statistical anomaly detector**, not a judge.
> A high score means the play *looks unusual compared to clean players at the same rating*.
> It does not prove cheating — gifted players, people using openings, or just people having a
> great day can score high.

---

## Part 7 — Key Config Values at a Glance

All tunable values live in `config.py`:

| Constant | Value | What it controls |
|---|---|---|
| `ENGINE_DEPTH` | 12 | How deep Stockfish thinks per move |
| `SKIP_OPENING_MOVES` | 10 | Moves ignored at the start of each game |
| `MAX_EVAL_CP` | 600 | Positions more decisive than this are skipped |
| `ONLY_MOVE_GAP_CP` | 75 | Gap between top-1 and top-2 to call a move "forced" |
| `MAX_MOVE_LOSS_CP` | 300 | Max centipawn loss recorded per move (blunder cap) |
| `MIN_MOVES` | 15 | Minimum moves needed after the opening for a game to count |
| `MIN_GAMES` | 10 | Minimum games the user can request |
| `MAX_GAMES` | 100 | Maximum games the user can request |
| `MAX_ELO` | 2000 | Players above this Elo are outside the model's training range |
| `CACHE_SECONDS` | 600 | Results are cached for 10 min to avoid re-analysing the same player |

---

## Summary Flowchart

```
                    TRAINING (done once)
                    ─────────────────────────────────────
dataset.csv (48k Lichess games with cheat labels)
        |
        v
  Filter Elo 1000–2000, balance clean/cheat tiers
        |
        v
  Stockfish Depth 12, MultiPV 3  -->  7 features per player per game
        |
        v
  70/15/15 game-level split (no leakage)
        |
        v
  Train Logistic Regression + HistGradientBoosting
        |
        v
  Best model saved --> models/cheat_model.joblib


                    LIVE (every analysis)
                    ─────────────────────────────────────
User enters username + game count
        |
        v
Chess.com API --> recent rated games (with clock data)
        |
        v
Stockfish Depth 12 --> 7 features per game
        |
        v
cheat_model.joblib --> probability per game (0–100%)
        |
        v
suspicion = 0.5 x mean + 0.5 x 85th_percentile
        |
        v
Verdict displayed on screen
```
