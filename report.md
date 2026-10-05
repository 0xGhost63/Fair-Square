# Fair Square: Semester AI Project Report
## Explainable Chess Engine Assistance Detector

**Author:** Semester AI Student  
**Version:** 2.0 (Final Deliverable)  
**Interface:** Command Line Interface (CLI)  

---

## 1. Executive Summary

This report documents the design, implementation, and evaluation of **Fair Square**, a moves-only chess engine assistance detection system built as a semester AI project.

Engine assistance detection on online chess platforms is typically controlled by private, proprietary algorithms. Fair Square creates an open, explainable, reproducible detection pipeline that processes player moves from standard PGN files, evaluates position dynamics using Stockfish, computes move-level and game-level features, and outputs a statistical suspicion score with actionable behavioral explanations.

---

## 2. Dataset Overview & Data Loading (Phase 1)

### 2.1 Datasets
1. **Real Tournament Dataset (`chess_fraud.parquet`):**
   - 38,510 total move rows from real online tournament games where players had access to a Stockfish helper.
   - Filtered for focal player moves (`is_used == True`, move 21 onward): **28,410 moves** across **1,010 games** and **49 unique players**.
   - Move-level label (`is_cheating_move`): 28.78% Cheating / 71.22% Fair.
   - Game-level label (`is_cheating_player_game`): 40.30% Cheating / 59.70% Fair.

2. **Synthetic Lichess Datasets (`synth_train.parquet` & `synth_test.parquet`):**
   - Extracted from public Lichess games with player-disjoint splits.
   - Filtered (`is_used == True`): **334,140 train moves** and **83,067 test moves**.
   - Labeled examples built by pairing player move (`move_player`, label 0) with Stockfish assisted move (`move_stockfish_15`, label 1).

---

## 3. Move-Level Feature Engineering (Phase 2)

For each move from move 21 onward, the feature extractor constructs a 10-dimensional feature vector:
1. `sf1_match`: Binary indicator whether played move equals Stockfish top choice at depth 1.
2. `sf9_match`: Binary indicator whether played move equals Stockfish choice at depth 9.
3. `sf15_match`: Binary indicator whether played move equals Stockfish choice at depth 15.
4. `eval_before`: Position evaluation before move (centipawns clipped to `[-1000, 1000]`).
5. `centipawn_loss`: $\max(0, eval_{before} - eval_{after})$.
6. `normalized_centipawn_loss`: $\max(0, p(eval_{before}) - p(eval_{after}))$ using win probability transform $p(e) = \frac{1}{1 + e^{-0.00368208 \cdot e}}$.
7. `position_difficulty`: Magnitude $|eval_{before}|$ measuring positional sharpness.
8. `player_elo`: Elo rating of the focal player from PGN header.
9. `elo_diff`: Rating difference (`player_elo - opponent_elo`).
10. `half_move`: Current ply number in the game.

---

## 4. Move-Level Classifier Evaluation (Phase 3)

### 4.1 Evaluation Methodology
To prevent data leakage across games played by the same individual, all evaluations on real data use **5-Fold GroupKFold grouped by `player_id`**.

### 4.2 Move-Level Results

| Model / Strategy | Precision | Recall | F1 Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression Baseline** (GroupKFold) | 0.4922 | 0.1534 | 0.2339 | 0.6852 |
| **HistGradientBoosting** (GroupKFold) | 0.4451 | 0.2714 | 0.3372 | 0.6394 |
| **Zero-Shot Synthetic Model** (Tested on Real) | 0.3826 | 0.3496 | 0.3654 | 0.6084 |

*Findings:* Synthetic pretraining achieves higher move-level recall because full synthetic engine moves represent 100% engine play, whereas real human cheaters cheat selectively on specific turns.

---

## 5. Game-Level Suspicion Score Evaluation (Phase 4)

### 5.1 Why Game-Average Features Fail
Averaging move match rates over an entire game dilutes the signal because real cheaters only toggle engine assistance on key moves. In baseline tests, game-average Stockfish match rates alone overlapped heavily between fair games (~43%) and cheating games (~55%).

### 5.2 Advanced Game Feature Aggregation
To capture selective cheating, move-level probabilities $p_m$ are aggregated into game-level features:
- `mean_move_score` & `max_move_score`
- `top10_mean_score`: Mean probability of the top 10 most suspicious moves.
- `max_sf_streak`: Longest consecutive streak of engine-matching moves.
- `streaks_len3_plus_count`: Count of engine-matching move streaks of length $\ge 3$.
- `critical_match_rate`: Engine match rate in positions with large eval swings ($|\text{eval}| \ge 200$ or normalized loss $\ge 0.05$).
- `game_avg_cpl` & `game_avg_norm_cpl`

### 5.3 Game-Level Model Comparison (GroupKFold by `player_id`)

| Model Variant | Precision | Recall | F1 Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline Game Model** (Game-Average Match & CPL Only) | 0.5159 | 0.4791 | 0.4968 | 0.6584 |
| **Advanced Game Model** (Top-10 Suspicion + Streaks + Critical Match) | **0.6102** | **0.5307** | **0.5677** | **0.6973** |

*Conclusion:* Aggregating move suspicion tails (Top 10 moves), consecutive Stockfish streaks, and critical position accuracy improves ROC-AUC from **0.6584 to 0.6973** (+3.89% boost).

---

## 6. CLI PGN Inference Pipeline (Phase 5)

The Command Line Interface (`cli.py`) reads any standard PGN file, invokes Stockfish 17 at depth 15, and outputs game statistics:

```bash
python cli.py --file sample_game.pgn --depth 15
```

### Example CLI Output:
```
==================================================
           FAIR SQUARE CLI DETECTION REPORT       
==================================================
Players:             PlayerA vs PlayerB
Event:               Semester AI Project Match
Moves Analyzed:      65 moves (move 21+)
Suspicion Score:     99.9%
Stockfish Match Rate:66.2%
Avg Centipawn Loss:  8.0 cp
Max Engine Streak:   11 consecutive moves
Streaks (>= 3 moves):5 streak(s)
Critical Position Match: 0.0%
==================================================
```

---

## 7. Project Limitations & Ethical Notice (Phase 6)

### 7.1 Key Limitations
1. **Synthetic vs Real Cheating Gap:** Synthetic cheaters use fixed engine recommendations, whereas real human cheaters exhibit complex, non-stationary toggle patterns and humanized move selection.
2. **Cohort Size:** The real tournament dataset contains 49 unique players and 1,010 games, which limits deep cross-player generalization.
3. **Time Control Specificity:** Tournament games in the real cohort are 5-minute blitz games; time dynamics vary across bullet, rapid, and classical time controls.
4. **Assistance Type:** Data reflects one primary form of screen/move recommendation assistance.

### 7.2 Statistical Disclaimer
**Notice:** All outputs generated by Fair Square represent statistical likelihood estimates derived from move accuracy, engine matching streaks, and position evaluation swings. They are intended strictly for educational, statistical, and academic research purposes and **do not constitute proof or a definitive verdict of cheating against any player**.
