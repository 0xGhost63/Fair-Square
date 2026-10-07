# Future ideas and features

These are ideas for growing the project after the semester submission, grouped by effort.

## Quick wins

Add a compare mode where two usernames are analysed side by side. Add a downloadable PDF or JSON report for a result. Show a small chart of suspicion over time so a sudden jump in recent games is visible. Add a rating filter so the user can analyse only blitz, only rapid or only games against higher rated opponents.

## Better detection

Add a position complexity feature using the engine evaluation spread between the best and the fifth best move, so the model can check whether the player thinks longer in genuinely hard positions. Add a feature for moves played instantly right after the opponent moves, which can reveal premove abuse or relayed moves. Detect streaks, for example many consecutive moves with almost identical thinking time. Look at performance against rating, such as a player whose accuracy is far above what their rating suggests. Track suspicious switching, such as strong play in some games and weak play in others, which is a known cheating pattern.

## Better modelling

Try gradient boosting and compare it with the random forest using the same grouped cross-validation. Calibrate the probabilities so that a score of 70 percent really means about 70 percent. Train a separate model per time control (bullet, blitz, rapid). Add a small chess.com validation set and fine tune the decision threshold on it to reduce the gap between the two sites. Experiment with a sequence model over the list of move times for each game, once the simple model is well understood.

## Product features

Add a leaderboard of most and least suspicious recent searches, with strong wording that the score is not proof. Add a browser extension that shows the score on a chess.com profile page. Add a background job that rechecks a watch list of players weekly. Add user accounts so analysis history is saved. Package the app with Docker so it can be deployed on a free hosting service.

## Responsible use

Add a clear explanation panel that shows which features pushed a game score up or down, using feature contributions, so users can see why the model reacted. Add a minimum game count warning when the interval is very wide. Keep the disclaimer visible in every result and never present the score as a verdict.
