# Testing folder

Two kinds of tests live here.

Code tests check that the program works. Run them with `pytest tests -q`. They need no model and no internet.

Model tests check how good the model is. First train with `python -m src.train`, then run `python tests/evaluate_model.py`. The script reads `tests/artifacts/test_set.csv` (players the model never saw) and writes charts plus `report.md` and `metrics.json` into `tests/reports/`.

How to read the report: accuracy alone is not enough. Look at the false positive rate (clean players wrongly flagged), the ROC curve and the player level AUC, which imitates what the app really does by averaging several games per player.
