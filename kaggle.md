# Kaggle Training Guide for Fair Square

This guide explains how to run the full large-scale Stockfish 16 MultiPV 3 feature extraction and model training on **Kaggle** using the move-level clean dataset.

---

## 1. Clean Move-Level Dataset (`dataset.csv`)

The training pipeline uses a clean, leak-free move-level dataset with the following characteristics:
- **Move-level ground truth**: "Liste cheat" binary strings indicate exactly which moves (from move 11 onwards) were assisted by Stockfish (`1`) versus played by Maia (`0`).
- **No synthetic vs human shortcut**: Both clean and cheating sides originate from the same Maia generation baseline, preventing the classifier from learning superficial distribution artifacts.
- **Dataset volume**: ~51,000 clean sides and ~46,600 assisted sides across ~49,000 games.
- **Elo rating scope**: 90% of Elo ratings sit inside 1000-2000, aligning with Fair Square's targeted scope.
- **Assist rate spectrum**: Varied assist rates between 10% and 100%, allowing robust evaluation of partial cheater detection.

---

## 2. Setup on Kaggle

1. Go to [Kaggle](https://www.kaggle.com/) and create a **New Notebook**.
2. In the right sidebar under **Notebook settings**:
   - **Accelerator**: `CPU` (4 vCPUs)
   - **Internet**: `ON` *(Required for downloading Stockfish and packages)*
   - **Persistence**: `Variables and Files` (recommended)

---

## 3. Notebook Execution Cells

### **Cell 1: Clone Repository & Install Dependencies**
```bash
!git clone https://github.com/0xGhost63/Fair-Square-Prototype.git Fair-Square
%cd Fair-Square
!pip install -r requirements.txt
```

### **Cell 2: Link Dataset**
Ensure `dataset.csv` is present in the `Fair-Square/` root directory:
```bash
!cp /kaggle/input/your-dataset-name/dataset.csv .
!ls -lh dataset.csv
```

### **Cell 3: Run End-to-End Extraction & Training**
Run the self-contained pipeline on 5,000 games (~2 hours across 4 cores):
```bash
!python kaggle_pipeline.py --max-games 5000 --workers 4
```

*Key Pipeline Features*:
- **Engine Consistency**: Fixed Stockfish 16 analysis at depth 12, Threads=1, Hash=32MB matching production inference.
- **Automatic Checkpointing**: Saves progress every 250 games to `data/features_checkpoint.parquet`. If your session disconnects, re-running the cell seamlessly resumes from the latest checkpoint.
- **Leakage-Free Splits**: Strict game-level grouped split (70% Train, 15% Val, 15% Test) ensuring no game appears in multiple splits.
- **Simulated Player Evaluation**: Simulates clean players vs partial cheaters with bootstrap confidence intervals to calibrate the 1% False Positive Rate threshold.

---

## 4. Retrieve Output Artifacts

After completion, download the following generated files from Kaggle's Output tab (`/kaggle/working/Fair-Square/`):

1. **`models/cheat_model.joblib`**: Place directly into your local `models/` directory.
2. **`tests/reports/recall_by_assist_level.png`**: Recall at 1% FPR bar chart.
3. **`tests/reports/player_level_roc_pr.png`**: ROC and Precision-Recall curves.
4. **`tests/reports/evaluation_metrics.json`**: Complete metrics summary with 95% bootstrap confidence intervals.

---

## 5. Run Live Local Detection

Once you place the new `cheat_model.joblib` in `models/`:
```bash
python app.py
```
Open `http://127.0.0.1:8000` to analyze any Chess.com account with the updated model.
