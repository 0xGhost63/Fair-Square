# Training & Kaggle Setup Guide

This document provides complete instructions for downloading dataset shards, extracting features, training the Machine Learning cheat detection model, and evaluating model accuracy both **locally** and on **Kaggle** (for large-scale training).

---

## 1. The Lichess Dataset

We use the public Lichess standard rated games dataset hosted on Hugging Face:
- **Dataset URL**: [https://huggingface.co/datasets/Lichess/standard-chess-games](https://huggingface.co/datasets/Lichess/standard-chess-games)
- **Format**: Parquet files organized as `data/year=YYYY/month=MM/train-XXXXX-of-XXXXX.parquet`.
- **Clock Features**: Each game contains move text with `[%clk]` tags, giving precise move time usage.

### Cheater Labels
The dataset does not contain an explicit cheater column. `src/lichess_labels.py` queries the Lichess User API to detect accounts marked with the `tosViolation` flag (closed for fair play violations). Flagged accounts form the positive (cheater) class and unflagged accounts form the clean class. Results are automatically cached in `data/flagged_cache.json`.

---

## 2. Training on Kaggle (Recommended for Large Datasets)

Kaggle provides **100% free** cloud compute (30 hours/week, 30 GB RAM, fast ~200+ MB/s internet speeds). It is the best environment for downloading massive dataset shards and running heavy training or Stockfish engine evaluations.

### Step 1: Create a Kaggle Notebook
1. Sign in to [Kaggle](https://www.kaggle.com/).
2. Click **Create** $\rightarrow$ **New Notebook**.
3. In the right sidebar under **Notebook settings**:
   - Set **Accelerator** to **CPU** (GPU is not needed).
   - Turn **Internet** $\rightarrow$ **ON** *(Essential for downloading datasets & Lichess API lookups)*.

### Step 2: Notebook Cell Setup

#### **Cell 1: Clone Repository & Install Dependencies**
```bash
!git clone https://github.com/your-username/Fair-Square.git
%cd Fair-Square
!pip install -r requirements.txt
```

#### **Cell 2 (Optional): Install Stockfish Engine**
*(Only needed if training with engine features `--engine`)*
```bash
!apt-get update && !apt-get install -y stockfish
!export STOCKFISH_PATH=/usr/games/stockfish
```

#### **Cell 3: Download Dataset Shards**
Download 10 to 50+ shards from Hugging Face:
```bash
!python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Lichess/standard-chess-games', repo_type='dataset', allow_patterns='data/year=2024/month=07/train-000*', local_dir='data/raw')"
```

#### **Cell 4: Build Feature Table**
For clock features only (fast, recommended):
```bash
!python -m src.build_dataset --parquet "data/raw/data/year=2024/month=07/*.parquet" --max-players 50000
```
*(With Stockfish engine features, add `--engine --max-rows 5000`)*.

#### **Cell 5: Train Model**
```bash
!python -m src.train
```

#### **Cell 6: Evaluate Model & Generate Reports**
```bash
!python tests/evaluate_model.py
```

### Step 3: Retrieve Output Artifacts
1. In the right sidebar under **Output** (`/kaggle/working/Fair-Square/`), expand `models/` and `tests/reports/`.
2. Download `cheat_model.joblib` and place it into your local project's `models/` folder.
3. Download the report charts (`roc_curve.png`, `confusion_matrix.png`, `report.md`).

---

## 3. Local Training Instructions

If you prefer to train locally on your machine:

### 1. Activate Environment & Install Dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Download Sample Dataset Shards
```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Lichess/standard-chess-games', repo_type='dataset', allow_patterns='data/year=2024/month=07/train-0000*', local_dir='data/raw')"
```

### 3. Extract Features & Build Dataset
```bash
python -m src.build_dataset --parquet "data/raw/data/year=2024/month=07/*.parquet" --max-players 10000
```

### 4. Train Model
```bash
python -m src.train
```

### 5. Evaluate Performance
```bash
python tests/evaluate_model.py
```

---

## 4. Running the Web Application

Once `models/cheat_model.joblib` exists, start the Flask server:

```bash
python app.py
```

Open your browser at **http://127.0.0.1:5000** to run live suspicion analysis on any chess.com username!

---

## 5. Performance Tips & Best Practices

- **Elo Limit (`MAX_ELO`)**: Maintained at 2000 in `config.py`. Above 2000 ELO, strong human move timing resembles engine consistency.
- **False Positive Rate**: When reading `tests/reports/report.md`, prioritize keeping the False Positive Rate low so honest players are not misflagged.
- **Engine Features**: If trained with `--engine`, make sure `STOCKFISH_PATH` environment variable points to the stockfish binary location on the host machine.
