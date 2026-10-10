# Chess Cheat Detector

A Flask web application that estimates how suspicious a chess.com player looks, based on their most recent games (10 to 100, selectable). The user enters a username, the app downloads the games, extracts numeric features from every game, and a trained machine learning model returns a suspicion rate from 0 to 100 percent.

The project is deliberately small and easy to explain in a viva. There is no deep learning, only a few hundred lines of readable Python and a classic scikit-learn model.

## What the model looks at

Time usage features. Human players spend more time on hard positions and less on obvious moves, so their thinking times vary a lot. Players who receive engine help often show unusually even thinking times. Each game is reduced to features such as the coefficient of variation of move time, the share of moves close to the median time, the correlation between time spent and the number of legal moves, the time spent on captures compared with quiet moves, and the share of sub-second moves. Times are normalised by the time control so bullet and rapid games are comparable.

Engine features (optional). If Stockfish is installed, the project can also compute the average centipawn loss, the share of moves matching the engine best move, and the blunder rate. These are the strongest signals for engine assistance, but they make training and prediction slower. The code works with or without them, and the trained model remembers which features it was trained on.

## Why the rating is limited to 2000

Strong human play looks similar to engine play, which makes cheating much harder to detect above roughly 2000. Limiting the app to players rated 2000 or below removes that confusing zone and raises both accuracy and confidence. The limit is a single value (`MAX_ELO`) in `config.py`.

## How the suspicion rate is produced

1. The last N usable games are downloaded from the public chess.com API. Only rated standard games with clock data are used.
2. Each game gives one feature row for the player's side.
3. The model outputs a cheating probability for each game.
4. The suspicion rate is the average of these probabilities. A 95 percent interval is shown, which is wide for few games and narrow for many games.
5. A verdict is attached: below 30 is low, 30 to 55 is moderate, above 55 is high.

## Project structure

- `app.py` Flask server with the page and the JSON API.
- `config.py` all settings in one place.
- `src/features.py` feature extraction, the core idea of the project.
- `src/engine.py` optional Stockfish helper.
- `src/chesscom.py` chess.com game download.
- `src/lichess_labels.py` finds cheater labels for training data.
- `src/build_dataset.py` turns the Lichess dataset into a feature table.
- `src/train.py` trains and saves the model.
- `src/predict.py` runs the model on a chess.com player.
- `templates/` and `static/` the web page, with a dark mode toggle.
- `tests/` code tests and the model performance report.
- `guide.md` dataset link and training instructions.
- `features/ideas.md` ideas for future work.

## Installation & Setup

### Prerequisites
- Python 3.10 or higher
- Stockfish chess engine (version 16 or newer recommended)

---

### Linux Setup

1. **Install Stockfish**:
   - **Debian / Ubuntu / Pop!_OS**:
     ```bash
     sudo apt update && sudo apt install -y stockfish
     ```
   - **Arch Linux / Manjaro**:
     ```bash
     sudo pacman -S stockfish
     ```
   - **Fedora**:
     ```bash
     sudo dnf install stockfish
     ```

2. **Clone and Create Virtual Environment**:
   ```bash
   git clone https://github.com/your-username/Fair-Square.git
   cd "Fair Square"
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Python Dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Verify Stockfish Path (Optional)**:
   The application automatically detects Stockfish at `/usr/bin/stockfish`, `/usr/games/stockfish`, or via PATH. If Stockfish is installed in a custom directory, export the `STOCKFISH_PATH` environment variable:
   ```bash
   export STOCKFISH_PATH=/path/to/stockfish
   ```

5. **Run the Application**:
   ```bash
   python app.py
   ```
   Open your browser at `http://127.0.0.1:5000`.

---

### Windows Setup

1. **Download Stockfish**:
   - Download the Windows binary from the official Stockfish site: [https://stockfishchess.org/download/](https://stockfishchess.org/download/).
   - Extract the downloaded archive.
   - Rename the executable (e.g. `stockfish-windows-x86-64-avx2.exe`) to `stockfish.exe`.
   - Place `stockfish.exe` directly inside the root folder of this project (`Fair Square/stockfish.exe`), or add its directory to your Windows System PATH.

2. **Clone and Create Virtual Environment**:
   Open Command Prompt (cmd) or PowerShell in the project directory:
   ```cmd
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install Python Dependencies**:
   ```cmd
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Custom Stockfish Path (Optional)**:
   If you prefer not to place `stockfish.exe` in the project root, set the environment variable:
   - In Command Prompt:
     ```cmd
     set STOCKFISH_PATH=C:\path\to\stockfish.exe
     ```
   - In PowerShell:
     ```powershell
     $env:STOCKFISH_PATH = "C:\path\to\stockfish.exe"
     ```

5. **Run the Application**:
   ```cmd
   python app.py
   ```
   Open your browser at `http://127.0.0.1:5000`.
## Honest limitations

The model is trained on Lichess games and applied to chess.com games. Clock behaviour is similar, but the sites are not identical, so scores should be treated as relative. Labels come from account closures, which contain some noise. Cheaters who deliberately randomise their timing are harder to catch with clock features alone, which is why the engine features exist.
