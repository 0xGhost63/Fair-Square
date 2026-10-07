"""Builds data/features.csv from Lichess parquet files (see guide.md).

Usage:
    python -m src.build_dataset --parquet "data/raw/*.parquet" --max-players 30000
    python -m src.build_dataset --parquet "data/raw/*.parquet" --engine --max-rows 4000
"""
import argparse
import glob
import io

import chess.pgn
import pandas as pd
from tqdm import tqdm

import config
from src.features import extract_features
from src.lichess_labels import get_flags

COLUMNS = ["White", "Black", "WhiteElo", "BlackElo", "TimeControl", "movetext"]


def load_games(files, max_players):
    frames = []
    target_rows = max_players * 4
    total_rows = 0
    for f in files:
        df = pd.read_parquet(f, columns=COLUMNS)
        df = df[df.movetext.str.contains("%clk", regex=False) & df.TimeControl.str.contains("+", regex=False)]
        frames.append(df)
        total_rows += len(df)
        if total_rows >= target_rows:
            break
    df = pd.concat(frames, ignore_index=True)
    if len(df) > target_rows:
        df = df.sample(n=target_rows, random_state=42)
    return df


def to_game(row):
    text = f'[TimeControl "{row.TimeControl}"]\n\n{row.movetext}'
    return chess.pgn.read_game(io.StringIO(text))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--max-players", type=int, default=30000)
    ap.add_argument("--max-pos-per-player", type=int, default=30)
    ap.add_argument("--max-neg-per-player", type=int, default=3)
    ap.add_argument("--max-rows", type=int, default=0, help="stop after this many feature rows (0 = no limit)")
    ap.add_argument("--engine", action="store_true", help="add Stockfish features (slow)")
    ap.add_argument("--out", default=config.DATA_PATH)
    args = ap.parse_args()

    df = load_games(sorted(glob.glob(args.parquet)), args.max_players)
    # one row per (game, side) so each player side is its own example
    sides = []
    for color, name_col, elo_col in ((True, "White", "WhiteElo"), (False, "Black", "BlackElo")):
        s = df[["TimeControl", "movetext", name_col, elo_col]].copy()
        s.columns = ["TimeControl", "movetext", "player", "rating"]
        s["color"] = color
        sides.append(s)
    sides = pd.concat(sides, ignore_index=True)
    sides = sides[sides.rating <= config.MAX_ELO]

    players = sides.player.drop_duplicates().sample(frac=1, random_state=42).head(args.max_players)
    sides = sides[sides.player.isin(players)]
    print(f"Checking {len(players)} players against the Lichess API...")
    flags = get_flags(list(players))
    sides["label"] = sides.player.str.lower().map(flags).astype(int)
    print("Flagged players found:", sum(flags.values()))

    # cap games per player so a few accounts cannot dominate the data
    sides = sides.sample(frac=1, random_state=42)
    cap = sides.label.map({1: args.max_pos_per_player, 0: args.max_neg_per_player})
    sides = sides[sides.groupby("player").cumcount() < cap]
    # balance classes by down-sampling the clean games
    n_pos = int(sides.label.sum())
    neg = sides[sides.label == 0].head(n_pos * 3)
    sides = pd.concat([sides[sides.label == 1], neg]).sample(frac=1, random_state=42)
    if args.max_rows:
        sides = sides.head(args.max_rows)

    analyzer = None
    if args.engine:
        from src.engine import Analyzer
        analyzer = Analyzer()
    rows = []
    for r in tqdm(sides.itertuples(), total=len(sides)):
        game = to_game(r)
        feats = extract_features(game, chess.WHITE if r.color else chess.BLACK, analyzer) if game else None
        if feats:
            rows.append({"player": r.player, "label": r.label, "rating": r.rating, **feats})
    if analyzer:
        analyzer.close()
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(f"Saved {len(out)} rows to {args.out} (cheater rows: {int(out.label.sum())})")


if __name__ == "__main__":
    main()
