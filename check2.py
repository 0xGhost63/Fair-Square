import pandas as pd

real = pd.read_parquet("chess_fraud.parquet")
real = real[real["is_used"]]
real["sf15_match"] = real["move_player"] == real["move_stockfish_15"]

# one row per game
game = real.groupby(["game_id", "is_cheating_player_game"]).agg(
    match=("sf15_match", "mean"),
    cpl=("centipawn_loss", "mean"),
).reset_index()

print(game.groupby("is_cheating_player_game")[["match", "cpl"]].mean())
print(game.groupby("is_cheating_player_game")[["match", "cpl"]].std())
