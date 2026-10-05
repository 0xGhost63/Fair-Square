import pandas as pd

real = pd.read_parquet("chess_fraud.parquet")
train = pd.read_parquet("synth_train.parquet")

# keep only rows meant for analysis (move 21 onward)
real = real[real["is_used"]]
train = train[train["is_used"]]

print("real rows:", len(real))
print("synth train rows:", len(train))

# how many real moves are cheating moves
print(real["is_cheating_move"].value_counts())

# how often the player's move equals Stockfish depth 15 move
print("real match SF15:", (real["move_player"] == real["move_stockfish_15"]).mean())
print("synth match SF15:", (train["move_player"] == train["move_stockfish_15"]).mean())
