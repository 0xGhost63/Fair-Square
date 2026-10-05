from datasets import load_dataset

# real tournament data (small, has real cheating labels)
real = load_dataset("artemlepin/chess-fraud", "chess_fraud")

# synthetic data (big, train/test split is already done)
synth = load_dataset("artemlepin/chess-fraud", "chess_fraud_synth")

print(real)
print(synth)

# save locally as parquet so you don't download again
real["full"].to_parquet("chess_fraud.parquet")
synth["train"].to_parquet("synth_train.parquet")
synth["test"].to_parquet("synth_test.parquet")
