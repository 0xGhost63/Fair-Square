import pandas as pd
from typing import Tuple, Dict, Any

def load_and_inspect_data(
    real_path: str = "chess_fraud.parquet",
    synth_train_path: str = "synth_train.parquet",
    synth_test_path: str = "synth_test.parquet"
) -> Dict[str, Any]:
    """
    Phase 1 Data Loader:
    Reads parquet files, filters rows where is_used == True,
    and computes basic statistics and class balance.
    """
    # 1. Load Real Tournament Data
    df_real_raw = pd.read_parquet(real_path)
    df_real = df_real_raw[df_real_raw["is_used"] == True].copy()

    # 2. Load Synthetic Data
    df_synth_train_raw = pd.read_parquet(synth_train_path)
    df_synth_train = df_synth_train_raw[df_synth_train_raw["is_used"] == True].copy()

    df_synth_test_raw = pd.read_parquet(synth_test_path)
    df_synth_test = df_synth_test_raw[df_synth_test_raw["is_used"] == True].copy()

    # 3. Class balance for real data
    move_balance = df_real["is_cheating_move"].value_counts(normalize=True).to_dict()
    game_balance = df_real.groupby("game_id")["is_cheating_player_game"].first().value_counts(normalize=True).to_dict()

    stats = {
        "real_raw_rows": len(df_real_raw),
        "real_used_rows": len(df_real),
        "real_num_players": df_real["player_id"].nunique(),
        "real_num_games": df_real["game_id"].nunique(),
        "real_move_class_balance": move_balance,
        "real_game_class_balance": game_balance,
        "synth_train_raw_rows": len(df_synth_train_raw),
        "synth_train_used_rows": len(df_synth_train),
        "synth_test_raw_rows": len(df_synth_test_raw),
        "synth_test_used_rows": len(df_synth_test),
        "columns_present": list(df_real.columns)
    }

    return stats, df_real, df_synth_train, df_synth_test

if __name__ == "__main__":
    stats, _, _, _ = load_and_inspect_data()
    print("=== PHASE 1 DATA INSPECTION SUMMARY ===")
    print(f"Real Data Total Rows (Raw): {stats['real_raw_rows']:,}")
    print(f"Real Data Filtered (is_used == True): {stats['real_used_rows']:,} moves")
    print(f"Number of Unique Players: {stats['real_num_players']}")
    print(f"Number of Unique Games: {stats['real_num_games']}")
    print(f"Move-Level Cheating Balance: {stats['real_move_class_balance'][True]:.2%} Cheating / {stats['real_move_class_balance'][False]:.2%} Fair")
    print(f"Game-Level Cheating Balance: {stats['real_game_class_balance'][True]:.2%} Cheating / {stats['real_game_class_balance'][False]:.2%} Fair")
    print(f"\nSynthetic Train (Filtered): {stats['synth_train_used_rows']:,} moves")
    print(f"Synthetic Test (Filtered): {stats['synth_test_used_rows']:,} moves")
