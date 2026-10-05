import os
import json
import sqlite3
import chess
import chess.engine
import chess.pgn
from typing import Dict, List, Any, Tuple, Optional
from fairsquare.config import (
    DEFAULT_ENGINE_PATH, CACHE_DB_PATH, CACHE_DIR,
    DEFAULT_DEPTH, DEFAULT_MULTIPV, DEFAULT_THREADS, DEFAULT_HASH_MB,
    BOOK_MOVE_CUTOFF
)

class StockfishAnalyzer:
    def __init__(self, engine_path: str = DEFAULT_ENGINE_PATH, depth: int = DEFAULT_DEPTH, multipv: int = DEFAULT_MULTIPV):
        self.engine_path = engine_path
        self.depth = depth
        self.multipv = multipv
        self._init_cache_db()

    def _init_cache_db(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        with sqlite3.connect(CACHE_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS fen_evals (
                    fen TEXT PRIMARY KEY,
                    depth INTEGER,
                    multipv_data TEXT
                )
            """)
            conn.commit()

    def _get_cached_eval(self, fen: str) -> Optional[List[Dict[str, Any]]]:
        try:
            with sqlite3.connect(CACHE_DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT depth, multipv_data FROM fen_evals WHERE fen = ?", (fen,))
                row = cursor.fetchone()
                if row and row[0] >= self.depth:
                    return json.loads(row[1])
        except Exception:
            pass
        return None

    def _save_cached_eval(self, fen: str, multipv_data: List[Dict[str, Any]]):
        try:
            with sqlite3.connect(CACHE_DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT OR REPLACE INTO fen_evals (fen, depth, multipv_data) VALUES (?, ?, ?)",
                    (fen, self.depth, json.dumps(multipv_data))
                )
                conn.commit()
        except Exception:
            pass

    def analyze_position(self, engine: chess.engine.SimpleEngine, board: chess.Board) -> List[Dict[str, Any]]:
        """
        Analyzes a single FEN position with Stockfish MultiPV.
        Returns candidate moves sorted by evaluation (cp or mate converted).
        """
        fen = board.fen()
        cached = self._get_cached_eval(fen)
        if cached:
            return cached

        # Run UCI analysis
        analysis_result = engine.analyse(
            board,
            chess.engine.Limit(depth=self.depth),
            multipv=self.multipv
        )

        candidates = []
        for entry in analysis_result:
            pv = entry.get("pv", [])
            if not pv:
                continue
            move = pv[0]
            score_obj = entry.get("score")
            
            if board.turn == chess.BLACK:
                score_cp = -score_obj.relative.score(mate_score=10000)
            else:
                score_cp = score_obj.relative.score(mate_score=10000)

            candidates.append({
                "move_uci": move.uci(),
                "move_san": board.san(move),
                "cp": score_cp,
                "pv_uci": [m.uci() for m in pv[:5]]
            })

        # Sort candidates descending by cp from turn perspective
        candidates.sort(key=lambda x: x["cp"], reverse=True)
        self._save_cached_eval(fen, candidates)
        return candidates

    def analyze_game(
        self,
        game: chess.pgn.Game,
        target_color: str = "both"
    ) -> List[Dict[str, Any]]:
        """
        Analyzes a full game move by move.
        Excludes opening book moves (first BOOK_MOVE_CUTOFF full moves) and forced moves.
        Returns list of move analysis records for specified target_color ('white', 'black', or 'both').
        """
        records = []
        board = game.board()

        # Check if Stockfish binary exists
        if not os.path.exists(self.engine_path):
            # Try system stockfish if bin/ stockfish not present
            engine_executable = "stockfish"
        else:
            engine_executable = self.engine_path

        try:
            engine = chess.engine.SimpleEngine.popen_uci(engine_executable)
            engine.configure({"Threads": DEFAULT_THREADS, "Hash": DEFAULT_HASH_MB})
        except Exception as e:
            raise RuntimeError(f"Could not launch Stockfish engine at '{engine_executable}': {str(e)}")

        try:
            ply = 0
            last_clock = {"white": None, "black": None}

            for node in game.mainline():
                move = node.move
                current_color = "white" if board.turn == chess.WHITE else "black"
                full_move_num = (ply // 2) + 1
                ply += 1

                # Extract move clock if available
                clock_remaining = node.clock()
                move_time_sec = None
                if clock_remaining is not None:
                    prev_clk = last_clock[current_color]
                    if prev_clk is not None and prev_clk >= clock_remaining:
                        move_time_sec = float(prev_clk - clock_remaining)
                    last_clock[current_color] = clock_remaining

                # Check color filter
                if target_color != "both" and current_color != target_color:
                    board.push(move)
                    continue

                # Filter 1: Opening book moves
                if full_move_num <= BOOK_MOVE_CUTOFF:
                    board.push(move)
                    continue

                # Filter 2: Forced moves (1 legal move)
                legal_moves_count = board.legal_moves.count()
                if legal_moves_count <= 1:
                    board.push(move)
                    continue

                # Run Stockfish position analysis
                candidates = self.analyze_position(engine, board)
                if not candidates:
                    board.push(move)
                    continue

                played_uci = move.uci()
                played_san = board.san(move)

                # Check move rank and played move evaluation
                best_candidate = candidates[0]
                best_cp = best_candidate["cp"]
                best_move_uci = best_candidate["move_uci"]

                # Find played move evaluation
                played_cp = None
                played_rank = None
                for idx, cand in enumerate(candidates):
                    if cand["move_uci"] == played_uci:
                        played_cp = cand["cp"]
                        played_rank = idx + 1
                        break

                # If played move is not in top MultiPV candidate lines, evaluate after pushing played move
                if played_cp is None:
                    board.push(move)
                    after_candidates = self.analyze_position(engine, board)
                    board.pop()
                    if after_candidates:
                        # Eval after played move is from opponent perspective, invert it
                        played_cp = -after_candidates[0]["cp"]
                    else:
                        played_cp = best_cp - 150  # Fallback penalty
                    played_rank = 3  # Lower rank

                # Calculate 2nd best candidate score (for position criticality)
                second_best_cp = candidates[1]["cp"] if len(candidates) > 1 else best_cp - 100

                records.append({
                    "ply": ply,
                    "move_number": full_move_num,
                    "color": current_color,
                    "fen": board.fen(),
                    "played_uci": played_uci,
                    "played_san": played_san,
                    "played_cp": played_cp,
                    "played_rank": played_rank,
                    "best_uci": best_move_uci,
                    "best_san": best_candidate["move_san"],
                    "best_cp": best_cp,
                    "second_best_cp": second_best_cp,
                    "cp_gap": max(0.0, float(best_cp - second_best_cp)),
                    "clock_remaining": clock_remaining,
                    "move_time_sec": move_time_sec,
                    "candidates": candidates
                })

                board.push(move)
        finally:
            engine.quit()

        return records
