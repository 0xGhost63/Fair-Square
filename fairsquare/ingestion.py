import io
import re
import requests
import chess.pgn
import zstandard as zstd
from typing import Generator, List, Dict, Any, Optional
from fairsquare.config import MIN_GAME_MOVES, DEFAULT_DATASET_PATH

class GameIngestion:
    @staticmethod
    def stream_lichess_pgn_zst(
        file_path: str = DEFAULT_DATASET_PATH,
        max_games: Optional[int] = None,
        time_control_filter: Optional[str] = "Blitz",
        min_moves: int = MIN_GAME_MOVES
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Memory-safe stream parser for compressed Lichess .pgn.zst file.
        Reads games one by one using zstandard stream reader.
        """
        count = 0
        with open(file_path, 'rb') as fh:
            dctx = zstd.ZstdDecompressor(max_window_size=2147483648)
            with dctx.stream_reader(fh) as reader:
                text_stream = io.TextIOWrapper(reader, encoding='utf-8', errors='replace')
                
                while True:
                    game = chess.pgn.read_game(text_stream)
                    if game is None:
                        break

                    headers = game.headers
                    termination = headers.get("Termination", "")
                    if termination != "Normal":
                        continue

                    event = headers.get("Event", "")
                    if time_control_filter and time_control_filter.lower() not in event.lower():
                        continue

                    moves = list(game.mainline_moves())
                    full_moves = len(moves) // 2
                    if full_moves < min_moves:
                        continue

                    white_rating = int(headers.get("WhiteElo", "1500"))
                    black_rating = int(headers.get("BlackElo", "1500"))

                    yield {
                        "event": event,
                        "white": headers.get("White", "Unknown"),
                        "black": headers.get("Black", "Unknown"),
                        "white_elo": white_rating,
                        "black_elo": black_rating,
                        "result": headers.get("Result", "*"),
                        "game_obj": game,
                        "full_moves": full_moves,
                        "time_control": headers.get("TimeControl", "")
                    }

                    count += 1
                    if max_games and count >= max_games:
                        break

    @staticmethod
    def fetch_chesscom_user_games(
        username: str,
        max_games: int = 50,
        time_controls: List[str] = ["blitz", "rapid"]
    ) -> List[Dict[str, Any]]:
        """
        Fetches recent games for a specified Chess.com username via public REST API.
        """
        headers = {
            "User-Agent": "FairSquare-ChessAssistanceDetector/1.0 (Contact: student_research@fairsquare.org)"
        }
        username_clean = username.strip().lower()
        archives_url = f"https://api.chess.com/pub/player/{username_clean}/games/archives"
        
        try:
            resp = requests.get(archives_url, headers=headers, timeout=10)
            if resp.status_code != 200:
                raise ValueError(f"Chess.com user '{username}' not found or API error (status {resp.status_code}).")
            
            archives = resp.json().get("archives", [])
            if not archives:
                return []
            
            games_list = []
            for archive_url in reversed(archives):
                if len(games_list) >= max_games:
                    break
                
                arc_resp = requests.get(archive_url, headers=headers, timeout=10)
                if arc_resp.status_code != 200:
                    continue
                
                month_games = arc_resp.json().get("games", [])
                for g in reversed(month_games):
                    if len(games_list) >= max_games:
                        break
                    
                    time_class = g.get("time_class", "")
                    if time_class not in time_controls:
                        continue
                    
                    pgn_text = g.get("pgn", "")
                    if not pgn_text:
                        continue
                    
                    pgn_io = io.StringIO(pgn_text)
                    game_obj = chess.pgn.read_game(pgn_io)
                    if not game_obj:
                        continue
                    
                    moves = list(game_obj.mainline_moves())
                    full_moves = len(moves) // 2
                    if full_moves < MIN_GAME_MOVES:
                        continue
                    
                    white_user = game_obj.headers.get("White", "").lower()
                    black_user = game_obj.headers.get("Black", "").lower()
                    user_color = "white" if username_clean in white_user else ("black" if username_clean in black_user else "unknown")
                    
                    user_elo = int(game_obj.headers.get("WhiteElo", 1500)) if user_color == "white" else int(game_obj.headers.get("BlackElo", 1500))
                    
                    games_list.append({
                        "id": g.get("url", ""),
                        "event": game_obj.headers.get("Event", time_class.capitalize()),
                        "white": game_obj.headers.get("White", "Unknown"),
                        "black": game_obj.headers.get("Black", "Unknown"),
                        "white_elo": int(game_obj.headers.get("WhiteElo", 1500)),
                        "black_elo": int(game_obj.headers.get("BlackElo", 1500)),
                        "target_user": username_clean,
                        "target_color": user_color,
                        "target_elo": user_elo,
                        "game_obj": game_obj,
                        "full_moves": full_moves,
                        "time_control": game_obj.headers.get("TimeControl", "")
                    })
                    
            return games_list
        except Exception as e:
            raise RuntimeError(f"Failed to fetch games for {username}: {str(e)}")

    @staticmethod
    def extract_pgn_player_names(pgn_text: str) -> Dict[str, Any]:
        """
        Extracts White and Black player names from raw PGN text headers.
        Uses regex parsing fallback if python-chess returns default placeholders.
        """
        white_name = "White Player"
        black_name = "Black Player"

        w_match = re.search(r'\[White\s+"([^"]+)"\]', pgn_text, re.IGNORECASE)
        if w_match:
            white_name = w_match.group(1).strip()

        b_match = re.search(r'\[Black\s+"([^"]+)"\]', pgn_text, re.IGNORECASE)
        if b_match:
            black_name = b_match.group(1).strip()

        if white_name == "White Player" or black_name == "Black Player":
            pgn_io = io.StringIO(pgn_text)
            game_obj = chess.pgn.read_game(pgn_io)
            if game_obj:
                if white_name == "White Player":
                    white_name = game_obj.headers.get("White", "White Player")
                if black_name == "Black Player":
                    black_name = game_obj.headers.get("Black", "Black Player")

        return {
            "white": white_name if white_name != "?" else "White Player",
            "black": black_name if black_name != "?" else "Black Player"
        }

    @staticmethod
    def parse_pgn_file(pgn_text: str) -> List[Dict[str, Any]]:
        """
        Parses a raw PGN text string (may contain single or multiple games).
        """
        pgn_io = io.StringIO(pgn_text)
        parsed_games = []
        
        while True:
            game_obj = chess.pgn.read_game(pgn_io)
            if not game_obj:
                break
            
            moves = list(game_obj.mainline_moves())
            full_moves = len(moves) // 2
            if full_moves < MIN_GAME_MOVES:
                continue
            
            parsed_games.append({
                "event": game_obj.headers.get("Event", "Custom PGN"),
                "white": game_obj.headers.get("White", "White Player"),
                "black": game_obj.headers.get("Black", "Black Player"),
                "white_elo": int(game_obj.headers.get("WhiteElo", 1500)),
                "black_elo": int(game_obj.headers.get("BlackElo", 1500)),
                "result": game_obj.headers.get("Result", "*"),
                "game_obj": game_obj,
                "full_moves": full_moves,
                "time_control": game_obj.headers.get("TimeControl", "")
            })
            
        return parsed_games
