"""API tests that need no model and no internet."""
from app import app


def test_health():
    r = app.test_client().get("/api/health")
    assert r.status_code == 200 and r.get_json()["status"] == "ok"


def test_rejects_bad_username():
    r = app.test_client().post("/api/analyze", json={"username": "bad name!", "games": 20})
    assert r.status_code == 400


def test_rejects_out_of_range_games():
    r = app.test_client().post("/api/analyze", json={"username": "someone", "games": 500})
    assert r.status_code == 400


def test_index_page_loads():
    assert app.test_client().get("/").status_code == 200


def test_analyze_response_structure(monkeypatch):
    from src import predict
    mock_data = {
        "username": "testplayer",
        "games_requested": 30,
        "games_scored": 30,
        "avg_rating": 1500,
        "suspicion": 22.5,
        "ci_low": 15.0,
        "ci_high": 30.0,
        "verdict": "Low suspicion",
        "flagged_games": 0,
        "avg_acpl": 45.2,
        "avg_top1": 48.0,
        "avg_blunder_rate": 3.5,
        "has_engine": True,
        "games": [
            {
                "url": "https://chess.com/game/1",
                "opponent": "opp",
                "rating": 1500,
                "time_class": "blitz",
                "date": "2026-10-10",
                "probability": 20.0,
                "acpl": 42.0,
                "top1_match": 50.0,
                "blunder_rate": 0.0,
            }
        ]
    }
    monkeypatch.setattr(predict, "analyze_player", lambda u, n, **kwargs: mock_data)
    r = app.test_client().post("/api/analyze", json={"username": "testplayer", "games": 30, "depth": 10})
    assert r.status_code == 200
    res = r.get_json()
    assert res["games_scored"] == 30
    assert res["avg_acpl"] == 45.2
    assert res["avg_top1"] == 48.0
    assert "acpl" in res["games"][0]
    assert "top1_match" in res["games"][0]


def test_analyze_stream_structure(monkeypatch):
    from src import predict
    def mock_stream(u, n, **kwargs):
        yield {"type": "step", "step": 1, "text": "Step 1"}
        yield {"type": "progress", "current": 1, "total": 10, "opponent": "opp"}
        yield {"type": "done", "result": {"username": u, "games_scored": 1}}

    monkeypatch.setattr(predict, "analyze_player_stream", mock_stream)
    r = app.test_client().get("/api/analyze/stream?username=testplayer&games=10&depth=10")
    assert r.status_code == 200
    assert "text/event-stream" in r.content_type
    body = r.data.decode()
    assert "data: " in body
    assert '"type": "progress"' in body
    assert '"type": "done"' in body
