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
