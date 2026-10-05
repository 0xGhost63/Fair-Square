import unittest
import json
from app import app, TASKS

class TestWebApp(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.client.testing = True

    def test_index_route(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'FAIR SQUARE', response.data)
        self.assertIn(b'CHESS ENGINE ASSISTANCE DETECTION PIPELINE', response.data)

    def test_chesscom_api_validation(self):
        response = self.client.post('/api/analyze/chesscom', json={})
        self.assertEqual(response.status_code, 400)

    def test_pgn_api_validation(self):
        response = self.client.post('/api/analyze/pgn', json={})
        self.assertEqual(response.status_code, 400)

if __name__ == "__main__":
    unittest.main()
