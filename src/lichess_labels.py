"""Finds which Lichess usernames were closed for breaking the fair play rules.

The Lichess user API marks such accounts with tosViolation. We use that flag as
our cheater label, because no public dataset ships with cheat labels.
Results are cached so the API is only asked once per player.
"""
import json
import os
import time

import requests
from tqdm import tqdm

CACHE = os.path.join("data", "flagged_cache.json")
URL = "https://lichess.org/api/users"


def load_cache():
    return json.load(open(CACHE)) if os.path.exists(CACHE) else {}


def get_flags(usernames):
    """Return {lowercase name: True/False} where True means flagged for a rules violation."""
    cache = load_cache()
    todo = [u for u in usernames if u.lower() not in cache]
    if todo:
        chunks = [todo[i:i + 300] for i in range(0, len(todo), 300)]
        print(f"Checking Lichess API for {len(todo)} users ({len(usernames) - len(todo)} cached)...")
        pbar = tqdm(chunks, desc="Lichess Batches", unit="batch")
        for chunk in pbar:
            while True:
                try:
                    r = requests.post(URL, data=",".join(chunk), headers={"Content-Type": "text/plain"}, timeout=30)
                    if r.status_code == 429:
                        for sec in range(60, 0, -1):
                            pbar.set_description(f"Rate limited (429)! Retrying in {sec}s")
                            time.sleep(1)
                        pbar.set_description("Lichess Batches")
                        continue
                    r.raise_for_status()
                    break
                except (requests.exceptions.RequestException, Exception) as err:
                    for sec in range(10, 0, -1):
                        pbar.set_description(f"Network error ({err.__class__.__name__})! Retrying in {sec}s")
                        time.sleep(1)
                    pbar.set_description("Lichess Batches")
            found = {u["id"]: bool(u.get("tosViolation")) for u in r.json()}
            for name in chunk:
                cache[name.lower()] = found.get(name.lower(), False)
            json.dump(cache, open(CACHE, "w"))
            time.sleep(1)
        pbar.close()
    return {u.lower(): cache[u.lower()] for u in usernames}
