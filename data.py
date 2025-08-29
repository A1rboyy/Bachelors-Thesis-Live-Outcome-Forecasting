## Not the big scale solution, but a quick way to get a timeline json


import os, gzip, json, time, requests
from pathlib import Path

MATCH_ID = "EUW1_7432465462"

OUT_DIR  = Path("raw")
OUT_DIR.mkdir(exist_ok=True)

API_KEY = "" # to be filled in by user

params = {"api_key": API_KEY}

url    = f"https://europe.api.riotgames.com/lol/match/v5/matches/{MATCH_ID}/timeline"
params = {"api_key": API_KEY}

resp = requests.get(url, params=params, timeout=10)
resp.raise_for_status()                                   # raises if not 200

out_path = OUT_DIR / f"{MATCH_ID}_timeline.json"
with out_path.open("w", encoding="utf-8") as f:
    json.dump(resp.json(), f, indent=2)    