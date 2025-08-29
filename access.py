import os, gzip, json, time, requests
from pathlib import Path
import sqlite3

OUT_DIR  = Path("raw")
OUT_DIR.mkdir(exist_ok=True)

API_KEY = "" # to be filled in by user

params = {"api_key": API_KEY}

puuid = "c-rE2qydrNAdftERlPCZaXP2xvJILeeVQ7uAXuvxW2rBiLdCjpNTSp7Mi1bfqMAQTtThLzc4RpOLxw"



DB_PATH = "riot_data.db"

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS puuids (
                        puuid TEXT PRIMARY KEY
                    )""")
        c.execute("""CREATE TABLE IF NOT EXISTS game_ids (
                        game_id TEXT PRIMARY KEY,
                        scanned BOOLEAN DEFAULT FALSE
                    )""")
        conn.commit()


def download_tl_from_db():
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT game_id FROM game_ids WHERE scanned = FALSE")
        rows = c.fetchall()

    game_list = [row[0] for row in rows]
    if not game_list:
        print("No unscanned games found.")
        return []
    saved = []
    for game_id in game_list:
        url = (f"https://europe.api.riotgames.com/lol/match/v5/matches/"
               f"{game_id}/timeline")

        try:
            r = requests.get(url, params=params, timeout=10)
            if r.ok:
                out = OUT_DIR / f"{game_id}.json.gz"
                with gzip.open(out, "wt") as f:
                    json.dump(r.json(), f)

                    puuids = set()

                data = r.json()
                participants = data.get("metadata", {}).get("participants", [])
                puuids.update(participants)

                with sqlite3.connect(DB_PATH) as conn:
                    c = conn.cursor()
                    for puuid in puuids:
                        try:
                            c.execute("INSERT OR IGNORE INTO puuids (puuid) VALUES (?)", (puuid,))
                        except sqlite3.Error as e:
                            print(f"DB error for puuid {puuid}: {e}")
                    conn.commit()

                saved.append(out)
                print(f"Saved {out.name}")
                print(f"Inserted {len(puuids)} puuids into database.")

                # Mark as scanned
                with sqlite3.connect(DB_PATH) as conn:
                    c = conn.cursor()
                    c.execute("UPDATE game_ids SET scanned = TRUE WHERE game_id = ?", (game_id,))
                    conn.commit()

            elif r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 2))
                print(f"Hit rate limit, waiting {wait}s…")
                time.sleep(wait)
                continue

            else:
                print(f"{game_id} failed ({r.status_code})")

        except requests.RequestException as e:
            print(f"Request failed for {game_id}: {e}")

    return saved


def extract_unique_puuids_from_timelines():
    puuids = set()
    for file in OUT_DIR.glob("*.json.gz"):
        with gzip.open(file, "rt") as f:
            data = json.load(f)
            participants = data.get("metadata", {}).get("participants", [])
            puuids.update(participants)

    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        for puuid in puuids:
            try:
                c.execute("INSERT OR IGNORE INTO puuids (puuid) VALUES (?)", (puuid,))
            except sqlite3.Error as e:
                print(f"DB error for puuid {puuid}: {e}")
        conn.commit()

    print(f"Inserted {len(puuids)} puuids into database.")
    return puuids


def fetch_and_store_match_ids_from_db_puuids():
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT puuid FROM puuids")
        puuid_rows = c.fetchall()

    if not puuid_rows:
        print("No PUUIDs found in database.")
        return

    for (puuid,) in puuid_rows:
        url = f"https://europe.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids"
        try:
            r = requests.get(url, params={**params, "start": 0}, timeout=10)
            
            if r.ok:
                match_ids = r.json()
                print(f"Found {len(match_ids)} matches for PUUID: {puuid}")

                with sqlite3.connect(DB_PATH) as conn:
                    c = conn.cursor()
                    for game_id in match_ids:
                        c.execute("INSERT OR IGNORE INTO game_ids (game_id, scanned) VALUES (?, FALSE)", (game_id,))
                    conn.commit()

            elif r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 2))
                print(f"Rate limit hit while processing {puuid}, waiting {wait}s…")
                time.sleep(wait)
                continue

            else:
                print(f"Failed to fetch match IDs for {puuid} (status {r.status_code})")

        except requests.RequestException as e:
            print(f"Request exception for {puuid}: {e}")


def save_match_ids_to_db(match_ids):
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        for game_id in match_ids:
            try:
                c.execute("INSERT OR IGNORE INTO game_ids (game_id) VALUES (?)", (game_id,))
            except sqlite3.Error as e:
                print(f"DB error for game ID {game_id}: {e}")
        conn.commit()
    print(f"Stored {len(match_ids)} unique game IDs in DB.")



if __name__ == "__main__":
    init_db()
    while True:
        fetch_and_store_match_ids_from_db_puuids()
        # save_match_ids_to_db(history)
        download_tl_from_db()