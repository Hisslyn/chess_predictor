#!/usr/bin/env python3
"""
src/parse.py

Reads raw HTML from data/raw/ and populates data/interim/chess.db (SQLite).

Tables populated
────────────────
  tournaments           – one row per tournament
  players_in_tournament – one row per player per tournament
  games                 – one row per game (reconstructed from art=5 crosstable)

Sources
───────
  art=1  (tnr{ID}_art1.html)  player list / starting ranks / tournament metadata
  art=5  (tnr{ID}_art5.html)  crosstable with per-round opponent+result cells
         → used for complete game reconstruction (art=2 only shows recent rounds
           for large tournaments, so art=5 is the authoritative source)

Round-cell encoding (art=5): "<opp_sno><color><score>"
  e.g. "111b1"  → opponent SNo=111, player is Black,  score=1   (win)
       "64w½"   → opponent SNo=64,  player is White,  score=0.5 (draw)
       "-0"     → bye / forfeit absent  (skip)
       "+"      → forfeit win           (skip – not a real game)
"""

import logging
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── paths ─────────────────────────────────────────────────────────────────────
RAW_DIR        = Path("data/raw")
CANDIDATES_CSV = Path("data/interim/tournament_candidates.csv")
DB_PATH        = Path("data/interim/chess.db")

BASE_URL = "https://s1.chess-results.com"


# ── helpers ───────────────────────────────────────────────────────────────────
def _soup(tid: int, art: int) -> BeautifulSoup:
    path = RAW_DIR / f"tnr{tid}_art{art}.html"
    return BeautifulSoup(path.read_text(encoding="utf-8"), "lxml")


def _parse_score(text: str) -> float | None:
    """Convert European decimal score string to float. '6,5' → 6.5"""
    text = text.strip().replace(",", ".")
    try:
        return float(text)
    except ValueError:
        # Handle "½" alone
        if text in ("½", "0.5"):
            return 0.5
        return None


def _parse_round_result(cell: str) -> tuple[int, str, float] | None:
    """
    Parse a single round-result cell from art=5.

    Returns (opponent_sno, color, score) or None for byes/forfeits/empty.

    Examples:
        "111b1"  → (111, 'b', 1.0)
        "64w½"   → (64,  'w', 0.5)
        "64w0.5" → (64,  'w', 0.5)  (rare variant)
        "-0", "+", "-", ""  → None
    """
    cell = cell.strip()
    if not cell or cell in ("-", "+", "-0", "0F", "F", "bye", "BYE"):
        return None
    if re.fullmatch(r"[-+0F]+", cell):
        return None

    m = re.match(r"^(\d+)([wb])([\d½\.]+)$", cell)
    if not m:
        return None

    opp_sno = int(m.group(1))
    color   = m.group(2)       # 'w' or 'b'
    raw_sc  = m.group(3)

    if raw_sc == "½":
        score = 0.5
    else:
        try:
            score = float(raw_sc)
        except ValueError:
            return None

    return opp_sno, color, score


# ── art=1 parsing ─────────────────────────────────────────────────────────────
def _parse_art1(tid: int, csv_date: str | None = None) -> tuple[dict, pd.DataFrame]:
    """
    Returns:
        meta   – dict with keys: name, start_date, end_date, n_rounds,
                 n_players, time_control, avg_rating, rating_std
        df     – DataFrame with one row per player:
                 seed_number, title, name, federation, starting_rating,
                 final_score, final_rank

    Notes
    ─────
    Two CSS-class variants exist across tournaments:
      • CRg-style  (CRg1/CRg2)  : Rk, SNo, title, Name, FED, Rtg, Club, Pts (8 cols)
      • CRng-style (CRng1/CRng2): Rk, SNo, empty, title, Name, FED, Rtg, Club, Pts (9+ cols)
    We detect which format is present from the header row.
    """
    soup = _soup(tid, 1)

    # ── tournament name ───────────────────────────────────────────────────
    name = ""
    for sel in ("h2", "h1"):
        tag = soup.select_one(sel)
        if tag:
            name = tag.get_text(" ", strip=True)
            break
    if not name:
        t = soup.find("title")
        if t:
            name = t.get_text(strip=True).split(" - ")[-1].strip()
    name = name or f"Tournament {tid}"

    # ── page text ─────────────────────────────────────────────────────────
    page_text = soup.get_text(" ", strip=True)

    # ── n_rounds ──────────────────────────────────────────────────────────
    m_rnd = re.search(r"(?:Final\s+Ranking|Ranking)\s+after\s+(\d+)\s+[Rr]ound", page_text)
    n_rounds = int(m_rnd.group(1)) if m_rnd else None

    # ── dates (from CSV; HTML hides them behind a server-load protection click) ──
    start_date = csv_date
    end_date   = csv_date

    # ── player table ──────────────────────────────────────────────────────
    t1 = soup.find("table", class_="CRs1")
    rows = []
    if t1:
        # Detect column layout by reading the header row.
        # Observed layouts:
        #   A  Rk, SNo, title, Name, FED, Rtg, Club, Pts, TB…
        #   B  Rk, SNo, title, Name, sex, FED, RtgI, RtgN, Club, Pts, TB…
        #   C  Rk, SNo, ?, ?, Name, Typ, sex, FED, RtgI, Pts, TB…
        # We locate columns by header text rather than fixed offsets.
        header_tr = t1.find("tr", class_=re.compile(r"CRg1b|CRng1b"))
        hdrs: list[str] = []
        if header_tr:
            hdrs = [c.get_text(strip=True) for c in header_tr.find_all(["th", "td"])]

        def _col(names: list[str]) -> int | None:
            """Return the first column index matching any of the given names."""
            for name in names:
                for i, h in enumerate(hdrs):
                    if h.strip().lower() == name.lower():
                        return i
            return None

        i_name  = _col(["Name"])
        i_fed   = _col(["FED"])
        i_rtg   = _col(["RtgI", "Rtg", "Elo", "Rating"])
        i_title = _col(["Tit.", "Title", ""])   # title is often blank header
        i_pts   = _col(["Pts.", "Pts", "Score"])

        # Fallback fixed offsets when headers are absent
        if i_name is None:
            i_name = 3
        if i_fed is None:
            i_fed = i_name + 1 if i_name is not None else 4
        if i_rtg is None:
            i_rtg = i_name + 2 if i_name is not None else 5
        if i_pts is None:
            i_pts = i_rtg + 2 if i_rtg is not None else 7

        for tr in t1.find_all("tr"):
            cls = tr.get("class", [])
            is_data = any(re.match(r"CRg[12]$|CRng[12]$", c) for c in cls)
            if not is_data:
                continue
            cells = [td.get_text(strip=True) for td in tr.find_all("td")]
            if len(cells) < max(i_name, i_fed, i_rtg, i_pts) + 1:
                continue
            # Title: the column just before Name (if it exists and is short)
            title_val = None
            if i_name and i_name >= 2:
                t_cand = cells[i_name - 1]
                if t_cand and len(t_cand) <= 4:   # CM, FM, IM, GM, WGM …
                    title_val = t_cand
            rows.append({
                "final_rank":      _parse_score(cells[0]),
                "seed_number":     _parse_score(cells[1]),
                "title":           title_val,
                "name":            cells[i_name],
                "federation":      cells[i_fed],
                "starting_rating": _parse_score(cells[i_rtg]),
                "final_score":     _parse_score(cells[i_pts]),
            })

    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["final_rank", "seed_number", "title", "name",
                 "federation", "starting_rating", "final_score"]
    )
    n_players = len(df)

    # ── rating stats ──────────────────────────────────────────────────────
    if not df.empty and "starting_rating" in df.columns:
        ratings = df["starting_rating"].dropna().astype(float)
        ratings = ratings[ratings > 0]
        avg_rating = float(ratings.mean())  if len(ratings) else None
        rating_std = float(ratings.std())   if len(ratings) > 1 else None
    else:
        avg_rating = rating_std = None

    meta = dict(
        name=name,
        start_date=start_date,
        end_date=end_date,
        n_rounds=n_rounds,
        n_players=n_players,
        time_control=None,   # not exposed in art=1 HTML
        avg_rating=avg_rating,
        rating_std=rating_std,
    )
    return meta, df


# ── art=5 parsing ─────────────────────────────────────────────────────────────
def _parse_art5(tid: int) -> tuple[pd.DataFrame, list[dict]]:
    """
    Returns:
        df_perf  – DataFrame with seed_number, performance_rating (may be NaN)
        games    – list of game dicts ready for the `games` table

    Notes
    ─────
    Two CSS-class variants:
      • CRng-style (CRng1/CRng2, CRng1b): used by most tournaments
      • CRg-style  (CRg1/CRg2,   CRg1b):  used by some (e.g. TUR, IRI)

    CRng rows: SNo, title, Name, Rtg, FED, round_cells..., Pts, Rk, TB...
    CRg rows:  No., Name, Rtg, FED, round_cells..., Pts, Rk, TB...  (no title col)
    """
    soup = _soup(tid, 5)

    # ── locate header row ─────────────────────────────────────────────────
    header_tr = soup.find("tr", class_=re.compile(r"CRng1b|CRg1b"))
    if not header_tr:
        log.warning("tnr%d art5: header row not found", tid)
        return pd.DataFrame(), []

    hcls = header_tr.get("class", [])
    is_crng = any("CRng" in c for c in hcls)

    hdrs = [td.get_text(strip=True) for td in header_tr.find_all(["th", "td"])]
    # Expected (CRng): No., '', Name, Rtg, FED, 1.Rd, ..., N.Rd, Pts., Rk., TB...
    # Expected (CRg):  No., Name, Rtg, FED, 1.Rd, ..., N.Rd, Pts., Rk., TB...
    round_indices = [i for i, h in enumerate(hdrs) if re.match(r"^\d+\.Rd$", h)]
    n_rounds = len(round_indices)
    if not round_indices:
        log.warning("tnr%d art5: no round columns found (hdrs=%s)", tid, hdrs)
        return pd.DataFrame(), []

    # ── parse player rows ─────────────────────────────────────────────────
    player_row_re = re.compile(r"^CRng[12]$") if is_crng else re.compile(r"^CRg[12]$")
    player_rows = soup.find_all("tr", class_=player_row_re)

    # CRng-style: SNo, title, Name, Rtg, FED, [rounds], Pts, Rk, TB...
    # CRg-style:  No., Name,  Rtg,  FED, [rounds], Pts, Rk, TB...  (no title col)
    players: dict[int, dict] = {}   # keyed by seed_number (SNo)
    for tr in player_rows:
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(cells) < 4 + n_rounds:
            continue

        try:
            sno = int(cells[0])
        except ValueError:
            continue

        # Round result cells start at round_indices[0]
        r0 = round_indices[0]
        round_cells = cells[r0: r0 + n_rounds]

        if is_crng:
            title  = cells[1] if cells[1] else None
            name   = cells[2]
            rating = _parse_score(cells[3])
            fed    = cells[4]
        else:
            # CRg-style has no title column
            title  = None
            name   = cells[1]
            rating = _parse_score(cells[2])
            fed    = cells[3]

        players[sno] = {
            "seed_number":       sno,
            "title":             title,
            "name":              name,
            "starting_rating":   rating,
            "federation":        fed,
            "round_cells":       round_cells,
            "final_score":       _parse_score(cells[r0 + n_rounds])     if len(cells) > r0 + n_rounds     else None,
            "final_rank":        _parse_score(cells[r0 + n_rounds + 1]) if len(cells) > r0 + n_rounds + 1 else None,
            "performance_rating": None,   # not present in standard art=5 HTML
        }

    # ── reconstruct games ─────────────────────────────────────────────────
    # Strategy: iterate every (player, round) cell.
    # When color == 'w', THIS player is White → emit a game row.
    # When color == 'b', skip – will be captured from opponent's 'w' perspective.
    # This guarantees each game is emitted exactly once.
    games: list[dict] = []

    for sno, pdata in players.items():
        rating_map = {s: p["starting_rating"] for s, p in players.items()}
        for rnd_idx, cell in enumerate(pdata["round_cells"], start=1):
            parsed = _parse_round_result(cell)
            if parsed is None:
                continue
            opp_sno, color, score = parsed

            if color != "w":
                continue   # emit from white's side only

            if opp_sno not in players:
                continue   # opponent dropped / not in our player list

            # white = sno, black = opp_sno
            white_score = score
            black_score = 1.0 - score if score in (0.0, 1.0) else 0.5

            # Encode result as standard string
            if white_score == 1.0:
                result = "1-0"
            elif white_score == 0.0:
                result = "0-1"
            else:
                result = "1/2-1/2"

            games.append({
                "tournament_id":              tid,
                "round_number":               rnd_idx,
                "white_seed":                 sno,
                "black_seed":                 opp_sno,
                "white_rating":               rating_map.get(sno),
                "black_rating":               rating_map.get(opp_sno),
                "result":                     result,
            })

    df_perf = pd.DataFrame([
        {"seed_number": p["seed_number"], "performance_rating": p["performance_rating"]}
        for p in players.values()
    ])
    return df_perf, games


# ── database setup ────────────────────────────────────────────────────────────
DDL = """
CREATE TABLE IF NOT EXISTS tournaments (
    id              INTEGER PRIMARY KEY,
    name            TEXT,
    location        TEXT,
    start_date      TEXT,
    end_date        TEXT,
    n_rounds        INTEGER,
    n_players       INTEGER,
    time_control    TEXT,
    avg_rating      REAL,
    rating_std      REAL
);

CREATE TABLE IF NOT EXISTS players_in_tournament (
    player_tournament_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    tournament_id         INTEGER NOT NULL REFERENCES tournaments(id),
    fide_id               INTEGER,
    name                  TEXT,
    title                 TEXT,
    federation            TEXT,
    starting_rating       REAL,
    seed_number           INTEGER,
    final_score           REAL,
    final_rank            INTEGER,
    performance_rating    REAL
);

CREATE TABLE IF NOT EXISTS games (
    game_id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    tournament_id                 INTEGER NOT NULL REFERENCES tournaments(id),
    round_number                  INTEGER,
    white_player_tournament_id    INTEGER REFERENCES players_in_tournament(player_tournament_id),
    black_player_tournament_id    INTEGER REFERENCES players_in_tournament(player_tournament_id),
    white_rating                  REAL,
    black_rating                  REAL,
    result                        TEXT
);
"""

# ── main ──────────────────────────────────────────────────────────────────────
def parse_tournament(tid: int, conn: sqlite3.Connection,
                     csv_date: str | None = None) -> dict:
    """Parse one tournament and insert into the DB. Returns summary counts."""
    log.info("── tnr%d ──────────────────────────────────────────────", tid)

    # ── 1. art=1 → tournament metadata + players ─────────────────────────
    meta, df_players = _parse_art1(tid, csv_date=csv_date)
    log.info(
        "  art1: name=%r  n_players=%d  n_rounds=%s  dates=%s→%s",
        meta["name"][:50], meta["n_players"],
        meta["n_rounds"], meta["start_date"], meta["end_date"],
    )

    # ── 2. art=5 → performance ratings + games ───────────────────────────
    df_perf, raw_games = _parse_art5(tid)
    log.info("  art5: %d players parsed, %d games reconstructed", len(df_perf), len(raw_games))

    # ── 3. Insert / update tournament row ─────────────────────────────────
    conn.execute("DELETE FROM tournaments WHERE id = ?", (tid,))
    conn.execute(
        """INSERT INTO tournaments
           (id, name, location, start_date, end_date, n_rounds, n_players,
            time_control, avg_rating, rating_std)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (tid, meta["name"], None, meta["start_date"], meta["end_date"],
         meta["n_rounds"], meta["n_players"], meta["time_control"],
         meta["avg_rating"], meta["rating_std"]),
    )

    # ── 4. Insert players ─────────────────────────────────────────────────
    conn.execute(
        "DELETE FROM players_in_tournament WHERE tournament_id = ?", (tid,)
    )

    # Merge art=1 and art=5 player data; art=5 is richer in names/ratings
    # but art=1 has final_rank/score aligned to the official ranking page.
    # Use art=1 as base, overlay performance_rating from art=5.
    if not df_perf.empty and "seed_number" in df_perf.columns:
        df_players = df_players.merge(
            df_perf[["seed_number", "performance_rating"]],
            on="seed_number", how="left",
        )
    else:
        df_players["performance_rating"] = None

    pit_rows = []
    for _, row in df_players.iterrows():
        pit_rows.append((
            tid,
            None,                              # fide_id — not in these pages
            row.get("name"),
            row.get("title"),
            row.get("federation"),
            row.get("starting_rating"),
            row.get("seed_number"),
            row.get("final_score"),
            row.get("final_rank"),
            row.get("performance_rating"),
        ))

    conn.executemany(
        """INSERT INTO players_in_tournament
           (tournament_id, fide_id, name, title, federation,
            starting_rating, seed_number, final_score, final_rank,
            performance_rating)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        pit_rows,
    )

    # ── 5. Build seed_number → player_tournament_id lookup ───────────────
    cur = conn.execute(
        """SELECT seed_number, player_tournament_id
           FROM players_in_tournament WHERE tournament_id = ?""",
        (tid,),
    )
    sno_to_ptid: dict[int, int] = {int(r[0]): r[1] for r in cur if r[0] is not None}

    # ── 6. Insert games ───────────────────────────────────────────────────
    conn.execute("DELETE FROM games WHERE tournament_id = ?", (tid,))

    game_rows = []
    for g in raw_games:
        w_ptid = sno_to_ptid.get(g["white_seed"])
        b_ptid = sno_to_ptid.get(g["black_seed"])
        game_rows.append((
            tid,
            g["round_number"],
            w_ptid,
            b_ptid,
            g["white_rating"],
            g["black_rating"],
            g["result"],
        ))

    conn.executemany(
        """INSERT INTO games
           (tournament_id, round_number,
            white_player_tournament_id, black_player_tournament_id,
            white_rating, black_rating, result)
           VALUES (?,?,?,?,?,?,?)""",
        game_rows,
    )
    conn.commit()

    return {
        "tid": tid,
        "players": len(pit_rows),
        "games":   len(game_rows),
    }


def main() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(DDL)
    conn.commit()

    df_cand = pd.read_csv(CANDIDATES_CSV)
    log.info("Parsing %d tournaments from %s", len(df_cand), CANDIDATES_CSV)

    # Build {tid: date_str} lookup from CSV
    date_map: dict[int, str] = dict(
        zip(df_cand["tournament_id"], df_cand["date"])
    )

    summaries = []
    for tid in df_cand["tournament_id"].tolist():
        try:
            s = parse_tournament(tid, conn, csv_date=date_map.get(tid))
            summaries.append(s)
        except Exception as exc:
            log.error("tnr%d FAILED: %s", tid, exc, exc_info=True)

    conn.close()

    # ── report ────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("Row counts per table")
    print("=" * 72)

    conn2 = sqlite3.connect(DB_PATH)
    for table in ("tournaments", "players_in_tournament", "games"):
        n = conn2.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table:<30s} {n:>6d} rows")

    print()
    print("Sample rows — tournaments:")
    df_t = pd.read_sql("SELECT * FROM tournaments LIMIT 5", conn2)
    print(df_t.to_string(index=False))

    print()
    print("Sample rows — players_in_tournament:")
    df_p = pd.read_sql(
        "SELECT * FROM players_in_tournament LIMIT 10", conn2
    )
    print(df_p.to_string(index=False))

    print()
    print("Sample rows — games:")
    df_g = pd.read_sql(
        """SELECT g.game_id, g.tournament_id, g.round_number,
                  wp.name AS white_name, g.white_rating,
                  g.result,
                  bp.name AS black_name, g.black_rating
           FROM games g
           LEFT JOIN players_in_tournament wp
                ON wp.player_tournament_id = g.white_player_tournament_id
           LEFT JOIN players_in_tournament bp
                ON bp.player_tournament_id = g.black_player_tournament_id
           LIMIT 10""",
        conn2,
    )
    print(df_g.to_string(index=False))

    print()
    print("Per-tournament summary:")
    df_s = pd.read_sql(
        """SELECT t.id, t.name, t.n_rounds, t.n_players,
                  COUNT(DISTINCT p.player_tournament_id) AS players_db,
                  COUNT(DISTINCT g.game_id) AS games_db
           FROM tournaments t
           LEFT JOIN players_in_tournament p ON p.tournament_id = t.id
           LEFT JOIN games g ON g.tournament_id = t.id
           GROUP BY t.id""",
        conn2,
    )
    print(df_s.to_string(index=False))
    print("=" * 72)

    conn2.close()


if __name__ == "__main__":
    main()
