#!/usr/bin/env python3
"""
src/validate.py

Runs quality checks on data/interim/chess.db and writes a report to
data/interim/validation_report.txt.

Checks performed
────────────────
  1. Missing starting_rating         (players_in_tournament)
  2. Missing / null performance_rating
  3. DNF players — played fewer rounds than their tournament has
  4. Bye / forfeit games             (games table)
  5. Tournaments where >MISSING_THRESHOLD of players have missing data
  6. Rating outliers per tournament  (|z-score| > Z_THRESHOLD)

Nothing is deleted.  Every flagged ID is recorded for the feature step.
"""

import sqlite3
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

# ── config ────────────────────────────────────────────────────────────────────
DB_PATH          = Path("data/interim/chess.db")
REPORT_PATH      = Path("data/interim/validation_report.txt")
MISSING_THRESHOLD = 0.10   # flag tournament if >10 % of players have issues
Z_THRESHOLD       = 3.0    # flag rating outliers beyond ±3 σ

# ── helpers ───────────────────────────────────────────────────────────────────

def _load(conn: sqlite3.Connection) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tournaments = pd.read_sql("SELECT * FROM tournaments", conn)
    players     = pd.read_sql("SELECT * FROM players_in_tournament", conn)
    games       = pd.read_sql("SELECT * FROM games", conn)
    return tournaments, players, games


def _section(title: str, width: int = 72) -> str:
    bar = "═" * width
    return f"\n{bar}\n{title}\n{bar}"


def _indent(text: str, spaces: int = 2) -> str:
    return textwrap.indent(text, " " * spaces)


# ── checks ────────────────────────────────────────────────────────────────────

def check_missing_starting_rating(players: pd.DataFrame) -> dict:
    """Players with missing or zero starting_rating."""
    mask = players["starting_rating"].isna() | (players["starting_rating"] == 0)
    bad  = players[mask]
    return {
        "count":   len(bad),
        "ptids":   bad["player_tournament_id"].tolist(),
        "by_tournament": (
            bad.groupby("tournament_id")["player_tournament_id"]
               .count()
               .to_dict()
        ),
    }


def check_missing_performance_rating(players: pd.DataFrame) -> dict:
    """Players with null performance_rating (expected until we compute it)."""
    bad = players[players["performance_rating"].isna()]
    return {
        "count":   len(bad),
        "ptids":   bad["player_tournament_id"].tolist(),
        "by_tournament": (
            bad.groupby("tournament_id")["player_tournament_id"]
               .count()
               .to_dict()
        ),
    }


def check_dnf_players(
    players: pd.DataFrame,
    games: pd.DataFrame,
    tournaments: pd.DataFrame,
) -> dict:
    """
    Players who played fewer games than their tournament's n_rounds.
    A player's game count = appearances in white_player_tournament_id
                           + appearances in black_player_tournament_id.
    """
    tid_to_rounds = tournaments.set_index("id")["n_rounds"].to_dict()

    white = games.groupby("white_player_tournament_id").size().rename("games_as_white")
    black = games.groupby("black_player_tournament_id").size().rename("games_as_black")

    played = (
        pd.concat([white, black], axis=1)
        .fillna(0)
        .assign(games_played=lambda d: d["games_as_white"] + d["games_as_black"])
        .reset_index()
        .rename(columns={"index": "player_tournament_id"})
    )

    merged = players[["player_tournament_id", "tournament_id", "name"]].merge(
        played, on="player_tournament_id", how="left"
    )
    merged["games_played"]  = merged["games_played"].fillna(0).astype(int)
    merged["expected_games"] = merged["tournament_id"].map(tid_to_rounds).fillna(0).astype(int)
    merged["shortfall"]      = merged["expected_games"] - merged["games_played"]

    dnf = merged[merged["shortfall"] > 0].copy()
    return {
        "count":   len(dnf),
        "ptids":   dnf["player_tournament_id"].tolist(),
        "by_tournament": (
            dnf.groupby("tournament_id")["player_tournament_id"]
               .count()
               .to_dict()
        ),
        "detail": dnf[["player_tournament_id", "tournament_id", "name",
                        "games_played", "expected_games", "shortfall"]],
    }


def check_bye_forfeit_games(games: pd.DataFrame) -> dict:
    """
    Games that are byes / forfeits:
      • white_player_tournament_id or black_player_tournament_id is NULL
      • result not in {'1-0', '0-1', '1/2-1/2'}
    """
    null_player = (
        games["white_player_tournament_id"].isna()
        | games["black_player_tournament_id"].isna()
    )
    bad_result = ~games["result"].isin({"1-0", "0-1", "1/2-1/2"})
    mask = null_player | bad_result

    bad = games[mask]
    return {
        "count":        len(bad),
        "game_ids":     bad["game_id"].tolist(),
        "null_player":  int(null_player.sum()),
        "bad_result":   int(bad_result.sum()),
        "result_counts": games["result"].value_counts().to_dict(),
    }


def check_high_missing_tournaments(
    players: pd.DataFrame,
    tournaments: pd.DataFrame,
    missing_ptids: set[int],
) -> dict:
    """
    Tournaments where >MISSING_THRESHOLD fraction of their players have
    any missing data (missing starting_rating OR missing performance_rating).
    """
    total = players.groupby("tournament_id")["player_tournament_id"].count()
    missing_count = (
        players[players["player_tournament_id"].isin(missing_ptids)]
        .groupby("tournament_id")["player_tournament_id"]
        .count()
    )
    frac = (missing_count / total).fillna(0)
    flagged = frac[frac > MISSING_THRESHOLD]

    rows = []
    for tid, f in flagged.items():
        name = tournaments.loc[tournaments["id"] == tid, "name"].values
        rows.append({
            "tournament_id":   tid,
            "name":            name[0] if len(name) else "?",
            "n_players":       int(total.get(tid, 0)),
            "n_missing":       int(missing_count.get(tid, 0)),
            "missing_frac":    round(float(f), 3),
        })

    df = pd.DataFrame(rows)
    return {
        "count":  len(df),
        "detail": df,
    }


def check_rating_outliers(
    players: pd.DataFrame,
    tournaments: pd.DataFrame,
) -> dict:
    """
    Players whose starting_rating is more than Z_THRESHOLD standard
    deviations from their tournament's mean (excluding unrated / 0 ratings).
    """
    rated = players[
        players["starting_rating"].notna()
        & (players["starting_rating"] > 0)
    ].copy()

    stats = (
        rated.groupby("tournament_id")["starting_rating"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "rtg_mean", "std": "rtg_std"})
    )
    rated = rated.join(stats, on="tournament_id")
    rated["z"] = (
        (rated["starting_rating"] - rated["rtg_mean"])
        / rated["rtg_std"].replace(0, np.nan)
    )
    outliers = rated[rated["z"].abs() > Z_THRESHOLD].copy()

    tid_names = tournaments.set_index("id")["name"].to_dict()
    outliers["tournament_name"] = outliers["tournament_id"].map(tid_names)

    return {
        "count":  len(outliers),
        "ptids":  outliers["player_tournament_id"].tolist(),
        "detail": outliers[["player_tournament_id", "tournament_id",
                             "tournament_name", "name",
                             "starting_rating", "rtg_mean", "rtg_std", "z"]],
    }


# ── report builder ────────────────────────────────────────────────────────────

def build_report(
    tournaments: pd.DataFrame,
    players: pd.DataFrame,
    games: pd.DataFrame,
) -> str:
    lines: list[str] = []

    lines.append("CHESS.DB VALIDATION REPORT")
    lines.append(f"Database : {DB_PATH}")
    lines.append(f"Thresholds: missing>{MISSING_THRESHOLD:.0%}  |z|>{Z_THRESHOLD}")
    lines.append("")
    lines.append(f"  tournaments            : {len(tournaments):>6}")
    lines.append(f"  players_in_tournament  : {len(players):>6}")
    lines.append(f"  games                  : {len(games):>6}")

    # ── 1. Missing starting_rating ────────────────────────────────────────
    r1 = check_missing_starting_rating(players)
    lines.append(_section("CHECK 1 · Missing / zero starting_rating"))
    lines.append(f"  Flagged players : {r1['count']}")
    if r1["count"]:
        for tid, cnt in sorted(r1["by_tournament"].items()):
            tname = tournaments.loc[tournaments["id"] == tid, "name"].values
            lines.append(f"    tnr{tid}  ({tname[0][:50] if len(tname) else '?'}) : {cnt} players")
        lines.append(f"  player_tournament_ids : {r1['ptids'][:20]}"
                     + (" …" if len(r1["ptids"]) > 20 else ""))

    # ── 2. Missing performance_rating ─────────────────────────────────────
    r2 = check_missing_performance_rating(players)
    lines.append(_section("CHECK 2 · Missing performance_rating"))
    lines.append(f"  Flagged players : {r2['count']}")
    lines.append("  (performance_rating is currently NULL for all players —")
    lines.append("   will be populated in the feature-engineering step)")

    # ── 3. DNF players ────────────────────────────────────────────────────
    r3 = check_dnf_players(players, games, tournaments)
    lines.append(_section("CHECK 3 · DNF players (played fewer rounds than expected)"))
    lines.append(f"  Flagged players : {r3['count']}")
    if r3["count"]:
        for tid, cnt in sorted(r3["by_tournament"].items()):
            tname = tournaments.loc[tournaments["id"] == tid, "name"].values
            lines.append(f"    tnr{tid}  ({tname[0][:50] if len(tname) else '?'}) : {cnt} DNFs")
        lines.append("")
        lines.append("  Detail (first 20 rows):")
        lines.append(_indent(
            r3["detail"].head(20).to_string(index=False), 4
        ))

    # ── 4. Bye / forfeit games ─────────────────────────────────────────────
    r4 = check_bye_forfeit_games(games)
    lines.append(_section("CHECK 4 · Bye / forfeit games"))
    lines.append(f"  Flagged games       : {r4['count']}")
    lines.append(f"    NULL player ref   : {r4['null_player']}")
    lines.append(f"    Non-standard result: {r4['bad_result']}")
    lines.append(f"  Result distribution :")
    for res, cnt in sorted(r4["result_counts"].items(), key=lambda x: -x[1]):
        lines.append(f"    {res!r:<15s} : {cnt}")
    if r4["game_ids"]:
        lines.append(f"  Flagged game_ids    : {r4['game_ids'][:20]}"
                     + (" …" if len(r4["game_ids"]) > 20 else ""))

    # ── 5. High-missing tournaments ───────────────────────────────────────
    # Exclude r2 (performance_rating) — it is intentionally NULL pre-feature-engineering
    all_missing_ptids = set(r1["ptids"])
    r5 = check_high_missing_tournaments(players, tournaments, all_missing_ptids)
    lines.append(_section(
        f"CHECK 5 · Tournaments with >{MISSING_THRESHOLD:.0%} missing-data players"
    ))
    lines.append(f"  Flagged tournaments : {r5['count']}")
    if r5["count"]:
        lines.append(_indent(r5["detail"].to_string(index=False), 4))

    # ── 6. Rating outliers ────────────────────────────────────────────────
    r6 = check_rating_outliers(players, tournaments)
    lines.append(_section(f"CHECK 6 · Rating outliers (|z| > {Z_THRESHOLD})"))
    lines.append(f"  Flagged players : {r6['count']}")
    if r6["count"]:
        lines.append(_indent(
            r6["detail"]
               .sort_values("z", key=abs, ascending=False)
               .head(30)
               .to_string(index=False),
            4
        ))

    # ── summary ───────────────────────────────────────────────────────────
    lines.append(_section("SUMMARY"))
    clean_games  = len(games) - r4["count"]
    clean_players = len(players) - len(
        set(r1["ptids"]) | set(r3["ptids"]) | set(r6["ptids"])
    )

    total_flagged_ptids = set(r1["ptids"]) | set(r3["ptids"]) | set(r6["ptids"])
    lines.append(f"  CHECK 1  missing starting_rating  : {r1['count']:>4} players flagged")
    lines.append(f"  CHECK 2  missing performance_rating: {r2['count']:>4} players flagged  (expected)")
    lines.append(f"  CHECK 3  DNF (incomplete rounds)  : {r3['count']:>4} players flagged")
    lines.append(f"  CHECK 4  bye/forfeit games         : {r4['count']:>4} games   flagged")
    lines.append(f"  CHECK 5  high-missing tournaments  : {r5['count']:>4} tournaments flagged")
    lines.append(f"  CHECK 6  rating outliers           : {r6['count']:>4} players flagged")
    lines.append("")
    lines.append(f"  Unique players flagged (any check) : {len(total_flagged_ptids)}")
    lines.append(f"  Clean players                      : {clean_players} / {len(players)}")
    lines.append(f"  Clean games (excl. bye/forfeit)    : {clean_games} / {len(games)}")
    lines.append("")

    return "\n".join(lines)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    tournaments, players, games = _load(conn)
    conn.close()

    report = build_report(tournaments, players, games)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")

    print(report)
    print(f"\nReport written → {REPORT_PATH}")


if __name__ == "__main__":
    main()
