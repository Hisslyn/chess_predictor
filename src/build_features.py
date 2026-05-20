#!/usr/bin/env python3
"""
src/build_features.py

1. Compute performance_rating (FIDE formula) and update chess.db.
2. Build data/processed/features.csv  — one row per (player, round).

Features
────────
Static       : player_rating, title_encoded, seed_percentile
Tournament   : tournament_avg_rating, tournament_rating_std,
               n_rounds_total, field_size, rating_gap_to_field_avg
Round state  : round_number, rounds_remaining, current_score,
               expected_score_so_far, score_delta, current_rank,
               gap_to_leader, avg_opponent_rating_so_far, color_balance
               (ONLY rounds 1..N-1 — no leakage)
Current round: opponent_rating, rating_diff, expected_score_this_game,
               playing_white, opponent_current_score

Byes/forfeits (NULL player IDs) are excluded throughout.
"""

import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from fide_utils import fide_dp as _fide_dp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DB_PATH  = Path("data/interim/chess.db")
OUT_PATH = Path("data/processed/features.csv")


def _elo_expected(player_rtg: float, opp_rtg: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((opp_rtg - player_rtg) / 400.0))


# ── data loading ──────────────────────────────────────────────────────────────

def load_data(conn: sqlite3.Connection) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (tournaments, players, games_resolved)."""
    tournaments = pd.read_sql("SELECT * FROM tournaments", conn)
    players = pd.read_sql("SELECT * FROM players_in_tournament", conn)

    # Games: join to get player ratings from players_in_tournament.
    # Exclude byes/forfeits (NULL player IDs on either side).
    games = pd.read_sql(
        """
        SELECT
            g.game_id,
            g.tournament_id,
            g.round_number,
            g.white_player_tournament_id  AS white_ptid,
            g.black_player_tournament_id  AS black_ptid,
            wp.starting_rating            AS white_rating,
            bp.starting_rating            AS black_rating,
            g.result
        FROM games g
        JOIN players_in_tournament wp
             ON wp.player_tournament_id = g.white_player_tournament_id
        JOIN players_in_tournament bp
             ON bp.player_tournament_id = g.black_player_tournament_id
        """,
        conn,
    )
    return tournaments, players, games


# ── performance rating ────────────────────────────────────────────────────────

def compute_performance_ratings(
    players: pd.DataFrame, games: pd.DataFrame
) -> pd.Series:
    """
    Return a Series indexed by player_tournament_id with performance_rating.

    Formula: Rp = avg_opponent_rating + dp(score_percentage)
    Requires ≥ 1 rated game. Players with no rated games get NaN.
    """
    # Build player-perspective records
    white = games[["white_ptid", "black_ptid", "white_rating", "black_rating", "result"]].copy()
    white = white.rename(columns={
        "white_ptid":   "ptid",
        "black_ptid":   "opp_ptid",
        "white_rating": "player_rtg",
        "black_rating": "opp_rtg",
    })
    white["score"] = white["result"].map({"1-0": 1.0, "1/2-1/2": 0.5, "0-1": 0.0})

    black = games[["black_ptid", "white_ptid", "black_rating", "white_rating", "result"]].copy()
    black = black.rename(columns={
        "black_ptid":   "ptid",
        "white_ptid":   "opp_ptid",
        "black_rating": "player_rtg",
        "white_rating": "opp_rtg",
    })
    black["score"] = black["result"].map({"1-0": 0.0, "1/2-1/2": 0.5, "0-1": 1.0})

    pg = pd.concat([white, black], ignore_index=True)
    pg = pg.dropna(subset=["opp_rtg"])   # need opponent rating

    grouped = pg.groupby("ptid").agg(
        avg_opp_rtg=("opp_rtg", "mean"),
        total_score=("score", "sum"),
        n_games=("score", "count"),
    )
    grouped["pct"] = grouped["total_score"] / grouped["n_games"] * 100.0
    grouped["performance_rating"] = grouped.apply(
        lambda r: r["avg_opp_rtg"] + _fide_dp(r["pct"]), axis=1
    ).round(1)

    return grouped["performance_rating"]


def update_performance_ratings(conn: sqlite3.Connection, perf: pd.Series) -> None:
    rows = [(float(v), int(k)) for k, v in perf.items()]
    conn.executemany(
        "UPDATE players_in_tournament SET performance_rating = ? WHERE player_tournament_id = ?",
        rows,
    )
    conn.commit()
    log.info("Updated performance_rating for %d players", len(rows))


# ── feature engineering ───────────────────────────────────────────────────────

TITLE_RANK = {"GM": 6, "WGM": 5, "IM": 5, "WIM": 4, "FM": 4, "WFM": 3,
              "CM": 3, "WCM": 2, "NM": 1}


def build_features(
    tournaments: pd.DataFrame,
    players: pd.DataFrame,
    games: pd.DataFrame,
) -> pd.DataFrame:
    """Build one row per (player, round) with no leakage."""

    # ── tournament-level lookup ───────────────────────────────────────────
    t_info = tournaments.set_index("id")[
        ["avg_rating", "rating_std", "n_rounds", "n_players"]
    ].rename(columns={"avg_rating": "t_avg_rtg", "rating_std": "t_rtg_std",
                      "n_rounds": "n_rounds_total", "n_players": "field_size_meta"})

    # ── player-level lookup ───────────────────────────────────────────────
    p = players[["player_tournament_id", "tournament_id", "name",
                 "starting_rating", "title", "seed_number",
                 "final_score", "final_rank"]].copy()
    p["title_encoded"] = p["title"].map(TITLE_RANK).fillna(0).astype(int)

    # seed_percentile per tournament (lower seed rank → higher percentile)
    def _seed_pct(sno: pd.Series) -> pd.Series:
        n = len(sno)
        return (n - sno) / max(n - 1, 1)

    p["seed_percentile"] = p.groupby("tournament_id")["seed_number"].transform(_seed_pct)

    # Merge tournament context onto players
    t_info_reset = t_info.reset_index().rename(columns={"id": "tournament_id"})
    p = p.merge(t_info_reset, on="tournament_id", how="left")
    p["rating_gap_to_field_avg"] = p["starting_rating"] - p["t_avg_rtg"]

    # field_size from actual DB count
    field_counts = (
        players.groupby("tournament_id")["player_tournament_id"]
        .count()
        .reset_index()
        .rename(columns={"player_tournament_id": "field_size"})
    )
    p = p.merge(field_counts, on="tournament_id", how="left")

    p_lookup = p.set_index("player_tournament_id")

    # ── build player-perspective game records ─────────────────────────────
    def _player_games(color: str) -> pd.DataFrame:
        if color == "white":
            g = games.rename(columns={
                "white_ptid": "ptid", "black_ptid": "opp_ptid",
                "white_rating": "p_rtg", "black_rating": "opp_rtg",
            }).copy()
            g["playing_white"] = 1
            g["score"] = g["result"].map({"1-0": 1.0, "1/2-1/2": 0.5, "0-1": 0.0})
        else:
            g = games.rename(columns={
                "black_ptid": "ptid", "white_ptid": "opp_ptid",
                "black_rating": "p_rtg", "white_rating": "opp_rtg",
            }).copy()
            g["playing_white"] = 0
            g["score"] = g["result"].map({"1-0": 0.0, "1/2-1/2": 0.5, "0-1": 1.0})
        return g[["tournament_id", "round_number", "ptid", "opp_ptid",
                  "p_rtg", "opp_rtg", "playing_white", "score"]]

    pg = pd.concat([_player_games("white"), _player_games("black")], ignore_index=True)
    pg = pg.sort_values(["tournament_id", "ptid", "round_number"]).reset_index(drop=True)

    # ── per-tournament, per-player: build cumulative prior-round state ────
    rows = []

    for (tid, ptid), grp in pg.groupby(["tournament_id", "ptid"]):
        grp = grp.sort_values("round_number").reset_index(drop=True)
        pi = p_lookup.loc[ptid] if ptid in p_lookup.index else None
        if pi is None:
            continue

        player_rtg  = pi["starting_rating"]
        title_enc   = pi["title_encoded"]
        seed_pct    = pi["seed_percentile"]
        t_avg       = pi["t_avg_rtg"]
        t_std       = pi["t_rtg_std"]
        n_rounds    = pi["n_rounds_total"]
        f_size      = pi["field_size"]
        rtg_gap     = pi["rating_gap_to_field_avg"]

        # We also need ALL players in this tournament for ranking
        tourn_players = p_lookup[p_lookup["tournament_id"] == tid]

        # Opponent current scores: we need each player's cumulative score
        # per-round for ranking — precompute later when we iterate rounds.
        # Build a score-by-round for ALL players in tournament from pg.
        # (done outside inner loop, shared within tournament group — deferred below)

        for i, row in grp.iterrows():
            rnd = int(row["round_number"])

            # ── prior-round features (rounds 1..rnd-1) ───────────────────
            prior = grp[grp["round_number"] < rnd]

            current_score = float(prior["score"].sum())

            if len(prior) > 0 and player_rtg and not np.isnan(player_rtg):
                exp_so_far = sum(
                    _elo_expected(float(player_rtg), float(r))
                    for r in prior["opp_rtg"].dropna()
                )
                avg_opp_rtg_so_far = float(prior["opp_rtg"].mean())
            else:
                exp_so_far = 0.0
                avg_opp_rtg_so_far = float("nan")

            score_delta   = current_score - exp_so_far
            color_balance = int(prior["playing_white"].map({1: 1, 0: -1}).sum())

            # ── current round features ────────────────────────────────────
            opp_ptid    = int(row["opp_ptid"])  if pd.notna(row["opp_ptid"]) else None
            opp_rtg     = float(row["opp_rtg"]) if pd.notna(row["opp_rtg"]) else float("nan")
            playing_w   = int(row["playing_white"])

            if player_rtg and opp_rtg and not np.isnan(player_rtg) and not np.isnan(opp_rtg):
                exp_this    = _elo_expected(float(player_rtg), opp_rtg)
                rtg_diff    = float(player_rtg) - opp_rtg
            else:
                exp_this  = float("nan")
                rtg_diff  = float("nan")

            rows.append({
                # identifiers
                "player_tournament_id": ptid,
                "tournament_id":        tid,
                "round_number":         rnd,
                # static
                "player_rating":        player_rtg,
                "title_encoded":        title_enc,
                "seed_percentile":      round(seed_pct, 4),
                # tournament context
                "tournament_avg_rating":  t_avg,
                "tournament_rating_std":  t_std,
                "n_rounds_total":         n_rounds,
                "field_size":             f_size,
                "rating_gap_to_field_avg": rtg_gap,
                # round state (prior rounds only)
                "rounds_remaining":       (n_rounds - rnd) if n_rounds else float("nan"),
                "current_score":          current_score,
                "expected_score_so_far":  round(exp_so_far, 4),
                "score_delta":            round(score_delta, 4),
                "avg_opponent_rating_so_far": avg_opp_rtg_so_far,
                "color_balance":          color_balance,
                # current round
                "opponent_rating":        opp_rtg,
                "rating_diff":            round(rtg_diff, 2) if not np.isnan(rtg_diff) else float("nan"),
                "expected_score_this_game": round(exp_this, 4) if not np.isnan(exp_this) else float("nan"),
                "playing_white":          playing_w,
                # filled in a second pass (needs full tournament state)
                "current_rank":           None,
                "gap_to_leader":          None,
                "opponent_current_score": None,
            })

    df = pd.DataFrame(rows)

    # ── second pass: current_rank, gap_to_leader, opponent_current_score ─
    # Build cumulative score at start of each round for every player.
    # cum_score[tid][ptid][round_start] = score entering that round
    # i.e. score after completing rounds 1..r-1

    # Create a score lookup: (tid, ptid, round_number) → score entering round
    score_entering = {}   # (tid, ptid, rnd) → cumulative score before rnd

    for (tid, ptid), grp in pg.groupby(["tournament_id", "ptid"]):
        grp = grp.sort_values("round_number")
        cum = 0.0
        for _, r in grp.iterrows():
            rnd = int(r["round_number"])
            score_entering[(tid, ptid, rnd)] = cum
            cum += float(r["score"])

    # Now fill current_rank, gap_to_leader, opponent_current_score
    for idx, row in df.iterrows():
        tid  = row["tournament_id"]
        ptid = row["player_tournament_id"]
        rnd  = row["round_number"]

        # Scores of all players entering this round (i.e. after rnd-1)
        tourn_ptids = p_lookup[p_lookup["tournament_id"] == tid].index.tolist()
        entering_scores = {
            pp: score_entering.get((tid, pp, rnd), 0.0)
            for pp in tourn_ptids
        }

        my_score = entering_scores.get(ptid, 0.0)
        all_scores = list(entering_scores.values())

        if all_scores:
            rank = sum(1 for s in all_scores if s > my_score) + 1
            leader_score = max(all_scores)
        else:
            rank = 1
            leader_score = my_score

        opp_ptid = None
        # find opponent from original pg data
        pg_match = pg[
            (pg["tournament_id"] == tid) &
            (pg["ptid"] == ptid) &
            (pg["round_number"] == rnd)
        ]
        if not pg_match.empty:
            raw_opp = pg_match.iloc[0]["opp_ptid"]
            if pd.notna(raw_opp):
                opp_ptid = int(raw_opp)

        opp_score = score_entering.get((tid, opp_ptid, rnd), float("nan")) if opp_ptid else float("nan")

        df.at[idx, "current_rank"]           = rank
        df.at[idx, "gap_to_leader"]          = round(my_score - leader_score, 1)
        df.at[idx, "opponent_current_score"] = opp_score

    return df


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)

    log.info("Loading data from %s …", DB_PATH)
    tournaments, players, games = load_data(conn)
    log.info(
        "Loaded: %d tournaments, %d players, %d games (excluding byes)",
        len(tournaments), len(players), len(games),
    )

    # ── Step 1: compute and store performance_rating ──────────────────────
    log.info("Computing performance ratings (FIDE formula) …")
    perf = compute_performance_ratings(players, games)
    update_performance_ratings(conn, perf)

    # Reload players with updated performance_rating
    players = pd.read_sql("SELECT * FROM players_in_tournament", conn)
    conn.close()

    # Show 10 sample performance ratings
    sample = players[players["performance_rating"].notna()][
        ["name", "tournament_id", "starting_rating", "final_score", "performance_rating"]
    ].head(10)
    print("\n── Performance rating sample (10 rows) ───────────────────────")
    print(sample.to_string(index=False))

    n_with = players["performance_rating"].notna().sum()
    n_total = len(players)
    print(f"\nperformance_rating populated: {n_with}/{n_total} players")

    # ── Step 2: build features ────────────────────────────────────────────
    log.info("Building features …")
    df = build_features(tournaments, players, games)

    df.to_csv(OUT_PATH, index=False)
    log.info("Saved %d rows to %s", len(df), OUT_PATH)

    # ── report ────────────────────────────────────────────────────────────
    print("\n── Feature matrix sample (5 rows) ───────────────────────────")
    print(df.head(5).to_string(index=False))

    print("\n── Summary stats ─────────────────────────────────────────────")
    desc = df.describe().T[["count", "mean", "std", "min", "max"]]
    print(desc.to_string())

    print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")


if __name__ == "__main__":
    main()
