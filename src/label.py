#!/usr/bin/env python3
"""
src/label.py

Adds action labels and training target to features.csv.

Action heuristic (configurable threshold at top of file)
─────────────────────────────────────────────────────────
rating_diff = player_rating - opponent_rating  (positive → player is stronger)

AGGRESSIVE  upset win as underdog          rating_diff < -T  and win
            decisive vs similar            |rating_diff| <= T and non-draw
SOLID       draw vs similar                |rating_diff| <= T and draw
            expected win vs weaker         rating_diff >  T  and win
            solid draw vs stronger         rating_diff < -T  and draw
PASSIVE     missed win vs weaker           rating_diff >  T  and draw
            unexpected loss vs weaker      rating_diff >  T  and loss
            expected loss vs stronger      rating_diff < -T  and loss

Priority when cases overlap: PASSIVE > SOLID > AGGRESSIVE
(e.g. loss as underdog is PASSIVE, not AGGRESSIVE "upset loss")

Outputs: data/processed/features_labeled.csv
"""

import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── configurable threshold ─────────────────────────────────────────────────────
THRESHOLD = 100   # rating points separating "similar" from "stronger/weaker"

# ── paths ──────────────────────────────────────────────────────────────────────
DB_PATH   = Path("data/interim/chess.db")
IN_PATH   = Path("data/processed/features.csv")
OUT_PATH  = Path("data/processed/features_labeled.csv")


# ── load player-perspective results from DB ────────────────────────────────────

def load_results(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Return a DataFrame with one row per (tournament_id, round_number,
    player_tournament_id) containing the player's result: 'win', 'draw', 'loss'.
    Only real games (both player IDs non-null) are included.
    """
    games = pd.read_sql(
        """
        SELECT tournament_id, round_number,
               white_player_tournament_id AS white_ptid,
               black_player_tournament_id AS black_ptid,
               result
        FROM games
        WHERE white_player_tournament_id IS NOT NULL
          AND black_player_tournament_id IS NOT NULL
        """,
        conn,
    )

    RESULT_WHITE = {"1-0": "win",   "1/2-1/2": "draw", "0-1": "loss"}
    RESULT_BLACK = {"1-0": "loss",  "1/2-1/2": "draw", "0-1": "win"}

    white = games[["tournament_id", "round_number", "white_ptid", "result"]].copy()
    white = white.rename(columns={"white_ptid": "player_tournament_id"})
    white["player_result"] = white["result"].map(RESULT_WHITE)

    black = games[["tournament_id", "round_number", "black_ptid", "result"]].copy()
    black = black.rename(columns={"black_ptid": "player_tournament_id"})
    black["player_result"] = black["result"].map(RESULT_BLACK)

    combined = pd.concat(
        [white[["tournament_id", "round_number", "player_tournament_id", "player_result"]],
         black[["tournament_id", "round_number", "player_tournament_id", "player_result"]]],
        ignore_index=True,
    )
    return combined


# ── action labelling ───────────────────────────────────────────────────────────

def assign_action(rating_diff: float, result: str, threshold: int = THRESHOLD) -> str:
    """
    Return 'aggressive', 'solid', or 'passive' for one game row.

    rating_diff = player_rating - opponent_rating (positive → player is stronger)

    < -T  win  → aggressive  (upset)
    < -T  draw → solid       (held against stronger)
    < -T  loss → solid       (expected loss — neutral, not a strategic failure)
    ±T    win  → aggressive  (competitive fight, went for it)
    ±T    draw → solid       (balanced outcome)
    ±T    loss → aggressive  (competitive fight, went for it but lost)
    > T   win  → solid       (expected win)
    > T   draw → passive     (missed opportunity vs weaker)
    > T   loss → passive     (underperformed vs weaker)
    """
    T = threshold

    if result == "win":
        if rating_diff < -T:
            return "aggressive"   # upset win
        elif abs(rating_diff) <= T:
            return "aggressive"   # competitive win vs similar
        else:                      # rating_diff > T
            return "solid"        # expected win vs weaker

    elif result == "draw":
        if rating_diff < -T:
            return "solid"        # held against stronger
        elif abs(rating_diff) <= T:
            return "solid"        # balanced outcome vs similar
        else:                      # rating_diff > T
            return "passive"      # missed win vs weaker

    else:  # loss
        if rating_diff < -T:
            return "solid"        # expected loss — neutral given the pairing
        elif abs(rating_diff) <= T:
            return "aggressive"   # competitive fight vs similar, came up short
        else:                      # rating_diff > T
            return "passive"      # underperformed vs weaker


def label(df: pd.DataFrame, threshold: int = THRESHOLD) -> pd.DataFrame:
    df = df.copy()

    mask_valid = df["rating_diff"].notna() & df["player_result"].notna()

    df["action"] = pd.Series(pd.NA, index=df.index, dtype=object)
    df.loc[mask_valid, "action"] = df.loc[mask_valid].apply(
        lambda r: assign_action(r["rating_diff"], r["player_result"], threshold),
        axis=1,
    )
    # Rows with NaN rating_diff (unrated opponent) get NaN action — drop them.
    before = len(df)
    df = df.dropna(subset=["action"]).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        log.info("Dropped %d rows with missing rating_diff or result", dropped)

    # One-hot encode
    df["action_aggressive"] = (df["action"] == "aggressive").astype(int)
    df["action_solid"]      = (df["action"] == "solid").astype(int)
    df["action_passive"]    = (df["action"] == "passive").astype(int)

    return df


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("Loading features from %s …", IN_PATH)
    features = pd.read_csv(IN_PATH)
    log.info("  %d rows, %d columns", *features.shape)

    conn = sqlite3.connect(DB_PATH)

    # ── join game results ──────────────────────────────────────────────────
    log.info("Loading per-player results from DB …")
    results = load_results(conn)

    features = features.merge(
        results,
        on=["tournament_id", "round_number", "player_tournament_id"],
        how="left",
    )
    missing_result = features["player_result"].isna().sum()
    if missing_result:
        log.warning("%d rows missing game result after join", missing_result)

    # ── join final_performance_rating (training target) ────────────────────
    log.info("Joining final_performance_rating …")
    perf = pd.read_sql(
        "SELECT player_tournament_id, performance_rating AS final_performance_rating"
        " FROM players_in_tournament",
        conn,
    )
    conn.close()

    features = features.merge(perf, on="player_tournament_id", how="left")
    null_target = features["final_performance_rating"].isna().sum()
    if null_target:
        log.warning("%d rows have NULL final_performance_rating", null_target)

    # ── apply heuristic labels ─────────────────────────────────────────────
    log.info("Assigning action labels (threshold=%d) …", THRESHOLD)
    df = label(features, threshold=THRESHOLD)

    # ── save ───────────────────────────────────────────────────────────────
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    log.info("Saved %d rows × %d cols to %s", *df.shape, OUT_PATH)

    # ── 1. class distribution ──────────────────────────────────────────────
    print(f"\n── Action class distribution  (threshold={THRESHOLD}) ──────────")
    dist = df["action"].value_counts().rename("count")
    dist_pct = (dist / len(df) * 100).round(1).rename("pct%")
    print(pd.concat([dist, dist_pct], axis=1).to_string())

    # ── 2. rows that changed vs old logic ─────────────────────────────────
    # Old rule: rating_diff < -T and loss → passive  (only difference)
    T = THRESHOLD
    mask_changed = (
        df["rating_diff"].notna() &
        (df["rating_diff"] < -T) &
        (df["player_result"] == "loss")
    )
    changed = df[mask_changed].copy()
    changed["old_action"] = "passive"
    print(f"\n── Rows where label changed  (old → new): {mask_changed.sum()} total ──")
    cols_diff = [
        "player_tournament_id", "round_number",
        "player_rating", "opponent_rating", "rating_diff",
        "player_result", "old_action", "action",
    ]
    print(changed[cols_diff].head(10).to_string(index=False))

    # ── 3. final_performance_rating completeness ───────────────────────────
    n_total    = len(df)
    n_non_null = df["final_performance_rating"].notna().sum()
    print(f"\n── final_performance_rating ────────────────────────────────────")
    print(f"  non-null: {n_non_null}/{n_total}  "
          f"({'OK — all rows covered' if n_non_null == n_total else 'WARNING: some NULL'})")
    print(f"  range:    {df['final_performance_rating'].min():.0f} – "
          f"{df['final_performance_rating'].max():.0f}")
    print(f"  mean:     {df['final_performance_rating'].mean():.1f}")


if __name__ == "__main__":
    main()
