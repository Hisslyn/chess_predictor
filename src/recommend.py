#!/usr/bin/env python3
"""
src/recommend.py

Given current game context, predicts final performance rating (Rp) under
each of the three strategic actions (aggressive / solid / passive) and
recommends the action that maximises Rp.

Usage
─────
from src.recommend import recommend
result = recommend(player_rating=2200, opponent_rating=2000, ...)
"""

from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
import shap

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────────
_MODEL_PATH = Path("models/xgb_rp_predictor.pkl")

# ── feature column order — must match train.py FEATURE_COLS exactly ────────────
FEATURE_COLS = [
    "player_rating", "title_encoded", "seed_percentile",
    "tournament_avg_rating", "tournament_rating_std",
    "n_rounds_total", "field_size", "rating_gap_to_field_avg",
    "round_number", "rounds_remaining",
    "current_score", "expected_score_so_far", "score_delta",
    "avg_opponent_rating_so_far",
    "color_balance",
    "opponent_rating", "rating_diff", "expected_score_this_game",
    "playing_white", "current_rank", "gap_to_leader",
    "opponent_current_score",
    "action_aggressive", "action_solid", "action_passive",
]

# Action index map — row 0 = aggressive, 1 = solid, 2 = passive
_ACTIONS = ["aggressive", "solid", "passive"]
_ACTION_ONEHOTS = {
    "aggressive": (1, 0, 0),
    "solid":      (0, 1, 0),
    "passive":    (0, 0, 1),
}

# ── lazy-loaded globals ────────────────────────────────────────────────────────
_model      = None
_explainer  = None


def _load() -> None:
    global _model, _explainer
    if _model is None:
        _model     = joblib.load(_MODEL_PATH)
        _explainer = shap.TreeExplainer(_model)


# ── helpers ───────────────────────────────────────────────────────────────────

def _elo_expected(player_rtg: float, opp_rtg: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((opp_rtg - player_rtg) / 400.0))


def _seed_percentile_est(player_rating: float,
                         tournament_avg_rating: float,
                         tournament_rating_std: float) -> float:
    """Approximate seed_percentile via normal CDF: fraction of field rated below."""
    from scipy.stats import norm
    if tournament_rating_std <= 0:
        return 0.5
    return float(norm.cdf((player_rating - tournament_avg_rating) / tournament_rating_std))


def _estimate_current_rank(current_score: float, gap_to_leader: float,
                            field_size: int, round_num: int) -> int:
    """
    Rough rank estimate from current standing.
    gap_to_leader = current_score - leader_score  (0 if leading, negative if behind).
    """
    if gap_to_leader >= 0:
        return 1
    pts_behind = -gap_to_leader
    rounds_done = max(round_num - 1, 1)
    # Fraction of field expected to be ahead: proportional to pts_behind / rounds_done
    frac_ahead  = min(0.99, pts_behind / rounds_done * 0.5)
    return max(1, int(round(1 + frac_ahead * field_size)))


# ── public API ────────────────────────────────────────────────────────────────

def recommend(
    player_rating: float,
    opponent_rating: float,
    round_num: int,
    n_rounds: int,
    current_score: float,
    gap_to_leader: float,
    playing_white: int,
    tournament_avg_rating: float,
    tournament_rating_std: float,
    field_size: int,
    title_encoded: int = 0,
    avg_opponent_rating_so_far: float | None = None,
    opponent_current_score: float | None = None,
    seed_number: int | None = None,
    prior_opponent_ratings: list[float] | None = None,
    color_balance: int = 0,
) -> dict:
    """
    Recommend a strategic action for the current round.

    Parameters
    ──────────
    player_rating              : player's current FIDE rating
    opponent_rating            : opponent's FIDE rating
    round_num                  : current round number (1-indexed)
    n_rounds                   : total rounds in tournament
    current_score              : player's score entering this round (e.g. 2.5)
    gap_to_leader              : current_score − leader_score (0 = leading, negative = behind)
    playing_white              : 1 if player has white, 0 if black
    tournament_avg_rating      : mean starting rating of all players
    tournament_rating_std      : std deviation of starting ratings
    field_size                 : number of players in tournament
    title_encoded              : 0=none, 1=NM, 2=WCM, 3=CM/WFM, 4=FM/WIM, 5=IM/WGM, 6=GM
    avg_opponent_rating_so_far : mean rating of opponents in prior rounds (None → 0.0)
    opponent_current_score     : opponent's score entering this round (None → estimated)
    seed_number                : player's seed number (1 = top seed); when provided,
                                 seed_percentile matches build_features.py exactly
    prior_opponent_ratings     : list of opponent ratings from all prior rounds; when
                                 provided, expected_score_so_far is the exact per-game
                                 Elo sum used in training
    color_balance              : cumulative color balance from prior rounds
                                 (+1 per white game, −1 per black game)

    Returns
    ───────
    dict with keys: aggressive, solid, passive (each with predicted_rp and delta_vs_best),
                    recommended, confidence, explanation
    """
    _load()

    # ── derive features ───────────────────────────────────────────────────
    rating_diff              = player_rating - opponent_rating
    expected_score_this_game = _elo_expected(player_rating, opponent_rating)
    rounds_remaining         = n_rounds - round_num
    rating_gap_to_field_avg  = player_rating - tournament_avg_rating

    # seed_percentile: use exact rank formula from build_features.py when seed_number
    # is known; fall back to normal-CDF approximation otherwise
    if seed_number is not None:
        seed_pct = (field_size - seed_number) / max(field_size - 1, 1)
    else:
        seed_pct = _seed_percentile_est(player_rating, tournament_avg_rating, tournament_rating_std)

    # avg_opponent_rating_so_far: round 1 → 0.0 (matches training imputation)
    if avg_opponent_rating_so_far is None:
        avg_opponent_rating_so_far = 0.0

    # expected_score_so_far: sum individual Elo expectations per prior game,
    # matching build_features.py exactly when prior_opponent_ratings is supplied
    rounds_done = round_num - 1
    if prior_opponent_ratings is not None:
        expected_score_so_far = sum(_elo_expected(player_rating, r) for r in prior_opponent_ratings)
    else:
        exp_per_round = _elo_expected(player_rating, avg_opponent_rating_so_far) \
                        if avg_opponent_rating_so_far > 0 else 0.5
        expected_score_so_far = rounds_done * exp_per_round
    score_delta = current_score - expected_score_so_far

    # current_rank estimate
    current_rank = _estimate_current_rank(current_score, gap_to_leader,
                                           field_size, round_num)

    # opponent_current_score: estimate as elo-weighted expected score over prior rounds
    if opponent_current_score is None:
        opp_exp = _elo_expected(opponent_rating, tournament_avg_rating)
        opponent_current_score = rounds_done * opp_exp

    # ── build 3 feature rows (one per action) ─────────────────────────────
    base = [
        player_rating, title_encoded, seed_pct,
        tournament_avg_rating, tournament_rating_std,
        n_rounds, field_size, rating_gap_to_field_avg,
        round_num, rounds_remaining,
        current_score, expected_score_so_far, score_delta,
        avg_opponent_rating_so_far, color_balance,
        opponent_rating, rating_diff, expected_score_this_game,
        playing_white, current_rank, gap_to_leader, opponent_current_score,
    ]

    rows = np.array([
        base + list(_ACTION_ONEHOTS[a]) for a in _ACTIONS
    ], dtype=np.float64)

    # ── predict ───────────────────────────────────────────────────────────
    preds = _model.predict(rows)

    best_idx = int(np.argmax(preds))
    best_rp  = float(preds[best_idx])

    rp_by_action = {a: float(preds[i]) for i, a in enumerate(_ACTIONS)}
    delta_by_action = {a: round(rp_by_action[a] - best_rp, 1) for a in _ACTIONS}

    # ── confidence ────────────────────────────────────────────────────────
    sorted_preds = sorted(preds, reverse=True)
    margin       = sorted_preds[0] - sorted_preds[1]
    confidence   = "strong" if margin >= 15.0 else "marginal"

    # ── SHAP explanation for the recommended action ────────────────────────
    best_row        = rows[[best_idx]]
    shap_vals       = _explainer.shap_values(best_row)[0]   # shape (n_features,)
    top3_idx        = np.argsort(np.abs(shap_vals))[::-1][:3]

    explanation_parts = []
    for i in top3_idx:
        fname = FEATURE_COLS[i]
        sval  = shap_vals[i]
        fval  = best_row[0, i]
        sign  = "+" if sval >= 0 else ""
        explanation_parts.append(f"{fname} {fval:.2f} (shap {sign}{sval:.1f})")

    explanation = (
        f"Recommended {_ACTIONS[best_idx].upper()} — top 3 drivers: "
        + ", ".join(explanation_parts)
    )

    # ── assemble result ───────────────────────────────────────────────────
    return {
        "aggressive": {
            "predicted_rp":   round(rp_by_action["aggressive"], 1),
            "delta_vs_best":  delta_by_action["aggressive"],
        },
        "solid": {
            "predicted_rp":   round(rp_by_action["solid"], 1),
            "delta_vs_best":  delta_by_action["solid"],
        },
        "passive": {
            "predicted_rp":   round(rp_by_action["passive"], 1),
            "delta_vs_best":  delta_by_action["passive"],
        },
        "recommended": _ACTIONS[best_idx],
        "confidence":  confidence,
        "explanation": explanation,
        "score_delta": round(score_delta, 4),
    }


def _pretty_print(scenario: str, kwargs: dict, result: dict) -> str:
    """Format the result as a readable narrative summary."""
    pr   = kwargs["player_rating"]
    opr  = kwargs["opponent_rating"]
    rnd  = kwargs["round_num"]
    nr   = kwargs["n_rounds"]
    sc   = kwargs["current_score"]
    gap  = kwargs["gap_to_leader"]
    pw   = kwargs["playing_white"]
    color = "white" if pw else "black"

    rec  = result["recommended"].upper()
    conf = result["confidence"]
    exp  = result["explanation"]

    rp_s  = result["solid"]["predicted_rp"]
    rp_a  = result["aggressive"]["predicted_rp"]
    rp_p  = result["passive"]["predicted_rp"]
    rd    = pr - opr

    gap_str = f"leading" if gap >= 0 else f"{gap:+.1f} vs leader"

    lines = [
        f"You're {pr:.0f}, round {rnd} of {nr}, {sc}/{rnd-1} pts, "
        f"vs {opr:.0f} with {color} ({gap_str}).",
        f"Recommendation: {rec}. "
        f"Predicted Rp: solid={rp_s:.0f}, aggressive={rp_a:.0f}, passive={rp_p:.0f}.",
        f"Confidence: {conf}.",
        f"Key factors: score_delta {result.get('score_delta', 'n/a')}, "
        f"gap_to_leader {gap:+.1f}, rating_diff {rd:+.0f}.",
        f"[{exp}]",
    ]
    return f"\n=== {scenario} ===\n" + "\n".join(lines)


if __name__ == "__main__":
    # Quick smoke test
    r = recommend(
        player_rating=2200, opponent_rating=2000,
        round_num=7, n_rounds=7,
        current_score=3.0, gap_to_leader=-2.5,
        playing_white=1,
        tournament_avg_rating=2000, tournament_rating_std=200, field_size=80,
    )
    print(r)
