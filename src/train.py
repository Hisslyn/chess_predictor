#!/usr/bin/env python3
"""
src/train.py

Trains and evaluates three models predicting final_performance_rating.

Split: 80/20 by tournament_id (no row-level leakage).

Models
──────
Baseline 0  – bracket-mean Rp (<1500, 1500-1800, 1800-2100, 2100+)
Baseline 1  – Elo expected-score formula: Rp ≈ avg_opp_rating + dp(expected_score_pct)
XGBoost     – all features + action one-hots, tuned via GroupKFold(5) CV

Outputs: models/xgb_rp_predictor.pkl, models/training_report.txt
"""

import logging
import textwrap
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from xgboost import XGBRegressor

from fide_utils import fide_dp as _dp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── paths ──────────────────────────────────────────────────────────────────────
DATA_PATH   = Path("data/processed/features_labeled.csv")
MODEL_PATH  = Path("models/xgb_rp_predictor.pkl")
REPORT_PATH = Path("models/training_report.txt")


# ── feature columns used by XGBoost ───────────────────────────────────────────
FEATURE_COLS = [
    "player_rating", "title_encoded", "seed_percentile",
    "tournament_avg_rating", "tournament_rating_std",
    "n_rounds_total", "field_size", "rating_gap_to_field_avg",
    "round_number", "rounds_remaining",
    "current_score", "expected_score_so_far", "score_delta",
    "avg_opponent_rating_so_far",   # NaN → 0 (round 1, no prior games)
    "color_balance",
    "opponent_rating", "rating_diff", "expected_score_this_game",
    "playing_white", "current_rank", "gap_to_leader",
    "opponent_current_score",
    "action_aggressive", "action_solid", "action_passive",
]

RATING_BRACKETS = [0, 1500, 1800, 2100, 9999]
BRACKET_LABELS  = ["<1500", "1500-1800", "1800-2100", "2100+"]


def _bracket(rating: float) -> str:
    for i in range(len(RATING_BRACKETS) - 1):
        if RATING_BRACKETS[i] <= rating < RATING_BRACKETS[i + 1]:
            return BRACKET_LABELS[i]
    return BRACKET_LABELS[-1]


# ── data loading & splitting ───────────────────────────────────────────────────

def load_and_split(path: Path, test_size: float = 0.2, random_state: int = 42):
    df = pd.read_csv(path)
    log.info("Loaded %d rows × %d cols from %s", *df.shape, path)

    # Impute NaN in avg_opponent_rating_so_far with 0 (no prior games played)
    df["avg_opponent_rating_so_far"] = df["avg_opponent_rating_so_far"].fillna(0.0)

    tournaments = sorted(df["tournament_id"].unique())
    rng = np.random.default_rng(random_state)
    shuffled = rng.permutation(tournaments)
    n_test = max(1, round(len(shuffled) * test_size))
    test_tids  = list(shuffled[-n_test:])
    train_tids = list(shuffled[:-n_test])

    train = df[df["tournament_id"].isin(train_tids)].copy()
    test  = df[df["tournament_id"].isin(test_tids)].copy()

    log.info(
        "Train: %d tournaments (%d rows)  |  Test: %d tournaments (%d rows)",
        len(train_tids), len(train), len(test_tids), len(test),
    )
    return df, train, test, train_tids, test_tids


# ── Baseline 0: rating-bracket mean Rp ────────────────────────────────────────

def fit_baseline0(train: pd.DataFrame) -> dict[str, float]:
    """Compute mean final_performance_rating per rating bracket from training data."""
    train = train.copy()
    train["bracket"] = train["player_rating"].apply(_bracket)
    bracket_means = (
        train.groupby("bracket")["final_performance_rating"].mean().to_dict()
    )
    # Fallback: global mean if a bracket has no training examples
    global_mean = train["final_performance_rating"].mean()
    for lbl in BRACKET_LABELS:
        bracket_means.setdefault(lbl, global_mean)
    log.info("Baseline 0 bracket means: %s",
             {k: round(v, 1) for k, v in bracket_means.items()})
    return bracket_means


def predict_baseline0(df: pd.DataFrame, bracket_means: dict[str, float]) -> np.ndarray:
    brackets = df["player_rating"].apply(_bracket)
    return brackets.map(bracket_means).values


# ── Baseline 1: Elo expected-score formula ────────────────────────────────────

def predict_baseline1(df: pd.DataFrame) -> np.ndarray:
    """
    For each player, compute predicted Rp = avg_opp_rating + dp(expected_pct).
    Uses expected_score_this_game (already in features) summed across all
    rounds, giving the same prediction for every row of that player.
    """
    player_preds = {}
    for ptid, grp in df.groupby("player_tournament_id"):
        total_exp   = grp["expected_score_this_game"].sum()
        n_rounds    = len(grp)
        avg_opp_rtg = grp["opponent_rating"].mean()
        exp_pct     = (total_exp / n_rounds * 100) if n_rounds > 0 else 50.0
        player_preds[ptid] = avg_opp_rtg + _dp(exp_pct)

    return df["player_tournament_id"].map(player_preds).values


# ── XGBoost with GroupKFold CV ────────────────────────────────────────────────

def train_xgboost(
    train: pd.DataFrame,
    n_cv_splits: int = 5,
    n_iter: int = 40,
    random_state: int = 42,
) -> XGBRegressor:

    X = train[FEATURE_COLS].values
    y = train["final_performance_rating"].values
    groups = train["tournament_id"].values

    n_groups = len(np.unique(groups))
    n_splits = min(n_cv_splits, n_groups)
    if n_splits < n_cv_splits:
        log.warning(
            "Only %d training tournaments — using %d-fold CV instead of %d",
            n_groups, n_splits, n_cv_splits,
        )

    cv = GroupKFold(n_splits=n_splits)

    param_dist = {
        "n_estimators":     randint(100, 501),
        "max_depth":        randint(3, 8),
        "learning_rate":    uniform(0.03, 0.17),
        "subsample":        uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "min_child_weight": randint(1, 8),
        "reg_alpha":        uniform(0.0, 1.0),
        "reg_lambda":       uniform(0.5, 2.0),
        "gamma":            uniform(0.0, 0.5),
    }

    base = XGBRegressor(
        objective="reg:squarederror",
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
        tree_method="hist",
        importance_type="gain",
    )

    search = RandomizedSearchCV(
        base,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="neg_mean_absolute_error",
        cv=cv,
        n_jobs=-1,
        random_state=random_state,
        refit=True,
        verbose=0,
    )

    log.info("Running RandomizedSearchCV  n_iter=%d  cv=%d-fold …", n_iter, n_splits)
    search.fit(X, y, groups=groups)

    best_cv_mae = -search.best_score_
    log.info("Best CV MAE: %.1f  params: %s", best_cv_mae,
             {k: round(v, 4) if isinstance(v, float) else v
              for k, v in search.best_params_.items()})

    return search.best_estimator_, search.best_params_, best_cv_mae


# ── evaluation ────────────────────────────────────────────────────────────────

def mae(y_true, y_pred) -> float:
    return mean_absolute_error(y_true, y_pred)


# ── report builder ────────────────────────────────────────────────────────────

def build_report(
    train_tids, test_tids,
    mae_b0, mae_b1, mae_xgb,
    best_cv_mae, best_params,
    feature_importances: pd.Series,
    bracket_means: dict,
    n_train: int, n_test: int,
    beats_both: bool,
) -> str:
    lines = []
    w = 72

    lines.append("=" * w)
    lines.append("CHESS PERFORMANCE RATING PREDICTOR — TRAINING REPORT")
    lines.append("=" * w)
    lines.append("")

    lines.append("TRAIN / TEST SPLIT")
    lines.append(f"  Train tournaments ({len(train_tids)}): "
                 + ", ".join(str(t) for t in sorted(train_tids)))
    lines.append(f"  Test  tournaments ({len(test_tids)}):  "
                 + ", ".join(str(t) for t in sorted(test_tids)))
    lines.append(f"  Train rows: {n_train}   Test rows: {n_test}")
    lines.append("")

    lines.append("MODEL PERFORMANCE (MAE — lower is better)")
    lines.append("-" * w)
    lines.append(f"  Baseline 0  bracket mean Rp      MAE = {mae_b0:>7.1f}")
    lines.append(f"  Baseline 1  Elo expected score   MAE = {mae_b1:>7.1f}")
    lines.append(f"  XGBoost     all features          MAE = {mae_xgb:>7.1f}  ← best CV MAE {best_cv_mae:.1f}")
    lines.append("")
    b0_delta = mae_b0  - mae_xgb
    b1_delta = mae_b1  - mae_xgb
    if beats_both:
        lines.append(f"  XGBoost beats Baseline 0 by {b0_delta:+.1f}  and Baseline 1 by {b1_delta:+.1f}")
    else:
        lines.append("  *** WARNING: XGBoost does NOT beat both baselines — see notes ***")
    lines.append("")

    lines.append("BASELINE 0 — RATING BRACKET MEANS (from training set)")
    lines.append("-" * w)
    for bracket, mean_rp in bracket_means.items():
        lines.append(f"  {bracket:<12s}  mean Rp = {mean_rp:.1f}")
    lines.append("")

    lines.append("XGBOOST BEST HYPERPARAMETERS")
    lines.append("-" * w)
    for k, v in sorted(best_params.items()):
        vstr = f"{v:.4f}" if isinstance(v, float) else str(v)
        lines.append(f"  {k:<22s} = {vstr}")
    lines.append("")

    lines.append("XGBOOST TOP-15 FEATURE IMPORTANCES (gain)")
    lines.append("-" * w)
    for rank, (feat, imp) in enumerate(feature_importances.items(), 1):
        lines.append(f"  {rank:>2}. {feat:<35s}  {imp:.4f}")
    lines.append("")

    lines.append("=" * w)
    return "\n".join(lines)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ── load & split ──────────────────────────────────────────────────────
    df, train, test, train_tids, test_tids = load_and_split(DATA_PATH)

    y_test = test["final_performance_rating"].values

    # ── Baseline 0 ────────────────────────────────────────────────────────
    log.info("Fitting Baseline 0 (bracket mean Rp) …")
    bracket_means = fit_baseline0(train)
    pred_b0 = predict_baseline0(test, bracket_means)
    mae_b0  = mae(y_test, pred_b0)
    log.info("Baseline 0 MAE: %.1f", mae_b0)

    # ── Baseline 1 ────────────────────────────────────────────────────────
    log.info("Fitting Baseline 1 (Elo expected score → Rp) …")
    pred_b1 = predict_baseline1(test)
    mae_b1  = mae(y_test, pred_b1)
    log.info("Baseline 1 MAE: %.1f", mae_b1)

    # ── XGBoost ───────────────────────────────────────────────────────────
    best_model, best_params, best_cv_mae = train_xgboost(train)

    X_test  = test[FEATURE_COLS].values
    pred_xgb = best_model.predict(X_test)
    mae_xgb  = mae(y_test, pred_xgb)
    log.info("XGBoost test MAE: %.1f", mae_xgb)

    # ── save model ────────────────────────────────────────────────────────
    joblib.dump(best_model, MODEL_PATH)
    log.info("Model saved to %s", MODEL_PATH)

    # ── feature importances (gain) ────────────────────────────────────────
    importances = pd.Series(
        best_model.feature_importances_, index=FEATURE_COLS
    ).sort_values(ascending=False)
    top15 = importances.head(15)

    # ── build and save report ─────────────────────────────────────────────
    beats_both = mae_xgb < mae_b0 and mae_xgb < mae_b1
    report = build_report(
        train_tids=train_tids, test_tids=test_tids,
        mae_b0=mae_b0, mae_b1=mae_b1, mae_xgb=mae_xgb,
        best_cv_mae=best_cv_mae, best_params=best_params,
        feature_importances=top15,
        bracket_means=bracket_means,
        n_train=len(train), n_test=len(test),
        beats_both=beats_both,
    )

    REPORT_PATH.write_text(report)
    log.info("Report saved to %s", REPORT_PATH)

    print("\n" + report)

    # ── investigation if XGBoost doesn't beat baselines ───────────────────
    if not beats_both:
        print("\n── INVESTIGATION: per-tournament test MAE breakdown ─────────")
        for tid in test_tids:
            mask = test["tournament_id"] == tid
            y_t  = test.loc[mask, "final_performance_rating"].values
            p_b0 = pred_b0[mask.values]
            p_b1 = pred_b1[mask.values]
            p_xg = pred_xgb[mask.values]
            print(f"  tnr{tid}  n={mask.sum():4d}  "
                  f"B0={mae(y_t,p_b0):.1f}  B1={mae(y_t,p_b1):.1f}  XGB={mae(y_t,p_xg):.1f}")
        print()
        print("  Possible causes:")
        print(f"  - Only {len(test_tids)} test tournaments → high variance in MAE estimate")
        print("  - XGBoost overfit to train tournaments (try more regularisation)")
        print("  - Baselines strong when test-tournament Rp distribution matches train brackets")


if __name__ == "__main__":
    main()
