# Chess Round Strategy Advisor

> "What should I aim for in this game — attack, consolidate, or just make a draw?"
> This tool answers that question using data from real tournaments.

---

## What This Project Does

Imagine you're rated 2500, it's round 4 of 7, you're leading with 2.5 points out of 3, and you're about to face a 2580-rated opponent with the black pieces. Should you go all-out and fight for a win? Play carefully and protect your lead? Or just trade pieces and aim for a solid draw?

This tool looks at your exact situation — your rating, your opponent's rating, what round it is, how you're doing in the tournament so far, and whether you have white or black — and gives you a concrete recommendation:

> **Play SOLID.** (marginal confidence — margin: 1.4 pts)
> Predicted tournament performance with each approach:
> solid = 2510 · aggressive = 2509 · passive = 2481

The number shown (e.g. 2510) is your *predicted performance rating* — a measure of how strong you played across the whole tournament, based on who you beat and who beat you. The tool picks the strategy that is predicted to produce the highest performance rating by the end of the event.

The three strategies it considers:

| Strategy | What it means in practice |
|---|---|
| **Aggressive** | Go for the win. Accept imbalanced positions, complications, and risk. |
| **Solid** | Play principled chess. Look for advantages but avoid unnecessary risk. |
| **Passive** | Aim to not lose. Exchange pieces, simplify, make a draw if possible. |

---

## Why It Works

**Elo ratings** are chess's way of measuring playing strength. Every player has a number — higher means stronger. When you beat someone rated much higher than you, your rating goes up a lot. Lose to someone much lower and it drops significantly. The system is designed so that, on average, your results match what your rating predicts.

**Swiss-system tournaments** pair players with similar scores against each other each round, regardless of who they've played before. This means that if you're doing well, you'll face tougher opponents; if you're struggling, you'll face easier ones. Your final standing depends not just on how many points you score, but on how strong your opponents were.

**Performance rating** is a tournament-by-tournament snapshot of how you actually played, independent of your pre-event rating. It's calculated from your results and the ratings of the players you faced. A performance rating higher than your current rating means you had a great event; lower means an off day.

The model has learned from thousands of real tournament games that your performance rating is shaped by more than just your rating. It matters whether you're ahead or behind in the standings, how many rounds are left, how strong the field is, and — crucially — whether you played aggressively or passively in each individual game.

---

## How Accurate Is It

The model's predictions are off by about **94.3 rating points** on average when tested on tournaments it had never seen before.

That sounds like a lot — but compare it to simpler methods:

| Approach | Average error |
|---|---|
| "Just use your rating bracket's historical average" | 168.1 rating points |
| "Use Elo math to predict your score" | 144.7 rating points |
| **This model** | **94.3 rating points** |

The model reduces prediction error by about **73.8 points** versus the simplest baseline and **50.3 points** versus using Elo alone. That means it's genuinely learning something useful from the tournament context — your standing, your opponents, your strategy — beyond what your rating alone can tell you.

The model was tuned so that its cross-validated error (measured on held-out folds the model never trained on) was even tighter: **88.5 rating points**. The gap between that and the 94 on held-out tournaments is small, which tells us the model generalises well — training on 118 tournaments gave it enough variety to avoid the overfitting we saw in earlier runs.

---

## How to Use It

You need Python installed with the packages listed in `requirements.txt`. Then:

```python
from src.recommend import recommend

result = recommend(
    player_rating         = 2500,   # Your FIDE rating before the tournament
    opponent_rating       = 2580,   # Your opponent's FIDE rating
    round_num             = 4,      # Which round this is (counting from 1)
    n_rounds              = 7,      # Total number of rounds in the event
    current_score         = 2.5,    # Your score entering this round (wins + half-points)
    gap_to_leader         = 0.0,    # Your score minus the leader's score (0 = you are leading)
    playing_white         = 0,      # 1 if you have white pieces, 0 if black
    tournament_avg_rating = 2350,   # Average rating of all players in the event
    tournament_rating_std = 200,    # How spread out the ratings are (rough guide: 150–250 for most opens)
    field_size            = 60,     # Total number of players in the event
)

print(result["recommended"])                 # e.g. "solid"
print(result["solid"]["predicted_rp"])       # e.g. 2540.0
print(result["aggressive"]["predicted_rp"])  # e.g. 2510.0
print(result["passive"]["predicted_rp"])     # e.g. 2485.0
print(result["confidence"])                  # "strong" or "marginal"
print(result["explanation"])                 # plain-language summary of the key factors
```

**What each input means:**

- **player_rating** — Your rating at the start of the tournament, as published in the official pairing list.
- **opponent_rating** — Your opponent's published rating for this event.
- **round_num** — The round you're about to play (1, 2, 3, …).
- **n_rounds** — How many rounds the tournament has in total.
- **current_score** — Your total points so far, *before* this round starts. A win = 1, draw = 0.5, loss = 0.
- **gap_to_leader** — Your score minus the current leader's score. Zero means you are in first place (or tied for it). −1.0 means you are a full point behind the leader.
- **playing_white** — 1 if you have the white pieces this round, 0 if you have black.
- **tournament_avg_rating** — The average rating of all participants. Usually shown in the tournament header on chess-results.com.
- **tournament_rating_std** — How much ratings vary across the field. If you don't know it, 200 is a reasonable default for an open Swiss.
- **field_size** — The total number of players in your section.

**Optional inputs** (the model estimates these if you don't provide them):

- **title_encoded** — 0 for no title, up through 6 for Grandmaster. Defaults to 0.
- **avg_opponent_rating_so_far** — Average rating of opponents you've faced in prior rounds. Useful to fill in if you know it.
- **opponent_current_score** — Your opponent's score entering this round.

The result also includes a **confidence** field. "Strong" means the recommended strategy is predicted to outperform the next-best option by 15 or more rating points. "Marginal" means the difference is smaller and the choice is less clear-cut.

---

## What Data It Learned From

The model was built from **147 Swiss tournaments** scraped from [chess-results.com](https://www.chess-results.com) (150 were scraped; 3 were dropped during labeling for insufficient data), covering **9,098 players** and **35,728 games** across the 147 retained tournaments (before unrated-opponent filtering), yielding 66,594 player-round observations. Of those 147, **118 were used for training** and **29 were held out for testing**. The data includes tournaments from Germany, Argentina, India, Turkey, Spain, the United Kingdom, and beyond, spanning a wide range of rating levels and event sizes.

For each player in each round, the data records their rating, their opponent's rating, their current score, their standing in the event, what color they had, and — crucially — whether their play in that round was closer to aggressive, solid, or passive (determined by how their actual result compared to what Elo math would predict).

The scraping was done automatically: the program searched chess-results.com for Swiss-system FIDE-rated events, downloaded the crosstables and standings, and stored everything in a local database.

---

## Limitations

- **Training coverage.** 147 tournaments is a meaningful dataset, but it is not exhaustive. Patterns learned from these events may not fully represent every time control, country, or rating band.

- **The model suggests a *style*, not a *move*.** "Aggressive" means aiming for complex, fighting positions — it doesn't tell you which opening to play or how to handle a specific position. The actual chess is still up to you.

- **Chess is unpredictable.** Even the best-prepared player can have an off day, and a single blunder can swing the result regardless of strategy. The performance ratings predicted here are statistical estimates, not guarantees.

- **The strategy labels are approximate.** Whether a game was "aggressive" or "passive" was inferred from how results compared to Elo expectations, not from actual move-by-move analysis. A lucky win in a drawn position might be labeled aggressive when it wasn't really.

- **Only covers Swiss tournaments.** Round-robin events, team leagues, or online tournaments have different dynamics and are not represented in the training data.

- **Prediction error is real.** The model is off by around 94 rating points on average in testing. Use the recommendation as one input to your thinking, not as a definitive answer.

---

## Project Structure

| Location | What it contains |
|---|---|
| `src/scrape_discovery.py` | Finds Swiss FIDE-rated tournaments on chess-results.com |
| `src/scrape_tournaments.py` | Downloads the actual tournament pages |
| `src/parse.py` | Reads the downloaded pages and stores results in a database |
| `src/validate.py` | Checks the data for missing ratings, unfinished players, etc. |
| `src/build_features.py` | Prepares the data for the model (calculates scores, standings, etc.) |
| `src/label.py` | Assigns aggressive / solid / passive labels to each game |
| `src/train.py` | Trains the XGBoost model and saves the accuracy report |
| `src/recommend.py` | The main tool — call `recommend()` here to get a strategy suggestion |
| `models/xgb_rp_predictor.pkl` | The trained model file |
| `models/training_report.txt` | Accuracy numbers and model details |
| `data/raw/` | Downloaded HTML pages from chess-results.com |
| `data/interim/` | Cleaned database and intermediate files |
| `data/processed/` | Final dataset used to train the model |
| `notebooks/` | Interactive analyses and result summaries |
| `run_pipeline.sh` | Runs pipeline steps 2–7 (scrape → train) after discovery is complete |
| `requirements.txt` | List of Python packages needed to run the project |

## Running the Pipeline

The pipeline has two stages. Run them in order:

```bash
# Step 1 — discover tournament IDs (writes data/interim/tournament_candidates.csv)
python src/scrape_discovery.py

# Steps 2–7 — scrape, parse, validate, build features, label, train
bash run_pipeline.sh
```

`run_pipeline.sh` requires `tournament_candidates.csv` to exist before it starts.
