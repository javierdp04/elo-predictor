# ELO Predictor

Predicts a chess player's Elo rating from how they played a single game.

The model uses Lichess games that include Stockfish evaluations (`[%eval]`). For each
player in a game it measures the quality of their moves (how much winning chance they
gave away, missed forced mates, …) and trains a regressor to estimate that player's Elo.

## Project structure

```
elo_predictor/
├── data/
│   └── transform_data.py        # Lichess PGN -> games/moves CSV converter
├── src/
│   ├── data_cleaning/main.py    # Filters games and moves
│   ├── data_transforming/main.py# Builds per-player features
│   ├── EDA/
│   │   ├── precleaning.ipynb    # Exploration of the raw CSVs
│   │   ├── postcleaning.ipynb   # Cleaning + feature building -> features.csv
│   │   └── helpers.py           # Outlier check, Spearman correlations
│   ├── model_testing.ipynb      # Model comparison + hyperparameter search
│   └── main.py                  # Loads features.csv and makes the train/test split
└── requirements.txt
```

Large data files (`*.pgn`, `*.pgn.zst`, `*.csv` in `data/`) are git-ignored.

## Pipeline

### 1. PGN to CSV (`data/transform_data.py`)

This step converts a Lichess monthly dump (by default
`lichess_db_standard_rated_2018-06.pgn`) into two files:

- `<name>_games.csv`: one row per game (players, result, Elo, time control, termination, number of plies, …).
- `<name>_moves.csv`: one row per half-move (SAN/UCI, eval in pawns from White's view, mate-in-N, clock, FEN).

The converter keeps only games with engine evaluations and parses them in parallel with `python-chess`.

```bash
python data/transform_data.py [path.pgn] [--out-dir DIR] [--limit N] [--workers N] [--batch-size N]
```

### 2. Cleaning (`src/data_cleaning/main.py`)

`clean_games_data` keeps only:
- *Rated Rapid* games;
- games where White and Black are different players;
- games with between 10 and 200 plies (exclusive);
- games that did not end by *Abandoned* or *Rules infraction*;
- games where both players' rating changed by at most ±40 points.

`clean_moves_data` keeps the moves of the remaining games and converts `eval` to centipawns.
Unused columns are dropped at the end.

### 3. Feature engineering (`src/data_transforming/main.py`)

`build_features_dataframe(moves, games)` returns **one row per player per game** with these columns:

| Feature | Description |
|---|---|
| `num_plies`, `base_time`, `increment_per_move` | Game length and time control |
| `termination_normal`, `termination_time` | How the game ended |
| `is_white` | Player's colour |
| `result_win`, `result_loss`, `result_draw` | Result from the player's point of view |
| `avg_win%_loss`, `highest_win%_loss` | Mean and maximum drop in winning probability caused by the player's moves (eval clipped at ±800 cp, converted to win% with a logistic curve) |
| `num_unseen_checkmates` | Times the player had a forced mate in ≤10 and lost it |
| `time_spent_1_to_5_moves`, `time_spent_6_to_10_moves` | Clock time the player used on moves 1–5 and 6–10 |
| `opp_*` | The same move-quality and time features for the opponent |
| `elo` | **Target**: the player's rating |

Steps 2 and 3 run from `src/EDA/postcleaning.ipynb`. The notebook reads the raw CSVs in
chunks, writes `cleaned_games.csv` and `cleaned_moves.csv`, and saves the final
`data/features.csv`.

### 4. Model (`src/model_testing.ipynb`)

- Split: 80/20 hold-out with `GroupShuffleSplit`, grouped by `game_id` so both players of a
  game stay in the same split (also done in `src/main.py`).
- Each model is a scikit-learn `Pipeline` that drops `game_id` before the regressor.
- Hyperparameters are tuned with `GridSearchCV` / `RandomizedSearchCV` over 5-fold
  `GroupKFold`, scored with **MAE**, and compared with `DummyRegressor` (mean/median) baselines.

## Results

| Model | CV MAE (Elo) |
|---|---|
| Dummy (mean) | 235.2825 |
| Dummy (median) | 235.2446 |
| Random Forest Regressor | 195.9247 |
| Linear Regressor (SVD) | 209.8862 |
| Linear Regressor (SGD) | 209.5262 |
| Hist gradient boosting regressor | 193.3666 |

| Model | TEST MAE (Elo) |
|---|---|
| Hist gradient boosting regressor | 193.3757 |


## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Download the source data from the [Lichess open database](https://database.lichess.org/)
(standard rated, 2018-06) and decompress it into `data/`:

```bash
zstd -d data/lichess_db_standard_rated_2018-06.pgn.zst
```

To try the pipeline quickly without the full dump, use the `--limit` option.
