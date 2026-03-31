# LALIGA Shot Analysis

End-to-end analysis of LALIGA shot events with exploratory analysis, player archetype clustering, and machine-learning models to predict goal probability.

## Project Summary

This project uses historical LALIGA shot data to:

1. Explore shot outcomes and match-context patterns.
2. Engineer football-specific features from shot coordinates and game state.
3. Cluster players into shooting archetypes.
4. Train and compare classification models for goal prediction.
5. Generate `submission.csv` with predicted `goal_prob` for the test set.

## Data

The repository includes:

- `laliga_shots_train.csv`: 72,116 rows, 19 columns (includes `result` target).
- `laliga_shots_test_no_result.csv`: 18,029 rows, 18 columns (no `result`).

Important columns include:

- Spatial: `X`, `Y`
- Context: `minute`, `h_a`, `situation`, `shotType`, `h_goals`, `a_goals`
- Entities: `player`, `player_id`, `h_team`, `a_team`
- Sequence/context: `player_assisted`, `lastAction`

## Workflow

All analysis is in [`LALIGA_Analysis_Challenge.ipynb`](./LALIGA_Analysis_Challenge.ipynb):

1. Data loading and quality checks: Missing-value assessment and dataset profiling.
2. Exploratory data analysis (EDA): Shot-result distribution, categorical breakdowns, and location analysis.
3. Feature engineering: `distance_to_goal`, `angle`, `score_diff`, `is_penalty`, `is_goal`.
4. Player clustering: K-Means on player-level aggregates to identify shooting archetypes.
5. Predictive modeling: Logistic Regression (baseline), Random Forest, and Tuned XGBoost (`GridSearchCV`).
6. Model comparison + diagnostics: Accuracy/Precision/Recall/F1/ROC-AUC, ROC curves, and confusion matrices.
7. Submission generation: Predicts `goal_prob` for test set and writes `submission.csv`.

## Model Performance

From the notebook evaluation:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7811 | 0.2971 | 0.7532 | 0.4261 | 0.8435 |
| Random Forest | 0.8053 | 0.3213 | 0.7237 | 0.4450 | 0.8424 |
| Tuned XGBoost | 0.8053 | 0.3236 | 0.7384 | 0.4500 | 0.8667 |

Best model: **Tuned XGBoost** (highest F1 and ROC-AUC).

Best XGBoost params:

- `n_estimators=450`
- `max_depth=6`
- `learning_rate=0.07`

## Key Insights

- Overall goal conversion rate is ~10.79%.
- Penalties have very high conversion (~75.67%) despite low frequency.
- Rebound-related context (`lastAction`) provides strong predictive signal.
- Engineered spatial features (`distance_to_goal`, `angle`) are the most influential predictors.

## Repository Structure

- [`LALIGA_Analysis_Challenge.ipynb`](./LALIGA_Analysis_Challenge.ipynb) - full analysis and modeling workflow
- [`laliga_shots_train.csv`](./laliga_shots_train.csv) - training data
- [`laliga_shots_test_no_result.csv`](./laliga_shots_test_no_result.csv) - test data
- [`submission.csv`](./submission.csv) - generated predictions
- [`requirements.txt`](./requirements.txt) - Python dependencies

## Setup

```bash
git clone https://github.com/NikhilaReddyKuntla/LALIGA-Shot-Analysis.git
cd LALIGA-Shot-Analysis
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install xgboost
```

Then run the notebook:

```bash
jupyter notebook LALIGA_Analysis_Challenge.ipynb
```

## Output

Running the full notebook creates `submission.csv` with two columns: `id` and `goal_prob`.
