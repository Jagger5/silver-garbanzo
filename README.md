# NCAA Basketball Ensemble Model

This repository implements an ensemble model tailored for NCAA Division I men's basketball betting outcomes. It ships with a real 2024 sample dataset and feature engineering helpers that turn typical box score / efficiency inputs into model-ready signals, while retaining the flexible scikit-learn stacking pipeline from the earlier multi-sport version.

## Features
- **NCAA-specific loader** that ingests real game data (teams, scores, tempo, rest, efficiencies) and derives leakage-free signals such as efficiency gaps, tempo averages, and rest differences.
- **Bundled 2024 sample** (`data/ncaa_2024_sample.csv`) with real game results to let you train immediately.
- **Mixed-type preprocessing** with imputers, scaling, and one-hot encoding for team/conference identifiers.
- **Stacked ensemble** of logistic regression and random forest models with a calibrated logistic stacker.
- **Cross-validated evaluation** (overall and by sport, defaulting to `NCAA_MBB`).
- **CLI training script** that saves evaluation reports, model artifacts, and validation predictions.

## Getting started
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Train on synthetic data:
   ```bash
   python -m sports_betting.train --generate-sample --output-dir outputs
   ```
3. Train on the bundled NCAA 2024 sample dataset:
   ```bash
   python -m sports_betting.train --ncaa-sample --output-dir outputs/ncaa
   ```

4. Train on your own NCAA CSV (case-insensitive column support for `home_team`, `away_team`, `home_score`, `away_score`, `home_adj_em`, `away_adj_em`, `home_adj_tempo`, `away_adj_tempo`, `home_rest_days`, `away_rest_days`, `neutral_site`):
   ```bash
   python -m sports_betting.train --data path/to/games.csv --sports-column sport --output-dir outputs/custom
   ```

Artifacts created in the output directory:
- `ensemble_model.joblib` – serialized model (including preprocessing and stacking logic).
- `evaluation.md` – ROC-AUC summary overall and per sport.
- `validation_predictions.csv` – sport-level probabilities for the validation split.

## Project structure
- `sports_betting/data.py` – dataset loading, NCAA-specific feature engineering, and synthetic dataset generation.
- `sports_betting/models.py` – preprocessing, ensemble definition, and evaluation helpers.
- `sports_betting/train.py` – CLI wiring for training/evaluation and artifact export.

## Extending the model
- Add engineered features (e.g., market odds, injury indicators, travel distance) to your CSV; they will be picked up automatically if numerical.
- Override `TrainingConfig.categorical_features` to one-hot encode more columns such as home/away teams or surfaces.
- Customize base estimators in `EnsembleSportsModel` by passing your own pipelines when instantiating the class.
