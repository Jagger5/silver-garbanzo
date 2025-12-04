"""Data utilities for training the ensemble sports betting model."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class TrainingConfig:
    """Configuration for training.

    Attributes:
        target: Name of the column containing the binary outcome to predict
            (1 if the home team covers/wins, 0 otherwise).
        sports_column: Optional column containing the sport name. This helps
            track performance across leagues and can be used for stratified
            splitting.
        categorical_features: List of feature columns treated as categorical.
        numerical_features: List of feature columns treated as numeric.
    """

    target: str = "target"
    sports_column: Optional[str] = "sport"
    categorical_features: List[str] = field(default_factory=list)
    numerical_features: List[str] = field(default_factory=list)

    def resolved_categoricals(self, columns: Sequence[str]) -> List[str]:
        if self.categorical_features:
            return self.categorical_features
        return [col for col in columns if col not in self.resolved_numericals(columns) and col != self.target]

    def resolved_numericals(self, columns: Sequence[str]) -> List[str]:
        if self.numerical_features:
            return self.numerical_features
        return [col for col in columns if col != self.target and col != self.sports_column]


def load_dataset(path: Path | str) -> pd.DataFrame:
    """Load a CSV dataset and standardize column names.

    The loader expects the dataset to contain a binary ``target`` column that
    marks whether the home team/outcome was successful. Numeric features can
    be any float/int columns. Categorical features can be strings such as team
    names or surface types.
    """

    frame = pd.read_csv(path)
    frame.columns = [col.strip() for col in frame.columns]
    return frame


def load_ncaa_basketball(path: Path | str) -> pd.DataFrame:
    """Load an NCAA men's basketball dataset and engineer betting-friendly features.

    Expected columns (case-insensitive, extra columns are preserved):
    - ``home_team`` / ``away_team``: Team names.
    - ``home_score`` / ``away_score``: Final scores (used to derive the binary target).
    - ``home_adj_em`` / ``away_adj_em``: Adjusted efficiency margin-style ratings.
    - ``home_adj_tempo`` / ``away_adj_tempo``: Adjusted tempos (possessions per 40).
    - ``home_rest_days`` / ``away_rest_days``: Days since each team's last game.
    - ``neutral_site``: 1 if played on a neutral court, else 0.

    If the CSV already includes a ``target`` column, it is used directly. Otherwise the
    target is derived from the game result (1 if the home team wins, else 0).
    A ``sport`` column is added (default value ``NCAA_MBB``) to make evaluation
    compatible with the multi-sport pipeline.
    """

    frame = load_dataset(path)

    # Normalize expected column names while keeping any user-provided extras.
    rename_map = {
        "home team": "home_team",
        "away team": "away_team",
        "home score": "home_score",
        "away score": "away_score",
        "home adj em": "home_adj_em",
        "away adj em": "away_adj_em",
        "home adj tempo": "home_adj_tempo",
        "away adj tempo": "away_adj_tempo",
        "home rest days": "home_rest_days",
        "away rest days": "away_rest_days",
    }
    lower_columns = {col.lower(): col for col in frame.columns}
    for normalized, target_name in rename_map.items():
        if normalized in lower_columns:
            frame = frame.rename(columns={lower_columns[normalized]: target_name})

    if "target" not in frame.columns and {"home_score", "away_score"}.issubset(frame.columns):
        frame["target"] = (frame["home_score"] > frame["away_score"]).astype(int)

    if "sport" not in frame.columns:
        frame["sport"] = "NCAA_MBB"

    # Derived signals that help the model without overwriting user-provided features.
    if {"home_adj_em", "away_adj_em"}.issubset(frame.columns):
        frame["efficiency_gap"] = frame["home_adj_em"] - frame["away_adj_em"]

    if {"home_adj_tempo", "away_adj_tempo"}.issubset(frame.columns):
        frame["tempo_gap"] = frame["home_adj_tempo"] - frame["away_adj_tempo"]
        frame["tempo_mean"] = (
            frame["home_adj_tempo"] + frame["away_adj_tempo"]
        ) / 2.0

    if {"home_rest_days", "away_rest_days"}.issubset(frame.columns):
        frame["rest_diff"] = frame["home_rest_days"] - frame["away_rest_days"]

    if "neutral_site" in frame.columns:
        frame["neutral_site"] = frame["neutral_site"].fillna(0).astype(int)

    required_columns = {"target", "sport"}
    if not required_columns.issubset(frame.columns):
        missing = required_columns - set(frame.columns)
        raise ValueError(f"Missing required columns for NCAA dataset: {missing}")

    return frame


def generate_synthetic_dataset(
    n_samples: int = 2500, random_state: int = 17, sports: Optional[Iterable[str]] = None
) -> pd.DataFrame:
    """Generate a synthetic multi-sport dataset for quick experimentation.

    The dataset includes:
    - ``sport``: The league/sport name (e.g., NBA, NFL, EPL).
    - ``home_elo`` and ``away_elo``: Synthetic Elo ratings.
    - ``pace``: Synthetic tempo indicator scaled by sport.
    - ``rest_days_home``/``rest_days_away``: Rest days since last game.
    - ``target``: Binary outcome indicating if the home side wins.
    """

    rng = np.random.default_rng(random_state)
    sports_list = list(sports) if sports else ["NBA", "NFL", "EPL", "MLB", "NHL"]
    sport_choices = rng.choice(sports_list, size=n_samples)

    pace_by_sport = {sport: idx + 1 for idx, sport in enumerate(sports_list)}

    base_home_elo = rng.normal(loc=1500, scale=120, size=n_samples)
    base_away_elo = rng.normal(loc=1500, scale=120, size=n_samples)
    pace = np.array([pace_by_sport[sport] for sport in sport_choices]) + rng.normal(scale=0.5, size=n_samples)
    rest_home = rng.integers(low=0, high=6, size=n_samples)
    rest_away = rng.integers(low=0, high=6, size=n_samples)

    elo_diff = base_home_elo - base_away_elo
    fatigue = 0.05 * (rest_away - rest_home)
    sport_bias = np.array([0.15 if sport in {"NBA", "EPL"} else -0.05 for sport in sport_choices])

    logits = 0.003 * elo_diff + fatigue + sport_bias + rng.normal(scale=0.5, size=n_samples)
    probs = 1 / (1 + np.exp(-logits))
    outcomes = rng.binomial(n=1, p=probs)

    frame = pd.DataFrame(
        {
            "sport": sport_choices,
            "home_elo": base_home_elo,
            "away_elo": base_away_elo,
            "pace": pace,
            "rest_days_home": rest_home,
            "rest_days_away": rest_away,
            "target": outcomes,
        }
    )

    return frame
