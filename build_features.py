from pathlib import Path

import pandas as pd


CSV_DIR = Path("csv_files")
BATTER_MATCH_IN = CSV_DIR / "batter_match_clean.csv"
FEATURES_OUT = CSV_DIR / "batter_match_features.csv"

# Columns to predict
TARGET_COLUMNS = [
    "runs_scored",
    "balls_faced",
    "strike_rate",
    "deliveries_seen",
    "fours",
    "sixes",
    "dots",
    "dismissed",
    "out_for_duck",
    "scored_20_plus",
    "scored_30_plus",
    "scored_50_plus",
]

# Columns that are only known after the match ends
POST_MATCH_COLUMNS = {
    "first_over_faced",
    "last_over_faced",
    "powerplay_balls",
    "middle_overs_balls",
    "death_overs_balls",
    "match_winner",
    "team_won",
    "result",
    "method",
    "dismissal_kind",
}

# Columns that describe the match before it starts
CONTEXT_COLUMNS = [
    "match_id",
    "season",
    "date",
    "venue",
    "city",
    "innings_number",
    "batting_team",
    "bowling_team",
    "batter",
    "batter_id",
    "batting_position",
    "toss_winner",
    "toss_decision",
]


def safe_rate(numerator, denominator):
    return numerator / denominator.replace(0, pd.NA)


def shifted_expanding_mean(df, group_cols, value_col):
    # Average all previous rows in the group, not the current row
    return df.groupby(group_cols, dropna=False)[value_col].transform(
        lambda s: s.shift().expanding(min_periods=1).mean()
    )


def shifted_expanding_sum(df, group_cols, value_col):
    # Sum all previous rows in the group
    return df.groupby(group_cols, dropna=False)[value_col].transform(
        lambda s: s.shift().expanding(min_periods=1).sum()
    )


def shifted_rolling_mean(df, group_cols, value_col, window):
    # Average only the last few previous rows in the group
    return df.groupby(group_cols, dropna=False)[value_col].transform(
        lambda s: s.shift().rolling(window=window, min_periods=1).mean()
    )


def prior_count(df, group_cols):
    return df.groupby(group_cols, dropna=False).cumcount()


def add_player_features(df):
    # Build player-level career and recent-form features
    group_cols = ["batter_id"]

    df["player_matches_before"] = prior_count(df, group_cols)
    df["player_career_runs_avg"] = shifted_expanding_mean(df, group_cols, "runs_scored")
    df["player_career_balls_avg"] = shifted_expanding_mean(df, group_cols, "balls_faced")
    df["player_career_sr_avg"] = shifted_expanding_mean(df, group_cols, "strike_rate")
    df["player_career_dismissal_rate"] = shifted_expanding_mean(df, group_cols, "dismissed")
    df["player_career_20_plus_rate"] = shifted_expanding_mean(df, group_cols, "scored_20_plus")
    df["player_career_30_plus_rate"] = shifted_expanding_mean(df, group_cols, "scored_30_plus")
    df["player_career_50_plus_rate"] = shifted_expanding_mean(df, group_cols, "scored_50_plus")

    # Calculate career boundary rate using prior totals
    prior_fours = shifted_expanding_sum(df, group_cols, "fours")
    prior_sixes = shifted_expanding_sum(df, group_cols, "sixes")
    prior_balls = shifted_expanding_sum(df, group_cols, "balls_faced")
    df["player_career_boundary_per_ball"] = safe_rate(prior_fours + prior_sixes, prior_balls)

    for window in (3, 5, 10):
        # Build recent-form features over the player's last 3, 5, and 10 matches
        df[f"player_last_{window}_runs_avg"] = shifted_rolling_mean(
            df, group_cols, "runs_scored", window
        )
        df[f"player_last_{window}_sr_avg"] = shifted_rolling_mean(
            df, group_cols, "strike_rate", window
        )
        df[f"player_last_{window}_balls_avg"] = shifted_rolling_mean(
            df, group_cols, "balls_faced", window
        )
        df[f"player_last_{window}_30_plus_rate"] = shifted_rolling_mean(
            df, group_cols, "scored_30_plus", window
        )

    return df


def add_context_features(df, group_cols, prefix):
    # Build history for a specific context like venue, opponent, or batting position
    df[f"{prefix}_matches_before"] = prior_count(df, group_cols)
    df[f"{prefix}_runs_avg"] = shifted_expanding_mean(df, group_cols, "runs_scored")
    df[f"{prefix}_sr_avg"] = shifted_expanding_mean(df, group_cols, "strike_rate")
    df[f"{prefix}_balls_avg"] = shifted_expanding_mean(df, group_cols, "balls_faced")
    df[f"{prefix}_30_plus_rate"] = shifted_expanding_mean(df, group_cols, "scored_30_plus")
    return df


def build_features(batter_match):
    df = batter_match.copy()
    df["date"] = pd.to_datetime(df["date"])

    bool_cols = [
        "dismissed",
        "out_for_duck",
        "scored_20_plus",
        "scored_30_plus",
        "scored_50_plus",
    ]
    for col in bool_cols:
        df[col] = df[col].astype(bool)

    # Sort chronologically so shifted history really means previous matches
    df = df.sort_values(
        ["date", "match_id", "innings_number", "batting_position", "batter"]
    ).reset_index(drop=True)

    # Add career, recent-form, venue, opponent, team, and position history
    df = add_player_features(df)
    df = add_context_features(df, ["batter_id", "venue"], "player_venue")
    df = add_context_features(df, ["batter_id", "bowling_team"], "player_vs_opponent")
    df = add_context_features(df, ["batter_id", "batting_team"], "player_for_team")
    df = add_context_features(df, ["batter_id", "batting_position"], "player_position")

    # Keep pre-match feature columns and remove post-match leakage columns
    feature_cols = [
        col
        for col in df.columns
        if col not in TARGET_COLUMNS
        and col not in POST_MATCH_COLUMNS
        and not col.startswith("target_")
    ]
    output_cols = feature_cols + [f"target_{col}" for col in TARGET_COLUMNS]

    # Add target columns with a target_ prefix for model training
    for col in TARGET_COLUMNS:
        df[f"target_{col}"] = df[col]

    # Save dates in CSV friendly format
    features = df[output_cols].copy()
    features["date"] = features["date"].dt.strftime("%Y-%m-%d")
    return features


def main() -> int:
    # Read batter-match data and write the final model feature table
    CSV_DIR.mkdir(exist_ok=True)
    batter_match = pd.read_csv(BATTER_MATCH_IN)
    features = build_features(batter_match)
    features.to_csv(FEATURES_OUT, index=False)

    holdout_2025 = features[features["season"] == 2025]
    print("Feature generation complete")
    print(f"Feature rows: {len(features):,}")
    print(f"Feature columns: {features.shape[1]:,}")
    print(f"2025 holdout rows: {len(holdout_2025):,}")
    print(f"Rows with player history: {(features['player_matches_before'] > 0).sum():,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

