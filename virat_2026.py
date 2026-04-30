from pathlib import Path

import pandas as pd

from build_features import build_features
from model_utils import (
    OUTPUT_COLUMNS,
    FIFTY_PLUS_TARGET_COLUMN,
    RUNS_TARGET_COLUMN,
    build_nn_classifier,
    build_nn_regressor,
    build_tree_classifier,
    build_tree_regressor,
    build_tree_upside_regressor,
    feature_columns,
    print_metrics,
    print_probability_metrics,
    print_run_totals,
    printable_report,
)


CSV_DIR = Path("csv_files")
BATTER_MATCH_IN = CSV_DIR / "batter_match_clean.csv"
FEATURES_IN = CSV_DIR / "batter_match_features.csv"
PREDICTIONS_OUT = CSV_DIR / "virat_2026.csv"

# Import Virat info
PLAYER_NAME = "V Kohli"
PLAYER_ID = "ba607b88"
PREDICTION_SEASON = 2026

# Columns needed to make the hardcoded 2026 matches look like batter_match_clean
BATTER_MATCH_COLUMNS = [
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
    "runs_scored",
    "balls_faced",
    "deliveries_seen",
    "fours",
    "sixes",
    "dots",
    "dismissed",
    "dismissal_kind",
    "first_over_faced",
    "last_over_faced",
    "powerplay_balls",
    "middle_overs_balls",
    "death_overs_balls",
    "toss_winner",
    "toss_decision",
    "match_winner",
    "result",
    "method",
    "strike_rate",
    "scored_20_plus",
    "scored_30_plus",
    "scored_50_plus",
    "out_for_duck",
    "team_won",
    "batting_position",
]

# Actual Virat innings completed through Apr 30, 2026
VIRAT_2026_MATCHES = [
    {
        "match_id": 1527674,
        "date": "2026-03-28",
        "venue": "M Chinnaswamy Stadium, Bengaluru",
        "city": "Bengaluru",
        "innings_number": 2,
        "batting_team": "Royal Challengers Bengaluru",
        "bowling_team": "Sunrisers Hyderabad",
        "batting_position": 2,
        "toss_winner": "Royal Challengers Bengaluru",
        "toss_decision": "field",
        "runs_scored": 69,
        "balls_faced": 38,
        "fours": 5,
        "sixes": 5,
        "dismissed": False,
    },
    {
        "match_id": 1527684,
        "date": "2026-04-05",
        "venue": "M Chinnaswamy Stadium, Bengaluru",
        "city": "Bengaluru",
        "innings_number": 1,
        "batting_team": "Royal Challengers Bengaluru",
        "bowling_team": "Chennai Super Kings",
        "batting_position": 2,
        "toss_winner": "Chennai Super Kings",
        "toss_decision": "field",
        "runs_scored": 28,
        "balls_faced": 18,
        "fours": 2,
        "sixes": 1,
        "dismissed": True,
    },
    {
        "match_id": 1527689,
        "date": "2026-04-10",
        "venue": "Barsapara Cricket Stadium, Guwahati",
        "city": "Guwahati",
        "innings_number": 1,
        "batting_team": "Royal Challengers Bengaluru",
        "bowling_team": "Rajasthan Royals",
        "batting_position": 2,
        "toss_winner": "Rajasthan Royals",
        "toss_decision": "field",
        "runs_scored": 32,
        "balls_faced": 16,
        "fours": 7,
        "sixes": 0,
        "dismissed": True,
    },
    {
        "match_id": 1527693,
        "date": "2026-04-12",
        "venue": "Wankhede Stadium, Mumbai",
        "city": "Mumbai",
        "innings_number": 1,
        "batting_team": "Royal Challengers Bengaluru",
        "bowling_team": "Mumbai Indians",
        "batting_position": 2,
        "toss_winner": "Mumbai Indians",
        "toss_decision": "field",
        "runs_scored": 50,
        "balls_faced": 38,
        "fours": 5,
        "sixes": 1,
        "dismissed": True,
    },
    {
        "match_id": 1529266,
        "date": "2026-04-15",
        "venue": "M Chinnaswamy Stadium, Bengaluru",
        "city": "Bengaluru",
        "innings_number": 2,
        "batting_team": "Royal Challengers Bengaluru",
        "bowling_team": "Lucknow Super Giants",
        "batting_position": 2,
        "toss_winner": "Royal Challengers Bengaluru",
        "toss_decision": "field",
        "runs_scored": 49,
        "balls_faced": 34,
        "fours": 6,
        "sixes": 1,
        "dismissed": True,
    },
    {
        "match_id": 1529269,
        "date": "2026-04-18",
        "venue": "M Chinnaswamy Stadium, Bengaluru",
        "city": "Bengaluru",
        "innings_number": 1,
        "batting_team": "Royal Challengers Bengaluru",
        "bowling_team": "Delhi Capitals",
        "batting_position": 2,
        "toss_winner": "Delhi Capitals",
        "toss_decision": "field",
        "runs_scored": 19,
        "balls_faced": 13,
        "fours": 3,
        "sixes": 0,
        "dismissed": True,
    },
    {
        "match_id": 1529277,
        "date": "2026-04-24",
        "venue": "M Chinnaswamy Stadium, Bengaluru",
        "city": "Bengaluru",
        "innings_number": 2,
        "batting_team": "Royal Challengers Bengaluru",
        "bowling_team": "Gujarat Titans",
        "batting_position": 2,
        "toss_winner": "Royal Challengers Bengaluru",
        "toss_decision": "field",
        "runs_scored": 81,
        "balls_faced": 44,
        "fours": 8,
        "sixes": 4,
        "dismissed": True,
    },
    {
        "match_id": 1529282,
        "date": "2026-04-27",
        "venue": "Arun Jaitley Stadium, Delhi",
        "city": "Delhi",
        "innings_number": 2,
        "batting_team": "Royal Challengers Bengaluru",
        "bowling_team": "Delhi Capitals",
        "batting_position": 2,
        "toss_winner": "Royal Challengers Bengaluru",
        "toss_decision": "field",
        "runs_scored": 23,
        "balls_faced": 15,
        "fours": 1,
        "sixes": 2,
        "dismissed": False,
    },
    {
        "match_id": 1529285,
        "date": "2026-04-30",
        "venue": "Narendra Modi Stadium, Ahmedabad",
        "city": "Ahmedabad",
        "innings_number": 1,
        "batting_team": "Royal Challengers Bengaluru",
        "bowling_team": "Gujarat Titans",
        "batting_position": 2,
        "toss_winner": "Gujarat Titans",
        "toss_decision": "field",
        "runs_scored": 28,
        "balls_faced": 13,
        "fours": 5,
        "sixes": 1,
        "dismissed": True,
    },
]


def complete_match_row(row):
    # Fill in columns that are normally created from ball-by-ball data
    completed = {
        "season": PREDICTION_SEASON,
        "batter": PLAYER_NAME,
        "batter_id": PLAYER_ID,
        "deliveries_seen": row["balls_faced"],
        "dots": 0,
        "dismissal_kind": None if not row["dismissed"] else "unknown",
        "first_over_faced": None,
        "last_over_faced": None,
        "powerplay_balls": None,
        "middle_overs_balls": None,
        "death_overs_balls": None,
        "match_winner": None,
        "result": None,
        "method": None,
    }
    completed.update(row)

    # Add the same outcome fields used by build_features.py
    completed["strike_rate"] = round(completed["runs_scored"] / completed["balls_faced"] * 100, 2)
    completed["scored_20_plus"] = completed["runs_scored"] >= 20
    completed["scored_30_plus"] = completed["runs_scored"] >= 30
    completed["scored_50_plus"] = completed["runs_scored"] >= 50
    completed["out_for_duck"] = completed["dismissed"] and completed["runs_scored"] == 0
    completed["team_won"] = None
    return completed


def virat_2026_batter_match():
    # Convert the hardcoded 2026 matches into batter-match rows
    rows = [complete_match_row(row) for row in VIRAT_2026_MATCHES]
    return pd.DataFrame(rows, columns=BATTER_MATCH_COLUMNS)


def build_2026_features():
    # Combine Virat's old history with his 2026 matches, then rebuild features
    batter_match = pd.read_csv(BATTER_MATCH_IN)
    virat_history = batter_match[batter_match["batter"] == PLAYER_NAME].copy()
    combined = pd.concat([virat_history, virat_2026_batter_match()], ignore_index=True)
    features = build_features(combined)
    return features[features["season"] == PREDICTION_SEASON].copy()


def main() -> int:
    # Read training data and build Virat's 2026 feature rows
    CSV_DIR.mkdir(exist_ok=True)
    train = pd.read_csv(FEATURES_IN)
    player_matches = build_2026_features()

    # Split model inputs into numeric and categorical features
    features = feature_columns(train)
    numeric_features = [
        col for col in features if pd.api.types.is_numeric_dtype(train[col])
    ]
    categorical_features = [col for col in features if col not in numeric_features]

    # Build out 5 models
    tree_regressor = build_tree_regressor(numeric_features, categorical_features)
    tree_upside_regressor = build_tree_upside_regressor(numeric_features, categorical_features)
    nn_regressor = build_nn_regressor(numeric_features, categorical_features)
    tree_classifier = build_tree_classifier(numeric_features, categorical_features)
    nn_classifier = build_nn_classifier(numeric_features, categorical_features)
    
    # Fit the models on data through 2025
    tree_regressor.fit(train[features], train[RUNS_TARGET_COLUMN])
    tree_upside_regressor.fit(train[features], train[RUNS_TARGET_COLUMN])
    nn_regressor.fit(train[features], train[RUNS_TARGET_COLUMN])
    tree_classifier.fit(train[features], train[FIFTY_PLUS_TARGET_COLUMN].astype(bool))
    nn_classifier.fit(train[features], train[FIFTY_PLUS_TARGET_COLUMN].astype(bool))

    # Predict runs and 50-plus probability for each 2026 match
    results = player_matches.copy()
    results["tree_predicted_runs"] = tree_regressor.predict(results[features]).round(1)
    results["tree_upside_predicted_runs"] = tree_upside_regressor.predict(
        results[features]
    ).round(1)
    results["nn_predicted_runs"] = nn_regressor.predict(results[features]).round(1)
    results["tree_50_plus_probability"] = tree_classifier.predict_proba(results[features])[
        :, 1
    ].round(3)
    results["nn_50_plus_probability"] = nn_classifier.predict_proba(results[features])[
        :, 1
    ].round(3)
    results["actual_runs"] = results[RUNS_TARGET_COLUMN]
    results["actual_50_plus"] = results[FIFTY_PLUS_TARGET_COLUMN].astype(bool)
    results["tree_absolute_error"] = (
        results["tree_predicted_runs"] - results["actual_runs"]
    ).abs()
    results["nn_absolute_error"] = (
        results["nn_predicted_runs"] - results["actual_runs"]
    ).abs()

    report = results[OUTPUT_COLUMNS].copy()
    report.to_csv(PREDICTIONS_OUT, index=False)

    # Print summary metrics and a compact match-by-match report
    print(f"{PLAYER_NAME} {PREDICTION_SEASON} predictions through 2026-04-30")
    print(f"Training rows: {len(train):,}")
    print(f"Predicted matches: {len(report):,}")
    print(f"Saved predictions: {PREDICTIONS_OUT}")
    print()
    print_run_totals(report)
    print()
    print_metrics(
        "Tree-based run prediction metrics",
        report["actual_runs"],
        report["tree_predicted_runs"],
    )
    print()
    print_metrics(
        "Neural-network run prediction metrics",
        report["actual_runs"],
        report["nn_predicted_runs"],
    )
    print()
    print_probability_metrics(
        "Tree-based 50-plus probability metrics",
        report["actual_50_plus"],
        report["tree_50_plus_probability"],
    )
    print()
    print_probability_metrics(
        "Neural-network 50-plus probability metrics",
        report["actual_50_plus"],
        report["nn_50_plus_probability"],
    )
    print()
    print(printable_report(report).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
