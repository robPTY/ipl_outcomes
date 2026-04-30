from pathlib import Path

import pandas as pd


CSV_DIR = Path("csv_files")
DELIVERIES_IN = CSV_DIR / "deliveries_clean.csv"
BATTER_MATCH_OUT = CSV_DIR / "batter_match_clean.csv"


def first_non_null(series):
    # Keep the first real value when grouping match-level fields
    values = series.dropna()
    return values.iloc[0] if not values.empty else None


def build_batting_positions(df):
    # Use the first delivery each batter faced to estimate batting order
    first_seen = (
        df.groupby(["match_id", "innings_number", "batting_team", "batter"], as_index=False)
        .agg(first_delivery=("delivery_number", "min"))
        .sort_values(["match_id", "innings_number", "batting_team", "first_delivery", "batter"])
    )
    first_seen["batting_position"] = (
        first_seen.groupby(["match_id", "innings_number", "batting_team"]).cumcount() + 1
    )
    return first_seen[
        ["match_id", "innings_number", "batting_team", "batter", "batting_position"]
    ]


def build_batter_match(df):
    df = df.copy()

    # Create helper columns used by the groupby aggregation
    df["is_legal_ball"] = df["is_legal_ball"].astype(bool)
    df["batter_dismissed"] = df["batter_dismissed"].astype(bool)
    df["dismissal_kind"] = df["wicket_kind"].where(df["batter_dismissed"])
    df["dot_ball"] = df["is_legal_ball"] & (df["total_runs"] == 0)
    df["powerplay_legal_ball"] = df["is_legal_ball"] & (df["phase"] == "powerplay")
    df["middle_overs_legal_ball"] = df["is_legal_ball"] & (df["phase"] == "middle")
    df["death_overs_legal_ball"] = df["is_legal_ball"] & (df["phase"] == "death")

    # These columns identify one batter innings within a match
    group_cols = [
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
    ]

    # Collapse ball-by-ball rows into one row per batter per match
    batter_match = (
        df.groupby(group_cols, dropna=False)
        .agg(
            runs_scored=("batter_runs", "sum"),
            balls_faced=("is_legal_ball", "sum"),
            deliveries_seen=("delivery_number", "count"),
            fours=("batter_runs", lambda s: (s == 4).sum()),
            sixes=("batter_runs", lambda s: (s == 6).sum()),
            dots=("dot_ball", "sum"),
            dismissed=("batter_dismissed", "max"),
            dismissal_kind=("dismissal_kind", first_non_null),
            first_over_faced=("over", "min"),
            last_over_faced=("over", "max"),
            powerplay_balls=("powerplay_legal_ball", "sum"),
            middle_overs_balls=("middle_overs_legal_ball", "sum"),
            death_overs_balls=("death_overs_legal_ball", "sum"),
            toss_winner=("toss_winner", first_non_null),
            toss_decision=("toss_decision", first_non_null),
            match_winner=("match_winner", first_non_null),
            result=("result", first_non_null),
            method=("method", first_non_null),
        )
        .reset_index()
    )

    # Add batter outcome columns that are useful for modeling
    batter_match["strike_rate"] = (
        batter_match["runs_scored"] / batter_match["balls_faced"].replace(0, pd.NA) * 100
    )
    batter_match["strike_rate"] = pd.to_numeric(batter_match["strike_rate"]).round(2)
    batter_match["scored_20_plus"] = batter_match["runs_scored"] >= 20
    batter_match["scored_30_plus"] = batter_match["runs_scored"] >= 30
    batter_match["scored_50_plus"] = batter_match["runs_scored"] >= 50
    batter_match["out_for_duck"] = batter_match["dismissed"] & (batter_match["runs_scored"] == 0)
    batter_match["team_won"] = batter_match["batting_team"] == batter_match["match_winner"]

    # Add batting position after aggregation
    positions = build_batting_positions(df)
    batter_match = batter_match.merge(
        positions,
        on=["match_id", "innings_number", "batting_team", "batter"],
        how="left",
    )

    # Keep the output columns in a predictable order
    ordered_cols = [
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
        "runs_scored",
        "balls_faced",
        "strike_rate",
        "deliveries_seen",
        "fours",
        "sixes",
        "dots",
        "dismissed",
        "dismissal_kind",
        "out_for_duck",
        "scored_20_plus",
        "scored_30_plus",
        "scored_50_plus",
        "first_over_faced",
        "last_over_faced",
        "powerplay_balls",
        "middle_overs_balls",
        "death_overs_balls",
        "toss_winner",
        "toss_decision",
        "match_winner",
        "team_won",
        "result",
        "method",
    ]
    return batter_match[ordered_cols].sort_values(
        ["date", "match_id", "innings_number", "batting_position", "batter"]
    )


def main() -> int:
    # Read cleaned deliveries and write one row per batter per match
    CSV_DIR.mkdir(exist_ok=True)
    deliveries = pd.read_csv(DELIVERIES_IN, low_memory=False)
    batter_match = build_batter_match(deliveries)
    batter_match.to_csv(BATTER_MATCH_OUT, index=False)

    print("Batter-match aggregation complete")
    print(f"Batter-match rows: {len(batter_match):,}")
    print(f"Matches represented: {batter_match['match_id'].nunique():,}")
    print(f"Missing batter IDs: {batter_match['batter_id'].isna().sum():,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
