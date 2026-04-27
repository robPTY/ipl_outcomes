from pathlib import Path

import pandas as pd
import yaml


DATA_DIR = Path("data")
MATCHES_OUT = Path("matches_clean.csv")
DELIVERIES_OUT = Path("deliveries_clean.csv")
PLAYER_MATCH_OUT = Path("player_match_clean.csv")
LEGACY_MASTER_OUT = Path("master_ipl_dataset.csv")

EXTRA_TYPES = ("wides", "noballs", "byes", "legbyes", "penalty")
NON_BATTER_DISMISSALS = {"retired hurt", "retired not out", "obstructing the field"}


def first_or_none(value):
    if isinstance(value, list):
        return value[0] if value else None
    return value


def normalize_date(value):
    if value is None:
        return None
    return str(value)


def season_from_info(info):
    season = info.get("season")
    if season is not None:
        return str(season)

    date = normalize_date(first_or_none(info.get("dates")))
    return date[:4] if date else None


def registry_id(registry, player_name):
    return registry.get(player_name)


def opponent_for(teams, team):
    opponents = [candidate for candidate in teams if candidate != team]
    return opponents[0] if len(opponents) == 1 else None


def parse_over_ball(over_ball):
    text = str(over_ball)
    if "." not in text:
        return int(text), None

    over_text, ball_text = text.split(".", 1)
    return int(over_text), int(ball_text)


def innings_phase(over_number):
    if over_number is None:
        return None
    if over_number < 6:
        return "powerplay"
    if over_number < 15:
        return "middle"
    return "death"


def wicket_values(ball_info):
    wickets = ball_info.get("wickets")
    if wickets is None:
        wicket = ball_info.get("wicket")
        wickets = [wicket] if wicket else []
    return wickets


def fielders_for(wicket):
    fielders = wicket.get("fielders", [])
    names = []
    for fielder in fielders:
        if isinstance(fielder, dict):
            names.append(fielder.get("name"))
        else:
            names.append(fielder)
    return ";".join(name for name in names if name)


def iter_innings(raw_innings):
    for index, inning in enumerate(raw_innings, start=1):
        if "team" in inning and "overs" in inning:
            yield index, f"{index} innings", inning
            continue

        for inning_name, inning_data in inning.items():
            yield index, inning_name, inning_data


def iter_deliveries(inning_data):
    if "overs" in inning_data:
        for over_data in inning_data.get("overs", []):
            over_number = over_data.get("over")
            for ball_index, ball_info in enumerate(over_data.get("deliveries", []), start=1):
                yield f"{over_number}.{ball_index}", ball_info
        return

    for delivery in inning_data.get("deliveries", []):
        for over_ball, ball_info in delivery.items():
            yield over_ball, ball_info


def build_match_row(match_id, info):
    dates = info.get("dates", [])
    teams = info.get("teams", [])
    toss = info.get("toss", {})
    outcome = info.get("outcome", {})
    outcome_by = outcome.get("by", {})

    return {
        "match_id": match_id,
        "season": season_from_info(info),
        "date": normalize_date(first_or_none(dates)),
        "competition": info.get("competition"),
        "match_type": info.get("match_type"),
        "gender": info.get("gender"),
        "city": info.get("city"),
        "venue": info.get("venue"),
        "team_1": teams[0] if len(teams) > 0 else None,
        "team_2": teams[1] if len(teams) > 1 else None,
        "toss_winner": toss.get("winner"),
        "toss_decision": toss.get("decision"),
        "match_winner": outcome.get("winner"),
        "result": outcome.get("result"),
        "win_by_runs": outcome_by.get("runs"),
        "win_by_wickets": outcome_by.get("wickets"),
        "method": outcome.get("method"),
        "eliminator": outcome.get("eliminator"),
        "overs": info.get("overs"),
        "balls_per_over": info.get("balls_per_over", 6),
        "neutral_venue": info.get("neutral_venue", 0),
        "player_of_match": ";".join(info.get("player_of_match", [])),
    }


def build_player_rows(match_id, info):
    registry = info.get("registry", {}).get("people", {})
    rows = []

    for team, players in info.get("players", {}).items():
        for player in players:
            rows.append(
                {
                    "match_id": match_id,
                    "team": team,
                    "player": player,
                    "player_id": registry_id(registry, player),
                }
            )

    return rows


def build_delivery_rows(match_id, info, innings):
    teams = info.get("teams", [])
    registry = info.get("registry", {}).get("people", {})
    base_match = build_match_row(match_id, info)
    rows = []

    for innings_number, innings_name, inning_data in iter_innings(innings):
        batting_team = inning_data.get("team")
        bowling_team = opponent_for(teams, batting_team)

        for delivery_number, (over_ball, ball_info) in enumerate(iter_deliveries(inning_data), start=1):
            over_number, ball_number = parse_over_ball(over_ball)
            runs = ball_info.get("runs", {})
            extras = ball_info.get("extras", {})
            wickets = wicket_values(ball_info)
            wicket_kinds = [wicket.get("kind") for wicket in wickets]
            players_out = [wicket.get("player_out") for wicket in wickets]
            batter = ball_info.get("batter") or ball_info.get("batsman")
            bowler = ball_info.get("bowler")
            non_striker = ball_info.get("non_striker")
            is_legal_ball = not extras.get("wides", 0) and not extras.get("noballs", 0)

            rows.append(
                {
                    **base_match,
                    "innings_number": innings_number,
                    "innings_name": innings_name,
                    "delivery_number": delivery_number,
                    "over_ball": over_ball,
                    "over": over_number,
                    "ball": ball_number,
                    "phase": innings_phase(over_number),
                    "batting_team": batting_team,
                    "bowling_team": bowling_team,
                    "batter": batter,
                    "batter_id": registry_id(registry, batter),
                    "bowler": bowler,
                    "bowler_id": registry_id(registry, bowler),
                    "non_striker": non_striker,
                    "non_striker_id": registry_id(registry, non_striker),
                    "batter_runs": runs.get("batter", runs.get("batsman", 0)),
                    "extra_runs": runs.get("extras", 0),
                    "total_runs": runs.get("total", 0),
                    "is_legal_ball": bool(is_legal_ball),
                    "wicket_count": len(wickets),
                    "player_out": ";".join(player for player in players_out if player),
                    "wicket_kind": ";".join(kind for kind in wicket_kinds if kind),
                    "fielder": ";".join(fielders_for(wicket) for wicket in wickets if fielders_for(wicket)),
                    "batter_dismissed": any(
                        wicket.get("player_out") == batter
                        and wicket.get("kind") not in NON_BATTER_DISMISSALS
                        for wicket in wickets
                    ),
                    **{extra_type: extras.get(extra_type, 0) for extra_type in EXTRA_TYPES},
                }
            )

    return rows


def write_outputs(yaml_files):
    match_rows = []
    delivery_rows = []
    player_rows = []
    skipped_files = []

    for file_path in yaml_files:
        with file_path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)

        match_id = file_path.stem
        info = data.get("info", {})

        if info.get("competition") != "IPL" or info.get("match_type") != "T20":
            skipped_files.append(file_path.name)
            continue

        match_rows.append(build_match_row(match_id, info))
        player_rows.extend(build_player_rows(match_id, info))
        delivery_rows.extend(build_delivery_rows(match_id, info, data.get("innings", [])))

    matches = pd.DataFrame(match_rows)
    deliveries = pd.DataFrame(delivery_rows)
    player_match = pd.DataFrame(player_rows)

    matches.to_csv(MATCHES_OUT, index=False)
    deliveries.to_csv(DELIVERIES_OUT, index=False)
    player_match.to_csv(PLAYER_MATCH_OUT, index=False)

    deliveries.rename(
        columns={
            "innings_name": "inning",
            "batter": "batsman",
            "batter_runs": "batsman_runs",
        }
    )[
        [
            "match_id",
            "season",
            "date",
            "venue",
            "inning",
            "batting_team",
            "over_ball",
            "batsman",
            "bowler",
            "non_striker",
            "batsman_runs",
            "extra_runs",
            "total_runs",
        ]
    ].to_csv(LEGACY_MASTER_OUT, index=False)

    print("Data cleaning complete")
    print(f"  Matches: {len(matches):,}")
    print(f"  Deliveries: {len(deliveries):,}")
    print(f"  Player-match rows: {len(player_match):,}")
    print(f"  Skipped non-IPL/T20 files: {len(skipped_files):,}")
    print(f"  Missing batter IDs: {deliveries['batter_id'].isna().sum():,}")
    print(f"  Missing bowler IDs: {deliveries['bowler_id'].isna().sum():,}")
    print(f"  Missing venues: {matches['venue'].isna().sum():,}")


def main() -> int:
    yaml_files = sorted(DATA_DIR.glob("*.yaml"))
    write_outputs(yaml_files)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
