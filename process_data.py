import yaml
import glob
import pandas as pd

def load_yaml(yaml_files):
    table_data = [] 
    for file_path in yaml_files:
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)

        info = data.get('info', {})
        venue = info.get('venue', 'Unknown')
        dates = info.get('dates', ['Unknown'])
        match_date = dates[0]
        season = int(str(match_date).split('-')[0]) if match_date != 'Unknown' else 'Unknown'

        innings = data.get('innings', [])
        for inning in innings:
            for inning_name, inning_data in inning.items():
                batting_team = inning_data.get('team')
                
                deliveries = inning_data.get('deliveries', [])
                for delivery in deliveries:
                    for over_ball, ball_info in delivery.items():
                        runs = ball_info.get('runs', {})
                        row = {
                            'season': season,
                            'date': match_date,
                            'venue': venue,
                            'inning': inning_name,
                            'batting_team': batting_team,
                            'over_ball': over_ball,
                            'batsman': ball_info.get('batsman'),
                            'bowler': ball_info.get('bowler'),
                            'non_striker': ball_info.get('non_striker'),
                            'batsman_runs': runs.get('batsman', 0),
                            'extra_runs': runs.get('extras', 0),
                            'total_runs': runs.get('total', 0)
                        }
                        table_data.append(row)
    df = pd.DataFrame(table_data)
    df.to_csv('master_ipl_dataset.csv', index=False)

def main() -> int:
    folder_path = 'data/*.yaml'
    yaml_files = glob.glob(folder_path)
    load_yaml(yaml_files)
    return 1

if __name__ == "__main__":
    main()