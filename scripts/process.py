import polars as pl

from helpers.parsers import GameMapping, get_game_events, get_game_mappings, get_game_stats
from helpers.storage import get_storage_options

RAW_DIR = "data/raw"
DELTA_DIR = "data/delta"
STATS_DIR = f"{DELTA_DIR}/stats"

# LEAGUES = ["vct-challengers", "game-changers", "vct-international"]
LEAGUES = ["game-changers"]
YEARS = [2024]


def write_to_table(game_summary: pl.DataFrame, league, year):
    storage_options = get_storage_options()
    target = "local"
    if storage_options:
        target = storage_options.pop("bucket")
    print(f"Writing to {target} table...")
    output = game_summary.select(pl.all(), pl.lit(year).alias("year"), pl.lit(league).alias("league_alias"))
    table_path = f"{target}/{STATS_DIR}" if target != "local" else STATS_DIR
    output.write_delta(
        table_path,
        mode="append",
        storage_options=storage_options,
        delta_write_options={"partition_by": ["league_alias", "year"]},
    )


def process_league_files(league: str, year: int | None = 2024):
    mapping_file = f"{RAW_DIR}/{league}/esports-data/mapping_data.json"
    games_folder = f"{RAW_DIR}/{league}/games/{year}"
    mappings = get_game_mappings(mapping_file)
    for i, mapping in enumerate(mappings):
        print(f"{i} of {len(mappings)}: ", end="")
        try:
            game_summary = process_game_file(games_folder, mapping, league, year)
            write_to_table(game_summary, league, year)
        except FileNotFoundError:
            print("File not found. Skipping...")
        except KeyError as e:
            if "causerId" in str(e):
                print("Bad game data. Skipping...")
            else:
                raise
        except ValueError as e:
            if "bad game data" in str(e).lower():
                print("Bad game data. Skipping...")
            else:
                raise
        print("---")


def process_game_file(games_folder: str, mapping: GameMapping, league: str, year: str) -> pl.DataFrame:
    game_file = f"{games_folder}/{mapping.platform_game_id}.json"
    print(f"{game_file}")
    events = get_game_events(game_file)
    game_stats = get_game_stats(events, mapping)
    q = (
        game_stats.lazy()
        .group_by("tournament_id", "esports_game_id", "team_id", "team_mode", "player_id")
        .agg([pl.sum("damage_dealt"), pl.sum("damage_taken"), pl.sum("players_killed")])
        .sort(["damage_dealt"])
    )
    game_summary = q.collect()
    print(
        f"Extracted events: {len(events)}, game stats rows: {len(game_stats)}, game summary rows: {len(game_summary)}"
    )
    if len(game_summary) != 20:
        print(game_summary)
        raise ValueError(f"Bad game data. Expected summary len = 20, but got {len(game_summary)}")

    return game_summary


def main():
    for league in LEAGUES:
        for year in YEARS:
            process_league_files(league, year)


if __name__ == "__main__":
    # load_dotenv()
    main()
