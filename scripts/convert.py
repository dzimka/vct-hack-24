import polars as pl
from dotenv import load_dotenv

from helpers.parsers import get_fixture_data
from helpers.storage import get_storage_options

RAW_DIR = "data/raw"
DELTA_DIR = "data/delta"

LEAGUES = ["game-changers", "vct-international", "vct-challengers"]


def convert_region_data():
    all_players = pl.DataFrame()
    for league in LEAGUES:
        players_in_league = get_fixture_data(f"{RAW_DIR}/{league}").select("league_region", "player_id").unique()
        all_players = pl.concat([all_players, players_in_league], how="vertical")
    storage_options = get_storage_options()
    bucket = storage_options.pop("bucket")
    table_path = f"{bucket}/{DELTA_DIR}/player_region"
    all_players.write_delta(table_path, storage_options=storage_options)


def main():
    load_dotenv()
    convert_region_data()
    print("Done!")


if __name__ == "__main__":
    main()
