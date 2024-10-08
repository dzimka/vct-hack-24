import random
from functools import cache
from typing import List

import polars as pl
from llama_index.core.tools import FunctionTool

from helpers.storage import get_storage_options

RAW_DIR = "data/raw"
DELTA_DIR = "data/delta"
STATS_DIR = f"{DELTA_DIR}/stats"
REGIONS_DIR = f"{DELTA_DIR}/player_region"

LEAGUES = ["game-changers", "vct-international"]
YEARS = [2024]


@cache
def scan_game_data() -> pl.LazyFrame:
    storage_options = get_storage_options()
    source = "local"
    if storage_options:
        source = storage_options.pop("bucket")
    table_path = f"{source}/{STATS_DIR}" if source != "local" else STATS_DIR
    print(f"reading game data from {table_path} table...")
    return pl.scan_delta(table_path, storage_options=storage_options)


@cache
def read_players_regions() -> pl.DataFrame:
    storage_options = get_storage_options()
    source = storage_options.pop("bucket")
    table_path = f"{source}/{REGIONS_DIR}"
    print(f"reading player region data from {table_path} table...")
    return (
        pl.read_delta(table_path, storage_options=storage_options)
        .select("league_region", pl.col("player_id").cast(pl.UInt64))
        .unique()
    )


@cache
def get_players_in_league(league: str) -> pl.DataFrame:
    f"""Retrieves players in a given league.

    Args:
        league (str): The name of the league. Must be one of {LEAGUES}

    Raises:
        ValueError: If the league is invalid.

    Returns:
        pl.DataFrame: A DataFrame containing the players in the specified league.
    """
    players = scan_game_data()
    query = players.filter(pl.col("league_alias") == league).select("player_id").unique()
    return query.collect()


def get_players_region(player_ids: List[int]) -> pl.DataFrame:
    """Retrieves the assigned region for a list of players.

    Args:
        player_id (List[int]): The list of the players IDs for which the region needs to be retrieved.

    Returns:
        pl.DataFrame: A DataFrame containing the players' regions.

    """
    players_list = pl.DataFrame(player_ids, schema={"player_id": pl.UInt64})
    return players_list.join(read_players_regions(), on="player_id", how="left")


def get_random_players(league: str, num_players: int) -> List[int]:
    """Reads all the players for the league and selects the requested number of random players.

    Args:
        league (str): The name of the league. Must be one of "game-changers", "vct-international" or "vct-challengers".
        num_players (int): The number of random players to select from the full list.

    Returns:
        List[int]: The list of players ids.
    """
    if league not in LEAGUES:
        raise ValueError(f"Invalid league: {league}. Choices are: {', '.join(LEAGUES)}")
    try:
        all_players = get_players_in_league(league)
        if len(all_players) < num_players:
            raise RuntimeWarning(
                "Not enough data. Do not retry with the same input. Modify the parameters and try again."
            )

        # select the num_players
        all_players_ids = list(all_players["player_id"])
        if len(all_players_ids) == num_players:
            return list(all_players_ids)

        selected_players = set()
        while len(selected_players) < num_players:
            idx = random.randint(0, len(all_players_ids))
            selected_players.add(all_players_ids[idx])
        return list(selected_players)

    except Exception as e:
        print(e.with_traceback)
        raise RuntimeWarning("The dataset is broken. Do not retry and instruct the user the fix the dataset.")


def get_player_stats(player_ids: List[int], league: str) -> pl.DataFrame:
    """Retrieves player statistics for a list of players.

    Args:
        player_id (List[int]): The list of the players IDs for which the stats needs to be retrieved.

    Returns:
        pl.DataFrame: A DataFrame containing the players' statistics.
    """
    player_stats = scan_game_data()
    query = player_stats.filter((pl.col("league_alias") == league) & pl.col("player_id").is_in(player_ids)).select(
        "year",
        "esports_game_id",
        "player_id",
        "damage_dealt",
        "damage_taken",
        "players_killed",
        pl.when(pl.col("team_mode") == "A")
        .then(pl.lit("attacking team"))
        .otherwise(pl.lit("defending team"))
        .alias("team_role"),
    )
    return query.collect()


def initialize_tools() -> List[FunctionTool]:
    tools = [
        FunctionTool.from_defaults(fn=get_players_region),
        FunctionTool.from_defaults(fn=get_player_stats),
        FunctionTool.from_defaults(fn=get_random_players),
    ]

    return tools


def warm_up_tools():
    print("warming up tools:", flush=True)
    _ = read_players_regions()
    _ = scan_game_data()
    for league in LEAGUES:
        _ = get_players_in_league(league)
    print("------done------")
