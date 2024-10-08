import json
from typing import Any, Dict, List

import polars as pl
from polars.exceptions import ComputeError
from pydantic import BaseModel


class GameMapping(BaseModel):
    platform_game_id: str  # e.g. "val:13d729bb-073f-4a26-ab3a-d36fec4ff421"
    esports_game_id: int  # e.g. 111890538382884654
    tournament_id: int  # e.g. 111890485752679867
    team_mapping: Dict[int, int]  # e.g. 17: 105720640249797517
    participant_mapping: Dict[int, int]  # e.g. 1: 111901586897234547


class GameEvent(BaseModel):
    seq_num: int  # sequence number
    metadata: Dict[Any, Any]  # event metadata
    name: str  # event name (e.g. "damageEvent")
    payload: Dict[Any, Any]  # event payload (e.g. victimId, damage, etc.)


def get_game_mappings(mapping_file: str) -> List[GameMapping]:
    game_mappings: List[GameMapping] = []
    with open(mapping_file, "r") as mf:
        maps = json.load(mf)
        for m in maps:
            try:
                game_mappings.append(
                    GameMapping(
                        platform_game_id=m["platformGameId"],
                        esports_game_id=int(m["esportsGameId"]),
                        tournament_id=int(m["tournamentId"]),
                        team_mapping={int(k): int(v) for k, v in m["teamMapping"].items()},
                        participant_mapping={int(k): int(v) for k, v in m["participantMapping"].items()},
                    )
                )
            except ValueError as e:
                print(f"Invalid mapping record: {m}\n{e}")
                print("Skipping...")
    return game_mappings


def get_game_events(game_file: str) -> List[GameEvent]:
    game_events: List[GameEvent] = []
    with open(game_file, "r") as gf:
        game_data = json.load(gf)
        for e in game_data:
            event_name = [x for x in list(e.keys()) if x not in ["metadata", "platformGameId"]][0]
            if event_name not in ["snapshot", "observerTarget"]:
                game_events.append(
                    GameEvent(
                        seq_num=e["metadata"]["sequenceNumber"],
                        metadata=e["metadata"],
                        name=event_name,
                        payload=e[event_name],
                    )
                )
    return game_events


def get_game_stats(events: List[GameEvent], mapping: GameMapping) -> pl.DataFrame:
    # mapping data
    tournament_id, game_id = mapping.tournament_id, mapping.esports_game_id
    # temp dict for rounds configuration
    rounds_config = {"round_num": [], "team_id": [], "team_mode": [], "player_id": []}
    # temp dict for damage events
    damage_stats = {
        "tournament_id": [],
        "esports_game_id": [],
        "round_num": [],
        "player_id": [],
        "damage_dealt": [],
        "damage_taken": [],
        "players_killed": [],
    }

    for e in events:
        match e.name:
            case "configuration":
                round_num = e.payload["spikeMode"]["currentRound"]
                if round_num not in rounds_config["round_num"]:
                    # add configuration event once per round
                    attacking_team = e.payload["spikeMode"]["attackingTeam"]["value"]
                    for team in e.payload["teams"]:
                        team_num = team["teamId"]["value"]
                        team_mode = "A" if team_num == attacking_team else "D"
                        team_players = [player["value"] for player in team["playersInTeam"]]
                        for player in team_players:
                            rounds_config["round_num"].append(round_num)
                            rounds_config["team_id"].append(int(mapping.team_mapping[team_num]))
                            rounds_config["team_mode"].append(team_mode)
                            rounds_config["player_id"].append(int(mapping.participant_mapping[player]))
            case "damageEvent":
                round_num = e.metadata["currentGamePhase"]["roundNumber"]
                causer = e.payload["causerId"]["value"]
                victim = e.payload["victimId"]["value"]
                damage = round(e.payload["damageDealt"])
                is_killed = 1 if e.payload["killEvent"] else 0
                # add causer
                damage_stats["tournament_id"].append(tournament_id)
                damage_stats["esports_game_id"].append(game_id)
                damage_stats["round_num"].append(round_num)
                damage_stats["player_id"].append(int(mapping.participant_mapping[causer]))
                damage_stats["damage_dealt"].append(damage)
                damage_stats["damage_taken"].append(0)
                damage_stats["players_killed"].append(is_killed)
                # then victim
                damage_stats["tournament_id"].append(tournament_id)
                damage_stats["esports_game_id"].append(game_id)
                damage_stats["round_num"].append(round_num)
                damage_stats["player_id"].append(int(mapping.participant_mapping[victim]))
                damage_stats["damage_dealt"].append(0)
                damage_stats["damage_taken"].append(damage)
                damage_stats["players_killed"].append(0)

    rounds_df = pl.DataFrame(data=rounds_config, schema_overrides={"team_id": pl.UInt64, "player_id": pl.UInt64})
    damage_df = pl.DataFrame(
        data=damage_stats,
        schema_overrides={"tournament_id": pl.UInt64, "esports_game_id": pl.UInt64, "player_id": pl.UInt64},
    )
    try:
        res = rounds_df.join(damage_df, on=["round_num", "player_id"], how="inner")
    except ComputeError as ce:
        print(rounds_df)
        print(damage_df)
        raise ValueError("Bad game data. Could not join dataframes.") from ce
    return res


def get_fixture_data(data_dir: str) -> pl.DataFrame:
    # leagues
    leagues = pl.read_json(f"{data_dir}/esports-data/leagues.json")
    leagues = leagues.select(
        pl.col("league_id").cast(pl.UInt64),
        pl.col("name").alias("league_name"),
        pl.col("region").alias("league_region"),
    )
    # tournaments
    tournaments = pl.read_json(f"{data_dir}/esports-data/tournaments.json")
    tournaments = tournaments.select(
        pl.col("id").cast(pl.UInt64).alias("tournament_id"),
        pl.col("league_id").cast(pl.UInt64),
        pl.col("name").alias("tournament_name"),
        pl.col("status").alias("tournament_status"),
        pl.col("time_zone").alias("tournament_tz"),
    )
    # teams
    teams = pl.read_json(f"{data_dir}/esports-data/teams.json")
    teams = (
        teams.select(
            pl.col("id").cast(pl.UInt64).alias("team_id"),
            pl.col("home_league_id").cast(pl.UInt64).alias("league_id"),
            pl.col("name").alias("team_name"),
        )
        .group_by(["league_id", "team_id"])
        .agg(pl.col("team_name").first())
    )
    # players
    players = pl.read_json(f"{data_dir}/esports-data/players.json")
    players = (
        players.select(
            pl.col("id").cast(pl.UInt64).alias("player_id"),
            pl.col("home_team_id").cast(pl.UInt64).alias("team_id"),
            pl.col("handle").alias("player_handle"),
            (pl.col("first_name") + " " + pl.col("last_name")).alias("player_name"),
            pl.when(pl.col("status") == "active").then(pl.lit(True)).otherwise(pl.lit(False)).alias("player_is_active"),
            # pl.col("created_at").str.to_datetime().alias("player_created"),
            # pl.col("updated_at").str.to_datetime().alias("player_updated"),
        )
        .group_by(["player_id", "team_id", "player_handle", "player_name", "player_is_active"])
        .agg(pl.all())
    )
    fixture_data = (
        leagues.join(tournaments, on="league_id", how="left")
        .join(teams, on="league_id", how="inner")
        .join(players, on="team_id", how="inner")
    )
    return fixture_data
