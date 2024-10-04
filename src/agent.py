from typing import List

import gradio as gr
import polars as pl
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.bedrock import Bedrock

DEFAULT_PROMPT = """
Build a team using only players from VCT Game Changers.
Assign roles to each player and explain why this composition
would be effective in a competitive match.
"""

agent: ReActAgent  # global agent


def initialize_settings():
    """
    Initialize global settings for LlamaIndex.
    This sets up the language model (LLM) and embedding model using Amazon Bedrock.
    """
    # Set the LLM to use Mistral model from Bedrock
    Settings.llm = Bedrock(
        model="mistral.mistral-large-2407-v1:0",
        region_name="us-west-2",
        context_size=2000,
    )


def agg_teams_by_region():
    """Aggregates teams by the region"""
    teams = pl.read_parquet("data/parquet/game-changers/esports-data/teams/data.parquet")
    leagues = pl.read_parquet("data/parquet/game-changers/esports-data/leagues/data.parquet")
    joined = teams.join(leagues, left_on="home_league_id", right_on="league_id", how="full")
    # region = ""
    q = joined.lazy().group_by("region").agg(pl.count("id").alias("count")).sort("count", descending=True)

    return q.collect()


LEAGUES_CHOICES = ["game_changers", "international"]


def get_players_in_league(league: str) -> pl.DataFrame:
    f"""Retrieves players in a given league.

    Args:
        league (str): The name of the league. Must be one of {LEAGUES_CHOICES}

    Raises:
        ValueError: If the league is invalid.

    Returns:
        pl.DataFrame: A DataFrame containing the players in the specified league.
    """
    if league not in LEAGUES_CHOICES:
        raise ValueError(f"Invalid league: {league}. Choices are: {', '.join(LEAGUES_CHOICES)}")

    players = pl.scan_delta("data/delta/game-changers/games/2024/stats")
    query = players.group_by("player_id", "team_mode").len()
    return query.collect()


def get_player_stats(player_ids: List[int]) -> pl.DataFrame:
    """Retrieves player statistics for a list of players.

    Args:
        player_id (List[int]): The list of the players IDs for which the stats needs to be retrieved.

    Returns:
        pl.DataFrame: A DataFrame containing the playerss' statistics.
    """
    player_stats = pl.scan_delta("data/delta/game-changers/games/2024/stats")
    query = player_stats.filter(pl.col("player_id").is_in(player_ids)).select(
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
        FunctionTool.from_defaults(fn=get_player_stats),
        FunctionTool.from_defaults(fn=get_players_in_league),
    ]
    # agg_by_region_tool = FunctionTool.from_defaults(fn=agg_teams_by_region)
    # tools.append(agg_by_region_tool)
    return tools


def run_task(prompt: str) -> str:
    global agent
    extended_prompt = f"""{prompt}
    Use a tool to fetch the list of players in the specified league,
    then choose up to 10 players randomly,
    then retrieve each player's related data,
    finally make a selection of exactly 5 players.
    Revise if needed.
    """
    return agent.chat(extended_prompt)


def main():
    initialize_settings()
    tools = initialize_tools()
    global agent
    agent = ReActAgent.from_tools(tools, verbose=True)

    with gr.Blocks(analytics_enabled=False) as demo:
        llm_input = gr.Text(value=DEFAULT_PROMPT, label="Prompt")
        run_task_button = gr.Button("Run Task")
        llm_output = gr.Textbox(label="Task Results")
        run_task_button.click(run_task, inputs=llm_input, outputs=llm_output)

    demo.launch(server_name="0.0.0.0", server_port=8080)


if __name__ == "__main__":
    load_dotenv()
    main()
