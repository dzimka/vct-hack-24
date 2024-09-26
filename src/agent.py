from typing import List

import gradio as gr
import polars as pl
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.bedrock import Bedrock

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


def initialize_tools() -> List[FunctionTool]:
    tools = []
    agg_by_region_tool = FunctionTool.from_defaults(fn=agg_teams_by_region)
    tools.append(agg_by_region_tool)
    return tools


def generate_answer(prompt: str):
    global agent
    extended_prompt = f"{prompt} Use a tool to calculate every step."
    return agent.chat(extended_prompt)


def main():
    initialize_settings()
    tools = initialize_tools()
    global agent
    agent = ReActAgent.from_tools(tools, verbose=True)

    with gr.Blocks(analytics_enabled=False) as demo:
        prompt_input = gr.Text(value="How many international teams", label="Prompt")
        generate_script_button = gr.Button("Answer")
        text_output = gr.Textbox(label="Generated Answer")
        generate_script_button.click(generate_answer, inputs=prompt_input, outputs=text_output)

    demo.launch(server_name="0.0.0.0", server_port=8080)


if __name__ == "__main__":
    load_dotenv()
    main()
