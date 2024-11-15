from typing import List

from llama_index.core import Settings
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.bedrock import Bedrock

from tools import initialize_tools, warm_up_tools

DEFAULT_MODEL = "mistral.mistral-large-2407-v1:0"
DEFAULT_CONTEXT_SIZE = 2000
DEFAULT_REGION = "us-west-2"


class TeamManager:
    agent: ReActAgent
    tools: List[FunctionTool]
    model: str
    region: str

    def __init__(
        self, model: str = DEFAULT_MODEL, context_size: int = DEFAULT_CONTEXT_SIZE, region: str = DEFAULT_REGION
    ):
        # init global settings for LlamaIndex via Bedrock
        self.model = model
        self.region = region
        Settings.llm = Bedrock(
            model=model,
            region_name=region,
            context_size=context_size,
        )
        # init tools
        self.tools = initialize_tools()
        warm_up_tools()
        self.agent = ReActAgent.from_tools(self.tools, verbose=True)

    def make_team(self, prompt: str) -> str:
        self.agent.reset()
        extended_prompt = f"""{prompt}
        Use provided tools to get the list of up to 10 players in the specified league,
        and optionally check their region if requested to do so.
        If the league is not specified, then combine the list from all available leagues.
        Then retrieve selected players' stats data,
        finally make a selection of exactly 5 players.
        Use Valorant game terminology and definitions when assigning roles.
        """
        return self.agent.chat(extended_prompt)
