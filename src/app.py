import logging
import os

import gradio as gr
from dotenv import find_dotenv, load_dotenv

from agent import TeamManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DEFAULT_PROMPT = """
Build a team using only players from VCT Game Changers.
Assign roles to each player and explain why this composition
would be effective in a competitive match.
"""

manager: TeamManager = None
num_calls: int = 100


def init_manager():
    model = os.environ.get("AWS_MODEL_ID")
    context_size = os.environ.get("AWS_MODEL_CONTEXT_SIZE")
    region = os.environ.get("AWS_MODEL_REGION")
    global manager
    manager = TeamManager(model, context_size, region)
    logger.info(f"Initialized manager using parameters:\nmodel={model}, context_size={context_size}, region={region}")


def run_task(prompt: str, history: str) -> str:
    global num_calls
    num_calls -= 1
    if num_calls <= 0:
        return "Number of runs exceeded. Please contact developer to reset the budget."
    if manager:
        return manager.make_team(prompt)
    return "Cannot execute prompt. Please try again later."


def main():
    with gr.Blocks(analytics_enabled=False) as demo:
        gr.Markdown(
            f"""
        model = {manager.model}, region = {manager.region}, number of calls left = {num_calls}
        """
        )
        llm_input = gr.Text(value=DEFAULT_PROMPT, label="Prompt")
        run_task_button = gr.Button("Run Task")
        llm_output = gr.Textbox(label="Task Results")
        run_task_button.click(run_task, inputs=[llm_input, llm_output], outputs=llm_output)

    demo.launch(server_name="0.0.0.0", server_port=8080)


if __name__ == "__main__":
    load_dotenv(find_dotenv(), verbose=True)
    init_manager()
    main()
