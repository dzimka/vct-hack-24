import gradio as gr
from dotenv import load_dotenv

from agent import TeamManager

DEFAULT_PROMPT = """
Build a team using only players from VCT Game Changers.
Assign roles to each player and explain why this composition
would be effective in a competitive match.
"""

manager: TeamManager = None


def run_task(prompt: str) -> str:
    if manager:
        return manager.make_team(prompt)
    return "Cannot execute prompt. Please try again later."


def main():
    global manager
    manager = TeamManager()

    with gr.Blocks(analytics_enabled=False) as demo:
        llm_input = gr.Text(value=DEFAULT_PROMPT, label="Prompt")
        run_task_button = gr.Button("Run Task")
        llm_output = gr.Textbox(label="Task Results")
        run_task_button.click(run_task, inputs=llm_input, outputs=llm_output)

    demo.launch(server_name="0.0.0.0", server_port=8080)


if __name__ == "__main__":
    load_dotenv(verbose=True)
    main()
