import os
import weave
import gradio as gr
from models.assistant import Assistant
from models.common import read_yaml, HOME_DIR
import omegaconf

CSS = """
.contain { display: flex; flex-direction: column; }
.gradio-container { height: 100vh !important; }
#component-0 { height: 100%; }
#chatbot { flex-grow: 1; overflow: auto;}
"""

def get_problem_description(problem_number):
    # Example implementation
    try:
        with open(f"problems/{problem_number}.txt", "r") as file:
            return f"<< Problem {problem_number} >>\n\n"+file.read()
    except FileNotFoundError:
        return f"No description found for problem {problem_number}."

def run_gradio(cfg):
    print(cfg)
    model = Assistant(
        name = cfg.name,
        description = cfg.description,
        cfg = cfg
        )
    with gr.Blocks(css=CSS) as demo:
        with gr.Row():  # Create a row to hold two columns
            with gr.Column(scale=1):  # Left column
                problem_description = gr.Markdown(
                    value="Problem description will appear here.",
                    elem_id="problem_description"
                )
                # Input box at the bottom left
                text_input = gr.Textbox(
                    placeholder="Enter your message here",
                    elem_id="text_input"
                )
            with gr.Column(scale=1):  # Right column
                history = gr.State([])  # Initialize history as a state variable
                chatbot = gr.Chatbot(
                    value=[],
                    elem_id="chatbot"
                )
                clear = gr.Button("Clear")

        # Function to handle user input
        def handle_user_input(user_message, history_state):
            assistant_outpu = model.handle_query(user_message)
            updated_history = history_state + [[user_message, assistant_outpu]]
            # model_response = updated_history[-1][1]
            problem_description_update = gr.update()
            recent_user_message = updated_history[-1][0]
            cmd_name = recent_user_message.split()[0]

            if cmd_name == "hint" or cmd_name == "explain":
                problem_number = " ".join(recent_user_message.split()[1:])
                problem_desc = get_problem_description(problem_number)
                problem_description_update = gr.update(value=problem_desc)
            chatbot_messages = [
                [pair[0], pair[1]] for pair in updated_history
            ]

            return "", updated_history, chatbot_messages, problem_description_update
        
        # Function to clear the conversation
        def clear_conversation():
            return "", [], [], gr.update(value="Problem description will appear here.")

        # Set up the interaction for when the user submits input
        text_input.submit(
            handle_user_input,
            inputs=[text_input, history],
            outputs=[text_input, history, chatbot, problem_description],
            queue=False,
        )
        # Set up the clear button to reset history and problem description
        clear.click(
            clear_conversation,
            inputs=None,
            outputs=[text_input, history, chatbot, problem_description],
            queue=False
        )

    demo.queue()
    demo.launch()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="gpt4o", help="Configuration name")
    args = parser.parse_args()
    
    cfg = omegaconf.OmegaConf.load(f"{HOME_DIR}/configs/{args.config}.yaml")
    
    weave.init('wandb-korea/boj-code-assistant')
    run_gradio(cfg)