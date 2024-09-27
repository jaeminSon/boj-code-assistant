import os
import json
import yaml
import subprocess
import glob
import weave
from typing import List, Dict

import gradio as gr
from openai import OpenAI
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings

CSS = """
.contain { display: flex; flex-direction: column; }
.gradio-container { height: 100vh !important; }
#component-0 { height: 100%; }
#chatbot { flex-grow: 1; overflow: auto;}
"""

homedir = os.path.dirname(os.path.realpath(__file__))


def read_yaml(path_yaml: str) -> Dict:
    with open(path_yaml, "r") as file:
        data = yaml.safe_load(file)
    return data


auto = read_yaml(os.path.join(homedir, "authentication/openai.yaml"))

# client = OpenAI(api_key=auto["api_key"])
# MODEL = "gpt-4o"
client = OpenAI(
    api_key="up_8RfbCiSaCEymY3vFb7CwFD0wpwTBv",
    base_url="https://api.upstage.ai/v1/solar"
)

MODEL = "solar-pro"

tags = json.load(open(os.path.join(homedir, "metainfo/problem_tag.json")))

prob_lists = json.load(open(os.path.join(homedir, "metainfo/problem_lists.json")))

vdb = None

path_draft = os.path.join(homedir, "draft.py")

@weave.op()
def change_directory(dest) -> None:
    try:
        os.chdir(dest)
        return os.getcwd()
    except IndexError:
        return "Set destination directory"
    except FileNotFoundError:
        return f"No such file or directory: {dest}"


def _vdb_exists(homedir) -> bool:
    return os.path.exists(os.path.join(homedir, "chroma.sqlite3"))


def initialize_vdb(homedir):
    def find_txt_files(directory: str) -> List[str]:
        return glob.glob(os.path.join(directory, "**/*.txt"), recursive=True)

    # save vdb to persistent directory
    docs = [TextLoader(f).load() for f in find_txt_files(homedir)]
    docs_list = [item for sublist in docs for item in sublist]
    Chroma.from_documents(
        documents=docs_list,
        collection_name="rag-chroma",
        embedding=GPT4AllEmbeddings(),
        persist_directory=homedir,
    )


def prepare_vdb(homedir) -> None:
    if not _vdb_exists(homedir):
        initialize_vdb(homedir)

    global vdb

    # load from chroma.sqlite3 under homedir
    vdb = Chroma(
        collection_name="rag-chroma",
        embedding_function=GPT4AllEmbeddings(),
        persist_directory=homedir,
    )

@weave.op()
def _exec_cmd(command):
    output = subprocess.run(command.split(), capture_output=True, text=True)
    if len(output.stderr) > 0:
        return output.stderr
    else:
        return output.stdout

@weave.op()
def _show_tag(algorithm):
    if algorithm in tags:
        return "\n".join(tags[algorithm])
    else:
        return (
            f"No tag '{algorithm}' exists.\n"
            + "=== Available tags ===\n"
            + "\n".join(tags.keys())
        )

@weave.op()
def _retrieve_description(algorithm):
    global vdb
    documents = vdb.similarity_search(algorithm, k=1)
    answer = "\n=======".join([d.page_content for d in documents])
    return answer

@weave.op()
def _general_advice():
    advices = []
    advices.append(
        "Analyze the Problem: Identify key objects in the problem and map them to appropriate data structures in computer science."
    )
    advices.append(
        "Figure out required operations: From the given information, derive the course of transformation that leads to the solution."
    )
    advices.append(
        "Filter Algorithms by Complexity: Based on the complexity constraints, narrow down to feasible algorithms that can efficiently solve the problem. Type 'complexity' for more details."
    )
    return "\n\n".join([f"{i+1}. {adv}" for i, adv in enumerate(advices)])

@weave.op()
def _write_io_codes(spec):
    chunks = spec.split()
    if len(chunks) == 1:
        problem_number = chunks[0]
        language = "python"
    elif len(chunks) == 2:
        problem_number, language = chunks
        language = language.lower()
    else:
        return "Unknown command. Type 'man' for usage."

    history_openai_format = [
        {
            "role": "system",
            "content": f"As an expert in {language}, write a code snippet to parse the input using {language}. Answer only the code.",
        }
    ]

    history_openai_format.append(
        {"role": "user", "content": _read_problem(problem_number)}
    )

    response = client.chat.completions.create(
        model=MODEL, messages=history_openai_format, temperature=1.0, stream=True
    )

    answer = ""
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            answer += chunk.choices[0].delta.content

    start_code = answer.find(f"```{language}")
    if start_code != -1:
        answer = answer[start_code + len(f"```{language}") :]
        end_code = answer.find("```")
        if end_code != -1:
            answer = answer[:end_code]
            _write_to_draft(answer)

    return "Finished writing to draft.py"


def _write_to_draft(content):
    with open(path_draft, "a") as f:
        f.write(content)

@weave.op()
def _manual():
    manuals = [
        "## change directory\n```cd </path/to/directory>```",
        "## show list directory content\n```ls```",
        "## execute python file\n```python </path/to/python_file>```",
        "## get a hint\n```hint <boj-prob-number>```",
        "## verify your logic\n```verify <boj-prob-number> <your-logic>```",
        "## show algorithm tags\n```tag <algorithm-name>```",
        "## write parsing codes on draft.py\n```io <boj-prob-number> <language>```",
        "## fetch algorithm explanation\n```explain <algorithm-name>```",
        "## show general strategy for problem solving\n```help```",
        "## write code on draft.py based on the discussion\n```code```",
    ]

    return "\n\n".join(manuals)

@weave.op()
def execute_command(history):
    command = history[-1][0]
    cmd_name = command.split()[0]
    if cmd_name == "cd":
        history[-1][1] = change_directory(" ".join(command.split()[1:]))
        return history
    elif cmd_name == "verify":
        history[-1][1] = _verify_logic(" ".join(command.split()[1:]))
        return history
    elif cmd_name == "hint":
        history[-1][1] = _give_hint(" ".join(command.split()[1:]))
        return history
    elif cmd_name == "tag":
        history[-1][1] = _show_tag(" ".join(command.split()[1:]))
        return history
    elif cmd_name == "io":
        history[-1][1] = _write_io_codes(" ".join(command.split()[1:]))
        return history
    elif cmd_name == "explain":
        history[-1][1] = _retrieve_description(" ".join(command.split()[1:]))
        return history
    elif cmd_name == "complexity":
        history[-1][1] = _retrieve_description("complexity analysis")
        return history
    elif cmd_name == "man":
        history[-1][1] = _manual()
        return history
    elif cmd_name == "help":
        history[-1][1] = _general_advice()
        return history
    else:
        history[-1][1] = _exec_cmd(command)
        return history


def is_command(prompt):
    commands = [
        "cd",
        "ls",
        # user-defined
        "man",
        "hint",
        "verify",
        "tag",
        "io",
        "explain",
        "complexity",
        "help",
    ]
    command = prompt.split()[0]
    return command in commands


def reference_to_code(prompt, commands_discussion=["hint", "verify"]):
    command = prompt.split()[0]
    return command in commands_discussion

@weave.op()
def default_answer(history: List[List]) -> List[List]:
    history[-1][1] = "I don't know your intention. Type 'man' for available commands."
    return history

@weave.op()
def handle_query(history: List[List]) -> str:
    print(history)
    recent_user_message = history[-1][0]
    if is_command(recent_user_message):
        return execute_command(history)
    else:
        return default_answer(history)


def read_txt(path: str):
    return open(path).read()

@weave.op()
def _problem_tags(problem_number):
    tags = [e["tags"] for e in prob_lists if e["problemId"] == int(problem_number)]

    if len(tags) == 0:
        raise ValueError("No tag found.")
    elif len(tags) > 1:
        raise ValueError("More than 1 tags exist.")
    else:
        return ", ".join(tags[0])

@weave.op()
def _read_problem(problem_num: str):
    try:
        return read_txt(os.path.join(homedir, f"problems/{problem_num}.txt"))
    except Exception:
        return "Not found"

@weave.op()
def _read_tags(problem_num: str):
    try:
        return _problem_tags(problem_num)
    except Exception:
        return "Not found"

@weave.op()
def _read_solution(problem_num: str):
    try:
        return read_txt(os.path.join(homedir, f"solutions/{problem_num}.txt"))
    except Exception:
        return "Not found"

@weave.op()
def _problem_prompt(problem_num: str):
    problem_desc = _read_problem(problem_num)
    ptags = _read_tags(problem_num)
    solution = _read_solution(problem_num)

    return (
        f"### problem: {problem_desc}\n\n"
        + f"### tags: {ptags}\n\n"
        + f"### solution code: {solution}"
    )

@weave.op()
def _verify_logic(prompt: str):
    try:
        chunks = prompt.split()
        problem_num = chunks[0]
        logic = prompt[prompt.index(problem_num) + len(problem_num) + 1 :]
    except Exception:
        return f"Unknown arguments {prompt} for 'verify' command."

    try:
        problem_num = str(int(problem_num))
    except Exception:
        problem_num = str(int(problem_num))
        return f"Unknown problem number {problem_num}"

    history_openai_format = [
        {
            "role": "system",
            "content": "You're an expert in algorithms. You are presented with a problem, algorithm tags, and a solution code. Your task is to guide the user by verifying the approach to solving the problem, taking into account the provided algorithm tags and solution code. Answer 'Correct' if the logic is accurate and conclude there. Answer 'Incorrect' if it isn't providing reasons briefly.",
        }
    ]

    history_openai_format.append(
        {
            "role": "user",
            "content": _problem_prompt(problem_num)
            + "\n\n"
            + f"### user's logic: {logic}",
        }
    )

    response = client.chat.completions.create(
        model=MODEL, messages=history_openai_format, temperature=1.0, stream=True
    )

    answer = ""
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            answer += chunk.choices[0].delta.content

    return answer

@weave.op()
def _give_hint(problem_num: str):
    try:
        problem_num = str(int(problem_num))
    except Exception:
        return f"Unknown problem number {problem_num}"

    history_openai_format = [
        {
            "role": "system",
            "content": "As an expert in algorithms, offer hints and valuable insights to a learner for the given problem. Consider the related algorithm tags and provided solution code. Do not write any code.",
        }
    ]

    history_openai_format.append(
        {"role": "user", "content": _problem_prompt(problem_num)}
    )

    response = client.chat.completions.create(
        model=MODEL, messages=history_openai_format, temperature=1.0, stream=True
    )

    answer = ""
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            answer += chunk.choices[0].delta.content

    return answer


def run_gradio():
    with gr.Blocks(css=CSS) as demo:
        history = gr.Chatbot(bubble_full_width=False, elem_id="chatbot")
        text_input = gr.Textbox()
        clear = gr.Button("Clear")

        def merge_user_input_to_history(user_message, history):
            return "", history + [[user_message, None]]

        text_input.submit(
            merge_user_input_to_history,
            [text_input, history],
            [text_input, history],
            queue=False,
        ).then(handle_query, history, history)
        clear.click(lambda: None, None, history, queue=False)

    demo.queue()
    demo.launch()


if __name__ == "__main__":
    weave.init('wandb-korea/boj-code-assistant')
    prepare_vdb(homedir=os.path.join(homedir, "descriptions"))

    run_gradio()
