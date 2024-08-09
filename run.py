import os
import json
import yaml
import subprocess
import glob
from typing import List, Dict

import gradio as gr
from openai import OpenAI
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings

homedir = os.path.dirname(os.path.realpath(__file__))


def read_yaml(path_yaml: str) -> Dict:
    with open(path_yaml, 'r') as file:
        data = yaml.safe_load(file)
    return data


auto = read_yaml(os.path.join(homedir,
                 "authentication/openai.yaml"))
client = OpenAI(api_key=auto["api_key"])

tags = json.load(
    open(os.path.join(homedir, "metainfo/problem_tag.json")))

prob_lists = json.load(
    open(os.path.join(homedir, "metainfo/problem_lists.json")))

vdb = None

path_draft = os.path.join(homedir, "draft.py")


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
        return glob.glob(os.path.join(directory, '**/*.txt'), recursive=True)

    # save vdb to persistent directory
    docs = [TextLoader(f).load() for f in find_txt_files(homedir)]
    docs_list = [item for sublist in docs for item in sublist]
    Chroma.from_documents(
        documents=docs_list,
        collection_name="rag-chroma",
        embedding=GPT4AllEmbeddings(),
        persist_directory=homedir
    )


def prepare_vdb(homedir) -> None:

    if not _vdb_exists(homedir):
        initialize_vdb(homedir)

    global vdb

    # load from chroma.sqlite3 under homedir
    vdb = Chroma(
        collection_name="rag-chroma",
        embedding_function=GPT4AllEmbeddings(),
        persist_directory=homedir
    )


def _exec_cmd(command):
    output = subprocess.run(command.split(), capture_output=True, text=True)
    if len(output.stderr) > 0:
        return output.stderr
    else:
        return output.stdout


def _exec_python_code(path_python_script):
    try:
        exec(open(path_python_script).read())
    except Exception as e:
        return str(e)


def _show_tag(algorithm):
    if algorithm in tags:
        return "\n".join(tags[algorithm])
    else:
        return f"No tag '{algorithm}' exists.\n" + "=== Available tags ===\n" + "\n".join(tags.keys())


def _retrieve_description(algorithm):
    global vdb
    documents = vdb.similarity_search(algorithm, k=1)
    answer = "\n=======".join([d.page_content for d in documents])
    return answer


def _general_advice():
    advices = []
    advices.append("Analyze the Problem: Identify and map the elements of the problem to appropriate data structures in computer science. Consider the allowed complexity constraints.")
    advices.append("Filter Algorithms by Complexity: Based on the complexity constraints, narrow down to feasible algorithms that can efficiently solve the problem. Type 'complexity' for more details.")
    return "\n\n".join([f"{i+1}. {adv}" for i, adv in enumerate(advices)])


def _write_io_codes(rows):
    with open(path_draft, "a") as f:
        for row in rows:
            row_chunks = row.split()
            if len(row_chunks) == 1:
                if row.isdigit():
                    code = " = int(input())"
                else:
                    code = " = input()"
            else:
                if all(e.isdigit() for e in row_chunks):
                    code = " = list(map(int, input().split())) or " + "," * \
                        (len(row_chunks)-1) + "= map(int, input().split())"
                else:
                    code = " = input().split() or" + ","*(len(row_chunks)-1) + "= input().split()"

            f.write(f"{code}\n")


def _write_to_draft(content):
    with open(path_draft, "a") as f:
        f.write(content)


def _manual():
    manuals = ["## change directory\n```cd </path/to/directory>```",
               "## show list directory content\n```ls```",
               "## execute python file\n```python </path/to/python_file>```",
               "## get a hint\n```hint <boj-prob-number>```",
               "## show algorithm tags\n```tag <algorithm-name>```",
               "## write parsing codes on draft.py\n```io <io-examples-after-newline>```",
               "## fetch algorithm explanation\n```explain <algorithm-name>```",
               "## show general strategy for problem solving\n```help```",
               "## write code on draft.py based on the discussion\n```code```"]

    return "\n\n".join(manuals)


def execute_command(history):
    command = history[-1][0]
    cmd_name = command.split()[0]
    if cmd_name == "cd":
        history[-1][1] = change_directory(" ".join(command.split()[1:]))
        return history
    elif cmd_name == "python":
        history[-1][1] = _exec_python_code(" ".join(command.split()[1:]))
        return history
    elif cmd_name == "code":
        return _write_code(history)
    elif cmd_name == "hint":
        return _give_hint(history)
    elif cmd_name == "tag":
        history[-1][1] = _show_tag(" ".join(command.split()[1:]))
        return history
    elif cmd_name == "io":
        history[-1][1] = _write_io_codes(command.split("\n")[1:])
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
        'cd', 'ls', 'python',

        # user-defined
        'man', 'hint', 'code', 'tag', 'io', 'explain', 'complexity', 'help'
    ]
    command = prompt.split()[0]
    return command in commands


def reference_to_code(prompt, commands_discussion=['hint', 'discuss']):
    command = prompt.split()[0]
    return command in commands_discussion


def default_answer(history):
    history[-1][1] = "I don't know your intention. Type 'man' for available commands."
    return history


def handle_query(history):
    recent_user_message = history[-1][0]
    if is_command(recent_user_message):
        return execute_command(history)
    else:
        return default_answer(history)


def _write_code(history):
    history_openai_format = [{"role": "system",
                              "content": "You're an expert in Python. Write a code-only solution to the problem discussed. At the end of your code, append if __name__ == '__main__': to call your function with sample arguments"}]
    for msg_user, msg_assistant in history:
        if msg_assistant and reference_to_code(msg_user):
            history_openai_format.append({"role": "user", "content": msg_user})
            history_openai_format.append(
                {"role": "assistant", "content": msg_assistant})

    final_user_prompt = history[-1][0]
    history_openai_format.append(
        {"role": "user", "content": final_user_prompt})

    response = client.chat.completions.create(model='gpt-4o',
                                              messages=history_openai_format,
                                              temperature=1.0,
                                              stream=True)

    history[-1][1] = ""
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            history[-1][1] += chunk.choices[0].delta.content

    answer = history[-1][1]
    start_code = answer.find('```python')
    if start_code != -1:
        answer = answer[start_code + len('```python'):]
        end_code = answer.find('```')
        if end_code != -1:
            answer = answer[:end_code]
            _write_to_draft(answer)

    return history


def read_txt(path):
    return open(path).read()


def _problem_tags(problem_number):

    tags = [e["tags"]
            for e in prob_lists if e["problemId"] == int(problem_number)]

    if len(tags) == 0:
        raise ValueError("No tag found.")
    elif len(tags) > 1:
        raise ValueError("More than 1 tags exist.")
    else:
        return ", ".join(tags[0])


def _problem_prompt(problem_num):
    try:
        problem_desc = read_txt(os.path.join(
            homedir, f"problems/{problem_num}.txt"))
    except Exception as e:
        problem_desc = "Not found"

    try:
        ptags = _problem_tags(problem_num)
    except Exception as e:
        ptags = "Not found"

    try:
        solution = read_txt(os.path.join(homedir,
                            f"solutions/{problem_num}.txt"))
    except Exception as e:
        solution = "Not found"

    return f"### problem: {problem_desc}\n\n" + \
        f"### tags: {ptags}\n\n" + f"### solution code: {solution}"


def _give_hint(history: str):

    try:
        user_prompt = history[-1][0]
        problem_num = str(int("".join(user_prompt.split()[1:])))
    except Exception as e:
        pnum = " ".join(user_prompt.split()[1:])
        history[-1][0] = "Failed command - " + history[-1][0]
        history[-1][1] = f"Uknown problem number {pnum}"
        return history

    history_openai_format = [{"role": "system",
                              "content": "You're an expert in algorithm. Given a problem, and algorithms that consist of a solution, and a solution code, advise to provide the best lessons. Never write codes."}]

    history[-1][0] = _problem_prompt(problem_num)
    history_openai_format.append({"role": "user", "content": history[-1][0]})

    response = client.chat.completions.create(model='gpt-4o',
                                              messages=history_openai_format,
                                              temperature=1.0,
                                              stream=True)

    history[-1][1] = ""
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            history[-1][1] += chunk.choices[0].delta.content

    return history


def run_gradio():
    with gr.Blocks(fill_height=True) as demo:
        history = gr.Chatbot(bubble_full_width=False)
        text_input = gr.Textbox()
        clear = gr.Button("Clear")

        def merge_user_input_to_history(user_message, history):
            return "", history + [[user_message, None]]

        text_input.submit(merge_user_input_to_history,
                          [text_input, history],
                          [text_input, history],
                          queue=False).then(handle_query, history, history)
        clear.click(lambda: None, None, history, queue=False)

    demo.queue()
    demo.launch()


if __name__ == "__main__":

    prepare_vdb(homedir=os.path.join(homedir, "descriptions"))

    run_gradio()
