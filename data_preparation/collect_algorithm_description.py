import os
import re
import json
import yaml

from openai import OpenAI


homedir = os.path.dirname(os.path.realpath(__file__))


def read_yaml(path_yaml: str):
    with open(path_yaml, 'r') as file:
        data = yaml.safe_load(file)
    return data


auto = read_yaml(os.path.join(homedir,
                 "../authentication/openai.yaml"))
client = OpenAI(api_key=auto["api_key"])


def ask_gpt(algorithm):
    history_openai_format = [{"role": "system",
                              "content": "You're an expert in explaining competitive programming algorithms. Give (1) a brief description in a single sentence and (2) complexity in large O notation with meanings of symbols explained."}]
    history_openai_format.append(
        {"role": "user", "content": f"Explain algorithm \"{algorithm}\". "})
    response = client.chat.completions.create(model='gpt-4o',
                                              messages=history_openai_format,
                                              temperature=1.0,
                                              stream=True)
    gpt_answer = ""
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            gpt_answer += chunk.choices[0].delta.content

    return gpt_answer


def save_descriptions(algorithm: str, savedir: str) -> None:
    filename = "".join(re.findall(r'[a-zA-Z]', algorithm))
    savepath = os.path.join(savedir, f"{filename}.txt")

    if not os.path.exists(savepath):
        gpt_answer = ask_gpt(algorithm)
        with open(savepath, "w") as f:
            f.write(f"{algorithm}\n\n")

        with open(savepath, "a") as f:
            f.write(gpt_answer)


if __name__ == "__main__":
    problem_tag = json.load(
        open(os.path.join(homedir, "../metainfo/problem_tag.json")))
    for tag in problem_tag:
        for algo in problem_tag[tag]:
            save_descriptions(algo, "descriptions")
