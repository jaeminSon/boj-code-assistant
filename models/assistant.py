import openai
import os
import subprocess
import weave
from typing import List
from models.common import HOME_DIR, AUTH, TAGS, PROB_LISTS, PATH_DRAFT, prepare_vdb
from langchain_community.vectorstores import Chroma


class Assistant(weave.Model):
    cfg: dict
    # def __init__(self, name, description, cfg):
    #     self.cfg = cfg

    @weave.op()
    def handle_query(self, input:str) -> str:
        if self.is_command(input):
            return self.execute_command(input)
        else:
            return self.default_answer(input)     
    
    @weave.op()
    def predict(self, question:str) -> str:

        language = "python"

        history_openai_format = [
            {
                "role": "system",
                "content": f"As an expert in {language}, write a code snippet to parse the input using {language}. Answer only the code.",
            }
        ]

        history_openai_format.append(
            {"role": "user", "content": question}
        )


        client = openai.Client(
            api_key=AUTH[self.cfg["provider"]],
            base_url=self.cfg["base_url"]
        )

        response = client.chat.completions.create(
            model=self.cfg["model_name"],
            messages=history_openai_format,
            temperature=1.0,
            stream=True
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
        return answer

    @weave.op()
    def default_answer(self, input:str) -> List[List]:
        output = "I don't know your intention. Type 'man' for available commands."
        return output

    @weave.op()
    def execute_command(self, input:str):
        command = input
        cmd_name = command.split()[0]
        if cmd_name == "cd":
            output = self.change_directory(" ".join(command.split()[1:]))
            return output
        elif cmd_name == "verify":
            output = self._verify_logic(" ".join(command.split()[1:]))
            return output
        elif cmd_name == "hint":
            output = self._give_hint(" ".join(command.split()[1:]))
            return output
        elif cmd_name == "tag":
            output = self._show_tag(" ".join(command.split()[1:]))
            return output
        elif cmd_name == "io":
            output = self._write_io_codes(" ".join(command.split()[1:]))
            return output
        elif cmd_name == "explain":
            output = self._retrieve_description(" ".join(command.split()[1:]))
            return output
        elif cmd_name == "complexity":
            output = self._retrieve_description("complexity analysis")
            return output
        elif cmd_name == "man":
            output = self._manual()
            return output
        elif cmd_name == "help":
            output = self._general_advice()
            return output
        elif cmd_name == "chat":
            output = self._chat(command.split()[1:])
            return output
        else:
            output = self._exec_cmd(command)
            return output

    @weave.op()
    def change_directory(self, dest) -> None:
        try:
            os.chdir(dest)
            return os.getcwd()
        except IndexError:
            return "Set destination directory"
        except FileNotFoundError:
            return f"No such file or directory: {dest}"

    @weave.op()
    def _exec_cmd(self, command):
        output = subprocess.run(command.split(), capture_output=True, text=True)
        if len(output.stderr) > 0:
            return output.stderr
        else:
            return output.stdout

    @weave.op()
    def _show_tag(self, algorithm):
        if algorithm in TAGS:
            return "\n".join(TAGS[algorithm])
        else:
            return (
                f"No tag '{algorithm}' exists.\n"
                + "=== Available tags ===\n"
                + "\n".join(TAGS.keys())
            )

    @weave.op()
    def _retrieve_description(self, algorithm):
        vdb = prepare_vdb()
        documents = vdb.similarity_search(algorithm, k=1)
        answer = "\n=======".join([d.page_content for d in documents])
        return answer

    @weave.op()
    def _general_advice(self):
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
    def _write_io_codes(self, spec):
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
            {"role": "user", "content": self._read_problem(problem_number)}
        )


        client = openai.Client(
            api_key=AUTH[self.cfg["provider"]],
            base_url=self.cfg["base_url"]
        )

        response = client.chat.completions.create(
            model=self.cfg["model_name"],
            messages=history_openai_format,
            temperature=1.0,
            stream=True
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
                self._write_to_draft(answer)

        return "Finished writing to draft.py"


    def _write_to_draft(self, content):
        with open(PATH_DRAFT, "a") as f:
            f.write(content)

    @weave.op()
    def _manual(self):
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


    def reference_to_code(self, prompt, commands_discussion=["hint", "verify"]):
        command = prompt.split()[0]
        return command in commands_discussion


    def read_txt(self, path: str):
        return open(path).read()

    @weave.op()
    def _problem_tags(self, problem_number):
        TAGS = [e["tags"] for e in PROB_LISTS if e["problemId"] == int(problem_number)]

        if len(TAGS) == 0:
            raise ValueError("No tag found.")
        elif len(TAGS) > 1:
            raise ValueError("More than 1 tags exist.")
        else:
            return ", ".join(TAGS[0])

    @weave.op()
    def _read_problem(self, problem_num: str):
        try:
            return self.read_txt(os.path.join(HOME_DIR, f"problems/{problem_num}.txt"))
        except Exception:
            return "Not found"

    @weave.op()
    def _read_tags(self, problem_num: str):
        try:
            return self._problem_tags(problem_num)
        except Exception:
            return "Not found"

    @weave.op()
    def _read_solution(self, problem_num: str):
        try:
            return self.read_txt(os.path.join(HOME_DIR, f"solutions/{problem_num}.txt"))
        except Exception:
            return "Not found"

    @weave.op()
    def _problem_prompt(self, problem_num: str):
        problem_desc = self._read_problem(problem_num)
        ptags = self._read_tags(problem_num)
        solution = self._read_solution(problem_num)

        return (
            f"### problem: {problem_desc}\n\n"
            + f"### tags: {ptags}\n\n"
            + f"### answer code: {solution}"
        )

    @weave.op()
    def _verify_logic(self, prompt: str):
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
                "content": "You're an expert in algorithms. You are presented with a problem, algorithm tags, and a answer code. Your task is to guide the user by verifying the approach to solving the problem, taking into account the provided algorithm tags and solution code. Answer 'Correct' if the logic is accurate and conclude there. Answer 'Incorrect' if it isn't providing reasons briefly.",
            }
        ]

        history_openai_format.append(
            {
                "role": "user",
                "content": self._problem_prompt(problem_num)
                + "\n\n"
                + f"### user's logic: {logic}",
            }
        )

        client = openai.Client(
            api_key=AUTH[self.cfg["provider"]],
            base_url=self.cfg["base_url"]
        )
        response = client.chat.completions.create(
            model=self.cfg["model_name"],
            messages=history_openai_format,
            temperature=1.0,
            stream=True
        )

        answer = ""
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                answer += chunk.choices[0].delta.content

        return answer

    @weave.op()
    def _give_hint(self, problem_num: str):
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
            {"role": "user", "content": self._problem_prompt(problem_num)}
        )

        client = openai.Client(
            api_key=AUTH[self.cfg["provider"]],
            base_url=self.cfg["base_url"]
        )
        
        response = client.chat.completions.create(
            model=self.cfg["model_name"],
            messages=history_openai_format,
            temperature=1.0,
            stream=True
        )

        answer = ""
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                answer += chunk.choices[0].delta.content

        return answer
    
    @weave.op()
    def _chat(self, input: str):
        history_openai_format = [
            {
                "role": "system",
                "content": f"Chat with the assistant to get help on solving problems. {input}"
            }
        ]

        client = openai.Client(
            api_key=AUTH[self.cfg["provider"]],
            base_url=self.cfg["base_url"]
        )
        
        response = client.chat.completions.create(
            model=self.cfg["model_name"],
            messages=history_openai_format,
            temperature=1.0,
            stream=True
        )

        answer = ""
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                answer += chunk.choices[0].delta.content

        return answer

    def is_command(self, prompt):
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
            "chat",
        ]
        command = prompt.split()[0]
        return command in commands