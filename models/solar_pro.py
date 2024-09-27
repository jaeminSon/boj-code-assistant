import os
import weave
from weave import Model
from typing import List
from openai import OpenAI
from models.common import _verify_logic, _give_hint, _show_tag, _write_io_codes, _retrieve_description, _manual, _general_advice
from models.common import _exec_cmd, is_command
from models.common import change_directory
from models.common import read_yaml, homedir
from pydantic import Field


class SolarPro(Model):
    client: OpenAI = Field(default_factory=lambda: OpenAI(
        api_key=read_yaml(os.path.join(homedir, "authentication/upstage.yaml"))["api_key"],
        base_url="https://api.upstage.ai/v1/solar"
        ))
    model_name: str = "solar-pro"

    @weave.op()
    def predict(self, input_data):
        print(input_data)
        self.handle_query(input_data)

    @weave.op()
    def handle_query(self, history: List[List]) -> str:
        recent_user_message = history[-1][0]
        if is_command(recent_user_message):
            return self.execute_command(history)
        else:
            return self.default_answer(history)

    @weave.op()
    def default_answer(self, history: List[List]) -> List[List]:
        history[-1][1] = "I don't know your intention. Type 'man' for available commands."
        return history

    @weave.op()
    def execute_command(self, history):
        command = history[-1][0]
        cmd_name = command.split()[0]
        if cmd_name == "cd":
            history[-1][1] = change_directory(" ".join(command.split()[1:]))
            return history
        elif cmd_name == "verify":
            history[-1][1] = _verify_logic(" ".join(command.split()[1:]), self.client, self.model_name)
            return history
        elif cmd_name == "hint":
            history[-1][1] = _give_hint(" ".join(command.split()[1:]), self.client, self.model_name)
            return history
        elif cmd_name == "tag":
            history[-1][1] = _show_tag(" ".join(command.split()[1:]))
            return history
        elif cmd_name == "io":
            history[-1][1] = _write_io_codes(" ".join(command.split()[1:]), self.client, self.model_name)
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


if __name__ == "__main__":
    
    weave.init('wandb-korea/boj-code-assistant')
    model = SolarPro()
    model.predict([['hint 1024', None]])