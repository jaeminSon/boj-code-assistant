from models.assistant import Assistant
from llm_judge import LLMJudge
import omegaconf
import argparse
import weave
import asyncio
import json

def get_dataset():
    problem_info = json.load(open('./metainfo/problem_lists.json'))
    import random

    # Get all unique tags
    all_tags = set()
    for problem in problem_info:
        all_tags.update(problem['tags'])

    # Create a dictionary to store the sampled problems
    sampled_problems = {}

    # Sample one problem per tag
    for tag in all_tags:
        # Get all problems with the current tag
        problems_with_tag = [problem for problem in problem_info if tag in problem['tags']]
        
        # Sample one problem randomly
        sampled_problem = random.choice(problems_with_tag)
        
        # Add the sampled problem to the dictionary
        sampled_problems[tag] = sampled_problem

    samples = []
    questions = []
    for d in sampled_problems.items():
        pid = d[1]['problemId']
        title = d[1]['titleKo']
        category = d[0]
        problem = open(f'./problems/{pid}.txt', 'r').read()
        samples.append({'problemId': pid, 'title': title, 'category': category, 'problem': problem})
        questions.append({"question": problem})
    return questions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="gpt-4o", help="Configuration name")
    args = parser.parse_args()
    
    cfg = omegaconf.OmegaConf.load(f"./configs/{args.config}.yaml")
    
    model = Assistant(
        name = cfg.name,
        description = cfg.description,
        cfg = cfg
        )
    judge = LLMJudge()
    questions = get_dataset()

    evaluation = weave.Evaluation(
        name = 'LLMJudge',
        dataset = questions,
        scorers=[judge],
    )
    weave.init('do-something/boj-code-assistant')
    asyncio.run(evaluation.evaluate(model))