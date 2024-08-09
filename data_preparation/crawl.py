import os
import random
import requests
import time
import json
import re

from tqdm import tqdm
from bs4 import BeautifulSoup


SEARCH_URL = "https://solved.ac/api/v3/search/problem"


def crawl_problem_list(page: int):
    querystring = {"query": "", "page": f"{page}"}

    headers = {"Content-Type": "application/json"}
    response = requests.request(
        "GET", SEARCH_URL, headers=headers, params=querystring)

    summarized_problem_lists = []
    problem_list = json.loads(response.text).get("items")
    for prob_desc in problem_list:
        if "problemId" not in prob_desc:
            continue

        if "titleKo" not in prob_desc:
            continue

        # retrieve tags
        retrieved_tags = []
        tags = prob_desc.get("tags", [])
        for tag in tags:
            displayNames = tag.get("displayNames", [])
            for display_name in displayNames:
                if display_name.get("language", "") == "en" and "name" in display_name:
                    retrieved_tags.append(display_name.get("name"))

        summarized_problem_lists.append(
            {
                "problemId": prob_desc["problemId"],
                "titleKo": prob_desc["titleKo"],
                "tags": retrieved_tags,
                "level": prob_desc["level"]
            }
        )

    return summarized_problem_lists


def crawl_problem_content(problem_id: int):
    user_agents_list = [
        "Mozilla/5.0 (iPad; CPU OS 12_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.83 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.51 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/535.11 (KHTML, like Gecko) Ubuntu/11.10 Chromium/17.0.963.65 Chrome/17.0.963.65 Safari/535.11",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/535.11 (KHTML, like Gecko) Ubuntu/11.04 Chromium/17.0.963.65 Chrome/17.0.963.65 Safari/535.11",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/535.11 (KHTML, like Gecko) Ubuntu/10.10 Chromium/17.0.963.65 Chrome/17.0.963.65 Safari/535.11",
        "Mozilla/5.0 (X11; Linux i686) AppleWebKit/535.11 (KHTML, like Gecko) Ubuntu/11.10 Chromium/17.0.963.65 Chrome/17.0.963.65 Safari/535.11",
        "Mozilla/5.0 (X11; Linux i686) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.65 Safari/535.11",
        "Mozilla/5.0 (X11; FreeBSD amd64) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.65 Safari/535.11",
        "Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.65 Safari/535.11",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_2) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.65 Safari/535.11",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_0) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.65 Safari/535.11",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_6_4) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.65 Safari/535.11",
    ]

    HEADERS = {"User-Agent": random.choice(user_agents_list)}

    response = requests.get(
        f"https://www.acmicpc.net/problem/{problem_id}", headers=HEADERS
    )
    soup = BeautifulSoup(response.text, "html.parser")
    problem_desc = soup.select("#problem_description")
    input_desc = soup.select("#problem_input")
    output_desc = soup.select("#problem_output")
    sample_input = soup.select("pre[id^=sample-input]")
    sample_output = soup.select("pre[id^=sample-output]")

    problem_str = "Problem Description: " + \
        "".join([e.text for e in problem_desc])
    input_str = "Input: " + "".join([e.text for e in input_desc])
    output_str = "Output: " + "".join([e.text for e in output_desc])
    sample_input_output = "\n".join(
        [
            f"Case#{n+1}\ninput:\n{i.text}output:\n{o.text}"
            for n, (i, o) in enumerate(zip(sample_input, sample_output))
        ]
    )

    return f"{problem_str}\n{input_str}\n{output_str}\n{sample_input_output}"


def read_json(path):
    with open(path, "r") as f:
        content = json.load(f)
    return content


if __name__ == "__main__":
    path_problem_lists = "problem_lists.json"
    second_wait_webpage = 2

    # save problem lists to single json file
    all_problem_lists = []
    for page in tqdm(range(1, 603)):
        problem_lists = crawl_problem_list(page)
        all_problem_lists.extend(problem_lists)
    with open(path_problem_lists, "w", encoding='utf8') as f:
        json.dump(all_problem_lists, f, indent="\t", ensure_ascii=False)

    # save problems to files
    dir_save = "problems"
    os.makedirs(dir_save, exist_ok=True)
    all_problem_lists = read_json(path_problem_lists)
    for problem_desc in tqdm(all_problem_lists):
        problem_id = problem_desc["problemId"]
        path_save = os.path.join(dir_save, f"{problem_id}.txt")
        if os.path.exists(path_save):
            continue

        try:
            problem_content = crawl_problem_content(problem_id)
            with open(path_save, "w") as f:
                f.write(problem_content)
        except Exception:
            pass

        time.sleep(second_wait_webpage)
