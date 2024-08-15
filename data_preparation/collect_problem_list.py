import requests
import json

from tqdm import tqdm


SEARCH_URL = "https://solved.ac/api/v3/search/problem"


def collect_problem_list(page: int):
    querystring = {"query": "", "page": f"{page}"}

    headers = {"Content-Type": "application/json"}
    response = requests.request("GET", SEARCH_URL, headers=headers, params=querystring)

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
                "level": prob_desc["level"],
            }
        )

    return summarized_problem_lists


def read_json(path):
    with open(path, "r") as f:
        content = json.load(f)
    return content


if __name__ == "__main__":
    path_problem_lists = "problem_lists.json"
    all_problem_lists = []
    for page in tqdm(range(1, 603)):
        problem_lists = collect_problem_list(page)
        all_problem_lists.extend(problem_lists)
    with open(path_problem_lists, "w", encoding="utf8") as f:
        json.dump(all_problem_lists, f, indent="\t", ensure_ascii=False)
