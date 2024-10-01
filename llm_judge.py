import weave
from weave.flow.scorer import Scorer
from weave import WeaveList
import openai
from models.common import AUTH

class LLMJudge(Scorer):
#Assess the code according to each criterion and return a dictionary that includes only the evaluation scores (1-5) for each criterion.
#'{"Correctness":5,"Efficiency":4,"Readability":3,"ErrorHandling":3,"EdgeCases":3,"OverallScore":3.6}'
    @weave.op()
    def score(self, query, answer):

        history_openai_format = [
            {
                "role": "system",
                "content": """As an LLM judge, your task is to evaluate the quality of Python code generated by different models based on the following evaluation criteria. You are provided with a task description and the generated code. Assess the code according to each criterion and return a overall score that calculates the average of all criteria scores.

Evaluation Criteria:

	1.	Correctness: Does the code fulfill the task requirements as described? (1-5)
	2.	Efficiency: Is the code optimized for performance, avoiding unnecessary computations and utilizing efficient algorithms? (1-5)
	3.	Readability: Does the code follow Pythonic conventions, including clear variable names, comments, and adherence to PEP 8 standards? (1-5)
	4.	Error Handling: Does the code include robust error handling, such as try-except blocks, to prevent unexpected crashes? (1-5)
	6.	Edge Cases: Does the code handle potential edge cases or unusual inputs gracefully? (1-5)


Task Description:
“Write a Python function that takes a list of integers and returns the sum of the even numbers in the list.”

Generated Code:
```python
def sum_of_evens(lst):
    return sum([x for x in lst if x % 2 == 0])
```

Example Evaluation Output:
3.6

Now, evaluate the following generated code based on the task description and criteria, and return the evaluation in the same Python dictionary format as the example above. You have to provide a number only. Do not include any other information in the response.
"""}
        ]

        history_openai_format.append(
            {"role": "user", "content": f"Task Description:\n{query}\n\nGenerated Code:\n```python\n{answer}\n```"}
        )


        client = openai.Client(
            api_key=AUTH["openai"],
            base_url="https://api.openai.com/v1/"
        )

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=history_openai_format,
            temperature=1.0,
        )
        try:
            return {"score":float(response.choices[0].message.content)}
        except:
            return {"score": None}

llm_judge = LLMJudge()