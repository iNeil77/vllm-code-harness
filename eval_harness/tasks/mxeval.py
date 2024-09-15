"""Multi-lingual Evaluation of Code Generation Models
https://openreview.net/forum?id=Bo7eeXm6An8

The MXEval dataset code to perform execution-based multi-lingual evaluation of code generation capabilities.

Homepage: https://github.com/amazon-science/mxeval
"""


import json
import os
import tempfile
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from eval_harness.base import Task
from eval_harness.tasks.custom_metrics.multiple_metrics.evaluation import evaluate_problem
from eval_harness.tasks.custom_metrics.multiple_metrics.single_experiment_pass_k import for_file


_CITATION = """
@inproceedings{DBLP:conf/iclr/AthiwaratkunGWL23,
  author       = {Ben Athiwaratkun and
                  Sanjay Krishna Gouda and
                  Zijian Wang and
                  Xiaopeng Li and
                  Yuchen Tian and
                  Ming Tan and
                  Wasi Uddin Ahmad and
                  Shiqi Wang and
                  Qing Sun and
                  Mingyue Shang and
                  Sujan Kumar Gonugondla and
                  Hantian Ding and
                  Varun Kumar and
                  Nathan Fulton and
                  Arash Farahani and
                  Siddhartha Jain and
                  Robert Giaquinto and
                  Haifeng Qian and
                  Murali Krishna Ramanathan and
                  Ramesh Nallapati},
  title        = {Multi-lingual Evaluation of Code Generation Models},
  booktitle    = {{ICLR}},
  publisher    = {OpenReview.net},
  year         = {2023}
}
"""

PL_LIST = [
    "csharp", 
    "go", 
    "java", 
    "javascript", 
    "perl", 
    "php", 
    "python", 
    "ruby", 
    "scala", 
    "swift", 
    "typescript"
]

EXT_MAP = {
    "csharp": "cs",
    "go": "go",
    "java": "java",
    "javascript": "js",
    "perl": "pl",
    "php": "php",
    "python": "py",
    "ruby": "rb",
    "scala": "scala",
    "swift": "swift",
    "typescript": "ts",
}


def create_all_tasks():
    """Creates a dictionary of tasks from a list of levels
    :return: {task_name: task}
        e.g. {multiple-py: Task, multiple-java: Task}
    """
    return {
        f"mxeval-{programming_language}": create_task(programming_language) 
        for programming_language in PL_LIST
    }


def create_task(nl, pl):
    class MXEval(GeneralMXEval):
        def __init__(self):
            super().__init__(nl, pl)

    return MXEval


class GeneralMXEval(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    DATASET_PATH = "mxeval/multi-humaneval"
    DATASET_NAME = None
    DATASET_REVISION = "b572eb39f2620835bcad352a122b15263519f612"

    def __init__(self, programming_language, num_workers=16):
        self.programming_language = programming_language
        self.workers = min(num_workers, os.cpu_count() - 1)
        self.DATASET_NAME = programming_language
        # we need the dataset to get stop words for each language
        self.dataset = load_dataset(
            GeneralMXEval.DATASET_PATH,
            self.DATASET_NAME,
            revision=self.DATASET_REVISION,
            trust_remote_code=True,
            split="test",
        )
        self.stop_words = self.dataset[0]["stop_tokens"] + ["<file_sep>"]
        self.requires_execution=True,


    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.dataset

    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from."""
        return doc["prompt"].strip()

    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        return doc["test"]

    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
            (not used for this task)
        """
        prompt = self.get_prompt(self.get_dataset()[idx])
        completion = generation[len(prompt) :]
        return prompt + self._stop_at_stop_token(completion, self.stop_words)

    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        """
        # get prompts and problem names
        prompts_names = [
            {"prompt": doc["prompt"], "task_id": doc["task_id"]}
            for i, doc in enumerate(self.get_dataset())
            if i < len(generations)
        ]
        # a common temp dir for all the problems
        temp_dir = tempfile.gettempdir()
        list_files = []
        for (prompt_name, generation, reference) in zip(
            prompts_names, generations, references
        ):
            problem = {
                "task_id": prompt_name["task_id"],
                "language": EXT_MAP[self.programming_language],
                "prompt": prompt_name["prompt"],
                "completions": generation,
                "tests": reference,
            }
            # each problem is save in a json file
            temp_file_name = os.path.join(temp_dir, f"{prompt_name['task_id']}.json")
            list_files.append(temp_file_name)
            with open(temp_file_name, "wt") as f:
                json.dump(problem, f)
        print(
            f"Saved {len(list_files)} problems in {temp_dir} for evaluation, each problem has {len(generations[0])} completions"
        )

        # execute the problems to evaluate them
        for file in tqdm(list_files):
            evaluate_problem(temp_dir, file, self.workers)

        # compute pass@k scores
        result_array = np.array(
            [for_file(p) for p in Path(temp_dir).glob("*.results.json")]
        )
        result = result_array.mean(axis=0)
        results = {
            f"pass@{k}": v
            for k, v in zip([1, 10, 25, 100], result)
            if k <= len(generations[0])
        }
        return results
