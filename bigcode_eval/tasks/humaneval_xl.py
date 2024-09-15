"""HumanEval-XL: A Multilingual Code Generation Benchmark for Cross-lingual Natural Language Generalization
https://aclanthology.org/2024.lrec-main.735/

The HumanEval-XL dataset offers a comprehensive evaluation platform for multilingual LLMs, allowing the 
assessment of the understanding of different NLs.

Homepage: https://github.com/FloatAI/HumanEval-XL
"""


import json
import os
import tempfile
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from bigcode_eval.base import Task
from bigcode_eval.tasks.custom_metrics.multiple_metrics.evaluation import evaluate_problem
from bigcode_eval.tasks.custom_metrics.multiple_metrics.single_experiment_pass_k import for_file


_CITATION = """
@inproceedings{peng-etal-2024-humaneval,
    title = "{H}uman{E}val-{XL}: A Multilingual Code Generation Benchmark for Cross-lingual Natural 
        Language Generalization",
    author = "Peng, Qiwei  and
        Chai, Yekun  and
        Li, Xuhong",
    editor = "Calzolari, Nicoletta  and
        Kan, Min-Yen  and
        Hoste, Veronique  and
        Lenci, Alessandro  and
        Sakti, Sakriani  and
        Xue, Nianwen",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, 
        Language Resources and Evaluation (LREC-COLING 2024)",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.lrec-main.735",
    pages = "8383--8394",
}
"""

NL_LIST = [
    'afrikaans',
    'arabic', 
    'bulgarian', 
    'chinese', 
    'dutch', 
    'english', 
    'estonian', 
    'finnish', 
    'french', 
    'german', 
    'greek', 
    'hebrew', 
    'hungarian', 
    'indonesian', 
    'italian', 
    'malay', 
    'persian', 
    'portuguese', 
    'russian', 
    'spanish', 
    'tagalog', 
    'turkish', 
    'vietnamese', 
]

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
        f"humaneval-xl-{programming_language}-{natural_language}": create_task(natural_language, programming_language) 
        for natural_language in NL_LIST for programming_language in PL_LIST
    }


def create_task(nl, pl):
    class HumanEvalXL(GeneralHumanEvalXL):
        def __init__(self):
            super().__init__(nl, pl)

    return HumanEvalXL


class GeneralHumanEvalXL(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    DATASET_PATH = "iNeil77/HumanEval-XL"
    DATASET_NAME = None
    DATASET_REVISION = "d2b8e0089411c7970e26b681ef0f2797b8e9ec28"

    def __init__(self, natural_language, programming_language, num_workers=16):
        self.natural_language = natural_language
        self.programming_language = programming_language
        self.workers = min(num_workers, os.cpu_count() - 1)
        self.DATASET_NAME = programming_language
        # we need the dataset to get stop words for each language
        self.dataset = load_dataset(
            GeneralHumanEvalXL.DATASET_PATH,
            self.DATASET_NAME,
            revision=self.DATASET_REVISION,
            trust_remote_code=True,
            split=natural_language,
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
        return doc["tests"]

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
                "language": EXT_MAP[self.language],
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
