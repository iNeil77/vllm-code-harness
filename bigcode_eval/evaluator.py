import json
import os
import warnings

from typing import List
from bigcode_eval import tasks
from bigcode_eval.generation import get_generations

_WARNING = """
################################################################################
                                  !!!WARNING!!!
################################################################################
The "code_eval"/"apps_metric" you are about to use, execute untrusted 
model-generated code in Python.
Although it is highly unlikely that model-generated code will do something
overtly malicious in response to this test suite, model-generated code may act
destructively due to a lack of model capability or alignment.
Users are strongly encouraged to sandbox this evaluation suite so that it
does not perform destructive actions on their host or network. For more
information on how OpenAI sandboxes its code, see the paper "Evaluating Large
Language Models Trained on Code" (https://arxiv.org/abs/2107.03374).
Once you have read this disclaimer and taken appropriate precautions, set the argument 
"allow_code_execution" to True.
################################################################################\
"""

class Evaluator:
    def __init__(self, model, tokenizer, args):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args

        # code evaluation permission
        self.allow_code_execution = args.allow_code_execution

    def generate_text(self, task_name):
        task = tasks.get_task(task_name, self.args)
        dataset = task.get_dataset()
        # if args.limit is None, use all samples
        # if args.limit is used, make sure args.limit_start + args.limit <= len(dataset)
        n_tasks = min(self.args.limit, len(dataset) - self.args.limit_start) if self.args.limit else len(dataset)
        # when args.limit is None
        # adjust n_tasks by args.limit_start to prevent out of bounds issues 
        if not self.args.limit:
            n_tasks -= self.args.limit_start
        references = [task.get_reference(dataset[i]) for i in range(self.args.limit_start, self.args.limit_start+n_tasks)]
        generations = get_generations(
            task,
            dataset,
            self.model,
            self.tokenizer,
            n_tasks=n_tasks,
            args=self.args
        )

        if len(generations[0]) > self.args.n_samples:
            generations = [l[: self.args.n_samples] for l in generations]
            warnings.warn(
                f"Number of tasks wasn't proportional to number of devices, we removed extra predictions to only keep nsamples={self.args.n_samples}"
            )
        return generations, references

    def evaluate(self, task_name):
        task = tasks.get_task(task_name, self.args)
        if task.requires_execution and not self.allow_code_execution:
            raise ValueError(_WARNING)

        generations, references = self.generate_text(task_name)

        if not self.args.load_generations_path:
            self.save_json_files(
                generations, 
                references, 
                self.args.save_generations_path, 
                self.args.save_references_path
            )

        if self.allow_code_execution and task.requires_execution:
            os.environ["HF_ALLOW_CODE_EVAL"] = "1"
        print("Evaluating generations...")
        results = task.process_results(generations, references)
        return results

    def save_json_files(
        self,
        generations: List[str],
        references: List[str],
        save_generations_path: str,
        save_references_path: str,
    ) -> None:
        if self.args.save_generations:
            print(f"saving generations at {save_generations_path}")
            os.makedirs(os.path.dirname(save_generations_path), mode=744, exist_ok=True)
            with open(save_generations_path, "w+") as fp:
                json.dump(generations, fp)
                print(f"generations were saved at {save_generations_path}")
        if self.args.save_references:
            print(f"saving references at {save_references_path}")
            os.makedirs(os.path.dirname(save_references_path), mode=744, exist_ok=True)
            with open(save_references_path, "w+") as fp:
                json.dump(references, fp)
                print(f"references were saved at {save_references_path}")
