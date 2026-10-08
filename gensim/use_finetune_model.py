import argparse
import os
from cliport import tasks
from cliport.dataset import RavensDataset
from cliport.environments.environment import Environment

from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import TerminalFormatter

import time
import random
import json

from gensim.llm import chat_completion, completion, configure_llm, get_llm_model
from gensim.utils import format_finetune_prompt



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default='build-car')
    parser.add_argument("--model", type=str, default=None)
    # davinci:ft-mit-cal:gensim-2023-08-06-16-00-56
    args = parser.parse_args()
    task = args.task
    configure_llm()
    model = get_llm_model()
    if args.model and args.model != model:
        raise ValueError(f"--model {args.model!r} does not match LLM_MODEL {model!r}; update .env.")
    prompt = format_finetune_prompt(task)
    legacy_completion_model = any(
        name in model.lower() for name in ("davinci", "curie", "babbage", "ada")
    )
    if legacy_completion_model:
        res = completion(prompt, temperature=0, max_tokens=1024)[0]
    else:
        messages = [
            {"role": "system", "content": "You are an AI in robot simulation code and task design."},
            {"role": "user", "content": prompt},
        ]
        res = chat_completion(messages, temperature=0, max_tokens=1024)[0]

    print("code!:", res)
    python_file_path = f"cliport/generated_tasks/finetune_{task.replace('-','_')}.py"
    print(f"saving task {args.task} to {python_file_path}")

    # evaluate and then save
    # with open(python_file_path, "w",
        #         ) as fhandle:
        # fhandle.write(res)

