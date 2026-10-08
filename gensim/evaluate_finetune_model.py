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
import traceback
import pybullet as p
import IPython
from gensim.topdown_sim_runner import TopDownSimulationRunner
import hydra
from datetime import datetime

from gensim.memory import Memory
from gensim.llm import chat_completion, completion, configure_llm, get_llm_model
from gensim.utils import format_finetune_prompt

@hydra.main(config_path='../cliport/cfg', config_name='data', version_base="1.2")
def main(cfg):
    task = cfg.target_task
    configure_llm()
    model = get_llm_model()
    requested_model = cfg.get('target_model')
    if requested_model and requested_model != model:
        raise ValueError(
            f"target_model {requested_model!r} does not match LLM_MODEL {model!r}; update .env."
        )
    prompt = format_finetune_prompt(task)
    # model_time = datetime.now().strftime("%d_%m_%Y_%H:%M:%S")

    #
    cfg['model_output_dir'] = os.path.join(cfg['output_folder'], cfg['prompt_folder'] + "_" + model)
    if 'seed' in cfg:
       cfg['model_output_dir'] = cfg['model_output_dir'] + f"_{cfg['seed']}"

    memory = Memory(cfg)
    simulation_runner = TopDownSimulationRunner(cfg, memory)

    for trial_i in range(cfg['trials']):
        legacy_completion_model = any(
            name in model.lower() for name in ("davinci", "curie", "babbage", "ada")
        )
        if 'new_finetuned_model' in cfg or not legacy_completion_model:
            messages = [
                {"role": "system", "content": "You are an AI in robot simulation code and task design."},
                {"role": "user", "content": prompt},
            ]
            res = chat_completion(messages, temperature=0.01, max_tokens=1000, stop=["\n```\n"])[0]
        else:
            res = completion(prompt, temperature=0, max_tokens=1800, stop=["\n```\n"])[0]

        simulation_runner.task_creation(res)
        simulation_runner.simulate_task()
        simulation_runner.print_current_stats()

    simulation_runner.save_stats()




# load few shot prompts


if __name__ == "__main__":
    main()
