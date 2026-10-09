import os

import numpy as np
import os
import hydra
import numpy as np
import random

from cliport import tasks
from cliport.dataset import RavensDataset
from cliport.environments.environment import Environment

from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import TerminalFormatter
import re

import IPython
import time
import pybullet as p
import traceback
from datetime import datetime
from pprint import pprint
import cv2
import re
import random
import json
import operator
import csv
import itertools
import warnings

from gensim.llm import (
    chat_completion,
    completion,
    get_context_budget,
    get_llm_model,
)

model = None

def set_gpt_model(gpt_model_name):
    """Deprecated shim; model selection comes from LLM_MODEL in root .env."""
    global model
    model = get_llm_model()
    warnings.warn(
        "set_gpt_model() is deprecated; set LLM_MODEL in the GenSim root .env.",
        DeprecationWarning,
        stacklevel=2,
    )
    if gpt_model_name and gpt_model_name != model:
        print(f"ignoring legacy model argument; using LLM_MODEL={model}")
    else:
        print("use LLM model:", model)

def mkdir_if_missing(dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)


def save_text(folder, name, out):
    mkdir_if_missing(folder)
    with open(os.path.join(folder, name + ".txt"), "w") as fhandle:
        fhandle.write(out)


def add_to_txt(full_interaction, message, with_print=False):
    """ Add the message string to the full interaction """
    full_interaction.append("\n\n"+message)
    if with_print:
        print("\n\n"+message)
    return full_interaction

def get_task_import_str():
    return "import numpy as np\n" + \
    "import os\n" + \
    "import pybullet as p\n" + \
    "import random\n" + \
    "from cliport.tasks import primitives\n" + \
    "from cliport.tasks.grippers import Spatula\n" + \
    "from cliport.tasks.task import Task\n" + \
    "from cliport.utils import utils\n"

def extract_code(res):
    """ parse code block """
    # Pattern to find string between ```
    pattern = r'```(.*?)```'

    # Use re.findall to get all substrings within ```
    code_string = re.findall(pattern, res, re.DOTALL)
    if len(code_string) == 0:
        print("\n".join(res.split("\n")))
        print("empty code string")
        return '', ''

    code_string = code_string[0]
    code_string = code_string.replace('python', '')
    code_lines = code_string.split("\n")

    if 'python' in code_string:
        code_lines = code_lines[1:] # skip the first line

    class_def = [line for line in code_lines if line.startswith('class')]
    task_name = class_def[0]
    task_name = task_name[task_name.find("class "): task_name.rfind("(Task)")][6:]

    print("task_name:", task_name)
    return get_task_import_str() + '\n'.join(code_lines).strip(), task_name

def extract_code_topdown(res):
    """ parse code block """
    # Pattern to find string between ```
    pattern = r'```python\n(.*?)```'
    # pattern = r'```python\n(.*?)'
    # Use re.findall to get all substrings within ```
    # code_string = re.findall(pattern, res, re.DOTALL)
    print(res)
    code_string = res[res.index("```python\n"):].strip()
    if len(code_string) == 0:
        print("\n".join(res.split("\n")))
        print("empty code string")
        return '', ''

    # code_string = code_string[0]
    code_string = code_string.replace('python', '')
    code_lines = code_string.split("\n")[1:]
    if code_lines[-1].strip().endswith(","):
        code_lines[-1] = code_lines[-1][:-1] + "))"
    if 'python' in code_string:
        code_lines = code_lines[1:] # skip the first line

    class_def = [line for line in code_lines if line.startswith('class')]
    task_name = class_def[0]
    task_name = task_name[task_name.find("class "): task_name.rfind("(Task)")][6:]
    # IPython.embed()
    print("task_name:", task_name)
    return '\n'.join(code_lines).strip(), task_name

def extract_code_topdown_offline(res):
    """ parse code block """
    # Pattern to find string between ```
    # pattern = r'```python\n(.*?)```'
    # pattern = r'```python\n(.*?)'
    # Use re.findall to get all substrings within ```
    # code_string = re.findall(pattern, res, re.DOTALL)
    # code_string = res[res.index("```python\n"):].strip()

    print(res)

    # if len(code_string) == 0:
        # try again without python
    pattern = r'```(.*?)```'
    code_string = res[res.index("```"):].strip()

    if len(code_string) == 0:
        print("\n".join(res.split("\n")))
        print("empty code string")
        return '', ''

    # code_string = code_string[0]
    code_string = code_string.replace('python', '')
    code_lines = code_string.split("\n")[1:]
    if code_lines[-1].strip().endswith(","):
        code_lines[-1] = code_lines[-1][:-1] + "))"
    if 'python' in code_string:
        code_lines = code_lines[1:] # skip the first line

    class_def = [line for line in code_lines if line.startswith('class')]
    task_name = class_def[0]
    task_name = task_name[task_name.find("class "): task_name.rfind("(Task)")][6:]
    print("task_name:", task_name)
    return '\n'.join(code_lines).strip(), task_name

def extract_dict(res, prefix="new_task"):
    """ parse task dictionary """
    pattern = r'{(.*?)}'
    code_string = re.findall(pattern, res, re.DOTALL)
    if len(code_string) == 0:
      return ''

    code_string = code_string[0]
    code_string = code_string.replace('python', '')

    return prefix + '={'+ code_string.replace("\n","").strip() + '}'



def extract_list(res, prefix="code_reference"):
    """ parse task dictionary """
    pattern = r'\[(.*?)\]'
    code_string = re.findall(pattern, res, re.DOTALL)
    if len(code_string) == 0:
      return ''

    code_string = code_string[0]
    return prefix + '=[' + code_string.strip() + ']'

def extract_assets(res):
    """ parse generated assets """
    pattern = r'<?xml(.*?)</robot>'
    code_string = re.findall(pattern, res, re.DOTALL)

    assets_pattern = r'robot name="(.*?)">'
    assets_string = re.findall(assets_pattern, res, re.DOTALL)
    if len(code_string) == 0:
        return {}

    try:
        new_urdf = {}
        for asset_path, code in zip(assets_string, code_string):
            new_urdf[asset_path] = "<?xml"+code

        # new_urdf_cmd ='new_urdf={' + code_string[0].rstrip() + '}'
        # exec(new_urdf_cmd)
        return new_urdf

    except:
        print("asset creation failure")
        print(str(traceback.format_exc()))
        return None

def save_stat(cfg, output_dir, env_names, syntax_rate, run_rate, env_success_rate, diversity_score):
    """ save run results """
    print("=========================================================")
    print(f"{cfg['prompt_folder']} | TOTAL SYNTAX_PASS_RATE: {syntax_rate * 100:.1f}% RUNTIME_PASS_RATE: {run_rate * 100:.1f}% ENV_PASS_RATE: {env_success_rate * 100:.1f}% DIVERSITE SCORE: {diversity_score:.3f}")
    print("=========================================================")

    with open(os.path.join(output_dir, "eval_results.csv"), "w") as f:
        writer = csv.writer(f)
        row_info_name = ["prompt", "metric", "success"]
        writer.writerow(row_info_name)
        for col, stat in zip(["syntax", "runtime", "env. completion"], [syntax_rate, run_rate, env_success_rate]):
            row_info = [cfg['prompt_folder'], col, stat]
            writer.writerow(row_info)

def format_dict_prompt(task_name_dict, sample_num=-1, sort_items=False):
    """ format a saved dictionary into prompt """
    if sort_items:
        task_name_dict = sorted(task_name_dict.items(), key=operator.itemgetter(0))
    prompt_replacement = ''
    sample_idx = list(range(len(task_name_dict)))
    random.shuffle(sample_idx)

    if sample_num > 0:
        sample_idx = np.random.choice(sample_idx, sample_num, replace=False)

    for idx, (task_name, task_desc) in enumerate(task_name_dict.items()):
        if idx in sample_idx:
            prompt_replacement += f'- {task_name}: {task_desc}\n'

    return prompt_replacement + "\n\n"

def format_list_prompt(task_list, sample_num=-1, sort_items=False):
    """ format a saved dictionary into prompt """

    # if sort_items:
    #     task_list = sorted(task_list, key=operator.itemgetter(0))
    prompt_replacement = ''
    sample_idx = list(range(len(task_list)))

    if sample_num > 0:
        sample_idx = np.random.choice(len(task_list), sample_num, replace=False)

    for idx, task in enumerate(task_list):
        if idx in sample_idx:
            prompt_replacement += f"- {task['task-name']}: {task['task-descriptions']}\n"

    return prompt_replacement + "\n\n"

def sample_list_reference(item_list, sample_num=-1):
    """ sample reference code from a list of python files """
    sample_idx = list(range(len(item_list)))
    prompt_replacement = ''

    if sample_num > 0:
        sample_idx = np.random.choice(len(item_list), sample_num, replace=False)

    print("reference files: ", [item_list[idx] for idx in sample_idx])
    for idx, item in enumerate(item_list):
        try:
            item_content = open(f"cliport/tasks/{item}").read()
        except:
            # one or the other
            item_content = open(f"cliport/generated_tasks/{item}").read()

        if idx in sample_idx:
            prompt_replacement += f'```\n{item_content}\n```\n\n'

    return prompt_replacement + "\n\n"


def compute_diversity_score_from_assets_old(task_assets):
    """ compute how many new asset combos are covered by previous by a proxy"""
    if len(task_assets) < 2:
        return 0

    existing_assets = []
    for asset in task_assets:
        new_asset_flag = True
        for existing_asset in existing_assets:
            # it's covered by any previous assets
            if set(asset).issubset(existing_asset):
                new_asset_flag = False
                break

        if new_asset_flag:
            existing_assets.append(asset)

    return len(existing_assets) / len(task_assets)

def iou_assets(asset1, asset2):
    asset1 = set(asset1)
    asset2 = set(asset2)
    return len(asset1 & asset2) / len(asset1 | asset2)

def compute_diversity_score_from_assets(task_assets, total_trials):
    """ compute the pairwise IOU for assets"""
    if len(task_assets) == 0:
        return 0

    score = 0
    pairs = list(itertools.combinations(range(len(task_assets)), 2))
    for j, k in pairs:
        score += 1. - iou_assets(task_assets[j], task_assets[k])

    if len(pairs) == 0:
        return 0

    return score / len(pairs)

_SYSTEM_MESSAGE_PROMPT = (
    "You are a helpful and expert assistant in robot simulation code writing and task design."
)


def _estimate_message_tokens(message):
    return (len(str(message.get("content", ""))) + 3) // 4


def truncate_message_for_token_limit(message_history, max_tokens=None):
    if max_tokens is None:
        max_tokens = get_context_budget()
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
        raise ValueError("max_tokens must be a positive integer.")
    history = list(message_history)
    current_prompt = None
    if history and history[-1].get("role") == "user":
        current_prompt = history.pop()

    if len(history) % 2:
        raise ValueError("message history must contain complete user/assistant exchanges.")
    exchanges = []
    for index in range(0, len(history), 2):
        user_message, assistant_message = history[index:index + 2]
        if (
            user_message.get("role") != "user"
            or assistant_message.get("role") != "assistant"
        ):
            raise ValueError("message history must contain complete user/assistant exchanges.")
        exchanges.append((user_message, assistant_message))

    system_tokens = _estimate_message_tokens({"content": _SYSTEM_MESSAGE_PROMPT})
    current_tokens = _estimate_message_tokens(current_prompt) if current_prompt else 0
    if system_tokens + current_tokens > max_tokens:
        raise ValueError(
            "Current prompt plus system message exceeds LLM_CONTEXT_BUDGET."
        )

    tokens = system_tokens + current_tokens
    selected_exchanges = []
    for exchange in reversed(exchanges):
        exchange_tokens = sum(_estimate_message_tokens(message) for message in exchange)
        if tokens + exchange_tokens > max_tokens:
            break
        selected_exchanges.append(exchange)
        tokens += exchange_tokens

    selected_exchanges.reverse()
    truncated_messages = [message for exchange in selected_exchanges for message in exchange]
    if current_prompt is not None:
        truncated_messages.append(current_prompt)
    return truncated_messages

def insert_system_message(message_history):
    message_history.insert(0, {"role": "system", "content": _SYSTEM_MESSAGE_PROMPT})

# globally always feed the previous reply as the assistant message back into the model
existing_messages = []
def generate_feedback(prompt, max_tokens=None, temperature=0.0, interaction_txt=None, retry_max=5, n=1, phase=None):
    """Generate chat feedback while keeping the legacy scalar/list return shape."""
    global existing_messages
    if get_llm_model() == "text-davinci-003":
        return generate_feedback_completion_only(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            interaction_txt=interaction_txt,
            retry_max=retry_max,
            n=n,
            phase=phase,
        )
    user_message = {"role": "user", "content": prompt}
    messages = truncate_message_for_token_limit([*existing_messages, user_message])
    insert_system_message(messages)
    responses = chat_completion(
        messages,
        max_tokens=max_tokens,
        temperature=temperature,
        n=n,
        phase=phase,
    )

    # retry_max remains accepted for compatibility; the SDK owns request retries.
    existing_messages.extend((user_message, {"role": "assistant", "content": responses[0]}))
    if interaction_txt is not None:
        add_to_txt(interaction_txt, ">>> Prompt: \n" + prompt, with_print=False)
        add_to_txt(interaction_txt, ">>> Answer: \n" + responses[0], with_print=False)

    to_print = highlight(f"{responses[0]}", PythonLexer(), TerminalFormatter())
    print(to_print)
    return responses if n > 1 else responses[0]

def clear_messages():
    global existing_messages
    existing_messages = []


def format_finetune_prompt(task_name):
    instruction_text = open('prompts/finetune_instructions_prompt.txt').read()
    instruction_text = instruction_text.replace("TASK_NAME_TEMPLATE", task_name)
    prompt_text = instruction_text
    return prompt_text

def format_finetune_prompt_codeonly(task_name, prompt_file='finetune_instructions_prompt_codeonly.txt'):
    instruction_text = open(f'prompts/{prompt_file}').read()
    instruction_text = instruction_text.replace("TASK_NAME_TEMPLATE", task_name)
    prompt_text = instruction_text
    return prompt_text

def generate_feedback_completion_only(prompt, max_tokens=None, temperature=0.0, interaction_txt=None, retry_max=5, n=1, phase=None):
    """Generate text using a legacy completion endpoint."""
    print("prompt size:", len(prompt))
    responses = completion(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        n=n,
        phase=phase,
    )
    if interaction_txt is not None:
        add_to_txt(interaction_txt, ">>> Prompt: \n" + prompt, with_print=False)
        add_to_txt(interaction_txt, ">>> Answer: \n" + responses[0], with_print=False)
    print(highlight(f"{responses[0]}", PythonLexer(), TerminalFormatter()))
    return responses if n > 1 else responses[0]
