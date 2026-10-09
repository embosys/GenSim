
# GenSim: Generating Robotic Simulation Tasks via Large Language Models

### Lirui Wang, Yiyang Ling, Zhecheng Yuan, Mohit Shridhar, Chen Bao, Yuzhe Qin, Bailin Wang, Huazhe Xu, Xiaolong Wang

[Project Page](https://liruiw.github.io/gensim) | [Arxiv](https://arxiv.org/abs/2310.01361) | [Gradio Demo](https://huggingface.co/spaces/Gen-Sim/Gen-Sim) | [Huggingface Dataset](https://huggingface.co/datasets/Gen-Sim/Gen-Sim) | [Finetuned Code-LLama Model](https://huggingface.co/Gen-Sim/Gen-Sim) | [GPTs](https://chat.openai.com/g/g-rqxeNpjxd-gensim)

This repo explores the use of an LLM code generation pipeline to write simulation environments and expert goals to augment diverse simulation tasks. Strongly recommend also checking out the [Gradio Demo](https://huggingface.co/spaces/Gen-Sim/Gen-Sim) and [GPTs](https://chat.openai.com/g/g-rqxeNpjxd-gensim).


![](media/gensim_teaser_v1.gif)

## ⚙️ Installation
```bash
uv sync --locked --extra llm
export GENSIM_ROOT="$PWD"
cp -n .env.example .env
```

Edit `.env` and set `LLM_MODEL` and `LLM_API_KEY` for your account. GenSim loads
this file from the repository root even when launched from another directory;
shell environment variables take precedence. `.env` is ignored by Git.
OpenAI, DeepSeek, Qwen, and other OpenAI-compatible Chat Completions endpoints
use the same client: change `LLM_BASE_URL`, `LLM_MODEL`, and `LLM_API_KEY`.
The commented examples in [.env.example](.env.example) show provider settings.
For Qwen, use the endpoint for your workspace/region and a key from that region.

All LLM settings are environment variables, not Hydra options. `gpt_model`,
`openai_key`, and `gpt_temperature` have been removed from `config.yaml`.
`LLM_MAX_OUTPUT_TOKENS` controls output length. Leave `LLM_TEMPERATURE` unset to
retain the original stage temperatures, or set it to `null` for models that do
not accept temperature. `LLM_EXTRA_BODY` is a JSON object for model-specific
parameters (e.g. thinking mode). These parameters cannot replace core request
fields. For models requiring `max_completion_tokens`, set
`LLM_TOKEN_LIMIT_PARAM=max_completion_tokens`.

Set `LLM_STREAM=true` for models requiring streaming; chunks are collected into
one final answer for the existing task parser. Reasoning content is not parsed
as task code. By default, multiple candidates use independent requests with
the same conversation. Set `LLM_SUPPORTS_N=true` only if the selected backend
and model support native `n`. Retries are handled once by the SDK, with
`LLM_TIMEOUT` and `LLM_MAX_RETRIES` controlling their limits.

`LLM_CONTEXT_BUDGET` is an approximate input budget (characters / 4, not a
model-specific tokenizer). Old exchanges are trimmed in pairs; an oversized
current prompt fails clearly instead of being silently removed. Choose this
budget conservatively for your model and language. Empty or truncated model
answers fail before task parsing.


## 🚶Getting Started
After the installation process, you can run: 
```
# basic bottom-up prompt
uv run --extra llm python gensim/run_simulation.py disp=True prompt_folder=vanilla_task_generation_prompt_simple

# bottom-up template generation
uv run --extra llm python gensim/run_simulation.py disp=True prompt_folder=bottomup_task_generation_prompt   save_memory=True load_memory=True  task_description_candidate_num=10 use_template=True

# top-down task generation
uv run --extra llm python gensim/run_simulation.py  disp=True  prompt_folder=topdown_task_generation_prompt save_memory=True load_memory=True task_description_candidate_num=10 use_template=True target_task_name="build-house"

# task-conditioned chain-of-thought generation
uv run --extra llm python gensim/run_simulation.py  disp=True  prompt_folder=topdown_chain_of_thought_prompt save_memory=True load_memory=True task_description_candidate_num=10 use_template=True target_task_name="build-car"
```

## 💾 Add and remove task
0. To remove a task (delete its code and remove it from the task and task code buffer), use ``python misc/purge_task.py -f color-sequenced-block-insertion``
1. To add a task (extract task description to add to buffer), use ``python misc/add_task_from_code.py -f ball_on_box_on_container``


## 🤖 LLM Generated Task Usage
1. All generated tasks in `cliport/generated_tasks` should have automatically been imported
2. Set the task name and then use `demo.py` for visualization. For instance, `python cliport/demos.py n=200 task=build-car mode=test disp=True`.
3.  The following is a guide for training everything from scratch (More details in [cliport](https://github.com/cliport/cliport)). All tasks follow a 4-phase workflow:
    1. Generate `train`, `val`, `test` datasets with `demos.py` 
    2. Train agents with `train.py` 
    3. Run validation with `eval.py` to find the best checkpoint on `val` tasks and save `*val-results.json`
    4. Evaluate the best checkpoint in `*val-results.json` on `test` tasks with `eval.py`


## 🎛️ LLM Finetune

The commands below document the original fine-tuning experiments and legacy
model IDs. Availability of those models and old fine-tuning CLI commands is
not restored by the SDK upgrade. Current API evaluation selects its model
with `LLM_MODEL` in `.env`; an optional old `target_model`/`--model` argument
must agree with it.
1. Prepare data using `python gensim/prepare_finetune_gpt.py`. Released dataset is [here](https://huggingface.co/datasets/Gen-Sim/Gen-Sim)

2. Finetune using openai api ` openai api fine_tunes.create --training_file output/finetune_data_prepared.jsonl --model davinci --suffix 'GenSim'`

3. Evaluate it using `python gensim/evaluate_finetune_model.py  +target_task=build-car +target_model=davinci:ft-mit-cal:gensim-2023-08-06-16-00-56`

4. Compare with `uv run --extra llm python gensim/run_simulation.py  disp=True  prompt_folder=topdown_task_generation_prompt_simple load_memory=True task_description_candidate_num=10 use_template=True target_task_name="build-house" trials=3`

5. Compare with `uv run --extra llm python gensim/run_simulation.py  disp=True  prompt_folder=topdown_task_generation_prompt_simple_singleprompt load_memory=True task_description_candidate_num=10  target_task_name="build-house"`

6. turbo finetuned models. `python gensim/evaluate_finetune_model.py  +target_task=build-car +target_model=ft:gpt-3.5-turbo-0613:  trials=3 disp=True  `

7. Finetune Code-LLAMA using hugging-face transformer library [here](https://github.com/liruiw/llama-recipes)

8. offline eval: `python -m gensim.evaluate_finetune_model_offline model_output_dir=after_finetune_CodeLlama-13b-Instruct-hf_fewshot_False_epoch_10_0`

## 🤖 Policy Benchmark
0. Note that the 100+ generated tasks by GenSim can be used for benchmarking algorithms in multitask policy training. See `scripts/task_list/GPT_*.json` for a list of benchmark settings. Pretrained multitask models can be found [here](https://drive.google.com/drive/folders/1RRSa4hXQKuN1ABuUVEdfV6urqZ99KZ57?usp=drive_link).
1. Generate multitask demonstrations. For example, run  `bash scripts/generate_datasets.sh data 'align-box-corner assembling-kits block-insertion' `
2. Single-task training  `sh scripts/train_test_multi_task.sh data "[align-rope,align-box-corner]`
3. Multi-task training   `sh scripts/train_test_single_task.sh data align-box-corner`


## ✅ Note
0. Temperature `0.5-0.8 `is good range for diversity, `0.0-0.2` is for stable results.
1. The generation pipeline will print out statistics regarding compilation, runtime, task design, and diversity scores. Note that these metric depend on the task compexity that LLM tries to generate.
2. Core prompting and code generation scripts are in `gensim` and training and task scripts are in `cliport`.
3. `prompts/` folder stores different kinds of prompts to get the desired environments. Each folder contains a sequence of prompts as well as a meta_data file. `prompts/data` stores the base task library and the generated task library.
4. The GPT-generated tasks are stored in `generated_tasks/`. Use `demo.py` to play with them.  `cliport/demos_gpt4.py` is an  all-in-one prompt script that can be converted into ipython notebook.
5. Raw text outputs are saved in `output/output_stats`, figure results saved in `output/output_figures`, policy evaluation results are saved in `output/cliport_output`.
6. To debug generated code, manually copy-paste ``generated_task.py`` then run 
``python cliport/demos.py n=50 task=gen-task disp=True``
7. This version of cliport should support `batchsize>1` and can run with more recent versions of pytorch and pytorch lightning.
8. Please use Github issue tracker to report bugs. For other questions please contact [Lirui Wang](wangliruisz@gmail.com)
9. blender rendering `python cliport/demos.py n=310 task=align-box-corner mode=test disp=True +record.blender_render=True record.save_video=True`

![](media/teaser_figure.png)

### Citation
If you find GenSim useful in your research, please consider citing:
```
@inproceedings{wang2023gen,
author    = {Lirui Wang and Yiyang Ling and Zhecheng Yuan and Mohit Shridhar and Chen Bao and Yuzhe Qin and Bailin Wang and Huazhe Xu and Xiaolong Wang},
title     = {GenSim: Generating Robotic Simulation Tasks via Large Language Models},
booktitle = {Arxiv},
year      = {2023}
}
```


### Scene-local UniSis experiment outputs

`scene_source=unisis` writes each trial into `<scene directory>/gensim/<timestamp>_s<seed>/`.
The original scene-generation mode keeps its existing output layout. Scene files and
assets are referenced in place. Older outputs are not moved.

```text
gensim/<run_id>/
  run_meta.json           # scene YAML hash, source commit, seeds, safe LLM settings
  result.json             # status, native_success, attempts, artifact counts
  run.log                 # console and simulator output
  llm/index.json
  llm/turn_0001_*.json     # actual messages, parameters, raw response, usage, timing/errors
  code/task.py
  trajectory/{color,depth,action,reward,info}/*.pkl
  artifacts/{eval_results.csv,full_interaction.txt}
  artifacts/videos/       # when record.save_video=True
  state/replay_001/       # optional independent oracle replay state
    manifest.json
    final_state.json
    final_state.bullet
```

Run from the repository root with `.env` configured:

```bash
GENSIM_ROOT="$PWD" uv run --locked --extra llm python -m gensim.run_simulation \
  scene_source=unisis unisis.scene_path=/path/to/002_living_room \
  trials=1 max_env_run_cnt=1 disp=False save_data=True
```

Optional overrides: `unisis.output_dir=/path/to/results`, `unisis.run_id=my_run`,
`unisis.seed=123`. Existing run IDs fail instead of overwriting. Multiple trials
append `_t001`, `_t002` to an explicit ID, create fresh task/LLM histories, and
use base seed + trial index; attempts add their own index to that trial seed.
`output_folder` and `data_dir` only control the original mode; UniSis paths are
owned by the run directory. Failed generation and execution also retain results
and available logs. Ordinary errors finish the remaining trials then exit nonzero.

Replay a saved generated task once and export its final PyBullet entity state:

```bash
uv run python scripts/replay_save_state.py /path/to/gensim/<run_id> \
  --seed 123 --settle-seconds 0.5
```

The script defaults to `<run_dir>/code/task.py` and uses the scene, robot setup,
and seed from `run_meta.json`; `--code` and `--scene` can override the paths.
`--vis` shows the PyBullet GUI. Each completed replay receives the next free
`state/replay_NNN/` directory. `final_state.json` stores every mapped scene
entity's root-frame position, WXYZ quaternion, and joint positions, including
entities from unsuccessful oracle sequences. A replay error does not create a
state directory. The manifest records the replay reward separately from the
original run's native result; reward does not filter state export.
`final_state.json` is the portable state
consumed by the shared converter. `final_state.bullet` is a PyBullet-native
backup for restoration with the matching scene and engine setup; the converter
does not read it.

`native_success` retains the GenSim reward (>0.99) and majority-of-attempts rule.
`success` is null until the shared experiment checker is connected. Only successful
native demonstrations are saved; actions are high-level oracle actions, not dense
joint trajectories. LLM call counts refer to logical SDK calls (including separate
fallback calls for `n`), not hidden SDK HTTP retries. Usage stays null if the
provider does not return it, including streams without usage chunks. The YAML
hash does not fingerprint referenced assets; source_dirty flags uncommitted edits.
No credentials or `.env` snapshot are included.
