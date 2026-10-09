import numpy as np
import os
import hydra
import random

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
import hashlib

from gensim.agent import Agent
from gensim.critic import Critic
from gensim.sim_runner import SimulationRunner
from gensim.memory import Memory
from gensim.llm import configure_llm
from gensim.utils import clear_messages


@hydra.main(config_path='../cliport/cfg', config_name='data', version_base="1.2")
def main(cfg):
    if str(cfg.get("scene_source", "original")).lower() == "unisis":
        return run_unisis(cfg)
    configure_llm()

    model_time = datetime.now().strftime("%d_%m_%Y_%H:%M:%S")
    cfg['model_output_dir'] = os.path.join(cfg['output_folder'], cfg['prompt_folder'] + "_" + model_time)
    if 'seed' in cfg:
       cfg['model_output_dir'] = cfg['model_output_dir'] + f"_{cfg['seed']}"

    memory = Memory(cfg)
    agent = Agent(cfg, memory)
    critic = Critic(cfg, memory)
    simulation_runner = SimulationRunner(cfg, agent, critic, memory)

    for trial_i in range(cfg['trials']):
        simulation_runner.task_creation()
        simulation_runner.simulate_task()
        simulation_runner.print_current_stats()
        clear_messages()

    simulation_runner.save_stats()

def run_unisis(cfg):
    from omegaconf import OmegaConf
    from gensim.run_output import RunOutput, capture_console
    from gensim.llm import set_run_output, get_llm_metadata
    failures = 0
    for trial in range(int(cfg["trials"])):
        trial_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        output = RunOutput(trial_cfg, trial)
        trial_cfg["model_output_dir"] = str(output.path / "artifacts")
        trial_cfg["data_dir"] = str(output.path / "trajectory")
        trial_cfg["record"]["save_video_path"] = str(output.path / "artifacts/videos")
        memory = None
        runner = None
        clear_messages()
        set_run_output(output)
        with capture_console(output.path / "run.log"):
            try:
                print("Run directory:", output.path)
                output.phase = "configuration"
                configure_llm()
                output.meta["llm"] = get_llm_metadata()
                output.phase = "scene_loading"
                output.meta["scene_sha256"] = hashlib.sha256(output.scene.read_bytes()).hexdigest()
                random.seed(output.seed)
                np.random.seed(output.seed)
                memory = Memory(trial_cfg)
                agent = Agent(trial_cfg, memory)
                output.meta["prompt_folder"] = trial_cfg["prompt_folder"]
                critic = Critic(trial_cfg, memory)
                runner = SimulationRunner(trial_cfg, agent, critic, memory)
                runner.output = output
                runner.task_creation()
                runner.simulate_task()
                runner.print_current_stats()
                output.phase = "saving"
                runner.save_stats()
            except BaseException as error:
                output.fail(error)
                if isinstance(error, (KeyboardInterrupt, SystemExit)):
                    output.result["status"] = "interrupted"
                    raise
                traceback.print_exc()
            finally:
                try:
                    if runner is not None and hasattr(runner, "env") and p.isConnected():
                        if trial_cfg["record"]["save_video"]:
                            runner.env.end_rec()
                    if p.isConnected():
                        p.disconnect()
                except Exception as error:
                    output.fail(error)
                    traceback.print_exc()
                try:
                    if memory is not None:
                        memory.save_run(getattr(runner, "generated_task", {}))
                except Exception as error:
                    output.phase = "saving"
                    output.fail(error)
                    traceback.print_exc()
                finally:
                    output.finish()
                    set_run_output(None)
                    clear_messages()
                print("Result:", output.result["status"], "native_success:", output.result["native_success"])
        failures += output.result["status"] in {"error", "interrupted"}
    if failures:
        raise RuntimeError(f"{failures} UniSis trial(s) failed; see scene-local result.json and run.log.")


if __name__ == '__main__':
    main()
