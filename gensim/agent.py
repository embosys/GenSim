import numpy as np
import os
import IPython
import random
import json
import traceback
import ast
import copy
import re
from pathlib import Path
import pybullet as p
from gensim.utils import (
    save_text,
    add_to_txt,
    extract_code,
    extract_dict,
    extract_list,
    extract_assets,
    format_dict_prompt,
    sample_list_reference,
    generate_feedback,
)


class Agent:
    """
    class that design new tasks and codes for simulation environments
    """
    def __init__(self, cfg, memory):
        self.cfg = cfg
        self.model_output_dir = cfg["model_output_dir"]
        self.scene_source = str(cfg.get("scene_source", "original")).lower()
        if self.scene_source not in {"original", "unisis"}:
            raise ValueError(
                f"Unsupported scene_source {self.scene_source!r}; expected 'original' or 'unisis'."
            )
        prompt_folder = cfg["prompt_folder"]
        # The UniSis prompt chain is the default only when the original
        # repository prompt folder is still selected. Custom folders are kept.
        if self.scene_source == "unisis" and prompt_folder == "vanilla_task_generation_prompt":
            prompt_folder = "unisis_task_execution_prompt"
            cfg["prompt_folder"] = prompt_folder
        self.prompt_folder = f"prompts/{prompt_folder}"
        self.memory = memory
        self.chat_log = memory.chat_log
        self.use_template = cfg['use_template']

    @property
    def is_unisis_scene(self):
        return self.scene_source == "unisis"

    @staticmethod
    def _task_name_slug(value):
        """Normalize a YAML task name for GenSim output and dataset paths."""
        slug = re.sub(r"[^a-z0-9]+", "-", str(value).strip().lower()).strip("-")
        return slug or "unisis-scene-task"

    @staticmethod
    def _compact_scene_entity(entity):
        """Keep prompt context useful without copying mesh or URDF contents."""
        fields = (
            "id", "name", "type", "position", "rotation", "scale", "fixed",
            "robot", "color", "category", "description",
        )
        compact = {key: copy.deepcopy(entity[key]) for key in fields if key in entity}
        asset = entity.get("file")
        if asset:
            compact["asset"] = Path(str(asset)).name
        return compact

    def _load_unisis_task(self):
        from cliport.environments.unisis_scene_loader import parse_scene_yaml

        scene_cfg = self.cfg.get("unisis", {})
        scene_path = scene_cfg.get("scene_path") if hasattr(scene_cfg, "get") else None
        if not scene_path:
            raise ValueError("scene_source=unisis requires unisis.scene_path.")
        resolved_path = Path(scene_path).expanduser()
        if resolved_path.is_dir():
            resolved_path = resolved_path / "scene.yaml"
        document = parse_scene_yaml(resolved_path)
        scene_task = copy.deepcopy(document.task or document.raw.get("metadata"))
        if not isinstance(scene_task, dict):
            raise ValueError(f"UniSis scene has no task mapping: {document.path}")

        task_name = (
            scene_task.get("task-name")
            or scene_task.get("task_name")
            or scene_task.get("name")
            or document.path.parent.name
            or document.path.stem
        )
        description = (
            scene_task.get("task-description")
            or scene_task.get("task_description")
            or scene_task.get("description")
            or scene_task.get("instruction")
            or scene_task.get("goal")
            or scene_task.get("language")
        )
        if not isinstance(description, str) or not description.strip():
            raise ValueError("UniSis task needs a non-empty description or instruction.")
        if "target_id" not in scene_task or "goal_point" not in scene_task:
            raise ValueError(
                "Supported UniSis task schema requires target_id and goal_point."
            )
        target_id = scene_task["target_id"]
        if not isinstance(target_id, (str, int)) or not str(target_id).strip():
            raise ValueError("UniSis task target_id must be a non-empty entity ID or name.")
        canonical_target_id = document.name_to_entity_id.get(str(target_id))
        if canonical_target_id is None:
            raise ValueError(f"UniSis target_id {target_id!r} does not name a scene entity.")
        target_entity = next(
            entity for entity in document.entities if entity["id"] == canonical_target_id
        )
        fixed_base = target_entity.get(
            "fixed",
            target_entity.get(
                "use_fixed_base",
                bool(target_entity.get("robot", False)) or target_entity["type"] == "plane",
            ),
        )
        if (
            bool(fixed_base)
            or bool(target_entity.get("robot", False))
            or target_entity["type"] == "plane"
        ):
            raise ValueError(
                f"UniSis target_id {target_id!r} must identify a dynamic non-robot entity."
            )
        goal_point = np.asarray(scene_task["goal_point"], dtype=np.float64)
        if goal_point.shape != (3,) or not np.all(np.isfinite(goal_point)):
            raise ValueError("UniSis goal_point must contain three finite world-coordinate values.")

        self.cfg["unisis"]["scene_path"] = str(document.path)
        self.scene_document = document
        return {
            "task-name": self._task_name_slug(task_name),
            "task-description": description.strip(),
            "assets-used": [],
            # Preserve the YAML declaration exactly. The task adapter checks
            # generated target arguments against these values at runtime.
            "scene-task": scene_task,
            # This compact catalog omits mesh data and full asset contents.
            "scene-entities": [
                self._compact_scene_entity(entity) for entity in document.entities
            ],
            "scene-entity-coordinate-convention": (
                "Entity positions use world XYZ. Entity rotations have been normalized "
                "to PyBullet XYZW; raw scene-task goal_point remains YAML world XYZ."
            ),
        }

    @staticmethod
    def _extract_unisis_code(response):
        """Parse and constrain a generated fixed-scene task class."""
        code_blocks = re.findall(
            r"```(?:python)?\s*(.*?)```", response, re.DOTALL | re.IGNORECASE
        )
        if not code_blocks:
            raise ValueError("Code generation returned no fenced Python block.")
        code = code_blocks[0].strip()
        tree = ast.parse(code)
        task_classes = [
            node for node in tree.body
            if isinstance(node, ast.ClassDef)
            and any(
                (isinstance(base, ast.Name) and base.id == "ExistingSceneTask")
                or (isinstance(base, ast.Attribute) and base.attr == "ExistingSceneTask")
                for base in node.bases
            )
        ]
        if len(task_classes) != 1:
            raise ValueError(
                "Generated UniSis code must define exactly one class inheriting ExistingSceneTask."
            )

        direct_mutation_methods = {
            "add_object", "remove_object", "set_color", "set_object_color", "reset",
            "step", "step_simulation", "movej", "movep", "_reset_scene",
            "createCollisionShape", "createVisualShape", "createMultiBody",
            "loadURDF", "loadSDF", "loadMJCF", "removeBody", "resetSimulation",
            "resetBasePositionAndOrientation", "resetJointState", "changeVisualShape",
            "changeDynamics", "setJointMotorControlArray",
        }
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            owner = node.func.value
            is_base_task_reset = (
                node.func.attr == "reset"
                and isinstance(owner, ast.Call)
                and isinstance(owner.func, ast.Name)
                and owner.func.id == "super"
            )
            if (
                (node.func.attr in direct_mutation_methods and not is_base_task_reset)
                or (
                    node.func.attr == "add_goal"
                    and isinstance(owner, ast.Name)
                    and owner.id == "env"
                )
            ):
                raise ValueError(
                    "Generated UniSis task directly mutates the loaded scene."
                )

        has_scene_goal_helper = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
            and node.func.attr == "add_scene_goal"
            for node in ast.walk(task_classes[0])
        )
        has_gen_sim_goal = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
            and node.func.attr == "add_goal"
            for node in ast.walk(task_classes[0])
        )
        if not (has_scene_goal_helper or has_gen_sim_goal):
            raise ValueError("Generated UniSis task must declare one validated goal.")
        return code, task_classes[0].name

    def propose_task(self, proposed_task_names):
        """Language descriptions for the task"""
        add_to_txt(self.chat_log, "================= Task and Asset Design!", with_print=True)

        if self.is_unisis_scene:
            self.new_task = self._load_unisis_task()
            print("Using fixed UniSis task:", self.new_task["task-name"])
            return self.new_task

        if self.use_template:
            task_prompt_text = open(f"{self.prompt_folder}/cliport_prompt_task.txt").read()
            task_asset_replacement_str = format_dict_prompt(self.memory.online_asset_buffer, self.cfg['task_asset_candidate_num'])
            task_prompt_text = task_prompt_text.replace("TASK_ASSET_PROMPT", task_asset_replacement_str)

            task_desc_replacement_str = format_dict_prompt(self.memory.online_task_buffer, self.cfg['task_description_candidate_num'])
            print("prompt task description candidates:")
            print(task_desc_replacement_str)
            task_prompt_text = task_prompt_text.replace("TASK_DESCRIPTION_PROMPT", task_desc_replacement_str)

            if len(self.cfg['target_task_name']) > 0:
                task_prompt_text = task_prompt_text.replace("TARGET_TASK_NAME", self.cfg['target_task_name'])

            # print("Template Task PROMPT: ", task_prompt_text)
        else:
            task_prompt_text = open(f"{self.prompt_folder}/cliport_prompt_task.txt").read()

        # maximum number
        print("online_task_buffer size:", len(self.memory.online_task_buffer))
        total_tasks = self.memory.online_task_buffer

        MAX_NUM = 10
        if len(total_tasks) > MAX_NUM:
            total_tasks = dict(random.sample(total_tasks.items(), MAX_NUM))

        task_prompt_text = task_prompt_text.replace("PAST_TASKNAME_TEMPLATE", format_dict_prompt(total_tasks))

        res = generate_feedback(
            task_prompt_text,
            temperature=0.8,
            interaction_txt=self.chat_log,
        )

        # Extract dictionary for task name, descriptions, and assets
        task_def = extract_dict(res, prefix="new_task")
        try:
            exec(task_def, globals())
            self.new_task = new_task
            return new_task
        except:
            self.new_task = {"task-name": "dummy", "assets-used": [], "task_descriptions": ""}
            print(str(traceback.format_exc()))
            return self.new_task

    def propose_assets(self):
        """Asset Generation. Not used for now."""
        if self.is_unisis_scene:
            return {}
        if os.path.exists(f"{self.prompt_folder}/cliport_prompt_asset_template.txt"):
            add_to_txt(self.chat_log, "================= Asset Generation!", with_print=True)
            asset_prompt_text = open(f"{self.prompt_folder}/cliport_prompt_asset_template.txt").read()

            if self.use_template:
                asset_prompt_text = asset_prompt_text.replace("TASK_NAME_TEMPLATE", self.new_task["task-name"])
                asset_prompt_text = asset_prompt_text.replace("ASSET_STRING_TEMPLATE", str(self.new_task["assets-used"]))
                print("Template Asset PROMPT: ", asset_prompt_text)

            res = generate_feedback(asset_prompt_text, temperature=0, interaction_txt=self.chat_log)
            print("Save asset to:", self.model_output_dir, task_name + "_asset_output")
            save_text(self.model_output_dir, f'{self.new_task["task-name"]}_asset_output', res)
            asset_list = extract_assets(res)
            # save_urdf(asset_list)
        else:
            asset_list = {}
        return asset_list

    def api_review(self):
        """review the task api"""
        if os.path.exists(f"{self.prompt_folder}/cliport_prompt_api_template.txt"):
            add_to_txt(
                self.chat_log, "================= API Preview!", with_print=True)
            api_prompt_text = open(
                f"{self.prompt_folder}/cliport_prompt_api_template.txt").read()
            if "task-name" in self.new_task:
                api_prompt_text = api_prompt_text.replace("TASK_NAME_TEMPLATE", self.new_task["task-name"])
            api_prompt_text = api_prompt_text.replace("TASK_STRING_TEMPLATE", str(self.new_task))

            res = generate_feedback(
                api_prompt_text, temperature=0, interaction_txt=self.chat_log)

    def template_reference_prompt(self):
        """ select which code reference to reference """
        if os.path.exists(f"{self.prompt_folder}/cliport_prompt_code_reference_selection_template.txt"):
            self.chat_log = add_to_txt(self.chat_log, "================= Code Reference!", with_print=True)
            code_reference_question = open(f'{self.prompt_folder}/cliport_prompt_code_reference_selection_template.txt').read()
            code_reference_question = code_reference_question.replace("TASK_NAME_TEMPLATE", self.new_task["task-name"])
            code_reference_question = code_reference_question.replace(
                "TASK_CODE_LIST_TEMPLATE",
                str(list(self.memory.online_code_buffer.keys())),
            )

            code_reference_question = code_reference_question.replace("TASK_STRING_TEMPLATE", str(self.new_task))
            res = generate_feedback(code_reference_question, temperature=0., interaction_txt=self.chat_log)
            if self.is_unisis_scene:
                # Reuse ordinary task code only as an API/style reference;
                # generated UniSis code is separately constrained to the
                # ExistingSceneTask adapter and fixed scene.
                match = re.search(r"code_reference\s*=\s*(\[.*?\])", res, re.DOTALL)
                if not match:
                    raise ValueError("Reference selection must return code_reference = [...].")
                references = ast.literal_eval(match.group(1))
                if not isinstance(references, list) or any(
                    not isinstance(reference, str) for reference in references
                ):
                    raise ValueError("code_reference must be a list of task-code filenames.")
                task_code_reference_replace_prompt = ""
                for key in references:
                    code = self.memory.online_code_buffer.get(key)
                    if code is None:
                        print("missing task reference code:", key)
                        continue
                    task_code_reference_replace_prompt += f"```\n{code}\n```\n\n"
                return task_code_reference_replace_prompt
            code_reference_cmd = extract_list(res, prefix='code_reference')
            exec(code_reference_cmd, globals())
            task_code_reference_replace_prompt = ''
            for key in code_reference:
                if key in self.memory.online_code_buffer:
                    task_code_reference_replace_prompt += f'```\n{self.memory.online_code_buffer[key]}\n```\n\n'
                else:
                    print("missing task reference code:", key)

        return task_code_reference_replace_prompt

    def implement_task(self):
        """Generate Code for the task"""
        code_prompt_text = open(f"{self.prompt_folder}/cliport_prompt_code_split_template.txt").read()
        code_prompt_text = code_prompt_text.replace("TASK_NAME_TEMPLATE", self.new_task["task-name"])
        if self.is_unisis_scene:
            code_prompt_text = code_prompt_text.replace(
                "TASK_STRING_TEMPLATE", str(self.new_task)
            )

        if self.use_template or os.path.exists(f"{self.prompt_folder}/cliport_prompt_code_reference_selection_template.txt"):
            task_code_reference_replace_prompt = self.template_reference_prompt()
            code_prompt_text = code_prompt_text.replace("TASK_CODE_REFERENCE_TEMPLATE", task_code_reference_replace_prompt)

        elif os.path.exists(f"{self.prompt_folder}/cliport_prompt_code_split_template.txt"):
            self.chat_log = add_to_txt(self.chat_log, "================= Code Generation!", with_print=True)
            code_prompt_text = code_prompt_text.replace("TASK_STRING_TEMPLATE", str(self.new_task))

        res = generate_feedback(
                code_prompt_text, temperature=0, interaction_txt=self.chat_log)
        if self.is_unisis_scene:
            code, task_name = self._extract_unisis_code(res)
            save_text(self.model_output_dir, f'{self.new_task["task-name"]}_code_output', code)
            print("Save code to:", self.model_output_dir, self.new_task["task-name"] + "_code_output")
            return code, task_name
        code, task_name = extract_code(res)
        print("Save code to:", self.model_output_dir, task_name + "_code_output")
        save_text(self.model_output_dir, task_name + "_code_output", code)

        if len(task_name) == 0:
            print("empty task name:", task_name)
            return None

        return code, task_name
