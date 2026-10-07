"""Task adapter for moving an existing UniSis scene entity to a fixed point."""

from __future__ import annotations

import numpy as np
import pybullet as p

from cliport.tasks.task import Task


class ExistingSceneTask(Task):
    """Bind GenSim's oracle and rewards to one already-loaded scene entity.

    The supported YAML task contract is a mapping with ``target_id`` and a
    three-value ``goal_point``. Scene construction and entity state belong to
    the YAML loader; generated tasks only declare the validated goal.
    """

    def __init__(self):
        super().__init__()
        self.max_steps = 1
        self.task_completed_desc = "done moving the scene object."
        self._scene_goal_added = False

    def reset(self, env):
        """Reset GenSim task state and copy the scene's observation geometry."""
        super().reset(env)
        if getattr(env, "document", None) is None:
            raise RuntimeError("ExistingSceneTask requires a loaded UniSis scene document.")
        if not isinstance(getattr(env, "scene_task", None), dict):
            raise RuntimeError("ExistingSceneTask requires a UniSis scene task mapping.")

        self.document = env.document
        self.scene_task = dict(env.scene_task)
        self._env = env
        self._expected_scene_target_id = self.scene_task.get("target_id")
        self._expected_scene_goal_point = tuple(self.scene_task.get("goal_point", ()))
        self.bounds = np.asarray(env.bounds, dtype=np.float32).copy()
        self.zone_bounds = self.bounds.copy()
        self.oracle_cams = env.oracle_cams
        self.pix_size = float(env.pix_size)
        self._scene_goal_added = False

        task_description = (
            self.scene_task.get("task-description")
            or self.scene_task.get("task_description")
            or self.scene_task.get("description")
            or self.scene_task.get("instruction")
            or self.scene_task.get("goal")
            or self.scene_task.get("language")
        )
        if isinstance(task_description, str) and task_description.strip():
            self.lang_template = task_description.strip()
            self.task_completed_desc = task_description.strip()

        declared_max_steps = self.scene_task.get("max_steps")
        if declared_max_steps is not None:
            self.max_steps = int(declared_max_steps)
            if self.max_steps <= 0:
                raise ValueError("scene task max_steps must be positive.")

    def add_scene_goal(self, target_id, goal_point, language_goal=None):
        """Bind the declared YAML target and goal point to a GenSim pose goal."""
        if not hasattr(self, "scene_task"):
            raise RuntimeError("Call ExistingSceneTask.reset(env) before adding a scene goal.")
        if self._scene_goal_added:
            raise RuntimeError("The supported UniSis task schema has one fixed move goal.")

        expected_target_id = self._expected_scene_target_id
        if expected_target_id is None:
            raise ValueError("UniSis task must define target_id.")
        if str(target_id) != str(expected_target_id):
            raise ValueError(
                f"Generated target_id {target_id!r} does not match YAML target_id "
                f"{expected_target_id!r}."
            )

        expected_point = np.asarray(self._expected_scene_goal_point, dtype=np.float64)
        generated_point = np.asarray(goal_point, dtype=np.float64)
        if expected_point.shape != (3,) or not np.all(np.isfinite(expected_point)):
            raise ValueError("YAML goal_point must be a finite three-value position.")
        if generated_point.shape != (3,) or not np.all(np.isfinite(generated_point)):
            raise ValueError("Generated goal_point must be a finite three-value position.")
        if not np.allclose(generated_point, expected_point, rtol=0.0, atol=1e-9):
            raise ValueError(
                f"Generated goal_point {generated_point.tolist()} does not match YAML "
                f"goal_point {expected_point.tolist()}."
            )

        lookup_id = self.document.name_to_entity_id.get(
            str(expected_target_id), str(expected_target_id)
        )
        body_id = self._env.get_scene_object(lookup_id)
        if body_id is None:
            raise RuntimeError(f"UniSis entity {expected_target_id!r} has no loaded body.")
        if p.getDynamicsInfo(body_id, -1)[0] <= 0:
            raise ValueError(
                f"UniSis target_id {expected_target_id!r} must be a dynamic entity."
            )
        _, target_orientation = p.getBasePositionAndOrientation(body_id)
        if language_goal is None:
            language_goal = (
                self.scene_task.get("task-description")
                or self.scene_task.get("task_description")
                or self.scene_task.get("description")
                or self.scene_task.get("instruction")
                or self.scene_task.get("goal")
                or self.scene_task.get("language")
                or "move the selected scene object to the goal point"
            )

        self.add_goal(
            objs=[body_id],
            matches=np.ones((1, 1), dtype=np.int32),
            targ_poses=[(tuple(float(value) for value in expected_point), target_orientation)],
            replace=False,
            rotations=True,
            metric="pose",
            params=None,
            step_max_reward=1.0,
            language_goal=str(language_goal),
        )
        self._scene_goal_added = True

    def validate_scene_goal(self):
        """Check the generated goal against the one-object YAML task contract."""
        if not self.goals or len(self.goals) != 1 or len(self.lang_goals) != 1:
            raise RuntimeError(
                "Generated ExistingSceneTask reset must add one goal and one language goal."
            )
        objs, matches, target_poses, _, _, metric, _, max_reward = self.goals[0]
        if len(objs) != 1 or len(target_poses) != 1 or metric != "pose":
            raise RuntimeError("The supported UniSis move task requires one pose target.")

        expected_target_id = self._expected_scene_target_id
        lookup_id = self.document.name_to_entity_id.get(
            str(expected_target_id), str(expected_target_id)
        )
        expected_body_id = self._env.get_scene_object(lookup_id)
        if p.getDynamicsInfo(expected_body_id, -1)[0] <= 0:
            raise RuntimeError("The YAML target body is not dynamic and cannot be manipulated.")
        generated_obj = objs[0]
        generated_body_id = generated_obj[0] if isinstance(generated_obj, tuple) else generated_obj
        if int(generated_body_id) != int(expected_body_id):
            raise RuntimeError(
                "Generated goal object does not match the YAML target_id."
            )

        match_matrix = np.asarray(matches)
        if match_matrix.shape != (1, 1) or not bool(match_matrix[0, 0]):
            raise RuntimeError("Generated goal matching must connect the target to its sole pose.")
        target_pose = target_poses[0]
        if (
            isinstance(target_pose, (tuple, list))
            and len(target_pose) == 2
            and np.asarray(target_pose[0]).shape == (3,)
        ):
            generated_point = np.asarray(target_pose[0], dtype=np.float64)
        else:
            generated_point = np.asarray(target_pose, dtype=np.float64)
        expected_point = np.asarray(self._expected_scene_goal_point, dtype=np.float64)
        if generated_point.shape != (3,) or not np.allclose(
            generated_point, expected_point, rtol=0.0, atol=1e-9
        ):
            raise RuntimeError(
                "Generated pose does not preserve the YAML goal_point."
            )
        if not np.isclose(float(max_reward), 1.0, rtol=0.0, atol=1e-9):
            raise RuntimeError("The supported single move goal must have reward 1.0.")
