"""Minimal PyBullet scene preview environment for UniSis YAML scenes.

This environment deliberately owns no task, oracle, or robot controller. It
loads the entities already described by UniSis and exposes their PyBullet body
IDs so a later GenSim task adapter can bind to them.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pybullet as p

from cliport.environments.unisis_scene_loader import (
    LoadedScene,
    SceneDocument,
    load_scene_entities,
    parse_scene_yaml,
)


class UnisisYAMLEnvironment:
    """Load and inspect one UniSis YAML scene in a dedicated PyBullet client."""

    def __init__(
        self,
        scene_path: str | Path,
        *,
        gui: bool = False,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.scene_path = Path(scene_path).expanduser().resolve()
        self.document: SceneDocument = parse_scene_yaml(self.scene_path)
        self.cache_dir = Path(cache_dir or (Path(__file__).resolve().parents[2] / ".cache" / "unisis"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.gui = gui
        self.connection_mode = p.GUI if gui else p.DIRECT
        self.client_id = p.connect(self.connection_mode)
        if self.client_id < 0:
            raise RuntimeError(f"Could not connect to PyBullet (gui={gui}).")
        self.loaded: LoadedScene | None = None
        self.warnings = list(getattr(self.document, "warnings", []))
        self.time_step = self._read_time_step(self.document.sim_options)
        self.gravity = self._read_gravity(self.document.sim_options)
        self._closed = False

    @staticmethod
    def _read_time_step(options: dict[str, Any]) -> float:
        for key in ("time_step", "timestep", "dt", "physics_timestep"):
            value = options.get(key)
            if value is not None:
                value = float(value)
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f"sim_options.{key} must be a positive finite number.")
                return value
        return 1.0 / 240.0

    @staticmethod
    def _read_gravity(options: dict[str, Any]) -> tuple[float, float, float]:
        value = options.get("gravity", (0.0, 0.0, -9.81))
        if isinstance(value, dict):
            value = [value.get(axis, default) for axis, default in zip(("x", "y", "z"), (0, 0, -9.81))]
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ValueError("sim_options.gravity must contain three numeric values.")
        result = tuple(float(component) for component in value)
        if not all(math.isfinite(component) for component in result):
            raise ValueError("sim_options.gravity values must be finite.")
        return result

    def reset(self) -> LoadedScene:
        """Rebuild the YAML scene and return the new entity/body mappings."""
        if self._closed or not p.isConnected(self.client_id):
            raise RuntimeError("The PyBullet client is closed.")
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId=self.client_id)
        try:
            p.resetSimulation(physicsClientId=self.client_id)
            p.setGravity(*self.gravity, physicsClientId=self.client_id)
            p.setTimeStep(self.time_step, physicsClientId=self.client_id)
            p.setPhysicsEngineParameter(enableFileCaching=0, physicsClientId=self.client_id)
            self.loaded = load_scene_entities(
                self.document,
                physics_client_id=self.client_id,
                cache_dir=self.cache_dir,
            )
            self.warnings = list(getattr(self.document, "warnings", []))
            self.warnings.extend(getattr(self.loaded, "warnings", []))
        finally:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.client_id)
        self._configure_viewer()
        return self.loaded

    def get_body_id(self, entity_id: str) -> int:
        """Return the body ID associated with a UniSis entity ID."""
        if self.loaded is None:
            raise RuntimeError("Call reset() before looking up entities.")
        try:
            return self.loaded.entity_id_to_body_id[entity_id]
        except KeyError as exc:
            raise KeyError(f"No loaded UniSis entity has id {entity_id!r}.") from exc

    @property
    def entity_id_to_body_id(self) -> dict[str, int]:
        return {} if self.loaded is None else self.loaded.entity_id_to_body_id

    @property
    def body_id_to_entity_id(self) -> dict[int, str]:
        return {} if self.loaded is None else self.loaded.body_id_to_entity_id

    @property
    def robot_ids(self) -> list[str]:
        return [] if self.loaded is None else self.loaded.robot_ids

    @property
    def obj_ids(self) -> dict[str, list[int]]:
        if self.loaded is None:
            return {"fixed": [], "rigid": [], "deformable": []}
        return self.loaded.obj_ids

    def step(self, count: int = 1) -> None:
        if count < 0:
            raise ValueError("step count must be non-negative")
        for _ in range(count):
            p.stepSimulation(physicsClientId=self.client_id)

    def _body_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        if self.loaded is None or not self.loaded.entity_id_to_body_id:
            return np.array([-0.5, -0.5, 0.0]), np.array([0.5, 0.5, 1.0])
        bounds = []
        for body_id in self.loaded.entity_id_to_body_id.values():
            link_count = p.getNumJoints(body_id, physicsClientId=self.client_id)
            for link_index in range(-1, link_count):
                lower, upper = p.getAABB(body_id, link_index, physicsClientId=self.client_id)
                lower_arr, upper_arr = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
                extent = upper_arr - lower_arr
                # Exclude infinite planes from automatic camera framing.
                if np.all(np.isfinite(lower_arr)) and np.all(np.isfinite(upper_arr)) and np.all(extent < 100):
                    bounds.append((lower_arr, upper_arr))
        if not bounds:
            return np.array([-0.5, -0.5, 0.0]), np.array([0.5, 0.5, 1.0])
        return np.min([pair[0] for pair in bounds], axis=0), np.max([pair[1] for pair in bounds], axis=0)

    def _camera_settings(self) -> tuple[np.ndarray, float, float, float]:
        camera = self.document.camera or (self.document.cameras[0] if self.document.cameras else {})
        if isinstance(camera, dict):
            target = camera.get("camera_lookat", camera.get("target", camera.get("target_position")))
            position = camera.get("camera_pos", camera.get("position"))
            if target is not None and position is not None:
                target_arr = np.asarray(target, dtype=float)
                delta = np.asarray(position, dtype=float) - target_arr
                distance = max(float(np.linalg.norm(delta)), 0.1)
                yaw = math.degrees(math.atan2(delta[1], delta[0]))
                pitch = -math.degrees(math.asin(float(np.clip(delta[2] / distance, -1.0, 1.0))))
                return target_arr, distance, yaw, pitch
            if target is not None and all(key in camera for key in ("distance", "yaw", "pitch")):
                return np.asarray(target, dtype=float), float(camera["distance"]), float(camera["yaw"]), float(camera["pitch"])

        lower, upper = self._body_bounds()
        center = (lower + upper) / 2
        radius = max(float(np.linalg.norm((upper - lower) / 2)), 0.5)
        return center, max(1.2, radius * 2.8), 45.0, -28.0

    def _configure_viewer(self) -> None:
        if not self.gui:
            return
        target, distance, yaw, pitch = self._camera_settings()
        p.resetDebugVisualizerCamera(
            cameraDistance=distance,
            cameraYaw=yaw,
            cameraPitch=pitch,
            cameraTargetPosition=target.tolist(),
            physicsClientId=self.client_id,
        )
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.client_id)

    def root_pose(self, entity_id: str) -> dict[str, list[float]]:
        """Read the entity's URDF root-link pose, separating inertial/COM offset."""
        body_id = self.get_body_id(entity_id)
        com_position, com_orientation = p.getBasePositionAndOrientation(
            body_id, physicsClientId=self.client_id
        )
        dynamics = p.getDynamicsInfo(body_id, -1, physicsClientId=self.client_id)
        inertial_position, inertial_orientation = dynamics[3], dynamics[4]
        inverse_position, inverse_orientation = p.invertTransform(inertial_position, inertial_orientation)
        root_position, root_orientation = p.multiplyTransforms(
            com_position,
            com_orientation,
            inverse_position,
            inverse_orientation,
        )
        return {
            "position": [float(value) for value in root_position],
            "rotation_xyzw": [float(value) for value in root_orientation],
            "com_position": [float(value) for value in com_position],
            "com_rotation_xyzw": [float(value) for value in com_orientation],
        }

    def robot_joint_states(self) -> dict[str, dict[str, Any]]:
        states: dict[str, dict[str, Any]] = {}
        for entity_id in self.robot_ids:
            body_id = self.get_body_id(entity_id)
            joints = []
            for joint_index in range(p.getNumJoints(body_id, physicsClientId=self.client_id)):
                info = p.getJointInfo(body_id, joint_index, physicsClientId=self.client_id)
                if info[3] < 0:
                    continue
                name = info[1].decode("utf-8", errors="replace")
                state = p.getJointState(body_id, joint_index, physicsClientId=self.client_id)
                joints.append({"index": joint_index, "name": name, "position": float(state[0])})
            config = self.loaded.entity_configs.get(entity_id, {}) if self.loaded is not None else {}
            robot_kwargs = config.get("robot_adapter_kwargs", {})
            expected_qpos = robot_kwargs.get("default_qpos") if isinstance(robot_kwargs, dict) else None
            actual_qpos = [joint["position"] for joint in joints]
            matches_config = None
            if expected_qpos is not None:
                expected_qpos = [float(value) for value in expected_qpos]
                matches_config = len(expected_qpos) == len(actual_qpos) and np.allclose(
                    expected_qpos, actual_qpos, atol=1e-5, rtol=0
                )
            states[entity_id] = {
                "movable_dof_count": len(joints),
                "joints": joints,
                "configured_default_qpos": expected_qpos,
                "matches_configured_default_qpos": matches_config,
            }
        return states

    def initial_snapshot(self) -> dict[str, Any]:
        if self.loaded is None:
            raise RuntimeError("Call reset() before taking a snapshot.")
        entity_poses = {
            entity_id: self.root_pose(entity_id)
            for entity_id in sorted(self.loaded.entity_id_to_body_id)
        }
        body_count = p.getNumBodies(physicsClientId=self.client_id)
        return {
            "entity_count": len(entity_poses),
            "body_count": body_count,
            "entity_poses": entity_poses,
            "robot_ids": list(self.robot_ids),
            "robot_joints": self.robot_joint_states(),
        }

    def render_rgb(self, width: int = 1280, height: int = 720) -> np.ndarray:
        if self.loaded is None:
            raise RuntimeError("Call reset() before rendering.")
        target, distance, yaw, pitch = self._camera_settings()
        view = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=target.tolist(),
            distance=distance,
            yaw=yaw,
            pitch=pitch,
            roll=0,
            upAxisIndex=2,
        )
        projection = p.computeProjectionMatrixFOV(
            fov=55,
            aspect=float(width) / float(height),
            nearVal=0.02,
            farVal=max(50.0, distance * 20),
        )
        image = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view,
            projectionMatrix=projection,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=self.client_id,
        )
        return np.asarray(image[2], dtype=np.uint8).reshape(height, width, 4)[..., :3].copy()

    def close(self) -> None:
        if not self._closed:
            if p.isConnected(self.client_id):
                p.disconnect(self.client_id)
            self._closed = True

    def __enter__(self) -> "UnisisYAMLEnvironment":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()
