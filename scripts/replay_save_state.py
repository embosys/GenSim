#!/usr/bin/env python3
"""Replay one saved GenSim task against its UniSis scene and export final state."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

p = None


def _resolve_scene(run_dir: Path, override: Path | None, run_meta: dict[str, Any]) -> Path:
    if override is not None:
        scene_path = override.expanduser().resolve()
    else:
        scene_value = run_meta.get("scene")
        if not scene_value:
            raise ValueError("run_meta.json does not contain a scene path; pass --scene.")
        scene_path = (run_dir / scene_value).resolve()
    if scene_path.is_dir():
        scene_path /= "scene.yaml"
    if not scene_path.is_file():
        raise FileNotFoundError(f"Scene YAML does not exist: {scene_path}")
    expected_hash = None if override is not None else run_meta.get("scene_sha256")
    actual_hash = hashlib.sha256(scene_path.read_bytes()).hexdigest()
    if expected_hash and actual_hash != expected_hash:
        raise ValueError(
            f"Scene hash differs from run_meta.json for {scene_path}: "
            f"expected {expected_hash}, got {actual_hash}."
        )
    return scene_path


def _task_class(code_path: Path):
    source = code_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(code_path))
    candidates = [
        node for node in tree.body
        if isinstance(node, ast.ClassDef)
        and any(
            (isinstance(base, ast.Name) and base.id == "ExistingSceneTask")
            or (isinstance(base, ast.Attribute) and base.attr == "ExistingSceneTask")
            for base in node.bases
        )
    ]
    if len(candidates) != 1:
        raise ValueError(
            "Expected generated code to define exactly one class inheriting ExistingSceneTask."
        )
    namespace: dict[str, Any] = {"__name__": "__gensim_replay_task__"}
    exec(compile(tree, str(code_path), "exec"), namespace)
    task_class = namespace[candidates[0].name]
    from cliport.tasks.task import Task

    if not isinstance(task_class, type) or not issubclass(task_class, Task):
        raise TypeError(f"Generated class {candidates[0].name!r} is not a GenSim Task.")
    return task_class


def _root_pose(body_id: int, client_id: int) -> tuple[list[float], list[float]]:
    """Return the URDF root-link frame, removing PyBullet's local inertial offset."""
    com_position, com_quaternion = p.getBasePositionAndOrientation(
        body_id, physicsClientId=client_id
    )
    dynamics = p.getDynamicsInfo(body_id, -1, physicsClientId=client_id)
    inertial_position, inertial_quaternion = dynamics[3], dynamics[4]
    inverse_position, inverse_quaternion = p.invertTransform(
        inertial_position, inertial_quaternion
    )
    root_position, root_quaternion_xyzw = p.multiplyTransforms(
        com_position,
        com_quaternion,
        inverse_position,
        inverse_quaternion,
    )
    quaternion_wxyz = [
        float(root_quaternion_xyzw[3]),
        float(root_quaternion_xyzw[0]),
        float(root_quaternion_xyzw[1]),
        float(root_quaternion_xyzw[2]),
    ]
    return [float(value) for value in root_position], quaternion_wxyz


def _export_state(env) -> dict[str, Any]:
    loaded = getattr(env, "loaded_scene", None)
    if loaded is None:
        raise RuntimeError("The scene was not loaded by the GenSim environment.")
    body_map = loaded.entity_id_to_body_id
    if not body_map:
        raise RuntimeError("The loaded scene has no mapped entities to export.")

    entities: dict[str, Any] = {}
    for entity_id, body_id in sorted(body_map.items()):
        position, quaternion = _root_pose(body_id, env.client_id)
        joints: dict[str, float] = {}
        for joint_index in range(p.getNumJoints(body_id, physicsClientId=env.client_id)):
            joint_info = p.getJointInfo(body_id, joint_index, physicsClientId=env.client_id)
            if joint_info[2] not in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                continue
            joint_name = joint_info[1].decode("utf-8", errors="replace")
            joint_state = p.getJointState(
                body_id, joint_index, physicsClientId=env.client_id
            )
            joints[joint_name] = float(joint_state[0])
        entities[entity_id] = {
            "position": position,
            "quaternion": quaternion,
            "joints": joints,
        }

    return {
        "schema_version": 1,
        "coordinate_system": "scene_world",
        "quaternion_order": "wxyz",
        "entities": entities,
    }


def _new_replay_dir(run_dir: Path) -> Path:
    state_root = run_dir / "state"
    state_root.mkdir(parents=True, exist_ok=True)
    index = 1
    while True:
        replay_dir = state_root / f"replay_{index:03d}"
        try:
            replay_dir.mkdir()
            return replay_dir
        except FileExistsError:
            index += 1


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _run(args: argparse.Namespace) -> Path:
    global p
    import numpy as np
    import pybullet

    p = pybullet
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise NotADirectoryError(f"Run directory does not exist: {run_dir}")
    meta_path = run_dir / "run_meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"GenSim run metadata does not exist: {meta_path}")
    run_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if run_meta.get("method") not in (None, "gensim"):
        raise ValueError(f"Expected a GenSim run, got method={run_meta.get('method')!r}.")
    scene_path = _resolve_scene(run_dir, args.scene, run_meta)

    code_path = args.code.expanduser().resolve() if args.code else run_dir / "code/task.py"
    if not code_path.is_file():
        raise FileNotFoundError(f"Generated task code does not exist: {code_path}")
    seed = int(run_meta.get("seed", 123) if args.seed is None else args.seed)
    if not math.isfinite(args.settle_seconds) or args.settle_seconds < 0:
        raise ValueError("--settle-seconds must be a finite non-negative number.")

    random.seed(seed)
    np.random.seed(seed)
    task_type = _task_class(code_path)

    from cliport.environments.environment import Environment

    env = None
    try:
        env = Environment(
            str(REPO_ROOT / "cliport/environments/assets"),
            disp=args.vis,
            shared_memory=False,
            hz=480,
            record_cfg={"save_video": False, "blender_render": False},
            scene_path=scene_path,
            end_effector=str(run_meta.get("end_effector", "suction")),
        )
        task = task_type()
        task.mode = str(run_meta.get("mode", "test"))
        env.set_task(task)
        obs = env.reset()
        if hasattr(task, "validate_scene_goal"):
            task.validate_scene_goal()
        expert = task.oracle(env)
        info = env.info
        total_reward = 0.0
        done = False
        steps = 0

        for _ in range(int(task.max_steps)):
            action = expert.act(obs, info)
            obs, reward, done, info = env.step(action)
            total_reward += float(reward)
            steps += 1
            print(
                f"Step {steps}: reward={float(reward):.3f}, "
                f"total_reward={total_reward:.3f}, done={bool(done)}"
            )
            if done:
                break

        physics = p.getPhysicsEngineParameters(physicsClientId=env.client_id)
        time_step = float(physics.get("fixedTimeStep", 1.0 / 240.0))
        settle_steps = math.ceil(args.settle_seconds / time_step) if args.settle_seconds else 0
        for _ in range(settle_steps):
            p.stepSimulation(physicsClientId=env.client_id)

        final_state = _export_state(env)
        replay_dir = _new_replay_dir(run_dir)
        p.saveBullet(
            str(replay_dir / "final_state.bullet"),
            physicsClientId=env.client_id,
        )
        _write_json(replay_dir / "final_state.json", final_state)

        original_result_path = run_dir / "result.json"
        original_result = None
        if original_result_path.is_file():
            try:
                result = json.loads(original_result_path.read_text(encoding="utf-8"))
                original_result = {
                    key: result.get(key)
                    for key in ("status", "native_success", "attempts")
                    if key in result
                }
            except json.JSONDecodeError:
                original_result = None

        source_ref = os.path.relpath(code_path, replay_dir)
        scene_ref = os.path.relpath(scene_path, replay_dir)
        env_config = {
            "auto_add_ground": False,
            "genesis_precision": "32",
        }
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "method": "gensim",
            "engine": "pybullet",
            "scene": scene_ref,
            "scene_sha256": hashlib.sha256(scene_path.read_bytes()).hexdigest(),
            "state_file": "final_state.json",
            "native_state_file": "final_state.bullet",
            "seed": seed,
            "settle_seconds": args.settle_seconds,
            "env_config": env_config,
            "source": {
                "code": source_ref,
                "code_sha256": hashlib.sha256(code_path.read_bytes()).hexdigest(),
                "task_class": task_type.__name__,
            },
            "execution": {
                "steps": steps,
                "done": bool(done),
                "native_reward": total_reward,
                "native_success": bool(total_reward > 0.99),
            },
            "original_run_result": original_result,
            "native_env_config": {
                "end_effector": str(run_meta.get("end_effector", "suction")),
                "time_step_seconds": time_step,
            },
        }
        for key in ("batch_hash", "batch_run_id", "batch_name"):
            if key in run_meta:
                manifest[key] = run_meta[key]
        _write_json(replay_dir / "manifest.json", manifest)
        print(f"Saved replay state: {replay_dir}")
        return replay_dir
    finally:
        if env is not None and p.isConnected(env.client_id):
            p.disconnect(env.client_id)
        elif p.isConnected():
            p.disconnect()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="completed GenSim run directory")
    parser.add_argument("--code", type=Path, help="task code path; defaults to run_dir/code/task.py")
    parser.add_argument("--scene", type=Path, help="override the scene path recorded in run_meta.json")
    parser.add_argument("--seed", type=int, help="replay seed; defaults to run_meta.json")
    parser.add_argument("--settle-seconds", type=float, default=0.5,
                        help="simulate this duration after the oracle sequence (default: 0.5)")
    parser.add_argument("--vis", action="store_true", help="show the PyBullet GUI while replaying")
    args = parser.parse_args()
    try:
        _run(args)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
