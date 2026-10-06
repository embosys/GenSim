"""Preview and validate a UniSis YAML scene in PyBullet without running a task."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pybullet as p

from cliport.environments.unisis_yaml_environment import UnisisYAMLEnvironment


def _resolve_scene_path(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if path.is_dir():
        path = path / "scene.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Scene YAML does not exist: {path}")
    return path


def _snapshot_consistent(first: dict[str, Any], current: dict[str, Any], atol: float = 1e-5) -> bool:
    if first["entity_count"] != current["entity_count"] or first["body_count"] != current["body_count"]:
        return False
    first_poses, current_poses = first["entity_poses"], current["entity_poses"]
    if first_poses.keys() != current_poses.keys():
        return False
    for entity_id in first_poses:
        for key in ("position", "rotation_xyzw"):
            if not np.allclose(first_poses[entity_id][key], current_poses[entity_id][key], atol=atol, rtol=0):
                return False
    return first["robot_joints"] == current["robot_joints"]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _build_report(env: UnisisYAMLEnvironment, snapshot: dict[str, Any], reset_count: int) -> dict[str, Any]:
    assert env.loaded is not None
    return {
        "scene_path": str(env.scene_path),
        "reset_count": reset_count,
        "entity_count": snapshot["entity_count"],
        "body_count": snapshot["body_count"],
        "entity_id_to_body_id": env.loaded.entity_id_to_body_id,
        "robot_ids": env.loaded.robot_ids,
        "object_counts": {key: len(value) for key, value in env.obj_ids.items()},
        "robot_joints": snapshot["robot_joints"],
        "entity_poses": snapshot["entity_poses"],
        "asset_info": _jsonable(getattr(env.loaded, "asset_info", [])),
        "warnings": env.warnings,
    }



def _wait_for_gui(client_id: int, hold_seconds: float | None) -> None:
    """Keep the paused viewer responsive, including macOS main-thread GUIs."""
    deadline = None if hold_seconds is None else time.monotonic() + hold_seconds
    while p.isConnected(client_id):
        if deadline is not None and time.monotonic() >= deadline:
            break
        try:
            # These commands pump PyBullet's GUI event loop on macOS without
            # advancing physics. isConnected() and sleep() alone do not.
            p.getMouseEvents(physicsClientId=client_id)
            p.getKeyboardEvents(physicsClientId=client_id)
        except p.error:
            if not p.isConnected(client_id):
                break
            raise
        delay = 1.0 / 60.0
        if deadline is not None:
            delay = min(delay, max(0.0, deadline - time.monotonic()))
        time.sleep(delay)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", help="UniSis scene.yaml path or its containing directory")
    parser.add_argument("--gui", action="store_true", help="show a PyBullet GUI window")
    parser.add_argument("--steps", type=int, default=0, help="simulate this many steps after initial-state checks")
    parser.add_argument("--hold-seconds", type=float, help="keep the GUI open for this duration; by default wait until closed")
    parser.add_argument("--cache-dir", type=Path, help="directory for converted mesh and MJCF assets")
    parser.add_argument("--report-json", type=Path, help="write the load report to this JSON path")
    parser.add_argument("--image", type=Path, help="save an RGB scene preview as a PNG")
    parser.add_argument("--reset-count", type=int, default=1, help="rebuild the scene this many times and check initial-state consistency")
    args = parser.parse_args()

    if args.steps < 0:
        parser.error("--steps must be non-negative")
    if args.reset_count < 1:
        parser.error("--reset-count must be at least 1")
    if args.hold_seconds is not None and args.hold_seconds < 0:
        parser.error("--hold-seconds must be non-negative")

    scene_path = _resolve_scene_path(args.scene)
    exit_code = 0
    try:
        with UnisisYAMLEnvironment(scene_path, gui=args.gui, cache_dir=args.cache_dir) as env:
            snapshots = []
            for _ in range(args.reset_count):
                env.reset()
                snapshots.append(env.initial_snapshot())
            first_snapshot = snapshots[0]
            consistent = all(_snapshot_consistent(first_snapshot, item) for item in snapshots[1:])
            report = _build_report(env, first_snapshot, args.reset_count)
            report["reset_consistent"] = consistent

            print(f"Loaded scene: {scene_path}")
            print(f"Entities: {report['entity_count']}; PyBullet bodies: {report['body_count']}")
            print(f"Objects: {report['object_counts']}")
            print(f"Robots: {report['robot_ids'] or '(none)'}")
            for robot_id, robot in report["robot_joints"].items():
                print(f"Robot {robot_id}: {robot['movable_dof_count']} movable joints")
                for joint in robot["joints"]:
                    print(f"  {joint['name']}: {joint['position']:.6f}")
                qpos_match = robot["matches_configured_default_qpos"]
                if qpos_match is not None:
                    print(f"  Matches configured default_qpos: {qpos_match}")
                    if not qpos_match:
                        exit_code = 2
            print(f"Repeated reset consistency ({args.reset_count} reset(s)): {consistent}")
            for warning in report["warnings"]:
                print(f"WARNING: {warning}")
            if not consistent:
                exit_code = 2

            if args.steps:
                env.step(args.steps)
                report["simulated_steps_after_initial_snapshot"] = args.steps
            if args.image:
                image_path = args.image.expanduser().resolve()
                image_path.parent.mkdir(parents=True, exist_ok=True)
                rgb = env.render_rgb()
                if not cv2.imwrite(str(image_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                    raise RuntimeError(f"Could not save preview image to {image_path}")
                print(f"Saved preview image: {image_path}")
            if args.report_json:
                report_path = args.report_json.expanduser().resolve()
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
                print(f"Saved load report: {report_path}")

            if args.gui:
                if args.hold_seconds is None:
                    print("GUI open. Close the PyBullet window or press Ctrl+C to exit.")
                _wait_for_gui(env.client_id, args.hold_seconds)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
