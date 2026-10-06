"""Load UniSis scene YAML files into a PyBullet client.

This module intentionally only materializes scene entities. It does not start a
simulation, add a ground plane, or create a task/robot controller.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


_SUPPORTED_TYPES = {"plane", "box", "sphere", "cylinder", "mesh", "urdf", "mjcf"}
_VECTOR_EPS = 1.0e-12


@dataclass(slots=True)
class SceneDocument:
    """Parsed scene data, with entity paths and poses normalized for PyBullet."""

    path: Path
    raw: dict[str, Any]
    entities: list[dict[str, Any]]
    task: dict[str, Any] | None = None
    cameras: list[dict[str, Any]] = field(default_factory=list)
    camera: dict[str, Any] | None = None
    sim_options: dict[str, Any] = field(default_factory=dict)
    rigid_options: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    name_to_entity_id: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class LoadedScene:
    """Bodies created from a parsed scene and their UniSis entity mapping."""

    entity_id_to_body_id: dict[str, int] = field(default_factory=dict)
    body_id_to_entity_id: dict[int, str] = field(default_factory=dict)
    robot_ids: list[str] = field(default_factory=list)
    obj_ids: dict[str, list[int]] = field(
        default_factory=lambda: {"fixed": [], "rigid": [], "deformable": []}
    )
    entity_configs: dict[str, dict[str, Any]] = field(default_factory=dict)
    asset_info: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    name_to_entity_id: dict[str, str] = field(default_factory=dict)


def parse_scene_yaml(path: str | Path) -> SceneDocument:
    """Parse a UniSis YAML scene, accepting both wrapped and bare scene forms.

    Relative asset paths resolve against the YAML file's directory. UniSis uses
    WXYZ quaternions while PyBullet uses XYZW; this parser converts the pose once
    so the loader and callers can use the normalized entity configuration.
    """

    yaml_path = Path(path).expanduser().resolve()
    if not yaml_path.is_file():
        raise FileNotFoundError(f"UniSis scene YAML does not exist: {yaml_path}")
    with yaml_path.open("r", encoding="utf-8") as stream:
        document_raw = yaml.safe_load(stream)
    if not isinstance(document_raw, dict):
        raise ValueError(f"Scene YAML root must be a mapping: {yaml_path}")

    wrapper = document_raw
    scene_raw = document_raw.get("scene", document_raw)
    if not isinstance(scene_raw, dict):
        raise ValueError("Scene 'scene' value must be a mapping")
    entities_raw = scene_raw.get("entities", [])
    if not isinstance(entities_raw, list):
        raise ValueError("Scene 'entities' must be a list")

    warnings: list[str] = []
    normalized_entities: list[dict[str, Any]] = []
    aliases: dict[str, str] = {}
    used_ids: set[str] = set()
    used_aliases: dict[str, str] = {}

    for index, raw_entity in enumerate(entities_raw):
        if not isinstance(raw_entity, dict):
            raise ValueError(f"Entity at index {index} must be a mapping")
        entity = dict(raw_entity)
        entity_id_raw = entity.get("id") or entity.get("name")
        if not isinstance(entity_id_raw, (str, int)) or not str(entity_id_raw).strip():
            raise ValueError(f"Entity at index {index} needs a non-empty id or name")
        entity_id = str(entity_id_raw)
        if entity_id in used_ids:
            raise ValueError(f"Duplicate scene entity id: {entity_id!r}")
        used_ids.add(entity_id)
        entity["id"] = entity_id

        alias = entity.get("name")
        if alias is not None:
            alias = str(alias)
            existing = used_aliases.get(alias)
            if existing is not None and existing != entity_id:
                raise ValueError(
                    f"Duplicate entity name alias {alias!r} for {existing!r} and {entity_id!r}"
                )
            used_aliases[alias] = entity_id
            aliases[alias] = entity_id
        # IDs themselves are also accepted as names by consumers.
        aliases[entity_id] = entity_id

        entity_type = str(entity.get("type", "")).strip().lower()
        if entity_type not in _SUPPORTED_TYPES:
            raise ValueError(
                f"Entity {entity_id!r} has unsupported type {entity_type!r}; "
                f"supported types are {sorted(_SUPPORTED_TYPES)}"
            )
        entity["type"] = entity_type

        if entity.get("file_preset"):
            raise ValueError(
                f"Entity {entity_id!r} uses file_preset, which PyBullet scene loading "
                "cannot resolve; provide an explicit file path"
            )
        if entity_type in {"mesh", "urdf", "mjcf"}:
            asset = entity.get("file")
            if not isinstance(asset, str) or not asset.strip():
                raise ValueError(f"Entity {entity_id!r} of type {entity_type} needs 'file'")
            asset_path = Path(asset).expanduser()
            if not asset_path.is_absolute():
                asset_path = yaml_path.parent / asset_path
            asset_path = asset_path.resolve()
            if not asset_path.is_file():
                raise FileNotFoundError(
                    f"Entity {entity_id!r} asset does not exist: {asset_path}"
                )
            entity["file"] = str(asset_path)

        entity["position"] = _normalize_position(
            entity.get("position", [0.0, 0.0, 0.0]), entity_id
        )
        entity["rotation"] = _normalize_rotation(entity)
        entity["scale"] = _normalize_scale(entity.get("scale", 1.0), entity_id)
        if "fixed" in entity and not isinstance(entity["fixed"], bool):
            raise ValueError(f"Entity {entity_id!r} field 'fixed' must be boolean")
        if "collision" in entity and entity["collision"] is not None:
            if not isinstance(entity["collision"], bool):
                raise ValueError(f"Entity {entity_id!r} field 'collision' must be boolean")
        material = entity.get("material")
        if material is not None and not isinstance(material, dict):
            raise ValueError(f"Entity {entity_id!r} material must be a mapping")
        if isinstance(material, dict):
            material_type = str(material.get("type", "Rigid")).lower()
            if material_type not in {"rigid", "static", "fixed"}:
                raise ValueError(
                    f"Entity {entity_id!r} material type {material.get('type')!r} "
                    "is not supported by the rigid PyBullet scene loader"
                )
        _validate_entity_physics(entity, entity_id)
        normalized_entities.append(entity)

    # An entity name may alias its own ID, but no alias may shadow another ID.
    for entity in normalized_entities:
        alias = entity.get("name")
        if alias is not None and alias != entity["id"] and alias in used_ids:
            raise ValueError(
                f"Entity name alias {alias!r} conflicts with another entity ID"
            )

    scene_offset = scene_raw.get("offset", wrapper.get("offset"))
    if scene_offset is not None:
        offset_values = _vec3(scene_offset, "Scene offset")
        if any(abs(value) > _VECTOR_EPS for value in offset_values):
            warnings.append(
                "Scene offset is nonzero and was not applied to physics bodies; "
                "UniSis documents it as a Unity visual-only offset"
            )

    cameras_raw = scene_raw.get("cameras", wrapper.get("cameras", []))
    if cameras_raw is None:
        cameras_raw = []
    if not isinstance(cameras_raw, list) or any(not isinstance(item, dict) for item in cameras_raw):
        raise ValueError("Scene 'cameras' must be a list of mappings")
    camera_raw = scene_raw.get("camera", wrapper.get("camera"))
    if camera_raw is not None and not isinstance(camera_raw, dict):
        raise ValueError("Scene 'camera' must be a mapping")
    normalized_options: dict[str, dict[str, Any]] = {}
    for key in ("sim_options", "rigid_options"):
        value = scene_raw.get(key, wrapper.get(key)) or {}
        if not isinstance(value, dict):
            raise ValueError(f"Scene {key!r} must be a mapping")
        normalized_options[key] = dict(value)
    if normalized_options["rigid_options"]:
        warnings.append(
            "rigid_options were preserved in SceneDocument but are not applied by "
            "the entity loader; the environment must map supported options explicitly"
        )

    task_raw = wrapper.get("task", scene_raw.get("task"))
    if task_raw is not None and not isinstance(task_raw, dict):
        raise ValueError("Scene 'task' must be a mapping")
    return SceneDocument(
        path=yaml_path,
        raw=dict(scene_raw),
        entities=normalized_entities,
        task=dict(task_raw) if task_raw is not None else None,
        cameras=[dict(camera) for camera in cameras_raw],
        camera=dict(camera_raw) if camera_raw is not None else None,
        sim_options=normalized_options["sim_options"],
        rigid_options=normalized_options["rigid_options"],
        warnings=warnings,
        name_to_entity_id=aliases,
    )


def load_scene_entities(
    document: SceneDocument,
    *,
    physics_client_id: int,
    cache_dir: str | Path,
) -> LoadedScene:
    """Create scene bodies in an existing PyBullet client.

    Static mesh collisions use their source triangle mesh. Dynamic meshes use
    PyBullet's convex mesh collision approximation and report that approximation
    in ``asset_info`` and ``warnings``. Mesh mass uses source mesh volume times
    ``rho`` and scale; for non-watertight source meshes, the convex-hull volume is
    used and explicitly reported.
    """

    try:
        import pybullet as p
    except ImportError as exc:  # pragma: no cover - depends on optional runtime
        raise RuntimeError("PyBullet is required to load a UniSis scene") from exc

    cache_path = Path(cache_dir).expanduser().resolve()
    cache_path.mkdir(parents=True, exist_ok=True)
    loaded = LoadedScene(
        warnings=list(document.warnings), name_to_entity_id=dict(document.name_to_entity_id)
    )
    for entity in document.entities:
        entity_id = entity["id"]
        entity_type = entity["type"]
        robot = bool(entity.get("robot", False))
        fixed = _entity_fixed(entity, robot=robot, entity_type=entity_type)
        position = tuple(float(v) for v in entity["position"])
        orientation = tuple(float(v) for v in entity["rotation"])
        scale = _scale_vec(entity["scale"])
        report: dict[str, Any] = {
            "entity_id": entity_id,
            "type": entity_type,
            "source_file": entity.get("file"),
            "fixed": fixed,
            "position": list(position),
            "rotation_xyzw": list(orientation),
            "scale": list(scale),
        }

        try:
            if entity_type in {"mesh", "urdf", "mjcf"}:
                from cliport.environments import unisis_assets

                source_path = Path(entity["file"])
                if entity_type == "mesh":
                    loaded_path = Path(
                        unisis_assets.prepare_mesh_asset(
                            source_path, cache_path,
                            file_meshes_are_zup=entity.get("file_meshes_are_zup"),
                        )
                    ).resolve()
                elif entity_type == "mjcf":
                    loaded_path = Path(
                        unisis_assets.prepare_mjcf_asset(source_path, cache_path)
                    ).resolve()
                else:
                    loaded_path = source_path
                if not loaded_path.is_file():
                    raise FileNotFoundError(
                        f"Asset converter returned a missing file for {entity_id!r}: {loaded_path}"
                    )
                report["loaded_file"] = str(loaded_path)
                conversion_report = loaded_path.parent / "conversion.json"
                if entity_type in {"mesh", "mjcf"} and conversion_report.is_file():
                    details = json.loads(conversion_report.read_text(encoding="utf-8"))
                    report["conversion"] = details
                    loaded.warnings.extend(
                        f"Entity {entity_id!r}: {warning}"
                        for warning in details.get("warnings", [])
                    )

                if entity_type == "mesh":
                    body_id, collision_kind, volume_note = _load_mesh(
                        p,
                        entity,
                        loaded_path,
                        position,
                        orientation,
                        scale,
                        fixed=fixed,
                        physics_client_id=physics_client_id,
                    )
                    report["collision_approximation"] = collision_kind
                    report.update(volume_note)
                else:
                    if not _uniform_scale(scale):
                        raise ValueError(
                            f"Entity {entity_id!r}: PyBullet URDF globalScaling requires uniform scale"
                        )
                    use_fixed_base = bool(entity.get("use_fixed_base", fixed))
                    if robot and "use_fixed_base" not in entity:
                        use_fixed_base = True
                    flags = p.URDF_USE_INERTIA_FROM_FILE
                    if entity.get("enable_self_collisions", entity.get("self_collision", False)):
                        flags |= p.URDF_USE_SELF_COLLISION
                    body_id = p.loadURDF(
                        str(loaded_path),
                        basePosition=position,
                        baseOrientation=orientation,
                        useFixedBase=use_fixed_base,
                        globalScaling=scale[0],
                        flags=flags,
                        physicsClientId=physics_client_id,
                    )
                    _apply_robot_joint_positions(
                        p, body_id, entity, physics_client_id, entity_id
                    )
                    if not use_fixed_base and entity.get("mass") is not None:
                        _override_articulated_mass(
                            p, body_id, float(entity["mass"]), physics_client_id
                        )
                        report["total_mass_override_kg"] = float(entity["mass"])
                    else:
                        report["authored_link_masses_preserved"] = True
                    if (
                        not use_fixed_base
                        and entity.get("mass") is None
                        and _material_value(entity, "rho") is not None
                    ):
                        loaded.warnings.append(
                            f"Entity {entity_id!r}: density is not applied to articulated asset; "
                            "authored per-link masses are retained"
                        )
                    _apply_body_dynamics(
                        p, body_id, entity, physics_client_id, fixed=use_fixed_base
                    )
                    report["fixed_base"] = use_fixed_base
            else:
                body_id, collision_kind = _load_primitive(
                    p,
                    entity,
                    position,
                    orientation,
                    scale,
                    fixed=fixed,
                    physics_client_id=physics_client_id,
                )
                report["collision_approximation"] = collision_kind
        except Exception as exc:
            raise RuntimeError(f"Failed loading UniSis entity {entity_id!r}: {exc}") from exc

        loaded.entity_id_to_body_id[entity_id] = int(body_id)
        loaded.body_id_to_entity_id[int(body_id)] = entity_id
        loaded.entity_configs[entity_id] = dict(entity)
        loaded.asset_info.append(report)
        if robot:
            loaded.robot_ids.append(entity_id)
        else:
            loaded.obj_ids["fixed" if fixed else "rigid"].append(int(body_id))
        if report.get("collision_approximation") == "convex_hull":
            loaded.warnings.append(
                f"Entity {entity_id!r}: dynamic mesh collision is a convex hull approximation"
            )
        if report.get("mass_volume_method") == "convex_hull":
            loaded.warnings.append(
                f"Entity {entity_id!r}: density-based mass uses convex-hull volume because the mesh is not watertight"
            )
    return loaded


def _validate_entity_physics(entity: dict[str, Any], entity_id: str) -> None:
    for key in ("mass", "rho", "density", "friction", "restitution"):
        if key in entity and entity[key] is not None:
            _positive_float(entity[key], f"Entity {entity_id!r} {key}", allow_zero=(key == "restitution"))
    material = entity.get("material") or {}
    params = material.get("parameters", {}) if isinstance(material, dict) else {}
    if params is not None and not isinstance(params, dict):
        raise ValueError(f"Entity {entity_id!r} material.parameters must be a mapping")
    if isinstance(material, dict):
        for key in ("rho", "density", "friction", "restitution"):
            if material.get(key) is not None:
                _positive_float(
                    material[key], f"Entity {entity_id!r} material.{key}",
                    allow_zero=(key == "restitution"),
                )
    if isinstance(params, dict):
        for key in ("rho", "density", "friction", "restitution"):
            if params.get(key) is not None:
                _positive_float(
                    params[key], f"Entity {entity_id!r} material.parameters.{key}",
                    allow_zero=(key == "restitution"),
                )
    if entity.get("surface_color") is not None:
        color = entity["surface_color"]
        if not isinstance(color, (list, tuple)) or len(color) not in (3, 4):
            raise ValueError(f"Entity {entity_id!r} surface_color must be RGB or RGBA")
        if any(float(v) < 0.0 or float(v) > 1.0 for v in color):
            raise ValueError(f"Entity {entity_id!r} surface_color values must be in [0, 1]")


def _positive_float(value: Any, label: str, *, allow_zero: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(result) or result < 0 or (result == 0 and not allow_zero):
        adjective = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{label} must be finite and {adjective}")
    return result


def _normalize_position(value: Any, entity_id: str) -> list[float]:
    if isinstance(value, dict):
        seed = value.get("seed")
        if seed is None:
            # A stable per-entity seed keeps scene materialization reproducible.
            digest = hashlib.sha256(entity_id.encode("utf-8")).digest()
            seed = int.from_bytes(digest[:4], "big")
        if "min" in value or "max" in value:
            low = _vec3(value.get("min"), f"Entity {entity_id!r} position.min")
            high = _vec3(value.get("max"), f"Entity {entity_id!r} position.max")
            ranges = list(zip(low, high))
        else:
            ranges = []
            for axis in ("x", "y", "z"):
                axis_value = value.get(axis)
                if isinstance(axis_value, (int, float)):
                    ranges.append((float(axis_value), float(axis_value)))
                elif isinstance(axis_value, (list, tuple)) and len(axis_value) == 2:
                    ranges.append((float(axis_value[0]), float(axis_value[1])))
                else:
                    raise ValueError(
                        f"Entity {entity_id!r} position.{axis} must be a number or [min, max]"
                    )
        rng = random.Random(int(seed))
        return [rng.uniform(min(a, b), max(a, b)) if a != b else float(a) for a, b in ranges]
    return _vec3(value, f"Entity {entity_id!r} position")


def _vec3(value: Any, label: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{label} must have exactly 3 values")
    result = [float(part) for part in value]
    if not all(math.isfinite(part) for part in result):
        raise ValueError(f"{label} must contain finite values")
    return result


def _normalize_rotation(entity: dict[str, Any]) -> list[float]:
    if entity.get("euler") is not None:
        if entity.get("rotation") is not None:
            # Match UniSis backend behavior: euler takes precedence over rotation.
            pass
        euler = _vec3(entity["euler"], f"Entity {entity.get('id')!r} euler")
        roll, pitch, yaw = (math.radians(angle) for angle in euler)
        cr, sr = math.cos(roll / 2), math.sin(roll / 2)
        cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
        cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
        # Intrinsic XYZ Euler, matching Unisis' Newton backend.
        w = cy * cp * cr + sy * sp * sr
        x = cy * cp * sr - sy * sp * cr
        y = sy * cp * cr + cy * sp * sr
        z = sy * cp * sr - cy * sp * cr
    else:
        raw = entity.get("rotation", [1.0, 0.0, 0.0, 0.0])
        if not isinstance(raw, (list, tuple)) or len(raw) != 4:
            raise ValueError(
                f"Entity {entity.get('id')!r} rotation must be quaternion [w, x, y, z]"
            )
        w, x, y, z = (float(v) for v in raw)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if not math.isfinite(norm) or norm < _VECTOR_EPS:
        raise ValueError(f"Entity {entity.get('id')!r} rotation quaternion is invalid")
    # PyBullet uses [x, y, z, w].
    return [x / norm, y / norm, z / norm, w / norm]


def _normalize_scale(value: Any, entity_id: str) -> float | list[float]:
    if isinstance(value, (int, float)):
        scale = float(value)
        values = [scale, scale, scale]
    elif isinstance(value, (list, tuple)) and len(value) == 3:
        values = [float(v) for v in value]
    else:
        raise ValueError(f"Entity {entity_id!r} scale must be a positive number or vec3")
    if not all(math.isfinite(v) and v > 0 for v in values):
        raise ValueError(f"Entity {entity_id!r} scale values must be finite and positive")
    return values[0] if values[0] == values[1] == values[2] else values


def _scale_vec(value: float | list[float]) -> tuple[float, float, float]:
    if isinstance(value, (int, float)):
        scale = float(value)
        return scale, scale, scale
    return float(value[0]), float(value[1]), float(value[2])


def _uniform_scale(scale: tuple[float, float, float]) -> bool:
    return math.isclose(scale[0], scale[1]) and math.isclose(scale[1], scale[2])


def _material_value(entity: dict[str, Any], name: str, default: float | None = None) -> float | None:
    material = entity.get("material") or {}
    params = material.get("parameters") or {}
    value = entity.get(name)
    if value is None:
        value = params.get(name)
    if value is None:
        value = material.get(name)
    if value is None and name == "rho":
        value = entity.get("density", params.get("density", material.get("density")))
    if value is None:
        return default
    return float(value)


def _entity_fixed(entity: dict[str, Any], *, robot: bool, entity_type: str) -> bool:
    if entity_type == "plane":
        return True
    if "fixed" in entity:
        return bool(entity["fixed"])
    if "use_fixed_base" in entity:
        return bool(entity["use_fixed_base"])
    return bool(robot)


def _entity_mass(entity: dict[str, Any], *, fixed: bool, volume: float | None = None) -> float:
    if fixed:
        return 0.0
    if entity.get("mass") is not None:
        return float(entity["mass"])
    rho = _material_value(entity, "rho", 300.0)
    if volume is None:
        raise ValueError(f"Entity {entity.get('id')!r}: no volume is available to derive mass")
    return float(rho) * volume


def _load_primitive(
    p: Any,
    entity: dict[str, Any],
    position: tuple[float, float, float],
    orientation: tuple[float, float, float, float],
    scale: tuple[float, float, float],
    *,
    fixed: bool,
    physics_client_id: int,
) -> tuple[int, str]:
    entity_type = entity["type"]
    collision_enabled = entity.get("collision", True) is not False
    collision_shape = -1
    visual_shape = -1
    volume: float | None = None
    collision_kind = "primitive"

    if entity_type == "plane":
        collision_shape = p.createCollisionShape(
            p.GEOM_PLANE,
            planeNormal=[0, 0, 1],
            physicsClientId=physics_client_id,
        ) if collision_enabled else -1
        plane_color = _color_for_entity(entity)
        visual_kwargs = {
            "shapeType": p.GEOM_PLANE,
            "planeNormal": [0, 0, 1],
            "physicsClientId": physics_client_id,
        }
        if plane_color is not None:
            visual_kwargs["rgbaColor"] = plane_color
        visual_shape = p.createVisualShape(**visual_kwargs)
        volume = None
    elif entity_type == "box":
        size = entity.get("size", entity.get("dimensions"))
        if size is None:
            raise ValueError("box entity needs 'size' (full extents)")
        extent = _vec3(size, f"Entity {entity['id']!r} box size")
        full_size = [extent[i] * scale[i] for i in range(3)]
        half_extents = [part / 2.0 for part in full_size]
        if any(part <= 0 for part in half_extents):
            raise ValueError("box size values must be positive")
        volume = full_size[0] * full_size[1] * full_size[2]
        if collision_enabled:
            collision_shape = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=half_extents, physicsClientId=physics_client_id
            )
        visual_shape = _primitive_visual(p, entity, p.GEOM_BOX, physics_client_id,
                                         halfExtents=half_extents)
    elif entity_type == "sphere":
        radius = float(entity.get("radius", 0.5))
        if not _uniform_scale(scale):
            raise ValueError("sphere geometry requires uniform scale")
        radius *= scale[0]
        if radius <= 0:
            raise ValueError("sphere radius must be positive")
        volume = (4.0 / 3.0) * math.pi * radius**3
        if collision_enabled:
            collision_shape = p.createCollisionShape(
                p.GEOM_SPHERE, radius=radius, physicsClientId=physics_client_id
            )
        visual_shape = _primitive_visual(p, entity, p.GEOM_SPHERE,
                                         physics_client_id, radius=radius)
    elif entity_type == "cylinder":
        if not math.isclose(scale[0], scale[1]):
            raise ValueError("cylinder geometry requires equal X/Y scale")
        radius = float(entity.get("radius", 0.5)) * scale[0]
        height = float(entity.get("height", 1.0)) * scale[2]
        if radius <= 0 or height <= 0:
            raise ValueError("cylinder radius and height must be positive")
        volume = math.pi * radius**2 * height
        if collision_enabled:
            collision_shape = p.createCollisionShape(
                p.GEOM_CYLINDER,
                radius=radius,
                height=height,
                physicsClientId=physics_client_id,
            )
        visual_shape = _primitive_visual(p, entity, p.GEOM_CYLINDER,
                                         physics_client_id, radius=radius, length=height)
    else:  # pragma: no cover - parser prevents this
        raise ValueError(f"Unsupported primitive type {entity_type!r}")

    mass = _entity_mass(entity, fixed=fixed, volume=volume) if entity_type != "plane" else 0.0
    body_id = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=position,
        baseOrientation=orientation,
        physicsClientId=physics_client_id,
    )
    _apply_body_dynamics(p, body_id, entity, physics_client_id, fixed=fixed)
    return int(body_id), collision_kind


def _primitive_visual(
    p: Any,
    entity: dict[str, Any],
    geometry: int,
    physics_client_id: int,
    **shape_args: Any,
) -> int:
    color = _color_for_entity(entity)
    kwargs = dict(shape_args)
    kwargs["shapeType"] = geometry
    kwargs["physicsClientId"] = physics_client_id
    if color is not None:
        kwargs["rgbaColor"] = color
    return int(p.createVisualShape(**kwargs))


def _color_for_entity(entity: dict[str, Any]) -> list[float] | None:
    color = entity.get("surface_color")
    if color is not None:
        rgba = [float(v) for v in color]
        if len(rgba) == 3:
            rgba.append(1.0)
        return rgba
    material = entity.get("material") or {}
    color = material.get("color") if isinstance(material, dict) else None
    if isinstance(color, (list, tuple)) and len(color) in (3, 4):
        rgba = [float(v) for v in color]
        if len(rgba) == 3:
            rgba.append(1.0)
        return rgba
    return None


def _load_mesh(
    p: Any,
    entity: dict[str, Any],
    mesh_path: Path,
    position: tuple[float, float, float],
    orientation: tuple[float, float, float, float],
    scale: tuple[float, float, float],
    *,
    fixed: bool,
    physics_client_id: int,
) -> tuple[int, str, dict[str, Any]]:
    collision_enabled = entity.get("collision", True) is not False
    collision_kind = "concave_static_mesh" if fixed else "convex_hull"
    # Static meshes need no volume: rugs and other valid planar surfaces may
    # have zero enclosed volume. Explicit dynamic mass also needs no estimate.
    volume = None
    volume_method = "not_needed" if fixed else "explicit_mass"
    if not fixed and entity.get("mass") is None:
        volume, volume_method = _mesh_volume(mesh_path, scale)
    mass = _entity_mass(entity, fixed=fixed, volume=volume)
    visual_shape = -1
    collision_shape = -1
    kwargs = {
        "shapeType": p.GEOM_MESH,
        "fileName": str(mesh_path),
        "meshScale": scale,
        "physicsClientId": physics_client_id,
    }
    color = _color_for_entity(entity)
    if color is not None:
        kwargs["rgbaColor"] = color
    visual_shape = p.createVisualShape(**kwargs)
    if collision_enabled:
        collision_kwargs = {
            "shapeType": p.GEOM_MESH,
            "fileName": str(mesh_path),
            "meshScale": scale,
            "physicsClientId": physics_client_id,
        }
        if fixed:
            collision_kwargs["flags"] = p.GEOM_FORCE_CONCAVE_TRIMESH
        collision_shape = p.createCollisionShape(**collision_kwargs)
    body_id = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=position,
        baseOrientation=orientation,
        physicsClientId=physics_client_id,
    )
    _apply_body_dynamics(p, body_id, entity, physics_client_id, fixed=fixed)
    note = {"mesh_volume_m3": volume, "mass_volume_method": volume_method}
    return int(body_id), collision_kind, note


def _mesh_volume(mesh_path: Path, scale: tuple[float, float, float]) -> tuple[float, str]:
    try:
        import trimesh
    except ImportError as exc:  # pragma: no cover - environment dependency
        raise RuntimeError(
            "trimesh is required to derive density-based mass for mesh entities"
        ) from exc
    loaded = trimesh.load(str(mesh_path), force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        geometries = list(loaded.dump())
        mesh = trimesh.util.concatenate(geometries)
    else:
        mesh = loaded
    if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
        raise ValueError(f"Could not read a triangle mesh from {mesh_path}")
    volume = abs(float(mesh.volume))
    method = "mesh"
    if not mesh.is_watertight or not math.isfinite(volume) or volume <= _VECTOR_EPS:
        try:
            volume = abs(float(mesh.convex_hull.volume))
        except Exception as exc:
            raise ValueError(f"Mesh has no usable closed volume: {mesh_path}") from exc
        method = "convex_hull"
    if not math.isfinite(volume) or volume <= _VECTOR_EPS:
        raise ValueError(f"Mesh volume is zero or invalid: {mesh_path}")
    volume *= abs(scale[0] * scale[1] * scale[2])
    return volume, method


def _apply_body_dynamics(
    p: Any,
    body_id: int,
    entity: dict[str, Any],
    physics_client_id: int,
    *,
    fixed: bool,
) -> None:
    lateral = _material_value(entity, "friction", 0.5)
    restitution = _material_value(entity, "restitution", 0.0)
    kwargs = {"physicsClientId": physics_client_id}
    if lateral is not None:
        kwargs["lateralFriction"] = lateral
    if restitution is not None:
        kwargs["restitution"] = restitution
    p.changeDynamics(body_id, -1, **kwargs)
    if entity.get("type") in {"urdf", "mjcf"}:
        joint_count = p.getNumJoints(body_id, physicsClientId=physics_client_id)
        for joint_index in range(joint_count):
            link_kwargs = dict(kwargs)
            p.changeDynamics(body_id, joint_index, **link_kwargs)


def _override_articulated_mass(
    p: Any, body_id: int, total_mass: float, physics_client_id: int
) -> None:
    """Scale authored link masses to a requested total while keeping proportions."""

    link_indices = [-1, *range(p.getNumJoints(body_id, physicsClientId=physics_client_id))]
    dynamics = [
        p.getDynamicsInfo(body_id, link, physicsClientId=physics_client_id)
        for link in link_indices
    ]
    masses = [float(info[0]) for info in dynamics]
    existing_total = sum(masses)
    if existing_total > _VECTOR_EPS:
        mass_scale = total_mass / existing_total
        adjusted = [mass * mass_scale for mass in masses]
    else:
        mass_scale = 1.0
        adjusted = [total_mass, *([0.0] * (len(link_indices) - 1))]
    for link, mass, info in zip(link_indices, adjusted, dynamics):
        kwargs: dict[str, Any] = {
            "mass": mass,
            "physicsClientId": physics_client_id,
        }
        inertia = info[2]
        if inertia is not None and len(inertia) == 3:
            kwargs["localInertiaDiagonal"] = [float(value) * mass_scale for value in inertia]
        p.changeDynamics(body_id, link, **kwargs)


def _apply_robot_joint_positions(
    p: Any,
    body_id: int,
    entity: dict[str, Any],
    physics_client_id: int,
    entity_id: str,
) -> None:
    joint_count = p.getNumJoints(body_id, physicsClientId=physics_client_id)
    movable: list[tuple[int, str]] = []
    for joint_index in range(joint_count):
        info = p.getJointInfo(body_id, joint_index, physicsClientId=physics_client_id)
        joint_type = info[2]
        joint_name = info[1].decode("utf-8") if isinstance(info[1], bytes) else str(info[1])
        if joint_type != p.JOINT_FIXED:
            if joint_type in (getattr(p, "JOINT_SPHERICAL", -100), getattr(p, "JOINT_PLANAR", -101)):
                raise ValueError(
                    f"Entity {entity_id!r} has multi-DOF joint {joint_name!r}; initial scalar joint positions are unsupported"
                )
            movable.append((joint_index, joint_name))

    robot_kwargs = entity.get("robot_adapter_kwargs") or {}
    if not isinstance(robot_kwargs, dict):
        raise ValueError(f"Entity {entity_id!r} robot_adapter_kwargs must be a mapping")
    default_qpos = robot_kwargs.get("default_qpos")
    values: dict[str, float] = {}
    if default_qpos is not None:
        if not isinstance(default_qpos, (list, tuple)):
            raise ValueError(f"Entity {entity_id!r} default_qpos must be a sequence")
        if len(default_qpos) != len(movable):
            raise ValueError(
                f"Entity {entity_id!r} default_qpos has {len(default_qpos)} values for "
                f"{len(movable)} movable joints"
            )
        values.update({name: float(value) for (_, name), value in zip(movable, default_qpos)})

    named_values = entity.get("initial_joints") or entity.get("joint_positions") or {}
    if not isinstance(named_values, dict):
        raise ValueError(f"Entity {entity_id!r} initial_joints must be a name-to-position mapping")
    valid_names = {name for _, name in movable}
    unknown = set(map(str, named_values)) - valid_names
    if unknown:
        raise ValueError(f"Entity {entity_id!r} has unknown joint names: {sorted(unknown)}")
    values.update({str(name): float(value) for name, value in named_values.items()})
    for joint_index, joint_name in movable:
        if joint_name in values:
            p.resetJointState(
                body_id,
                joint_index,
                targetValue=values[joint_name],
                targetVelocity=0.0,
                physicsClientId=physics_client_id,
            )
