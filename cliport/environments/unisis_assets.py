"""UniSis mesh/MJCF asset conversion helpers for GenSim's PyBullet scene loader.

GLB scene graph transforms and source origins are retained in the combined OBJ.
Mesh up axes are resolved to match UniSis/Genesis before entity scale and pose.
MJCF is compiled by MuJoCo before exporting an articulated URDF. Conversion
limits are exposed as module constants and written to conversion.json beside
each cached result. See those limits before using converted materials or
dynamics as an exact replacement for the source format.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

_CONVERTER_VERSION = "unisis-assets-v5"

MESH_CONVERSION_LIMITATIONS = (
    "OBJ preserves scene-node transforms and source origins; Y-up meshes are converted to Z-up as in UniSis/Genesis without recentering.",
    "Base-color textures are copied to PNG and referenced through MTL when trimesh exposes them.",
    "OBJ/MTL cannot represent all glTF PBR channels such as normal, metallic, roughness, emissive, or alpha modes.",
    "PyBullet OBJ material support varies by build; geometry conversion does not depend on materials.",
)
MJCF_CONVERSION_LIMITATIONS = (
    "Actuators, tendons, equality constraints, sensors, contact exclusions/pairs, and solver settings are not represented in URDF.",
    "Per-joint damping, armature, friction loss, and actuator limits are not represented.",
    "Body/joint frames, mesh and supported primitive geometry, visual colors, collision inclusion, mass, and principal inertia are exported.",
    "Hfield, SDF, plane, ball-joint, free-joint, and multiple joints on one body are rejected explicitly.",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest_files(paths: Iterable[Path], salt: str) -> str:
    digest = hashlib.sha256(salt.encode("utf-8"))
    for path in sorted({Path(path).resolve() for path in paths}, key=str):
        digest.update(str(path).encode("utf-8"))
        if path.is_file():
            digest.update(_sha256_file(path).encode("ascii"))
        else:
            digest.update(b"<missing>")
    return digest.hexdigest()


def _safe_name(value: str, fallback: str = "asset") -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._")
    return name or fallback


def _load_trimesh():
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError("GLB conversion requires the GenSim trimesh dependency.") from exc
    return trimesh


def _rgba(value: Any, default: tuple[float, float, float, float] = (0.7, 0.7, 0.7, 1.0)):
    try:
        values = [float(channel) for channel in value]
    except (TypeError, ValueError):
        return default
    if len(values) < 3:
        return default
    if max(values[:4], default=0.0) > 1.0:
        values = [channel / 255.0 for channel in values]
    if len(values) < 4:
        values.append(1.0)
    return tuple(max(0.0, min(1.0, channel)) for channel in values[:4])


def _material_color(material: Any):
    if material is not None:
        for attribute in ("baseColorFactor", "diffuse", "main_color"):
            value = getattr(material, attribute, None)
            if value is not None:
                color = _rgba(value, default=None)
                if color is not None:
                    return color
    return (0.7, 0.7, 0.7, 1.0)


def _save_texture(material: Any, texture_dir: Path) -> str | None:
    image = getattr(material, "baseColorTexture", None)
    if image is None:
        image = getattr(material, "image", None)
    if image is None or not hasattr(image, "save"):
        return None
    try:
        texture_dir.mkdir(parents=True, exist_ok=True)
        pixels = image.convert("RGBA").tobytes()
        texture_hash = hashlib.sha256(repr(image.size).encode("ascii") + pixels).hexdigest()[:16]
        texture_path = texture_dir / f"base_color_{texture_hash}.png"
        if not texture_path.exists():
            image.save(texture_path, format="PNG")
        return texture_path.relative_to(texture_dir.parent).as_posix()
    except Exception:
        return None


def _write_combined_obj(
    source_path: Path, output_path: Path, *, is_mesh_zup: bool
) -> dict[str, Any]:
    trimesh = _load_trimesh()
    try:
        scene = trimesh.load(source_path, force="scene", process=False)
    except Exception as exc:
        raise ValueError(f"Could not load mesh asset {source_path}: {exc}") from exc
    if isinstance(scene, trimesh.Trimesh):
        wrapper = trimesh.Scene()
        wrapper.add_geometry(scene, node_name=source_path.stem, geom_name=source_path.stem)
        scene = wrapper
    if not isinstance(scene, trimesh.Scene) or not scene.graph.nodes_geometry:
        raise ValueError(f"Mesh asset contains no geometry: {source_path}")

    texture_dir = output_path.parent / f"{output_path.stem}_textures"
    obj_lines = [f"mtllib {output_path.with_suffix('.mtl').name}"]
    materials: dict[str, tuple[tuple[float, float, float, float], str | None]] = {}
    material_keys: dict[tuple[Any, ...], str] = {}
    resolved_materials: dict[
        tuple[int, bool], tuple[tuple[float, float, float, float], str | None]
    ] = {}
    warnings: list[str] = []
    vertex_offset = 0
    uv_offset = 0
    instance_count = 0
    triangle_count = 0

    def use_material(source_material: Any, node_label: str, face_color: Any, uv_valid: bool) -> str:
        if face_color is not None:
            color = _rgba(face_color)
            texture = None
            key: tuple[Any, ...] = ("face", *color)
        else:
            material_identity = id(source_material) if source_material is not None else 0
            resolved_key = (material_identity, uv_valid)
            if resolved_key not in resolved_materials:
                color = _material_color(source_material)
                has_base_texture = (
                    source_material is not None
                    and (
                        getattr(source_material, "baseColorTexture", None) is not None
                        or getattr(source_material, "image", None) is not None
                    )
                )
                texture = _save_texture(source_material, texture_dir) if uv_valid else None
                if has_base_texture and uv_valid and texture is None:
                    warnings.append(f"Could not export a base-color texture from node {node_label!r}.")
                resolved_materials[resolved_key] = (color, texture)
            color, texture = resolved_materials[resolved_key]
            key = ("material", material_identity, color, texture, node_label if source_material is None else "")
        if key in material_keys:
            return material_keys[key]
        name = _safe_name(f"material_{len(materials)}_{node_label}")
        material_keys[key] = name
        materials[name] = (color, texture)
        return name

    for node_index, node_name in enumerate(scene.graph.nodes_geometry):
        transform, geometry_name = scene.graph[node_name]
        mesh = scene.geometry[geometry_name]
        if not hasattr(mesh, "vertices") or not hasattr(mesh, "faces"):
            warnings.append(f"Skipped non-triangle geometry on node {node_name!r}.")
            continue
        baked = mesh.copy()
        baked.apply_transform(transform)
        if not is_mesh_zup:
            # Genesis Y_UP_TRANSFORM.T: (x, y, z) -> (x, -z, y).
            # Apply after GLB node transforms and before YAML scale/rotation.
            baked.apply_transform([
                [1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1],
            ])
        vertices = baked.vertices
        faces = baked.faces
        if len(vertices) == 0 or len(faces) == 0:
            warnings.append(f"Skipped empty geometry on node {node_name!r}.")
            continue

        label = _safe_name(str(node_name), f"node_{node_index}")
        obj_lines.append(f"o {label}")
        for vertex in vertices:
            obj_lines.append("v " + " ".join(f"{float(value):.9g}" for value in vertex))
        uv = getattr(getattr(mesh, "visual", None), "uv", None)
        uv_valid = uv is not None and len(uv) == len(vertices)
        if uv_valid:
            for value in uv:
                obj_lines.append(f"vt {float(value[0]):.9g} {float(value[1]):.9g}")

        visual = getattr(mesh, "visual", None)
        face_colors = None
        if visual is not None and getattr(visual, "kind", None) in {"face", "vertex"}:
            try:
                colors = visual.face_colors
                if len(colors) == len(faces):
                    face_colors = colors
            except Exception:
                pass
        source_material = getattr(visual, "material", None) if visual is not None else None
        face_materials = getattr(visual, "face_materials", None) if visual is not None else None
        sub_materials = getattr(source_material, "materials", None)
        has_face_materials = face_materials is not None and len(face_materials) == len(faces)
        active_material = None
        for face_index, face in enumerate(faces):
            selected_material = source_material
            if has_face_materials and sub_materials:
                mat_index = int(face_materials[face_index])
                if 0 <= mat_index < len(sub_materials):
                    selected_material = sub_materials[mat_index]
            color = face_colors[face_index] if face_colors is not None else None
            material_name = use_material(selected_material, label, color, uv_valid)
            if material_name != active_material:
                obj_lines.append(f"usemtl {material_name}")
                active_material = material_name
            indices = [int(index) + vertex_offset + 1 for index in face]
            if uv_valid:
                uv_indices = [int(index) + uv_offset + 1 for index in face]
                obj_lines.append("f " + " ".join(f"{v}/{vt}" for v, vt in zip(indices, uv_indices)))
            else:
                obj_lines.append("f " + " ".join(map(str, indices)))
        vertex_offset += len(vertices)
        if uv_valid:
            uv_offset += len(uv)
        instance_count += 1
        triangle_count += len(faces)

    if instance_count == 0:
        raise ValueError(f"Mesh asset contains no triangle faces: {source_path}")
    mtl_lines: list[str] = []
    for name, (color, texture) in materials.items():
        r, g, b, alpha = color
        mtl_lines.extend((f"newmtl {name}", f"Kd {r:.7g} {g:.7g} {b:.7g}", f"d {alpha:.7g}", "illum 2"))
        if texture:
            mtl_lines.append(f"map_Kd {texture}")
        mtl_lines.append("")
    output_path.write_text("\n".join(obj_lines) + "\n", encoding="utf-8")
    output_path.with_suffix(".mtl").write_text("\n".join(mtl_lines) + "\n", encoding="utf-8")
    return {
        "source_mesh_is_zup": is_mesh_zup,
        "mesh_axis_conversion": "none" if is_mesh_zup else "y_up_to_z_up",
        "instances": instance_count,
        "triangles": triangle_count,
        "materials": len(materials),
        "warnings": sorted(set(warnings)),
        "limitations": list(MESH_CONVERSION_LIMITATIONS),
    }


def _cache_complete(cache_root: Path, output_name: str) -> bool:
    report_path = cache_root / "conversion.json"
    if not (cache_root / output_name).is_file() or not report_path.is_file():
        return False
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        return all((cache_root / value).is_file() for value in report.get("outputs", ()))
    except (OSError, json.JSONDecodeError, TypeError):
        return False


def _write_report(cache_root: Path, report: dict[str, Any]) -> None:
    report["outputs"] = [
        path.relative_to(cache_root).as_posix()
        for path in sorted(cache_root.rglob("*"))
        if path.is_file() and path.name != "conversion.json"
    ]
    (cache_root / "conversion.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _new_cache_dir(cache_dir: Path, key: str) -> tuple[Path, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / key, Path(tempfile.mkdtemp(prefix=f".{key}.", dir=cache_dir))


@contextmanager
def _cache_lock(cache_dir: Path, key: str):
    """Serialize validation and publication for one content-addressed asset."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Keep lock files in place: unlinking one could let waiters lock different inodes.
    with (cache_dir / f".{key}.lock").open("a") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def prepare_mesh_asset(
    source_path: Path, cache_dir: Path, *, file_meshes_are_zup: bool | None = None
) -> Path:
    """Return a cached OBJ with the same asset-axis interpretation as UniSis.

    Graph node transforms encoded inside the GLB are baked into OBJ vertices.
    GLB/GLTF default to Y-up and other formats default to Z-up, matching
    Genesis Mesh. ``file_meshes_are_zup`` explicitly overrides the convention.
    The entity-level scale in UniSis YAML remains external for the environment
    loader to apply. Colors and base-color textures are exported through MTL
    where possible; conversion.json records limitations and warnings.
    """
    source_path = Path(source_path).expanduser().resolve()
    cache_dir = Path(cache_dir).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"UniSis mesh asset does not exist: {source_path}")
    if file_meshes_are_zup is not None and not isinstance(file_meshes_are_zup, bool):
        raise ValueError("file_meshes_are_zup must be boolean or null")
    is_mesh_zup = (
        source_path.suffix.lower() not in {".glb", ".gltf"}
        if file_meshes_are_zup is None else file_meshes_are_zup
    )
    digest = _digest_files(
        [source_path], _CONVERTER_VERSION + f":mesh:zup={is_mesh_zup}"
    )
    key = f"unisis-mesh-{digest[:24]}"
    output_name = f"{_safe_name(source_path.stem)}.obj"
    with _cache_lock(cache_dir, key):
        final_dir = cache_dir / key
        if _cache_complete(final_dir, output_name):
            return final_dir / output_name

        final_dir, temp_dir = _new_cache_dir(cache_dir, key)
        try:
            output_path = temp_dir / output_name
            details = _write_combined_obj(source_path, output_path, is_mesh_zup=is_mesh_zup)
            _write_report(
                temp_dir,
                {
                    "kind": "glb_to_obj",
                    "source": str(source_path),
                    "source_sha256": _sha256_file(source_path),
                    "converter_version": _CONVERTER_VERSION,
                    **details,
                },
            )
            if final_dir.exists():
                shutil.rmtree(final_dir)
            os.replace(temp_dir, final_dir)
        except Exception:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise
        return final_dir / output_name


def _mjcf_dependencies(source_path: Path) -> list[Path]:
    """Resolve local XML includes and asset file references for cache invalidation."""
    dependencies: set[Path] = set()
    pending = [source_path.resolve()]
    parsed: set[Path] = set()
    while pending:
        xml_path = pending.pop()
        if xml_path in parsed:
            continue
        parsed.add(xml_path)
        dependencies.add(xml_path)
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError as exc:
            raise ValueError(f"Invalid MJCF XML in {xml_path}: {exc}") from exc
        compiler = root.find("compiler")
        meshdir = compiler.get("meshdir", ".") if compiler is not None else "."
        assetdir = compiler.get("assetdir", ".") if compiler is not None else "."
        for element in root.iter():
            file_value = element.get("file")
            if not file_value:
                continue
            if element.tag == "include":
                pending.append((xml_path.parent / file_value).resolve())
                continue
            base = xml_path.parent
            if element.tag == "mesh":
                base = (xml_path.parent / meshdir).resolve()
            elif element.tag in {"texture", "hfield", "skin"}:
                base = (xml_path.parent / assetdir).resolve()
            dependencies.add((base / file_value).resolve())
    return sorted(dependencies, key=str)


def _compile_mjcf(source_path: Path):
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError(
            "MJCF conversion needs mujoco>=3.2,<4 in the GenSim uv environment."
        ) from exc
    try:
        return mujoco, mujoco.MjModel.from_xml_path(str(source_path))
    except Exception as exc:
        raise ValueError(f"MuJoCo could not compile MJCF asset {source_path}: {exc}") from exc


def _quat_to_rpy(quaternion: Any) -> tuple[float, float, float]:
    w, x, y, z = (float(value) for value in quaternion)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0:
        return (0.0, 0.0, 0.0)
    w, x, y, z = (value / norm for value in (w, x, y, z))
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sin_pitch) if abs(sin_pitch) >= 1.0 else math.asin(sin_pitch)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def _quat_multiply(left: Any, right: Any):
    import numpy as np

    lw, lx, ly, lz = (float(value) for value in left)
    rw, rx, ry, rz = (float(value) for value in right)
    return np.asarray(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dtype=float,
    )


def _rotation_matrix(quaternion: Any):
    import numpy as np

    w, x, y, z = (float(value) for value in quaternion)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0:
        return np.eye(3)
    w, x, y, z = (value / norm for value in (w, x, y, z))
    return np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=float,
    )


def _fmt(values: Any) -> str:
    return " ".join(f"{float(value):.10g}" for value in values)


def _set_origin(parent: ET.Element, xyz: Any, quaternion: Any) -> None:
    roll, pitch, yaw = _quat_to_rpy(quaternion)
    ET.SubElement(parent, "origin", {"xyz": _fmt(xyz), "rpy": _fmt((roll, pitch, yaw))})


def _validate_bodies(model: Any, mujoco: Any) -> tuple[dict[int, str], dict[int, Any]]:
    import numpy as np

    names: dict[int, str] = {}
    shifts: dict[int, Any] = {}
    seen: set[str] = set()
    for body_id in range(1, model.nbody):
        raw_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        name = _safe_name(raw_name or f"body_{body_id}", f"body_{body_id}")
        if name in seen:
            raise ValueError(f"Duplicate MJCF body name {name!r} cannot be represented in URDF.")
        seen.add(name)
        names[body_id] = name
        count = int(model.body_jntnum[body_id])
        if count > 1:
            raise ValueError(
                f"MJCF body {raw_name or body_id!r} has {count} joints; "
                "the exporter supports at most one joint per body."
            )
        shifts[body_id] = (
            model.jnt_pos[int(model.body_jntadr[body_id])].copy()
            if count == 1
            else np.zeros(3, dtype=float)
        )
    return names, shifts


def _write_mesh_obj(path: Path, vertices: Any, faces: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        stream.write("# Compiled MuJoCo mesh for GenSim.\n")
        for vertex in vertices:
            stream.write("v " + _fmt(vertex) + "\n")
        for face in faces:
            stream.write("f " + " ".join(str(int(value) + 1) for value in face) + "\n")


def _geom_rgba(model: Any, geom_id: int):
    material_id = int(model.geom_matid[geom_id])
    rgba = model.mat_rgba[material_id] if material_id >= 0 else model.geom_rgba[geom_id]
    return tuple(max(0.0, min(1.0, float(value))) for value in rgba)


def _emit_geom(
    model: Any,
    mujoco: Any,
    geom_id: int,
    link: ET.Element,
    link_shift: Any,
    mesh_paths: dict[int, str],
    mesh_dir: Path,
    link_name: str,
) -> None:
    import numpy as np

    geom_type = int(model.geom_type[geom_id])
    geom_pos = np.asarray(model.geom_pos[geom_id], dtype=float) - np.asarray(link_shift, dtype=float)
    geom_quat = np.asarray(model.geom_quat[geom_id], dtype=float)
    size = np.asarray(model.geom_size[geom_id], dtype=float)
    raw_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
    name = _safe_name(raw_name or f"{link_name}_geom_{geom_id}", "geom")
    rgba = _geom_rgba(model, geom_id)
    collision = int(model.geom_contype[geom_id]) != 0 or int(model.geom_conaffinity[geom_id]) != 0
    visual = int(model.geom_group[geom_id]) != 3 and rgba[3] > 0.0
    if not collision and not visual:
        return

    shapes: list[tuple[str, dict[str, str], Any, Any]] = []
    if geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
        mesh_id = int(model.geom_dataid[geom_id])
        if mesh_id not in mesh_paths:
            raise ValueError(f"MJCF geom {name!r} references missing compiled mesh {mesh_id}.")
        # MuJoCo's compiler has already folded the mesh reference transform
        # into compiled geom_pos/geom_quat and centered mesh_vert. Applying
        # mesh_pos/mesh_quat again would double the source-origin transform.
        shape_pos = geom_pos
        shape_quat = geom_quat
        shapes.append(
            (
                "mesh",
                {
                    "filename": mesh_paths[mesh_id],
                    "scale": "1 1 1",
                },
                shape_pos,
                shape_quat,
            )
        )
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
        shapes.append(("box", {"size": _fmt(2.0 * size[:3])}, geom_pos, geom_quat))
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        shapes.append(("sphere", {"radius": f"{size[0]:.10g}"}, geom_pos, geom_quat))
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        shapes.append(
            ("cylinder", {"radius": f"{size[0]:.10g}", "length": f"{2.0 * size[1]:.10g}"}, geom_pos, geom_quat)
        )
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        radius, half_len = float(size[0]), float(size[1])
        for sign in (-1.0, 1.0):
            cap_pos = geom_pos + _rotation_matrix(geom_quat) @ np.asarray((0.0, 0.0, sign * half_len))
            shapes.append(("sphere", {"radius": f"{radius:.10g}"}, cap_pos, geom_quat))
        if half_len > 0:
            shapes.append(
                ("cylinder", {"radius": f"{radius:.10g}", "length": f"{2.0 * half_len:.10g}"}, geom_pos, geom_quat)
            )
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        try:
            import trimesh

            sphere = trimesh.creation.icosphere(subdivisions=3)
            ellipsoid_path = mesh_dir / f"ellipsoid_{geom_id}.obj"
            _write_mesh_obj(ellipsoid_path, sphere.vertices * size[:3], sphere.faces)
        except Exception as exc:
            raise RuntimeError(f"Could not tessellate ellipsoid geom {name!r}: {exc}") from exc
        shapes.append(
            (
                "mesh",
                {
                    "filename": f"meshes/{ellipsoid_path.name}",
                    "scale": "1 1 1",
                },
                geom_pos,
                geom_quat,
            )
        )
    else:
        enum_name = next(
            (
                candidate
                for candidate in dir(mujoco.mjtGeom)
                if candidate.startswith("mjGEOM_") and int(getattr(mujoco.mjtGeom, candidate)) == geom_type
            ),
            str(geom_type),
        )
        raise ValueError(
            f"Unsupported MuJoCo geom {enum_name} on {name!r}; supported: mesh, box, sphere, "
            "capsule, cylinder, ellipsoid."
        )

    for shape_index, (shape_tag, attributes, origin_pos, origin_quat) in enumerate(shapes):
        if visual:
            visual_element = ET.SubElement(link, "visual", {"name": f"{name}_visual_{shape_index}"})
            _set_origin(visual_element, origin_pos, origin_quat)
            geometry = ET.SubElement(visual_element, "geometry")
            ET.SubElement(geometry, shape_tag, attributes)
            material = ET.SubElement(visual_element, "material", {"name": f"{name}_material"})
            ET.SubElement(material, "color", {"rgba": _fmt(rgba)})
        if collision:
            collision_element = ET.SubElement(link, "collision", {"name": f"{name}_collision_{shape_index}"})
            _set_origin(collision_element, origin_pos, origin_quat)
            geometry = ET.SubElement(collision_element, "geometry")
            ET.SubElement(geometry, shape_tag, attributes)


def _export_mjcf(source_path: Path, urdf_path: Path) -> dict[str, Any]:
    import numpy as np

    mujoco, model = _compile_mjcf(source_path)
    body_names, link_shifts = _validate_bodies(model, mujoco)
    robot = ET.Element("robot", {"name": _safe_name(source_path.stem, "mujoco_robot")})
    base_name = "unisis_mjcf_base"
    base_link = ET.SubElement(robot, "link", {"name": base_name})
    base_inertial = ET.SubElement(base_link, "inertial")
    ET.SubElement(base_inertial, "mass", {"value": "0"})
    ET.SubElement(base_inertial, "inertia", {
        "ixx": "0", "iyy": "0", "izz": "0", "ixy": "0", "ixz": "0", "iyz": "0",
    })
    links = {
        body_id: ET.SubElement(robot, "link", {"name": name})
        for body_id, name in body_names.items()
    }

    # A synthetic base keeps the supplied YAML pose as the worldbody origin and
    # retains any body pose attached directly below MuJoCo worldbody.
    for body_id in range(1, model.nbody):
        parent_id = int(model.body_parentid[body_id])
        child_quat = np.asarray(model.body_quat[body_id], dtype=float)
        child_pos = np.asarray(model.body_pos[body_id], dtype=float)
        if parent_id == 0:
            parent_name = base_name
            if int(model.body_jntnum[body_id]) == 1:
                child_pos = child_pos + _rotation_matrix(child_quat) @ link_shifts[body_id]
            joint_origin_pos, joint_origin_quat = child_pos, child_quat
        else:
            parent_name = body_names[parent_id]
            joint_origin_pos = (
                -np.asarray(link_shifts[parent_id], dtype=float)
                + child_pos
                + _rotation_matrix(child_quat) @ np.asarray(link_shifts[body_id], dtype=float)
            )
            joint_origin_quat = child_quat

        if int(model.body_jntnum[body_id]) == 1:
            joint_id = int(model.body_jntadr[body_id])
            joint_kind = int(model.jnt_type[joint_id])
            if joint_kind == int(mujoco.mjtJoint.mjJNT_HINGE):
                joint_type = "revolute" if bool(model.jnt_limited[joint_id]) else "continuous"
            elif joint_kind == int(mujoco.mjtJoint.mjJNT_SLIDE):
                joint_type = "prismatic"
            else:
                joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
                raise ValueError(
                    f"Unsupported MuJoCo joint {joint_name!r}; only hinge and slide are supported."
                )
            raw_joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            name = _safe_name(raw_joint_name or f"joint_{joint_id}", f"joint_{joint_id}")
        else:
            joint_type = "fixed"
            name = _safe_name(f"fixed_{body_names[body_id]}", f"fixed_{body_id}")

        joint = ET.SubElement(robot, "joint", {"name": name, "type": joint_type})
        ET.SubElement(joint, "parent", {"link": parent_name})
        ET.SubElement(joint, "child", {"link": body_names[body_id]})
        _set_origin(joint, joint_origin_pos, joint_origin_quat)
        if joint_type != "fixed":
            ET.SubElement(joint, "axis", {"xyz": _fmt(model.jnt_axis[joint_id])})
            if joint_type == "continuous":
                ET.SubElement(joint, "limit", {"effort": "1000", "velocity": "1000"})
            else:
                if bool(model.jnt_limited[joint_id]):
                    lower, upper = (float(value) for value in model.jnt_range[joint_id])
                else:
                    lower, upper = -1e6, 1e6
                ET.SubElement(
                    joint,
                    "limit",
                    {
                        "lower": f"{lower:.10g}",
                        "upper": f"{upper:.10g}",
                        "effort": "1000",
                        "velocity": "1000",
                    },
                )

    mesh_dir = urdf_path.parent / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    mesh_paths: dict[int, str] = {}
    used_meshes = {
        int(model.geom_dataid[geom_id])
        for geom_id in range(model.ngeom)
        if int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_MESH)
    }
    for mesh_id in sorted(used_meshes):
        raw_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id)
        name = _safe_name(raw_name or f"mesh_{mesh_id}", f"mesh_{mesh_id}")
        relative_path = Path("meshes") / f"{mesh_id:03d}_{name}.obj"
        output_mesh = urdf_path.parent / relative_path
        va = int(model.mesh_vertadr[mesh_id])
        vn = int(model.mesh_vertnum[mesh_id])
        fa = int(model.mesh_faceadr[mesh_id])
        fn = int(model.mesh_facenum[mesh_id])
        vertices = np.asarray(model.mesh_vert[va : va + vn], dtype=float).copy()
        faces = np.asarray(model.mesh_face[fa : fa + fn], dtype=int).copy()
        _write_mesh_obj(output_mesh, vertices, faces)
        mesh_paths[mesh_id] = relative_path.as_posix()

    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        if body_id == 0:
            raise ValueError("MJCF worldbody geoms are not robot link geometry and are unsupported here.")
        _emit_geom(
            model,
            mujoco,
            geom_id,
            links[body_id],
            link_shifts[body_id],
            mesh_paths,
            mesh_dir,
            body_names[body_id],
        )

    for body_id, link in links.items():
        mass = float(model.body_mass[body_id])
        inertial = ET.SubElement(link, "inertial")
        position = np.asarray(model.body_ipos[body_id], dtype=float) - np.asarray(link_shifts[body_id], dtype=float)
        _set_origin(inertial, position, model.body_iquat[body_id])
        ET.SubElement(inertial, "mass", {"value": f"{mass:.10g}"})
        ix, iy, iz = (float(value) for value in model.body_inertia[body_id])
        ET.SubElement(
            inertial,
            "inertia",
            {
                "ixx": f"{ix:.10g}",
                "iyy": f"{iy:.10g}",
                "izz": f"{iz:.10g}",
                "ixy": "0",
                "ixz": "0",
                "iyz": "0",
            },
        )

    ET.indent(robot, space="  ")
    ET.ElementTree(robot).write(urdf_path, encoding="utf-8", xml_declaration=True)
    return {
        "kind": "mjcf_to_urdf",
        "source": str(source_path),
        "dependencies": [str(path) for path in _mjcf_dependencies(source_path)],
        "body_count": int(model.nbody - 1),
        "joint_count": int(model.njnt),
        "mesh_count": len(used_meshes),
        "converter_version": _CONVERTER_VERSION,
        "limitations": list(MJCF_CONVERSION_LIMITATIONS),
    }


def prepare_mjcf_asset(source_path: Path, cache_dir: Path) -> Path:
    """Compile MJCF using MuJoCo and return a cached articulated URDF.

    URDF and exported mesh files are self-contained in a content-addressed
    cache directory. This converts the kinematic/dynamic model, but does not
    create a Franka controller or encode MJCF actuators.
    """
    source_path = Path(source_path).expanduser().resolve()
    cache_dir = Path(cache_dir).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"UniSis MJCF asset does not exist: {source_path}")
    dependencies = _mjcf_dependencies(source_path)
    missing = [path for path in dependencies if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing MJCF dependency files: " + ", ".join(map(str, missing)))
    digest = _digest_files(dependencies, _CONVERTER_VERSION + ":mjcf")
    key = f"unisis-mjcf-{digest[:24]}"
    output_name = f"{_safe_name(source_path.stem)}.urdf"
    with _cache_lock(cache_dir, key):
        final_dir = cache_dir / key
        if _cache_complete(final_dir, output_name):
            return final_dir / output_name

        final_dir, temp_dir = _new_cache_dir(cache_dir, key)
        try:
            output = temp_dir / output_name
            details = _export_mjcf(source_path, output)
            _write_report(
                temp_dir,
                {
                    **details,
                    "source_sha256": _digest_files(dependencies, "mjcf-source"),
                },
            )
            if final_dir.exists():
                shutil.rmtree(final_dir)
            os.replace(temp_dir, final_dir)
        except Exception:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise
        return final_dir / output_name
