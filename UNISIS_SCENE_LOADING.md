# UniSis YAML scene loading

The standalone preview loads UniSis scenes into a dedicated PyBullet client.
It does not invoke GenSim's UR5 task reset, oracle, or action controller.

## Run

From the GenSim repository:

```bash
uv sync --locked
uv run python -m cliport.preview_unisis_scene /path/to/scene_directory --gui
```

A directory resolves to `scene.yaml`; a YAML file can also be passed directly.
The GUI displays the initial scene without advancing physics. Close the window
or press Ctrl+C to exit. `--hold-seconds 10` closes it after ten seconds.

Headless loading, repeated-reset checks, and preview outputs:

```bash
uv run python -m cliport.preview_unisis_scene /path/to/scene.yaml \
  --reset-count 2 --steps 0 \
  --image output/unisis_scene_preview/scene.png \
  --report-json output/unisis_scene_preview/report.json
```

`--steps N` explicitly advances physics after initial-state checks. It does not
run a robot controller or verify task success.

## Supported scene data

- Wrapped `scene: {entities: ...}` and bare `entities: ...` YAML.
- Asset paths relative to the YAML, entity IDs, positions, scale, WXYZ rotations,
  and Euler angles in degrees. YAML world coordinates are retained.
- Rigid planes, boxes, spheres, cylinders, meshes, URDF, and MJCF.
- Fixed/dynamic bodies, material density and friction, and robot joint values
  from `robot_adapter_kwargs.default_qpos` or named `initial_joints`.
- Entity-to-body mappings through `UnisisYAMLEnvironment.get_body_id()` and
  `entity_id_to_body_id`. The task mapping is retained in `env.document.task`.

GLB scene-node transforms are baked into OBJ without recentering or adding
entity scale twice. As in UniSis/Genesis, GLB/GLTF default to Y-up and their
mesh vertices are converted to Z-up before applying YAML scale and entity pose:
`(x, y, z) -> (x, -z, y)`. Other mesh formats default to Z-up.
`file_meshes_are_zup` can explicitly override the source up axis; it is included
in the conversion cache key. Entity/world coordinates and robot assets retain
their YAML conventions. MJCF is compiled with MuJoCo and exported as an articulated
URDF; this supports the supplied Franka model. Source assets are never rewritten.
Conversion products are cached under `.cache/unisis` by default; use
`--cache-dir` to override this. Each cache entry includes `conversion.json`.

## Limits

Dynamic mesh collision uses a convex hull; density-derived mass for open meshes
uses convex-hull volume. These approximations are recorded in the JSON report.
OBJ/MTL cannot preserve every GLB PBR channel. MJCF actuators, tendons, equality
constraints, contact rules, and solver settings are not transferred to URDF.
Ball/free joints and multiple joints on one MJCF body are rejected.

`file_preset` must be resolved to explicit assets before loading. Scene offsets
and backend-specific rigid options are reported when unsupported. Camera
position/look-at values can guide the preview, but this is not a reproduction
of UniSis sensor outputs. No additional robot, table, or floor is inserted.

Franka task execution, GenSim task binding, and success checking remain separate
future integration work.
