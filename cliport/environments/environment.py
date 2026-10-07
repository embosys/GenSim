"""Environment class."""

import os
import tempfile
import time
import cv2
import imageio

import gym
import numpy as np
from cliport.tasks import cameras
from cliport.utils import pybullet_utils
from cliport.utils import utils
import string
import pybullet as p
import tempfile
import random
import sys
from pathlib import Path

PLACE_STEP = 0.0003
PLACE_DELTA_THRESHOLD = 0.005

UR5_URDF_PATH = 'ur5/ur5.urdf'
UR5_WORKSPACE_URDF_PATH = 'ur5/workspace.urdf'
PLANE_URDF_PATH = 'plane/plane.urdf'


class Environment(gym.Env):
    """OpenAI Gym-style environment class."""

    def __init__(self,
                 assets_root,
                 task=None,
                 disp=False,
                 shared_memory=False,
                 hz=240,
                 record_cfg=None,
                 scene_path=None,
                 end_effector='suction',
                 cache_dir=None):
        """Creates OpenAI Gym-style environment with PyBullet.

        Args:
          assets_root: root directory of assets.
          task: the task to use. If None, the user must call set_task for the
            environment to work properly.
          disp: show environment with PyBullet's built-in display viewer.
          shared_memory: run with shared memory.
          hz: PyBullet physics simulation step speed. Set to 480 for deformables.

        Raises:
          RuntimeError: if pybullet cannot load fileIOPlugin.
        """
        self.pix_size = 0.003125
        self.obj_ids = {'fixed': [], 'rigid': [], 'deformable': []}
        self.objects = self.obj_ids # make a copy

        self.homej = np.array([-1, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi
        self.agent_cams = cameras.RealSenseD415.CONFIG
        self.oracle_cams = cameras.Oracle.CONFIG
        self.record_cfg = record_cfg
        self.save_video = False
        self.step_counter = 0

        self.assets_root = assets_root
        self.scene_path = Path(scene_path).expanduser().resolve() if scene_path else None
        self.end_effector = end_effector
        self.cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir else (
            Path(__file__).resolve().parents[2] / '.cache' / 'unisis'
        )
        self.document = None
        self.scene_task = None
        self.loaded_scene = None
        self.entity_id_to_body_id = {}
        self.body_id_to_entity_id = {}
        self.scene_warnings = []
        if self.scene_path is not None:
            from cliport.environments.unisis_scene_loader import parse_scene_yaml

            self.document = parse_scene_yaml(self.scene_path)
            self.scene_task = self.document.task or self.document.raw.get('metadata')

        color_tuple = [
            gym.spaces.Box(0, 255, config['image_size'] + (3,), dtype=np.uint8)
            for config in self.agent_cams
        ]
        depth_tuple = [
            gym.spaces.Box(0.0, 20.0, config['image_size'], dtype=np.float32)
            for config in self.agent_cams
        ]
        self.observation_space = gym.spaces.Dict({
            'color': gym.spaces.Tuple(color_tuple),
            'depth': gym.spaces.Tuple(depth_tuple),
        })
        self.position_bounds = gym.spaces.Box(
            low=np.array([0.25, -0.5, 0.], dtype=np.float32),
            high=np.array([0.75, 0.5, 0.28], dtype=np.float32),
            shape=(3,),
            dtype=np.float32)
        self.bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.3]])

        self.action_space = gym.spaces.Dict({
            'pose0':
                gym.spaces.Tuple(
                    (self.position_bounds,
                     gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32))),
            'pose1':
                gym.spaces.Tuple(
                    (self.position_bounds,
                     gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)))
        })

        # Start PyBullet.
        disp_option = p.DIRECT
        if disp:
            disp_option = p.GUI
            if shared_memory:
                disp_option = p.SHARED_MEMORY
        client = p.connect(disp_option)
        self.client_id = client
        file_io = p.loadPlugin('fileIOPlugin', physicsClientId=client)
        if file_io < 0:
            raise RuntimeError('pybullet: cannot load FileIO!')
        if file_io >= 0:
            p.executePluginCommand(
                file_io,
                textArgument=assets_root,
                intArgs=[p.AddFileIOAction],
                physicsClientId=client)

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setPhysicsEngineParameter(enableFileCaching=0)
        p.setAdditionalSearchPath(assets_root)
        p.setAdditionalSearchPath(tempfile.gettempdir())
        p.setTimeStep(1. / hz)

        # If using --disp, move default camera closer to the scene.
        if disp:
            target = p.getDebugVisualizerCamera()[11]
            p.resetDebugVisualizerCamera(
                cameraDistance=1.1,
                cameraYaw=90,
                cameraPitch=-25,
                cameraTargetPosition=target)

        if task:
            self.set_task(task)

    def _scene_body_aabb(self, body_id):
        """Return a finite world-space AABB spanning every link of a body."""
        bounds = []
        for link_index in range(-1, p.getNumJoints(body_id, physicsClientId=self.client_id)):
            try:
                lower, upper = p.getAABB(
                    body_id, link_index, physicsClientId=self.client_id
                )
            except p.error:
                continue
            lower = np.asarray(lower, dtype=np.float64)
            upper = np.asarray(upper, dtype=np.float64)
            extent = upper - lower
            if (np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))
                    and np.all(extent >= 0) and np.all(extent < 100)):
                bounds.append((lower, upper))
        if not bounds:
            return None
        return np.min([item[0] for item in bounds], axis=0), np.max(
            [item[1] for item in bounds], axis=0
        )

    @staticmethod
    def _quaternion_from_rotation_matrix(rotation):
        """Convert a 3x3 rotation matrix to a PyBullet XYZW quaternion."""
        matrix = np.asarray(rotation, dtype=np.float64)
        trace = float(np.trace(matrix))
        if trace > 0:
            scale = np.sqrt(trace + 1.0) * 2.0
            quaternion = np.array([
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
                0.25 * scale,
            ])
        elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quaternion = np.array([
                0.25 * scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
            ])
        elif matrix[1, 1] > matrix[2, 2]:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quaternion = np.array([
                (matrix[0, 1] + matrix[1, 0]) / scale,
                0.25 * scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
            ])
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quaternion = np.array([
                (matrix[0, 2] + matrix[2, 0]) / scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
                0.25 * scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ])
        quaternion /= np.linalg.norm(quaternion)
        return tuple(float(value) for value in quaternion)

    @classmethod
    def _look_at_config(cls, position, target, *, image_size=(480, 640), focal=450.0,
                        zrange=(0.01, 10.0)):
        """Build a camera config using the render_camera local +Z view convention."""
        position = np.asarray(position, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        forward = target - position
        forward /= np.linalg.norm(forward)
        up_hint = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(forward, up_hint))) > 0.98:
            up_hint = np.array([0.0, 1.0, 0.0])
        up = up_hint - np.dot(up_hint, forward) * forward
        up /= np.linalg.norm(up)
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        rotation = np.column_stack((right, -up, forward))
        height, width = image_size
        intrinsics = (
            focal, 0.0, width / 2.0,
            0.0, focal, height / 2.0,
            0.0, 0.0, 1.0,
        )
        return {
            'image_size': image_size,
            'intrinsics': intrinsics,
            'position': tuple(float(value) for value in position),
            'rotation': cls._quaternion_from_rotation_matrix(rotation),
            'zrange': zrange,
            'noise': False,
        }

    def _configure_scene_workspace(self):
        """Frame the declared manipulation target, support surface, and goal."""
        task = self.scene_task if isinstance(self.scene_task, dict) else {}
        target_id = task.get('target_id')
        raw_surface = next((
            task[key] for key in (
                'surface_id', 'support_surface_id', 'goal_surface_id', 'surface'
            ) if task.get(key) is not None
        ), None)
        surface_refs = []
        surface_points = []
        if isinstance(raw_surface, dict):
            surface_ref = next((raw_surface[key] for key in (
                'entity_id', 'id', 'name'
            ) if raw_surface.get(key) is not None), None)
            if surface_ref is not None:
                surface_refs.append(str(surface_ref))
            position = raw_surface.get('position', raw_surface.get('center'))
            if isinstance(position, (list, tuple)) and len(position) == 3:
                surface_points.append(np.asarray(position, dtype=np.float64))
        elif isinstance(raw_surface, (list, tuple)):
            if len(raw_surface) == 3 and all(
                isinstance(value, (int, float)) for value in raw_surface
            ):
                surface_points.append(np.asarray(raw_surface, dtype=np.float64))
            else:
                surface_refs.extend(str(value) for value in raw_surface)
        elif raw_surface is not None:
            surface_refs.append(str(raw_surface))

        def _body_bounds_for_reference(reference):
            canonical_id = self.document.name_to_entity_id.get(
                str(reference), str(reference)
            )
            body_id = self.entity_id_to_body_id.get(canonical_id)
            if body_id is None:
                return None
            return self._scene_body_aabb(body_id)

        target_bounds = _body_bounds_for_reference(target_id) if target_id is not None else None
        surface_bounds = [
            bounds for reference in surface_refs
            if (bounds := _body_bounds_for_reference(reference)) is not None
        ]
        goal_point = task.get('goal_point')
        if isinstance(goal_point, (list, tuple)) and len(goal_point) == 3:
            goal_point = np.asarray(goal_point, dtype=np.float64)
            if not np.all(np.isfinite(goal_point)):
                goal_point = None
        else:
            goal_point = None

        xy_anchors = []
        z_lows = []
        z_highs = []
        if target_bounds is not None:
            target_lower, target_upper = target_bounds
            xy_anchors.extend((target_lower[:2], target_upper[:2]))
            z_lows.append(float(target_lower[2]))
            z_highs.append(float(target_upper[2]))
        if goal_point is not None:
            xy_anchors.append(goal_point[:2])
            z_lows.append(float(goal_point[2]))
            z_highs.append(float(goal_point[2]))
        xy_anchors.extend(point[:2] for point in surface_points)
        z_lows.extend(float(point[2]) for point in surface_points)
        z_highs.extend(float(point[2]) for point in surface_points)

        # Include only the local patch of a declared support surface around the
        # target/goal. This avoids framing unrelated room geometry such as rugs
        # and sofas while keeping the actual placement region visible.
        if xy_anchors:
            anchor_lower = np.min(xy_anchors, axis=0)
            anchor_upper = np.max(xy_anchors, axis=0)
            for surface_lower, surface_upper in surface_bounds:
                patch_lower = np.maximum(surface_lower[:2], anchor_lower - 0.2)
                patch_upper = np.minimum(surface_upper[:2], anchor_upper + 0.2)
                if np.all(patch_lower <= patch_upper):
                    xy_anchors.extend((patch_lower, patch_upper))
                # A support surface's top is the relevant vertical anchor;
                # its legs/base would otherwise pull the map down to the floor.
                z_lows.append(float(surface_upper[2]) - 0.1)
                z_highs.append(float(surface_upper[2]))
            anchor_lower = np.min(xy_anchors, axis=0)
            anchor_upper = np.max(xy_anchors, axis=0)
        else:
            # Scenes without the fixed-target schema get a compact workspace
            # around the robot instead of aggregating nearby room furniture.
            robot_config = self.loaded_scene.entity_configs[self.scene_robot_entity_id]
            robot_position = np.asarray(robot_config['position'], dtype=np.float64)
            anchor_lower = robot_position[:2] - 0.25
            anchor_upper = robot_position[:2] + 0.25
            z_lows.append(max(0.0, float(robot_position[2])))
            z_highs.append(float(robot_position[2]) + 0.25)
            if surface_bounds:
                nearest_surface = min(
                    surface_bounds,
                    key=lambda bounds: float(np.linalg.norm(
                        np.clip(robot_position[:2], bounds[0][:2], bounds[1][:2])
                        - robot_position[:2]
                    )),
                )
                surface_lower, surface_upper = nearest_surface
                anchor_lower = np.maximum(anchor_lower, surface_lower[:2])
                anchor_upper = np.minimum(anchor_upper, surface_upper[:2])
                z_lows.append(float(surface_upper[2]) - 0.1)
                z_highs.append(float(surface_upper[2]))

        margin = 0.2
        xy_lower = anchor_lower - margin
        xy_upper = anchor_upper + margin
        for axis in range(2):
            if xy_upper[axis] - xy_lower[axis] < 0.5:
                center = (xy_lower[axis] + xy_upper[axis]) / 2.0
                xy_lower[axis] = center - 0.25
                xy_upper[axis] = center + 0.25

        if not z_lows:
            z_lows.append(0.0)
        if not z_highs:
            z_highs.append(0.25)
        z_lower = max(0.0, min(z_lows) - 0.05)
        z_upper = max(z_highs) + 0.55
        if z_upper <= z_lower + 0.3:
            z_upper = z_lower + 0.6
        self.bounds = np.array([
            [xy_lower[0], xy_upper[0]],
            [xy_lower[1], xy_upper[1]],
            [z_lower, z_upper],
        ], dtype=np.float64)
        self.position_bounds = gym.spaces.Box(
            low=self.bounds[:, 0].astype(np.float32),
            high=self.bounds[:, 1].astype(np.float32),
            shape=(3,), dtype=np.float32,
        )
        self.action_space = gym.spaces.Dict({
            'pose0': gym.spaces.Tuple((
                self.position_bounds,
                gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32),
            )),
            'pose1': gym.spaces.Tuple((
                self.position_bounds,
                gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32),
            )),
        })

        center_xy = (xy_lower + xy_upper) / 2.0
        center_z = (z_lower + z_upper) / 2.0
        span_x = float(xy_upper[0] - xy_lower[0])
        span_y = float(xy_upper[1] - xy_lower[1])
        camera_height = max(z_upper + 1.5, center_z + 2.0)
        camera_distance = camera_height - center_z
        image_size = cameras.Oracle.CONFIG[0]['image_size']
        image_height, image_width = image_size
        focal = min(
            image_height * camera_distance / max(span_x, 1e-3),
            image_width * camera_distance / max(span_y, 1e-3),
        ) * 0.95
        oracle_camera = dict(cameras.Oracle.CONFIG[0])
        oracle_camera.update({
            'position': (float(center_xy[0]), float(center_xy[1]), camera_height),
            'intrinsics': (
                focal, 0.0, image_width / 2.0,
                0.0, focal, image_height / 2.0,
                0.0, 0.0, 1.0,
            ),
            'zrange': (0.01, camera_height - z_lower + 0.1),
            'noise': False,
        })
        self.oracle_cams = [oracle_camera]

        radius = max(0.8, 0.85 * max(span_x, span_y))
        eye_height = max(0.7, 0.75 * radius)
        target = np.array([center_xy[0], center_xy[1], center_z])
        camera_targets = [
            center_xy + np.array([radius, 0.0]),
            center_xy + np.array([-0.5 * radius, radius * 0.87]),
            center_xy + np.array([-0.5 * radius, -radius * 0.87]),
        ]
        self.agent_cams = []
        for xy in camera_targets:
            eye = np.array([xy[0], xy[1], center_z + eye_height])
            self.agent_cams.append(self._look_at_config(
                eye, target, zrange=(0.01, max(4.0, camera_height - z_lower + 1.0))
            ))
        color_tuple = [
            gym.spaces.Box(0, 255, config['image_size'] + (3,), dtype=np.uint8)
            for config in self.agent_cams
        ]
        depth_tuple = [
            gym.spaces.Box(0.0, 20.0, config['image_size'], dtype=np.float32)
            for config in self.agent_cams
        ]
        self.observation_space = gym.spaces.Dict({
            'color': gym.spaces.Tuple(color_tuple),
            'depth': gym.spaces.Tuple(depth_tuple),
        })

    def _reset_scene(self):
        """Reload a UniSis scene in this environment's existing PyBullet client."""
        from cliport.environments.unisis_scene_loader import load_scene_entities
        from cliport.environments.robot_adapters import create_robot_adapter

        if self.end_effector != 'suction':
            raise ValueError(
                f"UniSis GenSim integration currently supports end_effector='suction'; "
                f"got {self.end_effector!r}."
            )
        self.obj_ids = {'fixed': [], 'rigid': [], 'deformable': []}
        self.objects = self.obj_ids
        p.configureDebugVisualizer(
            p.COV_ENABLE_RENDERING, 0, physicsClientId=self.client_id
        )
        try:
            p.resetSimulation(
                flags=p.RESET_USE_DEFORMABLE_WORLD, physicsClientId=self.client_id
            )
            sim_options = self.document.sim_options
            gravity = sim_options.get('gravity', (0.0, 0.0, -9.81))
            if isinstance(gravity, dict):
                gravity = [gravity.get(axis, default) for axis, default in zip(
                    ('x', 'y', 'z'), (0.0, 0.0, -9.81)
                )]
            p.setGravity(*[float(value) for value in gravity], physicsClientId=self.client_id)
            time_step = next((sim_options[key] for key in
                              ('time_step', 'timestep', 'dt', 'physics_timestep')
                              if sim_options.get(key) is not None), 1.0 / 240.0)
            p.setTimeStep(float(time_step), physicsClientId=self.client_id)
            p.setPhysicsEngineParameter(
                enableFileCaching=0, physicsClientId=self.client_id
            )
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.loaded_scene = load_scene_entities(
                self.document,
                physics_client_id=self.client_id,
                cache_dir=self.cache_dir,
            )
            self.obj_ids = self.loaded_scene.obj_ids
            self.objects = self.obj_ids
            self.entity_id_to_body_id = dict(self.loaded_scene.entity_id_to_body_id)
            self.body_id_to_entity_id = dict(self.loaded_scene.body_id_to_entity_id)
            self.scene_warnings = list(self.loaded_scene.warnings)

            if len(self.loaded_scene.robot_ids) != 1:
                raise ValueError(
                    'UniSis GenSim integration requires exactly one Franka Panda robot; '
                    f"scene has {len(self.loaded_scene.robot_ids)} robot entities."
                )
            self.scene_robot_entity_id = self.loaded_scene.robot_ids[0]
            robot_config = self.loaded_scene.entity_configs[self.scene_robot_entity_id]
            robot_type = str(robot_config.get('robot_adapter', '')).strip().lower()
            if not robot_type:
                identity = ' '.join((
                    self.scene_robot_entity_id,
                    str(robot_config.get('file', '')),
                )).lower()
                if 'franka' in identity or 'panda' in identity:
                    robot_type = 'franka'
            if robot_type not in {
                'franka', 'franka_panda', 'franka_emika_panda', 'panda'
            }:
                raise ValueError(
                    'Only Franka Panda scenes are supported by the GenSim environment; '
                    f"robot entity {self.scene_robot_entity_id!r} declares "
                    f"robot_adapter={robot_type or '<missing>'!r}."
                )
            adapter_options = robot_config.get('robot_adapter_kwargs') or {}
            default_qpos = adapter_options.get('default_qpos') if isinstance(
                adapter_options, dict
            ) else None
            if default_qpos is not None:
                default_qpos = [float(value) for value in default_qpos]
                if len(default_qpos) not in (7, 9):
                    raise ValueError(
                        f"Robot {self.scene_robot_entity_id!r} default_qpos must contain "
                        '7 arm values or 9 arm-and-finger values.'
                    )
            self.robot_adapter = create_robot_adapter(
                self.entity_id_to_body_id[self.scene_robot_entity_id],
                self.obj_ids,
                self.assets_root,
                robot_type=robot_type,
                end_effector=self.end_effector,
                default_qpos=default_qpos,
            )
            self.ur5 = self.robot_adapter.robot_id
            self.joints = list(self.robot_adapter.joints)
            self.homej = np.asarray(self.robot_adapter.homej, dtype=np.float32)
            self.ee = self.robot_adapter.ee
            self.ee_tip = self.robot_adapter.ee_tip
            self._configure_scene_workspace()
            p.configureDebugVisualizer(
                p.COV_ENABLE_GUI, 0, physicsClientId=self.client_id
            )
            self.ee.release()
            self.task.reset(self)
        finally:
            p.configureDebugVisualizer(
                p.COV_ENABLE_RENDERING, 1, physicsClientId=self.client_id
            )

        obs, _, _, _ = self.step()
        return obs

    def __del__(self):
        if hasattr(self, 'video_writer'):
            self.video_writer.close()

    @property
    def is_static(self):
        """Return true if objects are no longer moving."""
        v = [np.linalg.norm(p.getBaseVelocity(i)[0])
             for i in self.obj_ids['rigid']]
        return all(np.array(v) < 5e-3)

    def fill_dummy_template(self, template):
        """check if there are empty templates that haven't been fulfilled yet. if so. fill in dummy numbers """
        full_template_path = os.path.join(self.assets_root, template)
        with open(full_template_path, 'r') as file:
            fdata = file.read()

        fill = False
        for field in ['DIMH', 'DIMR', 'DIMX', 'DIMY', 'DIMZ', 'DIM']:
            # usually 3 should be enough
            if field in fdata:
                default_replace_vals = np.random.uniform(0.03, 0.05, size=(3,)).tolist() # [0.03,0.03,0.03]
                for i in range(len(default_replace_vals)):
                    fdata = fdata.replace(f'{field}{i}', str(default_replace_vals[i]))
                fill = True

        for field in ['HALF']:
            # usually 3 should be enough
            if field in fdata:
                default_replace_vals = np.random.uniform(0.01, 0.03, size=(3,)).tolist() # [0.015,0.015,0.015]
                for i in range(len(default_replace_vals)):
                    fdata = fdata.replace(f'{field}{i}', str(default_replace_vals[i]))
                fill = True

        if fill:
            alphabet = string.ascii_lowercase + string.digits
            rname = ''.join(random.choices(alphabet, k=16))
            tmpdir = tempfile.gettempdir()
            template_filename = os.path.split(template)[-1]
            fname = os.path.join(tmpdir, f'{template_filename}.{rname}')
            with open(fname, 'w') as file:
                file.write(fdata)
            # print("fill-in dummys")

            return fname
        else:
            return template

    def add_object(self, urdf, pose, category='rigid', color=None, **kwargs):
        """List of (fixed, rigid, or deformable) objects in env."""
        fixed_base = 1 if category == 'fixed' else 0

        if 'template' in urdf:
            if not os.path.exists(os.path.join(self.assets_root, urdf)):
                urdf = urdf.replace("-template", "")

            urdf = self.fill_dummy_template(urdf)

        if not os.path.exists(os.path.join(self.assets_root, urdf)):
          print(f"missing urdf error: {os.path.join(self.assets_root, urdf)}. use dummy block.")
          urdf = 'stacking/block.urdf'

        if len(pose) == 3 and (not hasattr(pose[0], '__len__')):
            # add default orientation if missing
            pose = (pose, (0,0,0,1))

        obj_id = pybullet_utils.load_urdf(
            p,
            os.path.join(self.assets_root, urdf),
            pose[0],
            pose[1],
            useFixedBase=fixed_base)

        if not obj_id is None:
            self.obj_ids[category].append(obj_id)

        if color is not None:
            if type(color) is str:
                color = utils.COLORS[color]
            color = color + [1.]
            p.changeVisualShape(obj_id, -1, rgbaColor=color)

        if  hasattr(self, 'record_cfg') and 'blender_render' in self.record_cfg and self.record_cfg['blender_render']:
            # print("urdf:", os.path.join(self.assets_root, urdf))
            # if color is None:
            #     color = (0.5,0.5,0.5,1) # by default
            print("color:", color)

            self.blender_recorder.register_object(obj_id, os.path.join(self.assets_root, urdf), color=color)

        return obj_id

    def set_color(self, obj_id, color):
        p.changeVisualShape(obj_id, -1, rgbaColor=color + [1])

    def set_object_color(self, *args, **kwargs):
        return self.set_color(*args, **kwargs)

    # ---------------------------------------------------------------------------
    # Standard Gym Functions
    # ---------------------------------------------------------------------------

    def seed(self, seed=None):
        self._random = np.random.RandomState(seed)
        return seed

    def reset(self):
        """Performs common reset functionality for all supported tasks."""
        if not self.task:
            raise ValueError('environment task must be set. Call set_task or pass '
                             'the task arg in the environment constructor.')
        if self.scene_path is not None:
            return self._reset_scene()
        self.obj_ids = {'fixed': [], 'rigid': [], 'deformable': []}
        p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
        p.setGravity(0, 0, -9.8)

        # Temporarily disable rendering to load scene faster.
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

        plane = pybullet_utils.load_urdf(p, os.path.join(self.assets_root, PLANE_URDF_PATH),
                                 [0, 0, -0.001])
        workspace = pybullet_utils.load_urdf(
            p, os.path.join(self.assets_root, UR5_WORKSPACE_URDF_PATH), [0.5, 0, 0])

        # Load UR5 robot arm equipped with suction end effector.
        # TODO(andyzeng): add back parallel-jaw grippers.
        self.ur5 = pybullet_utils.load_urdf(
            p, os.path.join(self.assets_root, UR5_URDF_PATH))
        self.ee = self.task.ee(self.assets_root, self.ur5, 9, self.obj_ids)
        self.ee_tip = 10  # Link ID of suction cup.

        if  hasattr(self, 'record_cfg') and 'blender_render' in self.record_cfg and self.record_cfg['blender_render']:
            from misc.pyBulletSimRecorder import PyBulletRecorder
            self.blender_recorder = PyBulletRecorder()

            self.blender_recorder.register_object(plane, os.path.join(self.assets_root, PLANE_URDF_PATH))
            self.blender_recorder.register_object(workspace, os.path.join(self.assets_root, UR5_WORKSPACE_URDF_PATH))
            self.blender_recorder.register_object(self.ur5, os.path.join(self.assets_root, UR5_URDF_PATH))

            self.blender_recorder.register_object(self.ee.base,  self.ee.base_urdf_path)
            if hasattr(self.ee, 'body'):
                self.blender_recorder.register_object(self.ee.body,  self.ee.urdf_path)


        # Get revolute joint indices of robot (skip fixed joints).
        n_joints = p.getNumJoints(self.ur5)
        joints = [p.getJointInfo(self.ur5, i) for i in range(n_joints)]
        self.joints = [j[0] for j in joints if j[2] == p.JOINT_REVOLUTE]

        # Move robot to home joint configuration.
        for i in range(len(self.joints)):
            p.resetJointState(self.ur5, self.joints[i], self.homej[i])

        # Reset end effector.
        self.ee.release()

        # Reset task.
        self.task.reset(self)

        # Re-enable rendering.
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        obs, _, _, _ = self.step()
        return obs

    def step(self, action=None):
        """Execute action with specified primitive.

        Args:
          action: action to execute.

        Returns:
          (obs, reward, done, info) tuple containing MDP step data.
        """
        if action is not None:
            timeout = self.task.primitive(self.movej, self.movep, self.ee, action['pose0'], action['pose1'])

            # Exit early if action times out. We still return an observation
            # so that we don't break the Gym API contract.
            if timeout:
                obs = {'color': (), 'depth': ()}
                for config in self.agent_cams:
                    color, depth, _ = self.render_camera(config)
                    obs['color'] += (color,)
                    obs['depth'] += (depth,)
                return obs, 0.0, True, self.info

        start_time = time.time()
        # Step simulator asynchronously until objects settle.
        while not self.is_static:
            self.step_simulation()
            if time.time() - start_time > 5: # timeout
                break

        # Get task rewards.
        reward, info = self.task.reward() if action is not None else (0, {})
        done = self.task.done()

        # Add ground truth robot state into info.
        info.update(self.info)

        obs = self._get_obs()

        return obs, reward, done, info

    def step_simulation(self):
        p.stepSimulation()
        self.step_counter += 1

        if self.save_video and self.step_counter % 5 == 0:
            self.add_video_frame()

    def render(self, mode='rgb_array'):
        # Render only the color image from the first camera.
        # Only support rgb_array for now.
        if mode != 'rgb_array':
            raise NotImplementedError('Only rgb_array implemented')
        color, _, _ = self.render_camera(self.agent_cams[0])
        return color

    def render_camera(self, config, image_size=None, shadow=1):
        """Render RGB-D image with specified camera configuration."""
        if not image_size:
            image_size = config['image_size']

        # OpenGL camera settings.
        lookdir = np.float32([0, 0, 1]).reshape(3, 1)
        updir = np.float32([0, -1, 0]).reshape(3, 1)
        rotation = p.getMatrixFromQuaternion(config['rotation'])
        rotm = np.float32(rotation).reshape(3, 3)
        lookdir = (rotm @ lookdir).reshape(-1)
        updir = (rotm @ updir).reshape(-1)
        lookat = config['position'] + lookdir
        focal_len = config['intrinsics'][0]
        znear, zfar = config['zrange']
        viewm = p.computeViewMatrix(config['position'], lookat, updir)
        fovh = (image_size[0] / 2) / focal_len
        fovh = 180 * np.arctan(fovh) * 2 / np.pi

        # Notes: 1) FOV is vertical FOV 2) aspect must be float
        aspect_ratio = image_size[1] / image_size[0]
        projm = p.computeProjectionMatrixFOV(fovh, aspect_ratio, znear, zfar)

        # Render with OpenGL camera settings.
        _, _, color, depth, segm = p.getCameraImage(
            width=image_size[1],
            height=image_size[0],
            viewMatrix=viewm,
            projectionMatrix=projm,
            shadow=shadow,
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            renderer=p.ER_BULLET_HARDWARE_OPENGL)

        # Get color image.
        color_image_size = (image_size[0], image_size[1], 4)
        color = np.array(color, dtype=np.uint8).reshape(color_image_size)
        color = color[:, :, :3]  # remove alpha channel
        if config['noise']:
            color = np.int32(color)
            color += np.int32(self._random.normal(0, 3, image_size))
            color = np.uint8(np.clip(color, 0, 255))

        # Get depth image.
        depth_image_size = (image_size[0], image_size[1])
        zbuffer = np.array(depth).reshape(depth_image_size)
        depth = (zfar + znear - (2. * zbuffer - 1.) * (zfar - znear))
        depth = (2. * znear * zfar) / depth
        if config['noise']:
            depth += self._random.normal(0, 0.003, depth_image_size)

        # Get segmentation image.
        if self.scene_path is not None:
            # PyBullet packs the link index into the high byte. Keep body IDs in
            # int32 and strip those link bits so task masks compare to body IDs.
            packed_segm = np.asarray(segm, dtype=np.int32).reshape(depth_image_size)
            segm = np.where(
                packed_segm < 0, -1, packed_segm & ((1 << 24) - 1)
            ).astype(np.int32, copy=False)
        else:
            segm = np.uint8(segm).reshape(depth_image_size)

        return color, depth, segm

    @property
    def info(self):
        """Environment info variable with object poses, dimensions, and colors."""

        # Some tasks create and remove zones, so ignore those IDs.
        # removed_ids = []
        # if (isinstance(self.task, tasks.names['cloth-flat-notarget']) or
        #         isinstance(self.task, tasks.names['bag-alone-open'])):
        #   removed_ids.append(self.task.zone_id)

        info = {}  # object id : (position, rotation, dimensions)
        for obj_ids in self.obj_ids.values():
            for obj_id in obj_ids:
                pos, rot = p.getBasePositionAndOrientation(obj_id)
                if self.scene_path is not None:
                    body_bounds = self._scene_body_aabb(obj_id)
                    if body_bounds is None:
                        continue
                    dim = tuple(body_bounds[1] - body_bounds[0])
                else:
                    dim = p.getVisualShapeData(obj_id)[0][3]
                info[obj_id] = (pos, rot, dim)

        info['lang_goal'] = self.get_lang_goal()
        return info

    def set_task(self, task):
        task.set_assets_root(self.assets_root)
        self.task = task

    def get_task_name(self):
        return type(self.task).__name__

    def get_lang_goal(self):
        if self.task:
            return self.task.get_lang_goal()
        else:
            raise Exception("No task for was set")

    # ---------------------------------------------------------------------------
    # Robot Movement Functions
    # ---------------------------------------------------------------------------

    def movej(self, targj, speed=0.01, timeout=5):
        """Move the robot; scene mode budgets simulated time, not wall time."""
        if self.scene_path is not None:
            return self._movej_scene(targj, speed, timeout)
        if self.save_video:
            timeout = timeout * 30 # 50?

        t0 = time.time()
        while (time.time() - t0) < timeout:
            currj = [p.getJointState(self.ur5, i)[0] for i in self.joints]
            currj = np.array(currj)
            diffj = targj - currj
            if all(np.abs(diffj) < 1e-2):
                return False

            # Move with constant velocity
            norm = np.linalg.norm(diffj)
            v = diffj / norm if norm > 0 else 0
            stepj = currj + v * speed
            gains = np.ones(len(self.joints))
            p.setJointMotorControlArray(
                bodyIndex=self.ur5,
                jointIndices=self.joints,
                controlMode=p.POSITION_CONTROL,
                targetPositions=stepj,
                positionGains=gains)
            self.step_counter += 1
            self.step_simulation()

        print(f'Warning: movej exceeded {timeout} second timeout. Skipping.')
        return True

    def _movej_scene(self, targj, speed, timeout):
        """Bound motion by physics steps so large meshes cannot expire it early."""
        target = np.asarray(targj, dtype=np.float64)
        if target.shape != (len(self.joints),) or not np.all(np.isfinite(target)):
            raise ValueError("Scene joint target must contain seven finite arm values.")
        if not np.isfinite(speed) or speed <= 0 or not np.isfinite(timeout) or timeout <= 0:
            raise ValueError("Scene motion speed and timeout must be positive and finite.")
        time_step = p.getPhysicsEngineParameters(
            physicsClientId=self.client_id
        )['fixedTimeStep']
        max_steps = max(1, int(np.ceil(timeout / time_step)))
        for _ in range(max_steps):
            current = np.asarray([
                p.getJointState(self.robot_adapter.robot_id, joint,
                                physicsClientId=self.client_id)[0]
                for joint in self.joints
            ])
            difference = target - current
            if np.all(np.abs(difference) < 1e-2):
                return False
            distance = np.linalg.norm(difference)
            command = current + difference / distance * min(speed, distance)
            p.setJointMotorControlArray(
                self.robot_adapter.robot_id,
                self.joints,
                p.POSITION_CONTROL,
                targetPositions=command,
                positionGains=np.ones(len(self.joints)),
                physicsClientId=self.client_id,
            )
            self.step_simulation()
        print(f'Warning: scene movej exceeded {max_steps} physics steps. Skipping.')
        return True

    def start_rec(self, video_filename):
        assert self.record_cfg

        # make video directory
        if not os.path.exists(self.record_cfg['save_video_path']):
            os.makedirs(self.record_cfg['save_video_path'])

        # close and save existing writer
        if hasattr(self, 'video_writer'):
            self.video_writer.close()

        # initialize writer
        self.video_writer = imageio.get_writer(os.path.join(self.record_cfg['save_video_path'],
                                                            f"{video_filename}.mp4"),
                                               fps=self.record_cfg['fps'],
                                               format='FFMPEG',
                                               codec='h264',)
        p.setRealTimeSimulation(False)
        self.save_video = True

    def end_rec(self):
        if hasattr(self, 'video_writer'):
            self.video_writer.close()

        p.setRealTimeSimulation(True)
        self.save_video = False

    def add_video_frame(self):
        # Render frame.
        config = self.agent_cams[0]
        image_size = (self.record_cfg['video_height'], self.record_cfg['video_width'])
        color, depth, _ = self.render_camera(config, image_size, shadow=0)
        color = np.array(color)

        if hasattr(self.record_cfg, 'blender_render') and  self.record_cfg['blender_render']:
            # print("add blender key frame")
            self.blender_recorder.add_keyframe()

        # Add language instruction to video.
        if self.record_cfg['add_text']:
            lang_goal = self.get_lang_goal()
            reward = f"Success: {self.task.get_reward():.3f}"

            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.65
            font_thickness =  1

            # Write language goal.
            line_length = 60
            for i in range(len(lang_goal) // line_length + 1):
                lang_textsize = cv2.getTextSize(lang_goal[i*line_length:(i+1)*line_length], font, font_scale, font_thickness)[0]
                lang_textX = (image_size[1] - lang_textsize[0]) // 2
                color = cv2.putText(color, lang_goal[i*line_length:(i+1)*line_length], org=(lang_textX, 570+i*30), # 600
                                fontScale=font_scale,
                                fontFace=font,
                                color=(0, 0, 0),
                                thickness=font_thickness, lineType=cv2.LINE_AA)

            ## Write Reward.
            # reward_textsize = cv2.getTextSize(reward, font, font_scale, font_thickness)[0]
            # reward_textX = (image_size[1] - reward_textsize[0]) // 2
            #
            # color = cv2.putText(color, reward, org=(reward_textX, 634),
            #                     fontScale=font_scale,
            #                     fontFace=font,
            #                     color=(0, 0, 0),
            #                     thickness=font_thickness, lineType=cv2.LINE_AA)

            color = np.array(color)

        if 'add_task_text' in self.record_cfg and self.record_cfg['add_task_text']:
            lang_goal = self.get_task_name()
            reward = f"Success: {self.task.get_reward():.3f}"

            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 1
            font_thickness =  2

            # Write language goal.
            lang_textsize = cv2.getTextSize(lang_goal, font, font_scale, font_thickness)[0]
            lang_textX = (image_size[1] - lang_textsize[0]) // 2

            color = cv2.putText(color, lang_goal, org=(lang_textX, 600),
                                fontScale=font_scale,
                                fontFace=font,
                                color=(255, 0, 0),
                                thickness=font_thickness, lineType=cv2.LINE_AA)

            color = np.array(color)

        self.video_writer.append_data(color)

    def movep(self, pose, speed=0.01):
        """Move UR5 to target end effector pose."""
        targj = self.solve_ik(pose)
        return self.movej(targj, speed)

    def solve_ik(self, pose):
        """Calculate joint configuration with inverse kinematics."""
        if self.scene_path is not None:
            return np.asarray(self.robot_adapter.solve_ik(pose), dtype=np.float32)
        joints = p.calculateInverseKinematics(
            bodyUniqueId=self.ur5,
            endEffectorLinkIndex=self.ee_tip,
            targetPosition=pose[0],
            targetOrientation=pose[1],
            lowerLimits=[-3 * np.pi / 2, -2.3562, -17, -17, -17, -17],
            upperLimits=[-np.pi / 2, 0, 17, 17, 17, 17],
            jointRanges=[np.pi, 2.3562, 34, 34, 34, 34],  # * 6,
            restPoses=np.float32(self.homej).tolist(),
            maxNumIterations=100,
            residualThreshold=1e-5)
        joints = np.float32(joints)
        joints[2:] = (joints[2:] + np.pi) % (2 * np.pi) - np.pi
        return joints

    def get_scene_object(self, entity_id):
        """Resolve a UniSis entity ID or name to its current PyBullet body ID."""
        if self.loaded_scene is None:
            raise RuntimeError('The UniSis scene has not been reset yet.')
        entity_id = str(entity_id)
        canonical_id = self.document.name_to_entity_id.get(entity_id, entity_id)
        try:
            return self.entity_id_to_body_id[canonical_id]
        except KeyError as exc:
            raise KeyError(f'No loaded UniSis entity has id or name {entity_id!r}.') from exc

    def _get_obs(self):
        # Get RGB-D camera image observations.
        obs = {'color': (), 'depth': ()}
        for config in self.agent_cams:
            color, depth, _ = self.render_camera(config)
            obs['color'] += (color,)
            obs['depth'] += (depth,)

        return obs

    def get_object_pose(self, obj_id):
        return p.getBasePositionAndOrientation(obj_id)

    def get_object_size(self, obj_id):
        """ approximate object's size using AABB """
        aabb_min, aabb_max = p.getAABB(obj_id)

        size_x = aabb_max[0] - aabb_min[0]
        size_y = aabb_max[1] - aabb_min[1]
        size_z = aabb_max[2] - aabb_min[2]
        return size_z * size_y * size_x



class EnvironmentNoRotationsWithHeightmap(Environment):
    """Environment that disables any rotations and always passes [0, 0, 0, 1]."""

    def __init__(self,
                 assets_root,
                 task=None,
                 disp=False,
                 shared_memory=False,
                 hz=240):
        super(EnvironmentNoRotationsWithHeightmap,
              self).__init__(assets_root, task, disp, shared_memory, hz)

        heightmap_tuple = [
            gym.spaces.Box(0.0, 20.0, (320, 160, 3), dtype=np.float32),
            gym.spaces.Box(0.0, 20.0, (320, 160), dtype=np.float32),
        ]
        self.observation_space = gym.spaces.Dict({
            'heightmap': gym.spaces.Tuple(heightmap_tuple),
        })
        self.action_space = gym.spaces.Dict({
            'pose0': gym.spaces.Tuple((self.position_bounds,)),
            'pose1': gym.spaces.Tuple((self.position_bounds,))
        })

    def step(self, action=None):
        """Execute action with specified primitive.

        Args:
          action: action to execute.

        Returns:
          (obs, reward, done, info) tuple containing MDP step data.
        """
        if action is not None:
            action = {
                'pose0': (action['pose0'][0], [0., 0., 0., 1.]),
                'pose1': (action['pose1'][0], [0., 0., 0., 1.]),
            }
        return super(EnvironmentNoRotationsWithHeightmap, self).step(action)

    def _get_obs(self):
        obs = {}

        color_depth_obs = {'color': (), 'depth': ()}
        for config in self.agent_cams:
            color, depth, _ = self.render_camera(config)
            color_depth_obs['color'] += (color,)
            color_depth_obs['depth'] += (depth,)
        cmap, hmap = utils.get_fused_heightmap(color_depth_obs, self.agent_cams,
                                               self.task.bounds, pix_size=0.003125)
        obs['heightmap'] = (cmap, hmap)
        return obs

