"""Robot adapters for scene-loaded robots used by GenSim primitives."""

from __future__ import annotations

import numpy as np
import pybullet as p

from cliport.tasks.grippers import Suction


_IDENTITY_QUATERNION = (0.0, 0.0, 0.0, 1.0)
_SUCTION_TIP_FROM_HEAD = ((0.0, 0.0, 0.029), _IDENTITY_QUATERNION)
_ORACLE_CONTACT_TO_TIP = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))


class SuctionToolAdapter:
    """Mount the original GenSim suction gripper and map its TCP convention.

    Oracle contact frames point toward the object. The physical suction tip
    frame is rotated 180 degrees about X from that virtual frame, while its
    position is the same. The existing URDF's tip link is 0.029 m from its
    head origin along local +Z.
    """

    def __init__(
        self,
        robot_id,
        hand_link,
        obj_ids,
        assets_root,
        *,
        hand_to_tip_transform=((0.0, 0.0, 0.14), _IDENTITY_QUATERNION),
        tip_from_head_transform=_SUCTION_TIP_FROM_HEAD,
        contact_to_tip_transform=_ORACLE_CONTACT_TO_TIP,
    ):
        self.robot_id = robot_id
        self.hand_link = hand_link
        self.tip_link = 0
        self.hand_to_tip = hand_to_tip_transform
        self.contact_to_tip = contact_to_tip_transform
        self.hand_to_contact = p.multiplyTransforms(
            self.hand_to_tip[0],
            self.hand_to_tip[1],
            *p.invertTransform(
                self.contact_to_tip[0], self.contact_to_tip[1]
            ),
        )
        head_from_tip = p.invertTransform(
            tip_from_head_transform[0], tip_from_head_transform[1]
        )
        hand_to_head = p.multiplyTransforms(
            self.hand_to_tip[0],
            self.hand_to_tip[1],
            head_from_tip[0],
            head_from_tip[1],
        )
        self.ee = Suction(
            assets_root,
            robot_id,
            hand_link,
            obj_ids,
            base_mount_transform=((0.0, 0.0, 0.0), _IDENTITY_QUATERNION),
            head_mount_transform=hand_to_head,
        )
        # The primitive interface uses this object for contact and grasp logic.
        self.primitive = self.ee

    def get_contact_pose(self, tip_pose):
        """Convert an actual suction-tip pose to the oracle contact frame."""
        contact_from_tip = p.invertTransform(
            self.contact_to_tip[0], self.contact_to_tip[1]
        )
        return p.multiplyTransforms(
            tip_pose[0], tip_pose[1], contact_from_tip[0], contact_from_tip[1]
        )

    def activate(self):
        return self.ee.activate()

    def release(self):
        return self.ee.release()

    def detect_contact(self):
        return self.ee.detect_contact()

    def check_grasp(self):
        return self.ee.check_grasp()


class FrankaSuctionAdapter:
    """Adapt a loaded Franka Panda and GenSim's original suction gripper.

    ``solve_ik`` accepts the oracle's virtual world contact pose and converts
    it to the named Franka hand link pose before asking PyBullet for inverse
    kinematics. The constructor reads the loaded joint state and never resets
    it.
    """

    def __init__(self, robot_id, obj_ids, assets_root, default_qpos=None):
        self.robot_id = robot_id
        self.obj_ids = obj_ids
        self.assets_root = assets_root

        joint_infos = [
            p.getJointInfo(robot_id, index)
            for index in range(p.getNumJoints(robot_id))
        ]
        self._joint_infos_by_name = {
            _decode_name(info[1]): info for info in joint_infos
        }
        self._link_indices_by_name = {
            _decode_name(info[12]): info[0] for info in joint_infos
        }

        self._arm_joint_names = _select_joint_names(
            self._joint_infos_by_name,
            tuple(f"joint{index}" for index in range(1, 8)),
            tuple(f"panda_joint{index}" for index in range(1, 8)),
        )
        self._finger_joint_names = _select_joint_names(
            self._joint_infos_by_name,
            ("finger_joint1", "finger_joint2"),
            ("panda_finger_joint1", "panda_finger_joint2"),
        )
        self.joints = [
            self._joint_infos_by_name[name][0] for name in self._arm_joint_names
        ]
        self.finger_joints = [
            self._joint_infos_by_name[name][0] for name in self._finger_joint_names
        ]
        self.ee_tip = _select_link_index(
            self._link_indices_by_name, ("hand", "panda_hand", "franka_hand")
        )
        self.hand_link = self.ee_tip

        for name in self._arm_joint_names:
            if self._joint_infos_by_name[name][2] != p.JOINT_REVOLUTE:
                raise ValueError(f"Franka arm joint {name!r} is not revolute")
        for name in self._finger_joint_names:
            if self._joint_infos_by_name[name][2] != p.JOINT_PRISMATIC:
                raise ValueError(f"Franka finger joint {name!r} is not prismatic")

        self._ik_joint_infos = sorted(
            (
                info for info in joint_infos
                if info[2] in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC)
            ),
            key=lambda info: info[3],
        )
        if len(self._ik_joint_infos) != 9:
            raise ValueError(
                "Expected seven Franka arm joints and two finger joints for IK; "
                f"found {len(self._ik_joint_infos)} movable joints"
            )

        if default_qpos is None:
            initial_qpos = {
                _decode_name(info[1]): float(p.getJointState(robot_id, info[0])[0])
                for info in self._ik_joint_infos
            }
            self.homej = np.asarray(
                [initial_qpos[name] for name in self._arm_joint_names],
                dtype=np.float32,
            )
        else:
            qpos = np.asarray(default_qpos, dtype=np.float32).reshape(-1)
            if qpos.size not in (7, 9):
                raise ValueError(
                    "Franka default_qpos must contain seven arm values or the "
                    f"seven arm and two finger values; got {qpos.size}"
                )
            if not np.all(np.isfinite(qpos)):
                raise ValueError("Franka default_qpos must contain finite values")
            self.homej = qpos[:7].copy()
            initial_qpos = dict(zip(self._arm_joint_names, qpos[:7], strict=True))
            if qpos.size == 9:
                initial_qpos.update(
                    zip(self._finger_joint_names, qpos[7:], strict=True)
                )
            else:
                initial_qpos.update(
                    {
                        name: float(
                            p.getJointState(
                                robot_id, self._joint_infos_by_name[name][0]
                            )[0]
                        )
                        for name in self._finger_joint_names
                    }
                )

        self._rest_poses = [
            float(initial_qpos[_decode_name(info[1])])
            if _decode_name(info[1]) in initial_qpos
            else float(p.getJointState(robot_id, info[0])[0])
            for info in self._ik_joint_infos
        ]
        self._lower_limits = []
        self._upper_limits = []
        for info in self._ik_joint_infos:
            lower, upper = float(info[8]), float(info[9])
            if lower >= upper:
                lower, upper = -2.0 * np.pi, 2.0 * np.pi
            self._lower_limits.append(lower)
            self._upper_limits.append(upper)
        self._joint_ranges = [
            upper - lower
            for lower, upper in zip(
                self._lower_limits, self._upper_limits, strict=True
            )
        ]

        self.tool = SuctionToolAdapter(
            robot_id,
            self.hand_link,
            obj_ids,
            assets_root,
        )
        self.ee = self.tool.ee
        self._disable_finger_geometry()

    def _disable_finger_geometry(self):
        """Clear space around the suction cup while keeping the hand visible."""
        for link_index in self.finger_joints:
            p.setCollisionFilterGroupMask(
                self.robot_id,
                link_index,
                collisionFilterGroup=0,
                collisionFilterMask=0,
            )
            p.changeVisualShape(
                self.robot_id, link_index, rgbaColor=(1.0, 1.0, 1.0, 0.0)
            )

    def solve_ik(self, pose):
        """Return the seven arm joints for a desired oracle contact pose."""
        position = np.asarray(pose[0], dtype=np.float64).reshape(-1)
        orientation = np.asarray(pose[1], dtype=np.float64).reshape(-1)
        if position.size != 3 or orientation.size != 4:
            raise ValueError(
                "A suction contact pose needs a 3D position and XYZW quaternion"
            )
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(orientation)):
            raise ValueError("A suction contact pose must contain finite values")
        quat_norm = np.linalg.norm(orientation)
        if quat_norm <= 1e-12:
            raise ValueError("A suction contact quaternion must have nonzero length")
        orientation = orientation / quat_norm

        contact_to_hand = p.invertTransform(
            self.tool.hand_to_contact[0], self.tool.hand_to_contact[1]
        )
        hand_position, hand_orientation = p.multiplyTransforms(
            position.tolist(),
            orientation.tolist(),
            contact_to_hand[0],
            contact_to_hand[1],
        )
        solution = p.calculateInverseKinematics(
            bodyUniqueId=self.robot_id,
            endEffectorLinkIndex=self.hand_link,
            targetPosition=hand_position,
            targetOrientation=hand_orientation,
            lowerLimits=self._lower_limits,
            upperLimits=self._upper_limits,
            jointRanges=self._joint_ranges,
            restPoses=self._rest_poses,
            maxNumIterations=200,
            residualThreshold=1e-5,
        )
        if len(solution) != len(self._ik_joint_infos):
            raise RuntimeError(
                "PyBullet returned an unexpected Franka IK result size: "
                f"expected {len(self._ik_joint_infos)}, got {len(solution)}"
            )
        positions_by_joint = {
            info[0]: float(value)
            for info, value in zip(self._ik_joint_infos, solution, strict=True)
        }
        return np.asarray(
            [positions_by_joint[index] for index in self.joints], dtype=np.float32
        )

    def get_contact_pose(self, tip_pose):
        """Return the oracle pose represented by a physical suction tip pose."""
        return self.tool.get_contact_pose(tip_pose)

    def activate(self):
        """Enable the mounted suction gripper."""
        return self.ee.activate()

    def release(self):
        """Release any object held by the mounted suction gripper."""
        return self.ee.release()


def create_robot_adapter(
    robot_id,
    obj_ids,
    assets_root,
    *,
    robot_type="franka",
    end_effector="suction",
    default_qpos=None,
):
    """Create a supported robot adapter, with explicit errors for reserved tools."""
    normalized_robot = str(robot_type).lower().replace("-", "_")
    normalized_end_effector = str(end_effector).lower().replace("-", "_")
    if normalized_end_effector in {"parallel", "parallel_gripper", "panda_hand"}:
        raise NotImplementedError(
            "The Franka parallel-gripper adapter is reserved but not implemented"
        )
    if normalized_end_effector != "suction":
        raise NotImplementedError(
            f"Unsupported GenSim end effector: {end_effector!r}"
        )
    if normalized_robot not in {
        "franka",
        "franka_panda",
        "franka_emika_panda",
        "panda",
    }:
        raise NotImplementedError(
            f"Unsupported GenSim robot adapter: {robot_type!r}"
        )
    return FrankaSuctionAdapter(
        robot_id, obj_ids, assets_root, default_qpos=default_qpos
    )


def _decode_name(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _select_joint_names(joints_by_name, *candidates):
    for names in candidates:
        if all(name in joints_by_name for name in names):
            return names
    expected = " or ".join(", ".join(names) for names in candidates)
    raise ValueError(f"Could not find expected Franka joints: {expected}")


def _select_link_index(links_by_name, candidates):
    for name in candidates:
        if name in links_by_name:
            return links_by_name[name]
    raise ValueError(
        "Could not find Franka hand link; expected one of "
        + ", ".join(candidates)
    )
