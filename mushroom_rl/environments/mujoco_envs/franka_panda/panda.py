import yaml
from pathlib import Path

import mujoco
import numpy as np

from mushroom_rl.environments.mujoco import MuJoCo, ObservationType


class Panda(MuJoCo):

    config_path = (
        Path(__file__).resolve().parent.parent / "data" / "panda" / "panda.yaml"
    ).as_posix()

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Access the loaded configuration
    WORKSPACE_LIMITS = config["workspace"]["limits"]
    WORKSPACE_MIN = np.array(config["workspace"]["min"])
    WORKSPACE_MAX = np.array(config["workspace"]["max"])

    JOINT_POS_LIMITS = config["joint_pos"]["limits"]
    JOINT_POS_MIN = np.array(config["joint_pos"]["min"])
    JOINT_POS_MAX = np.array(config["joint_pos"]["max"])

    JOINT_VEL_LIMITS = config["joint_vel"]["limits"]
    JOINT_VEL_MIN = np.array(config["joint_vel"]["min"])
    JOINT_VEL_MAX = np.array(config["joint_vel"]["max"])

    TORQUE_LIMITS = config["torque"]["limits"]
    TORQUE_MIN = np.array(config["torque"]["min"])
    TORQUE_MAX = np.array(config["torque"]["max"])

    def __init__(
        self,
        xml_path,
        gamma,
        horizon,
        n_substeps,
        additional_data_spec=None,
        collision_groups=None,
        actuation_spec=None,
        **viewer_params,
    ):
        actuation_spec = actuation_spec or [
            "actuator1",
            "actuator2",
            "actuator3",
            "actuator4",
            "actuator5",
            "actuator6",
            "actuator7",
            "actuator8",
        ]

        observation_spec = [
            ("joint1_pos", "joint1", ObservationType.JOINT_POS),
            ("joint2_pos", "joint2", ObservationType.JOINT_POS),
            ("joint3_pos", "joint3", ObservationType.JOINT_POS),
            ("joint4_pos", "joint4", ObservationType.JOINT_POS),
            ("joint5_pos", "joint5", ObservationType.JOINT_POS),
            ("joint6_pos", "joint6", ObservationType.JOINT_POS),
            ("joint7_pos", "joint7", ObservationType.JOINT_POS),
            ("finger_joint1_pos", "finger_joint1", ObservationType.JOINT_POS),
            ("finger_joint2_pos", "finger_joint2", ObservationType.JOINT_POS),
            ("gripper_pos", "gripper", ObservationType.SITE_POS),
            ("joint1_vel", "joint1", ObservationType.JOINT_VEL),
            ("joint2_vel", "joint2", ObservationType.JOINT_VEL),
            ("joint3_vel", "joint3", ObservationType.JOINT_VEL),
            ("joint4_vel", "joint4", ObservationType.JOINT_VEL),
            ("joint5_vel", "joint5", ObservationType.JOINT_VEL),
            ("joint6_vel", "joint6", ObservationType.JOINT_VEL),
            ("joint7_vel", "joint7", ObservationType.JOINT_VEL),
        ]

        additional_data_spec = additional_data_spec or []

        additional_data_spec += [
            ("joint1_pos", "joint1", ObservationType.JOINT_POS),
            ("joint2_pos", "joint2", ObservationType.JOINT_POS),
            ("joint3_pos", "joint3", ObservationType.JOINT_POS),
            ("joint4_pos", "joint4", ObservationType.JOINT_POS),
            ("joint5_pos", "joint5", ObservationType.JOINT_POS),
            ("joint6_pos", "joint6", ObservationType.JOINT_POS),
            ("joint7_pos", "joint7", ObservationType.JOINT_POS),
            ("finger_joint1_pos", "finger_joint1", ObservationType.JOINT_POS),
            ("finger_joint2_pos", "finger_joint2", ObservationType.JOINT_POS),
            ("gripper_pos", "gripper", ObservationType.SITE_POS),
            ("hand_rot", "hand", ObservationType.BODY_ROT),
            ("left_fingertip_pos", "left_fingertip", ObservationType.SITE_POS),
            ("right_fingertip_pos", "right_fingertip", ObservationType.SITE_POS),
            ("left_cube_pos", "left_cube", ObservationType.SITE_POS),
            ("right_cube_pos", "right_cube", ObservationType.SITE_POS),
        ]

        collision_groups = collision_groups or []

        collision_groups += [
            ("hand", ["hand_c"]),
            (
                "left_finger",
                [
                    "left_fingertip_pad_collision_1",
                    "left_fingertip_pad_collision_2",
                    "left_fingertip_pad_collision_3",
                    "left_fingertip_pad_collision_4",
                    "left_fingertip_pad_collision_5",
                ],
            ),
            (
                "right_finger",
                [
                    "right_fingertip_pad_collision_1",
                    "right_fingertip_pad_collision_2",
                    "right_fingertip_pad_collision_3",
                    "right_fingertip_pad_collision_4",
                    "right_fingertip_pad_collision_5",
                ],
            ),
            (
                "robot",
                [
                    "link0_c",
                    "link1_c",
                    "link2_c",
                    "link3_c",
                    "link4_c",
                    "link5_c0",
                    "link5_c1",
                    "link5_c2",
                    "link6_c",
                    "link7_c",
                ],
            ),
            ("floor", ["floor"]),
        ]

        super().__init__(
            xml_path,
            gamma=gamma,
            horizon=horizon,
            actuation_spec=actuation_spec,
            observation_spec=observation_spec,
            additional_data_spec=additional_data_spec,
            collision_groups=collision_groups,
            n_substeps=n_substeps,
            **viewer_params,
        )

    def _load_keyframe(self, name: str):
        keyframe = self._model.keyframe(name)
        mujoco.mj_resetDataKeyframe(self._model, self._data, keyframe.id)  # type: ignore

    def setup(self, obs):
        super().setup(obs)
        self._load_keyframe("home")

    def _compute_action(self, obs, action):
        action = super()._compute_action(obs, action)
        self._data.qfrc_applied[:7] = self._data.qfrc_bias[:7]
        return action

    def _get_random_workspace_pos(self, x=None, y=None, z=None):
        x_pos = x or np.random.uniform(*self.WORKSPACE_LIMITS["x"])
        y_pos = y or np.random.uniform(*self.WORKSPACE_LIMITS["y"])
        z_pos = z or np.random.uniform(*self.WORKSPACE_LIMITS["z"])
        return np.array([x_pos, y_pos, z_pos])
