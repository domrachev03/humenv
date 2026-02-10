# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import mujoco
from typing import Tuple

# Number of elements in qpos and qvel for each humanoid type
_QPOS_LEN = {
    "smpl": 76,
    "skeleton": 38,  # 7 (freejoint) + 31 (hinge joints)
}
_QVEL_LEN = {
    "smpl": 75,
    "skeleton": 37,  # 6 (freejoint) + 31 (hinge joints)
}

# Keep old constants for backward compatibility
QPOS_LEN_FOR_SMPL = 76
QVEL_LEN_FOR_SMPL = 75


def _get_dims(humanoid_type: str) -> Tuple[int, int]:
    return _QPOS_LEN[humanoid_type], _QVEL_LEN[humanoid_type]


def tpose(
    model: mujoco.MjModel, data: mujoco.MjData, random: np.random.RandomState, humanoid_type: str = "smpl"
) -> Tuple[np.ndarray, np.ndarray]:
    qpos_len, qvel_len = _get_dims(humanoid_type)
    qpos = np.zeros_like(data.qpos)
    qvel = np.zeros_like(data.qvel)

    if humanoid_type == "smpl":
        qpos[2] = 0.94
        # Only reset human model position. Keep other objects as defined in the XML
        qpos[qpos_len:] = data.qpos[qpos_len:]
        qvel[qvel_len:] = data.qvel[qvel_len:]
        z_rot = 0
        euler = np.array([90, 0, z_rot])
        rad = euler * np.pi / 180
        quat = np.zeros(4)
        mujoco.mju_euler2Quat(quat, rad, "XYZ")
        qpos[3] = quat[0]
        qpos[4:7] = quat[1:]
    elif humanoid_type == "skeleton":
        # Skeleton model is Y-up with initial quat rotation.
        # Standing height ~0.975 in skeleton coords → ~0.975 Z in world after rotation
        qpos[2] = 0.975
        qpos[qpos_len:] = data.qpos[qpos_len:]
        qvel[qvel_len:] = data.qvel[qvel_len:]
        # Use the skeleton's default orientation (Y-up to Z-up rotation)
        qpos[3] = 0.7071067811865475
        qpos[4] = 0.7071067811865475
        qpos[5] = 0.0
        qpos[6] = 0.0

    return qpos, qvel


def fall(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    random: np.random.RandomState,
    action_size: int,
    integration_steps: int,
    humanoid_type: str = "smpl",
) -> Tuple[np.ndarray, np.ndarray]:
    qpos_len, qvel_len = _get_dims(humanoid_type)
    # Only reset human model position. Keep other objects as defined in the XML
    data.qpos[:qpos_len] = 0
    data.qvel[:qvel_len] = 0
    z = 1
    orientation = random.random(4)
    data.qpos[2] = z
    data.qpos[3:7] = np.array(orientation)
    mujoco.mj_forward(model, data)
    n_steps = random.integers(0, 5, 1).item()
    for _ in range(n_steps):
        action = (random.random(action_size) - 0.5) * 1
        data.ctrl[:action_size] = action
        mujoco.mj_step(model, data, integration_steps)
    return data.qpos, data.qvel


def default_and_fall(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    random: np.random.RandomState,
    fall_prob: float,
    action_size: int,
    integration_steps: int,
    humanoid_type: str = "smpl",
) -> Tuple[np.ndarray, np.ndarray]:
    if random.random() < fall_prob:
        return fall(model, data, random, action_size, integration_steps, humanoid_type=humanoid_type)
    else:
        return tpose(model, data, random, humanoid_type=humanoid_type)
