# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Universal Robots.

The following configuration parameters are available:

* :obj:`UR10_CFG`: The UR10 arm without a gripper.

Reference: https://github.com/ros-industrial/universal_robot
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
import os

##
# Configuration
##
DOCKER_ISAACLAB_PATH = os.environ.get("DOCKER_ISAACLAB_PATH", "/home/arr/IsaacLab")

SEED_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{DOCKER_ISAACLAB_PATH}/scripts/my_models/seed_r7_description/noid_mercury/noid_mercury.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
            retain_accelerations=False,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "l_shoulder_p_joint": 0.0,
            "l_shoulder_r_joint": 0.0,
            "l_elbow_joint": 0.0,
            "l_wrist_y_joint": 0.0,
            "l_wrist_p_joint": 0.0,
            "l_wrist_r_joint": 0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            velocity_limit=100.0,
            effort_limit=87.0,
            stiffness=10000.0,
            damping=1000.0,
        ),
    },
)
"""Configuration of UR-10 arm using implicit actuator models."""
