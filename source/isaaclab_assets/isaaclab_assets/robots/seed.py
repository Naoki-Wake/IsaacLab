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

##
# Configuration
##


SEED_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/workspace/isaaclab/scripts/my_models/seed_r7_description/noid_mercury/noid_mercury.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "l_shoulder_p_joint": 0.0,  # 以前の "shoulder_pan_joint"
            "l_shoulder_r_joint": 0.0,  # 以前の "shoulder_lift_joint"
            "l_elbow_joint": 0.0,  # 以前の "elbow_joint"
            "l_wrist_y_joint": 0.0,  # 以前の "wrist_1_joint"
            "l_wrist_p_joint": 0.0,  # 以前の "wrist_2_joint"
            "l_wrist_r_joint": 0.0,  # 以前の "wrist_3_joint"
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            velocity_limit=100.0,
            effort_limit=87.0,
            stiffness=800.0,
            damping=40.0,
        ),
    },
)
"""Configuration of UR-10 arm using implicit actuator models."""
