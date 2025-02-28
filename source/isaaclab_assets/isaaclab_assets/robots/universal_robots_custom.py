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


DOCKER_ISAACLAB_PATH = os.environ["DOCKER_ISAACLAB_PATH"]
UR10SHADOW_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{DOCKER_ISAACLAB_PATH}/scripts/my_models/ur10_shadow_group.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.712,
            "elbow_joint": 1.712,
            "wrist_1_joint": 0.0,
            "wrist_2_joint": 0.0,
            "wrist_3_joint": 0.0,
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
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=["rh_WR.*", "rh_(FF|MF|RF|LF|TH)J(3|2|1)", "rh_(LF|TH)J4", "rh_THJ0"],
            effort_limit={
                "rh_WRJ1": 4.785,
                "rh_WRJ0": 2.175,
                "rh_(FF|MF|RF|LF)J1": 0.7245,
                "rh_FFJ(3|2)": 0.9,
                "rh_MFJ(3|2)": 0.9,
                "rh_RFJ(3|2)": 0.9,
                "rh_LFJ(4|3|2)": 0.9,
                "rh_THJ4": 2.3722,
                "rh_THJ3": 1.45,
                "rh_THJ(2|1)": 0.99,
                "rh_THJ0": 0.81,
            },
            stiffness={
                "rh_WRJ.*": 5.0,
                "rh_(FF|MF|RF|LF|TH)J(3|2|1)": 1.0,
                "rh_(LF|TH)J4": 1.0,
                "rh_THJ0": 1.0,
            },
            damping={
                "rh_WRJ.*": 0.5,
                "rh_(FF|MF|RF|LF|TH)J(3|2|1)": 0.1,
                "rh_(LF|TH)J4": 0.1,
                "rh_THJ0": 0.1,
            },
        ),
    },
)
"""Configuration of UR-10 arm using implicit actuator models."""
