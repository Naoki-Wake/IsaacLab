# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Universal Robots.

The following configuration parameters are available:

* :obj:`NEXTAGE_CFG`: The Nextage + Shadow.

"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg


##
# Configuration
##


NEXTAGE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"scripts/my_models/nextage/nextage_env.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=12, solver_velocity_iteration_count=1
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.3),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={
        "head": ImplicitActuatorCfg(
            joint_names_expr=[
                "LARM_.*",
            ],
            effort_limit=None,
            velocity_limit=None,
            stiffness=None,
            damping=None,
        ),
        "left-arm": ImplicitActuatorCfg(
            joint_names_expr=["LARM_JOINT[0-5]"],
            effort_limit=50.0,
            velocity_limit=1.5,
            stiffness=100.0,
            damping=5.0,
        ),
        "right-arm": ImplicitActuatorCfg(
            joint_names_expr=["RARM_JOINT[0-5]"],
            effort_limit=50.0,
            velocity_limit=1.5,
            stiffness=100.0,
            damping=5.0,
        ),
        "left-hand": ImplicitActuatorCfg(
            joint_names_expr=["lh_.*"],
            effort_limit=20.0,
            velocity_limit=1.0,
            stiffness=50.0,
            damping=2.0,
        ),
        "right-hand": ImplicitActuatorCfg(
            joint_names_expr=["rh_.*"],
            effort_limit=20.0,
            velocity_limit=1.0,
            stiffness=50.0,
            damping=2.0,
        ),
    },

)
"""Configuration of Nextage robot with ShadowHands"""
