# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p $TSS_ISAAC_PATH/scripts/scenes/create_scene_nextage.py --enable_cameras

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher


# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the interactive scene interface.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import math
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.math import quat_from_euler_xyz
from isaaclab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns, ImuCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.sim.schemas import CollisionPropertiesCfg

from tasqsym_isaac.assets.robots.nextage import NEXTAGE_CFG


import roslibpy
import numpy as np

from PIL import Image
import io
import base64


import os
tss_isaac_path = os.environ.get("TSS_ISAAC_PATH")

img_width = 640 #1280
img_height = int(img_width * 720 / 1280)


##
# Pre-defined configs
##

@configclass
class nextageSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane",
                          spawn=sim_utils.GroundPlaneCfg(),
                          init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0))
                          )

    robot = NEXTAGE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Lights
    distant_light = AssetBaseCfg(
        prim_path="/World/lightDistant", spawn=sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print(ISAAC_NUCLEUS_DIR)

    # table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/ThorlabsTable/table_instanceable.usd",
            scale=(1.0, 1.0, 1.0)
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(1.75, 0.0, 0.8), rot=(0,0,0,1))
    )

    box = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[1.08, 0, 0.8 + 0.02], rot=[1, 0, 0, 0]),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{tss_isaac_path}/scripts/models/test_objects/S_NvidiaCube_edited.usda",
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=0.1, #1000.0,
                max_linear_velocity=0.1, #1000.0,
        	max_depenetration_velocity=1.0, #5.0,
                disable_gravity=False,
            ),
            collision_props=CollisionPropertiesCfg(collision_enabled=True),
        ),
    )

    # from isaaclab.sim.spawners.shapes import CuboidCfg
    # box = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Object",
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=[1.08, 0, 0.8 + 0.045], rot=[1, 0, 0, 0]),
    #     spawn=CuboidCfg(
    #         size=(0.04, 0.1, 0.07),
    #         rigid_props=RigidBodyPropertiesCfg(
    #             solver_position_iteration_count=16,
    #             solver_velocity_iteration_count=1,
    #             max_angular_velocity=0.1,
    #             max_linear_velocity=0.1,
    #     	max_depenetration_velocity=5.0, #5.0,
    #             disable_gravity=False,
    #         ),
    #         collision_props=CollisionPropertiesCfg(collision_enabled=True),
    #         physics_material=sim_utils.RigidBodyMaterialCfg(
    #             static_friction=100.0,
    #             dynamic_friction=100.0,
    #             restitution=0.0,
    #             friction_combine_mode="max",
    #             restitution_combine_mode="max"
    #         )
    #     ),
    # )

base_pose = None
def navcb(msg):
    global base_pose
    base_pose = msg['transform']

joint_state = None
def jointscb(msg):
    global joint_state
    joint_state = msg

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    rosclient = roslibpy.Ros('localhost', 9090)
    try:
        rosclient.run(30)
    except:
        print('could not connect to ROS')

    if not rosclient.is_connected: return

    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.
    robot = scene["robot"]

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0

    js = [0.0 for x in range(len(robot.data.joint_names))]

    # Simulation loop
    while simulation_app.is_running():

        joint_pos = torch.Tensor([js])
        joint_vel = robot.data.default_joint_vel.clone()
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        # robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)

        scene.write_data_to_sim()


        sim.step()
        # Increment counter
        count += 1
        # Update buffers
        scene.update(sim_dt)


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.0, -1.4, 1.3], [0.0, 0.0, 0.0])
    # Design scene
    scene_cfg = nextageSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
