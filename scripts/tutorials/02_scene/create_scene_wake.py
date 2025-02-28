# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/02_scene/create_scene_wake.py

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
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab_assets import UR10_CFG, SHADOW_HAND_CFG, UR10SHADOW_CFG  # UR10用設定（Isaac-Reach-UR10-v0)
from isaaclab.utils.math import quat_from_euler_xyz
##
# Pre-defined configs
##

@configclass
class CartpoleSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", 
                          spawn=sim_utils.GroundPlaneCfg(),
                          init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.80))
                          )

    # # lights
    # dome_light = AssetBaseCfg(
    #     prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    # )
# 
    # # articulation <isaaclab.assets.articulation.articulation.Articulation object at 0x76d00ae85000>
    robot = UR10SHADOW_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
# 
    # hand = SHADOW_HAND_CFG.replace(prim_path="{ENV_REGEX_NS}/Hand")
    # # robot
    # robot = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=f"/workspace/isaaclab/scripts/my_models/ur10_shadow2.usd", 
    #         scale=(1.0,1.0,1.0)
    #     ),
    # )
    # # mount
    mount = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Mount",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd", 
            scale=(2.0, 2.0, 2.0)
        ),
    )

    # table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/ThorlabsTable/table_instanceable.usd", 
            scale=(1.0, 1.0, 1.0)
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.3, 0.0, 0.0))
    )

    # spawn a YCB object
    ycb = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/YCB",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/009_gelatin_box.usd", 
            scale=(1.0, 1.0, 1.0)
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.7, 0.0, 0.0237), 
                                                rot=quat_from_euler_xyz(torch.tensor(-(math.pi/2.0)), torch.tensor(0.0), torch.tensor(math.pi/2.0)))
    )
        # use isaaclab.utils.math.quat_from_euler_xyz
        
def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.
    robot = scene["robot"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulation loop
    reset_joints = torch.tensor([-0.3536, -1.1882,  1.5369,  0.5527,  1.3457, -0.2750])
    while simulation_app.is_running():
        if count == 0:
            #robot.reset()
            import pdb;pdb.set_trace()
            pass
        # Reset
        #joint_pos = robot.data.default_joint_pos.clone()
        #joint_pos = reset_joints
        #joint_vel = robot.data.default_joint_vel.clone()
        #robot.write_joint_state_to_sim(position=joint_pos, velocity=joint_vel)
        if count % 500 == 0:
            # joint_pos = robot.data.default_joint_pos.clone()
            # joint_pos = reset_joints
            # joint_vel = robot.data.default_joint_vel.clone()
            # robot.write_joint_state_to_sim(position=joint_pos, velocity=joint_vel)
            # reset counter
            # count = 0
            # reset the scene entities
            # root state
            # we offset the root state by the origin since the states are written in simulation world frame
            # if this is not done, then the robots will be spawned at the (0, 0, 0) of the simulation world
            # root_state = robot.data.default_root_state.clone()
            # root_state[:, :3] += scene.env_origins
            #robot.write_root_pose_to_sim(root_state[:, :7])
            #robot.write_root_velocity_to_sim(root_state[:, 7:])
            # set joint positions with some noise
            # joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            # joint_pos += torch.rand_like(joint_pos) * 0.1
            #robot.write_joint_state_to_sim(joint_pos, joint_vel)
            # clear internal buffers
            # scene.reset()
            print("[INFO]: Resetting robot state...")
        # Apply random action
        # -- generate random joint efforts
        #efforts = torch.randn_like(robot.data.joint_pos) * 5.0
        # -- apply action to the robot
        #robot.set_joint_effort_target(efforts)
        # -- write data to sim
        # scene.write_data_to_sim()
        # Perform step
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
    scene_cfg = CartpoleSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
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
