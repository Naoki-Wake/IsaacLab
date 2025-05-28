# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/02_scene/create_scene_seed.py --enable_cameras

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
from isaaclab_assets import SEED_CFG
from isaaclab.utils.math import quat_from_euler_xyz
from isaaclab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

##
# Pre-defined configs
##

@configclass
class CartpoleSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", 
                          spawn=sim_utils.GroundPlaneCfg(),
                          init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0))
                          )

    # # lights
    # dome_light = AssetBaseCfg(
    #     prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    # )
# 
    # # articulation <isaaclab.assets.articulation.articulation.Articulation object at 0x76d00ae85000>
    # robot = UR10SHADOW_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
# 
    # hand = SHADOW_HAND_CFG.replace(prim_path="{ENV_REGEX_NS}/Hand")
    # # robot
    # robot = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=f"/workspace/isaaclab/scripts/my_models/seed_r7_description/noid_mercury/noid_mercury.usd", 
    #         scale=(1.0,1.0,1.0)
    #     ),
    # )
    robot = SEED_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Lights
    distant_light = AssetBaseCfg(
        prim_path="/World/lightDistant", spawn=sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/ThorlabsTable/table_instanceable.usd", 
            scale=(1.0, 1.0, 1.0)
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.7, 0.0, 0.8))
    )

    # spawn a YCB object
    # ycb = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/YCB",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/009_gelatin_box.usd", 
    #         scale=(1.0, 1.0, 1.0)
    #     ),
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=(0.48, 0.0, 0.8237), 
    #                                             rot=quat_from_euler_xyz(torch.tensor(-(math.pi/2.0)), torch.tensor(0.0), torch.tensor(math.pi/2.0)))
    # )
    
    # use isaaclab.utils.math.quat_from_euler_xyz
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/head_link/front_cam",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.1 , 0.0, 0.0), rot=(0.5,0.5,-0.5,-0.5), convention="ros"),
    )

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.
    robot = scene["robot"]
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0

    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    # Define goals for the arm
    ee_goals = [
        [0.73, 0.0, 0.82, 0.5, 0.5, -0.5, 0.5],
        [0.73, -0.02, 0.82, 0.5, 0.5, -0.5, 0.5],
        [0.73, 0.02, 0.82, 0.5, 0.5, -0.5, 0.5],
    ]
    ee_goals = torch.tensor(ee_goals, device=sim.device)
    # Track the given command
    current_goal_idx = 0
    # Create buffers to store actions
    ik_commands = torch.zeros(scene.num_envs, diff_ik_controller.action_dim, device=robot.device)
    ik_commands[:] = ee_goals[current_goal_idx]
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=[".*"], body_names=["r_wrist_top"])
    robot_entity_cfg.resolve(scene)
    if robot.is_fixed_base:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
    else:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0]
    # Simulation loop
    while simulation_app.is_running():
        if count % 500 == 0:
            scene.reset()
            robot.reset()
            # Reset
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            ik_commands[:] = ee_goals[current_goal_idx]
            robot.write_joint_state_to_sim(position=joint_pos, velocity=joint_vel)
            joint_pos_des = joint_pos[:, robot_entity_cfg.joint_ids].clone()
            # reset controller
            diff_ik_controller.reset()
            diff_ik_controller.set_command(ik_commands)
            # change goal
            current_goal_idx = (current_goal_idx + 1) % len(ee_goals)
        else:
            # obtain quantities from simulation
            jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
            ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
            root_pose_w = robot.data.root_state_w[:, 0:7]
            joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
            # compute frame in root frame
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
            # compute the joint commands
            joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
        # apply actions
        robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
        scene.write_data_to_sim()
        if count % 500 == 0:
            pass
            #import pdb;pdb.set_trace()
            #joint_pos = robot.data.default_joint_pos.clone()
            # joint_pos = reset_joints
            #joint_vel = robot.data.default_joint_vel.clone()
            #robot.write_joint_state_to_sim(position=joint_pos, velocity=joint_vel)
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
            #print(scene["camera"])
        print("Received shape of rgb   image: ", scene["camera"].data.output["rgb"].shape)
        print("Received shape of depth image: ", scene["camera"].data.output["distance_to_image_plane"].shape)

        # set looking at
        #scene["camera"].set_world_poses_from_view
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
        # obtain quantities from simulation
        ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
        # update marker positions
        ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        goal_marker.visualize(ik_commands[:, 0:3] + scene.env_origins, ik_commands[:, 3:7])

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
