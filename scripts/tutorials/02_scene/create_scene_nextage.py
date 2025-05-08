# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/02_scene/create_scene_nextage.py --enable_cameras

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
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_assets import NEXTAGE_CFG
from isaaclab.utils.math import quat_from_euler_xyz, transform_points
from isaaclab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.utils.math import quat_from_euler_xyz
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
##
# Pre-defined configs
##

@configclass
class CartpoleSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""


    # Corrected robot_base definition
    #robot_base = RigidObjectCfg(
    #        prim_path="{ENV_REGEX_NS}/robot_base",  # Recommended: Use ENV_REGEX_NS for multi-env compatibility
    #        spawn=sim_utils.CuboidCfg(
    #            size=(0.5, 0.5, 0.5),  # Cube dimensions (length, width, height)
    #            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)),
    #            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
    #            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    #            collision_props=sim_utils.CollisionPropertiesCfg(),
    #        ),
    #        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.25)),  # Position the cube
    #    )

    robot = NEXTAGE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, -0.3], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.35]),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # Set Cube as object
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, -0.135], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            scale=(0.8, 0.8, 0.8),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
    )
    # use isaaclab.utils.math.quat_from_euler_xyz
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/HEAD_JOINT1_Link/LEFT_CAMERA/front_cam",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.0,0.0,0.0), rot=(0.58965,0.39028,-0.39028,-0.58965), convention="opengl"),
        #offset=CameraCfg.OffsetCfg(pos=(0.0,0.0,0.0), rot=quat_from_euler_xyz(torch.tensor(0.0), torch.deg2rad(torch.tensor(-67.0)), torch.tensor(-math.pi/2.0)), convention="opengl"),
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
        [0.53, 0.0, 0.1, 0.92,0,-0.38,0],
        [0.53, -0.02, 0.1, 0.92,0,-0.38,0],
        [0.53, 0.02, 0.1, 0.92,0,-0.38,0],
    ]
    ee_goals = torch.tensor(ee_goals, device=sim.device)
    # Track the given command
    current_goal_idx = 0
    # Create buffers to store actions
    ik_commands = torch.zeros(scene.num_envs, diff_ik_controller.action_dim, device=robot.device)
    ik_commands[:] = ee_goals[current_goal_idx]
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=[".*"], body_names=["RARM_JOINT5_Link"])
    robot_entity_cfg.resolve(scene)
    if robot.is_fixed_base:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
    else:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0]
    # Simulation loop
    while simulation_app.is_running():
        if count % 300 == 0:
            scene.reset()
            robot.reset()
            # Reset
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            ik_commands[:] = ee_goals[current_goal_idx]
            robot.write_joint_state_to_sim(position=joint_pos, velocity=joint_vel)
            joint_pos_des = joint_pos[:, robot_entity_cfg.joint_ids].clone()

            # change goal
            current_goal_idx = (current_goal_idx + 1) % len(ee_goals)
            root_pos_w = robot.data.root_state_w[:, :3]
            root_quat_w = robot.data.root_state_w[:, 3:7]
            goal_w = ee_goals[current_goal_idx]          # (7,)  ← 1D なのでそのまま [:3] でOK
            goal_pos_b, goal_quat_b = subtract_frame_transforms(
                root_pos_w, root_quat_w,
                goal_w[:3].unsqueeze(0),                 # (1,3)
                goal_w[3:].unsqueeze(0)                  # (1,4)
            )

            #ik_commands[:, :3] = goal_pos_b
            #ik_commands[:, 3:7] = goal_quat_b
            target = torch.zeros_like(ik_commands, device=robot.device)
            target[:, :3] = goal_pos_b
            target[:, 3:7] = goal_quat_b
            # reset controller
            diff_ik_controller.reset()
            diff_ik_controller.set_command(target)
            goal_pos_w, goal_quat_w = subtract_frame_transforms(
                root_pos_w, root_quat_w, goal_pos_b, goal_quat_b
            )
        else:
            # obtain quantities from simulation
            jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
            ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
            joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
            # compute frame in root frame
            ee_pos_b, ee_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w,
                                                            ee_pose_w[:, :3], ee_pose_w[:, 3:7])
            joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b,
                                                    jacobian, joint_pos)
        # apply actions
        robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
        scene.write_data_to_sim()
        #print("Received shape of rgb   image: ", scene["camera"].data.output["rgb"].shape)
        #print("Received shape of depth image: ", scene["camera"].data.output["distance_to_image_plane"].shape)

        #print("[INFO]: Resetting robot state...")
        sim.step()
        # Increment counter
        count += 1
        # Update buffers
        scene.update(sim_dt)
        # obtain quantities from simulation
        ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
        # update marker positions
        ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])

        #goal_marker.visualize(ik_commands[:, 0:3] + scene.env_origins, ik_commands[:, 3:7])
        goal_marker.visualize(goal_pos_w, goal_quat_w)
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
