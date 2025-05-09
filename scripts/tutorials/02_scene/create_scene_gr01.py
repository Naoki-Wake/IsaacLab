# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/02_scene/create_scene_gr01.py --enable_cameras

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
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import torchvision.transforms as T
from PIL import Image
import torch
import numpy as np
from isaaclab_assets.robots.fourier import GR1T2_CFG  # isort: skip
import time
from gr00t_helper import RobotInferenceClient
import pandas as pd
import torch.nn.functional as F
import threading
import sys
import math

output_status = False

def get_joint_idx(scene, joint_name: str) -> int:
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=[joint_name], body_names=[".*"])
    robot_entity_cfg.resolve(scene)
    return robot_entity_cfg.joint_ids[0]

task_description_input = ["Move arms upwards."]  # default
def input_thread():
    global task_description_input
    print("Type new task descriptions anytime and press Enter.")
    while True:
        user_input = sys.stdin.readline().strip()
        if user_input:
            task_description_input = [user_input]
            print(f"[INFO] Updated task_description: {user_input}")
##
# Scene definition
##
@configclass
class MySceneCfg(InteractiveSceneCfg):

    # Table
    packing_table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/PackingTable",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.55, 0.0], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    # Object
    # object = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Object",
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.35, 0.40, 1.0413], rot=[1, 0, 0, 0]),
    #     spawn=sim_utils.CylinderCfg(
    #         radius=0.018,
    #         height=0.35,
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=0.3),
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.1), metallic=1.0),
    #         physics_material=sim_utils.RigidBodyMaterialCfg(
    #             friction_combine_mode="max",
    #             restitution_combine_mode="min",
    #             static_friction=0.9,
    #             dynamic_friction=0.9,
    #             restitution=0.0,
    #         ),
    #     ),
    # )
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0, 0.40, 1.0413], rot=[1, 0, 0, 0]),
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
    # Humanoid robot w/ arms higher
    robot: ArticulationCfg = GR1T2_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0, 0, 0.93),
            rot=(0.7071, 0, 0, 0.7071),
            joint_pos={
                # right-arm
                "right_shoulder_pitch_joint": 0.0,
                "right_shoulder_roll_joint": 0.0,
                "right_shoulder_yaw_joint": 0.0,
                "right_elbow_pitch_joint": -1.5708,
                "right_wrist_yaw_joint": 0.0,
                "right_wrist_roll_joint": 0.0,
                "right_wrist_pitch_joint": 0.0,
                # left-arm
                "left_shoulder_pitch_joint": 0.0,
                "left_shoulder_roll_joint": 0.0,
                "left_shoulder_yaw_joint": 0.0,
                "left_elbow_pitch_joint": -1.5708,
                "left_wrist_yaw_joint": 0.0,
                "left_wrist_roll_joint": 0.0,
                "left_wrist_pitch_joint": 0.0,
                # --
                "head_.*": 0.0,
                "waist_.*": 0.0,
                ".*_hip_.*": 0.0,
                ".*_knee_.*": 0.0,
                ".*_ankle_.*": 0.0,
                "R_.*": 0.0,
                "L_.*": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
    )

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(),
    )

    # Lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # use isaaclab.utils.math.quat_from_euler_xyz
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/head_yaw_link/front_cam",
        update_period=0.1,
        height=440,
        width=640,
        data_types=["rgb"],# "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=69, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.1,0.0,0.1), rot=(0.58965,0.39028,-0.39028,-0.58965), convention="opengl"),
        #offset=CameraCfg.OffsetCfg(pos=(0.0,0.0,0.0), rot=quat_from_euler_xyz(torch.tensor(0.0), torch.deg2rad(torch.tensor(-67.0)), torch.tensor(-math.pi/2.0)), convention="opengl"),
    )
def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.
    robot = scene["robot"]
    camera = scene["camera"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    # wrist_prim = get_prim_at_path("/World/envs/env_0/Robot/right_wrist_roll_joint")
    # drive = UsdPhysics.DriveAPI.Apply(wrist_prim, "angular")
    # drive.CreateDampingAttr(50.0)  # 適切なダンピング値に設定
    # drive.CreateStiffnessAttr(0.0)  # 必要に応じて設定
    # right_arm indexes
    right_arm_idx = np.array([
        get_joint_idx(scene, "right_shoulder_pitch_joint"),
        get_joint_idx(scene, "right_shoulder_roll_joint"),
        get_joint_idx(scene, "right_shoulder_yaw_joint"),
        get_joint_idx(scene, "right_elbow_pitch_joint"),
        get_joint_idx(scene, "right_wrist_yaw_joint"),
        get_joint_idx(scene, "right_wrist_roll_joint"),
        get_joint_idx(scene, "right_wrist_pitch_joint"),
    ])
    # left_arm indexes
    left_arm_idx = np.array([
        get_joint_idx(scene, "left_shoulder_pitch_joint"),
        get_joint_idx(scene, "left_shoulder_roll_joint"),
        get_joint_idx(scene, "left_shoulder_yaw_joint"),
        get_joint_idx(scene, "left_elbow_pitch_joint"),
        get_joint_idx(scene, "left_wrist_yaw_joint"), # prolem
        get_joint_idx(scene, "left_wrist_roll_joint"),
        get_joint_idx(scene, "left_wrist_pitch_joint"),
    ])
    print("left_arm_idx", left_arm_idx)
    # left hand indexes
    left_hand_idx = np.array([
        get_joint_idx(scene, "L_pinky_proximal_joint"),
        get_joint_idx(scene, "L_ring_proximal_joint"),
        get_joint_idx(scene, "L_middle_proximal_joint"),
        get_joint_idx(scene, "L_index_proximal_joint"),
        get_joint_idx(scene, "L_thumb_proximal_pitch_joint"),
        get_joint_idx(scene, "L_thumb_proximal_yaw_joint"),
    ])
    # right hand indexes
    right_hand_idx = np.array([
        get_joint_idx(scene, "R_pinky_proximal_joint"),
        get_joint_idx(scene, "R_ring_proximal_joint"),
        get_joint_idx(scene, "R_middle_proximal_joint"),
        get_joint_idx(scene, "R_index_proximal_joint"),
        get_joint_idx(scene, "R_thumb_proximal_pitch_joint"),
        get_joint_idx(scene, "R_thumb_proximal_yaw_joint"),
    ])
    # waist indexes
    waist_idx = np.array([
        get_joint_idx(scene, "waist_yaw_joint"),
        get_joint_idx(scene, "waist_pitch_joint"),
        get_joint_idx(scene, "waist_roll_joint"),
    ])
    head_pitch_idx = np.array([
        get_joint_idx(scene, "head_pitch_joint"),
    ])

    policy_client = RobotInferenceClient(host="10.137.70.15", port=5555)

    print("Available modality config available:")
    modality_configs = policy_client.get_modality_config()
    print(modality_configs.keys())
    action = None


    # Simulation loop
    while simulation_app.is_running():
        if count == 0:
            scene.reset()
            robot.reset()
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(position=joint_pos, velocity=joint_vel)
        if count%16 == 0:
            ego_view = scene["camera"].data.output["rgb"]
            # get robot state
            joint_pos = robot.data.joint_pos.clone()
            left_arm = joint_pos[:, left_arm_idx].cpu().numpy()
            right_arm = joint_pos[:, right_arm_idx].cpu().numpy()
            left_hand = joint_pos[:, left_hand_idx].cpu().numpy()
            right_hand = joint_pos[:, right_hand_idx].cpu().numpy()
            waist = joint_pos[:, waist_idx].cpu().numpy()
            # Making prediction...
            # - obs: video.ego_view: (1, 256, 256, 3)
            # - obs: state.left_arm: (1, 7)
            # - obs: state.right_arm: (1, 7)
            # - obs: state.left_hand: (1, 6)
            # - obs: state.right_hand: (1, 6)
            # - obs: state.waist: (1, 3)
            # reshape to (1, 256, 256, 3)

            ego_view = ego_view.squeeze(0).detach().cpu()  # [H, W, 3]
            image_np = ego_view.numpy()
            image_pil = Image.fromarray(image_np)
            # resize 640x440 to 256x176
            resized_image = image_pil.resize((256, 176), resample=Image.BILINEAR)
            resized_np = np.array(resized_image)
            canvas_np = np.zeros((256, 256, 3), dtype=np.uint8)
            y_start = 40
            y_end = y_start + 176
            canvas_np[y_start:y_end, :] = resized_np
            # save image
            canvas_pil = Image.fromarray(canvas_np)
            canvas_pil.save("image.png")
            # import pdb; pdb.set_trace()
            ego_view = canvas_np[None, ...]
            # print("Saved image as image.png")
            # Convert to numpy
            left_arm = left_arm.astype(np.float64)
            right_arm = right_arm.astype(np.float64)
            left_hand = left_hand.astype(np.float64)
            right_hand = right_hand.astype(np.float64)
            waist = waist.astype(np.float64)
            obs = {
                "video.ego_view": ego_view,
                "state.left_arm": left_arm,
                "state.right_arm": right_arm,
                "state.left_hand": left_hand,
                "state.right_hand": right_hand,
                "state.waist": waist,
                "annotation.human.action.task_description": task_description_input,
                #"annotation.human.action.task_description": ["Move the right arm upwards."],
            }
            time_start = time.time()
            action = policy_client.get_action(obs)
            if output_status:
                print(f"Total time taken to get action from server: {time.time() - time_start} seconds")
        assert action is not None, "Action is None. Please check the server connection."
        # - action: action.left_arm: (16, 7)
        # - action: action.right_arm: (16, 7)
        # - action: action.left_hand: (16, 6)
        # - action: action.right_hand: (16, 6)
        # - action: action.waist: (16, 3)
        #for key, value in action.items():
        #    print(f"Action: {key}: {value.shape}")
        #import pdb; pdb.set_trace()
        step_within_sim = count%16
        #joint_pos_des = robot.data.default_joint_pos.clone()
        if count == 0:
            joint_pos_des = robot.data.default_joint_pos.clone()
        else:
            joint_pos_des = robot.data.joint_pos.clone()
        # import pdb; pdb.set_trace()
        if output_status:
            print("joint_pos_des", joint_pos_des)
        right_arm_action = torch.tensor(action["action.right_arm"][step_within_sim], device=joint_pos_des.device)
        left_arm_action = torch.tensor(action["action.left_arm"][step_within_sim], device=joint_pos_des.device)
        action_left_hand = torch.tensor(action["action.left_hand"][step_within_sim], device=joint_pos_des.device)
        action_right_hand = torch.tensor(action["action.right_hand"][step_within_sim], device=joint_pos_des.device)
        waist_action = torch.tensor(action["action.waist"][step_within_sim], device=joint_pos_des.device)
        
        # use Torch.remainder(angle + π, 2π) - π to wrap angles
        # right_arm_action = torch.remainder(right_arm_action + torch.pi, 2 * torch.pi) - torch.pi
        # left_arm_action = torch.remainder(left_arm_action + torch.pi, 2 * torch.pi) - torch.pi
        # right_hand_action = torch.remainder(right_hand_action + torch.pi, 2 * torch.pi) - torch.pi
        # left_hand_action = torch.remainder(left_hand_action + torch.pi, 2 * torch.pi) - torch.pi
        # waist_action = torch.remainder(waist_action + torch.pi, 2 * torch.pi) - torch.pi

        #def avg_first_16(arr_np):
        #    return torch.as_tensor(arr_np[:16],            # (16, DOF)
        #                        device=joint_pos_des.device,
        #                        dtype=torch.float32).mean(dim=0)  # (DOF,)
#
        #right_arm_action  = avg_first_16(action["action.right_arm"])
        #left_arm_action   = avg_first_16(action["action.left_arm"])
        #left_hand_action  = avg_first_16(action["action.left_hand"])
        #right_hand_action = avg_first_16(action["action.right_hand"])
        #waist_action      = avg_first_16(action["action.waist"])

        action_right_hand[0:4] = -1* action_right_hand[0:4]
        action_right_hand[0:4] = torch.clamp(action_right_hand[0:4], -math.pi/2, 0.0)
        action_right_hand[4] = torch.clamp(action_right_hand[4], 0.0, math.pi/3)
        action_right_hand[5] = -1* action_right_hand[5]
        action_right_hand[5] = torch.clamp(action_right_hand[5], -math.pi/2, 0.0)
        #action_right_hand[5] = -math.pi/2
        action_left_hand[0:4] = -1* action_left_hand[0:4]
        action_left_hand[0:4] = torch.clamp(action_left_hand[0:4], -math.pi/2, 0.0)
        action_left_hand[4] = torch.clamp(action_left_hand[4], 0.0, math.pi/3)
        action_left_hand[5] = -1* action_left_hand[5]
        action_left_hand[5] = torch.clamp(action_left_hand[5], -math.pi/2, 0.0)
        #action_left_hand[5] = -math.pi/2
        # set joint pos
        joint_pos_des[:, right_arm_idx] = right_arm_action
        joint_pos_des[:, left_arm_idx] = left_arm_action
        joint_pos_des[:, left_hand_idx] = action_left_hand
        joint_pos_des[:, right_hand_idx] = action_right_hand
        joint_pos_des[:, waist_idx] = waist_action
        print(action_right_hand)
        # tilt head
        joint_pos_des[:, head_pitch_idx] = math.pi/15
        #joint_ids_des = np.concatenate((right_arm_idx, left_arm_idx, left_hand_idx, right_hand_idx, waist_idx), axis=0)
        #target_des = joint_pos_des[:, joint_ids_des]   
        # apply actions
        # Store data for visualization
        if count == 0:
            data = {
            "step": [],
            "joint_pos_des": []
            }

        # Append current data
        data["step"].append(count)
        data["joint_pos_des"].append(joint_pos_des.cpu().numpy().tolist()[0])

        # Save to CSV periodically
        if count % 100 == 0:
            df = pd.DataFrame(data)
            df.to_csv("simulation_data.csv", index=False)
            # print(f"Saved data to simulation_data.csv at step {count}")

        # Apply actions
        # Apply moving average to actions
        moving_average = True
        if moving_average:
            if count >= 16:
                recent_joint_pos = torch.tensor(data["joint_pos_des"][-16:], device=joint_pos_des.device, dtype=torch.float32)  # shape: (16, DOF)
                avg_joint_pos = recent_joint_pos.mean(dim=0)  # shape: (DOF,)
                # Set averaged joint positions for the controlled joints
                joint_pos_des[:, right_arm_idx] = avg_joint_pos[right_arm_idx]
                joint_pos_des[:, left_arm_idx] = avg_joint_pos[left_arm_idx]
                joint_pos_des[:, left_hand_idx] = avg_joint_pos[left_hand_idx]
                joint_pos_des[:, right_hand_idx] = avg_joint_pos[right_hand_idx]
                joint_pos_des[:, waist_idx] = avg_joint_pos[waist_idx]
            else:
                joint_pos_des[:, right_arm_idx] = right_arm_action
                joint_pos_des[:, left_arm_idx] = left_arm_action
                joint_pos_des[:, left_hand_idx] = action_left_hand
                joint_pos_des[:, right_hand_idx] = action_right_hand
                joint_pos_des[:, waist_idx] = waist_action
            
        # Set joint position target
        robot.set_joint_position_target(joint_pos_des)

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
    sim.set_camera_view([-0.6, 0.93, 2.4], [0.0, 0.0, 1.0])
    # Design scene
    scene_cfg = MySceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")

    threading.Thread(target=input_thread, daemon=True).start()
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
