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

task_description_input = ["Move the right arm upwards."]  # default
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
from pxr import UsdPhysics

def set_joint_drive_parameters(sim: sim_utils.SimulationContext, joint_path: str, stiffness=500.0, damping=50.0, max_force=1000.0):
    stage = sim.stage
    joint_prim = stage.GetPrimAtPath(joint_path)
    if not joint_prim.IsValid():
        print(f"[WARNING] Invalid prim at {joint_path}")
        return
    drive_api = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
    drive_api.CreateTypeAttr().Set("position")       # corrected method
    drive_api.CreateStiffnessAttr().Set(stiffness)
    drive_api.CreateDampingAttr().Set(damping)
    drive_api.CreateMaxForceAttr().Set(max_force)
    print(f"[INFO] Set drive params for {joint_path}")


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
    # Replace with your joint path
    joint_prim_path = "/World/envs/env_0/Robot/joints/right_wrist_yaw_joint"
    #set_joint_drive_parameters(sim, joint_prim_path, stiffness=10000.0, damping=500.0, max_force=100.0)
    joint_prim_path = "/World/envs/env_0/Robot/joints/left_wrist_yaw_joint"
    #set_joint_drive_parameters(sim, joint_prim_path, stiffness=10000.0, damping=500.0, max_force=100.0)
    joint_prim_path = "/World/envs/env_0/Robot/joints/R_pinky_proximal_joint"

    #get_joint_idx(scene, "R_pinky_proximal_joint"),
    #get_joint_idx(scene, "R_ring_proximal_joint"),
    #get_joint_idx(scene, "R_middle_proximal_joint"),
    #get_joint_idx(scene, "R_index_proximal_joint"),
    #get_joint_idx(scene, "R_thumb_proximal_pitch_joint"),
    #get_joint_idx(scene, "L_thumb_proximal_yaw_joint"),
    # right_arm indexes
    import numpy as np
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
    print("right_hand_idx", right_hand_idx)
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

    # load replay data
    import pandas as pd
    import os
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "gt_action_joints_across_time.csv"), header=None)
    #gt_action_joints = np.concatenate(
    #    [
    #        gt_action_right_arm,
    #        gt_action_left_arm,
    #        gt_action_waist,
    #        gt_action_right_hand,
    #        gt_action_left_hand,
    #    ],
    #    axis=0,
    #)
    # import pdb; pdb.set_trace()
    # get the first row as example
    all_actions = df.to_numpy(dtype=np.float32)
    # print(example)
    # print(example[0])
    # Simulation loop
    read_idx = 150
    image_buffer = []
    while simulation_app.is_running():
        if count == 0:
            scene.reset()
            robot.reset()
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(position=joint_pos, velocity=joint_vel)
        if count%1 == 0:
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
            action = all_actions[read_idx]
            read_idx += 1
            image_buffer.append(ego_view)
            print("read_idx", read_idx)

            if read_idx >= len(all_actions):
                import cv2
                import numpy as np
                # Save the images as a video (mp4)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # or use 'XVID' for .avi
                # Convert images to uint8 if they are not already
                images = [image.astype(np.uint8) for image in image_buffer]
                # Ensure images are in the correct format (BGR) for OpenCV
                # color conversion
                images = [cv2.cvtColor(image, cv2.COLOR_RGB2BGR) for image in images]
                # Create a video writer object
                # Assuming all images are the same size, use the first image's dimensions
                video_writer = cv2.VideoWriter("ego_view_images.mp4", fourcc, 30, (images[0].shape[1], images[0].shape[0]))

                for image in images:
                    video_writer.write(image)

                video_writer.release()
                simulation_app.close()
                exit(0)
            # print("action", action)
            assert action is not None, "Action is None. Please check the server connection."
            action_right_arm = action[0:7]
            action_left_arm = action[7:14]
            action_waist = action[14:17]
            action_right_hand = action[17:23]
            action_left_hand = action[23:29]
 
            # clamp action and flip values except for the last one (thumb_proximal_pitch_joint)
            action_right_hand[0:4] = -1* action_right_hand[0:4]
            action_right_hand[0:4] = np.clip(action_right_hand[0:4], -math.pi/2, 0.0)
            action_right_hand[4] = np.clip(action_right_hand[4], 0.0, math.pi/3)
            action_right_hand[5] = -1* action_right_hand[5]
            action_right_hand[5] = np.clip(action_right_hand[5], -math.pi/2, 0.0)
            action_left_hand[0:4] = -1* action_left_hand[0:4]
            action_left_hand[0:4] = np.clip(action_left_hand[0:4], -math.pi/2, 0.0)
            action_left_hand[4] = np.clip(action_left_hand[4], 0.0, math.pi/3)
            action_left_hand[5] = -1* action_left_hand[5]
            action_left_hand[5] = np.clip(action_left_hand[5], -math.pi/2, 0.0)


        print("action_right_hand (clamped)", action_right_hand)
        #joint_pos_des = robot.data.default_joint_pos.clone()
        if count == 0:
            joint_pos_des = robot.data.default_joint_pos.clone()
        else:
            joint_pos_des = robot.data.default_joint_pos.clone() #robot.data.joint_pos.clone()
        # import pdb; pdb.set_trace()
        if output_status:
            print("joint_pos_des", joint_pos_des)
        right_arm_action = torch.tensor(action_right_arm, device=joint_pos_des.device, dtype=joint_pos_des.dtype)
        left_arm_action = torch.tensor(action_left_arm, device=joint_pos_des.device, dtype=joint_pos_des.dtype)
        left_hand_action = torch.tensor(action_left_hand, device=joint_pos_des.device, dtype=joint_pos_des.dtype)
        right_hand_action = torch.tensor(action_right_hand, device=joint_pos_des.device, dtype=joint_pos_des.dtype)
        waist_action = torch.tensor(action_waist, device=joint_pos_des.device, dtype=joint_pos_des.dtype)
        #right_hand_action = -right_hand_action
        #right_hand_action = torch.clamp(right_hand_action, -math.pi / 2, 0.0)
        #right_hand_action[:-1] = -right_hand_action[:-1]
        #left_hand_action = -left_hand_action
        #left_hand_action = torch.clamp(left_hand_action, -math.pi / 2, 0.0)
        #left_hand_action[:-1] = -left_hand_action[:-1]
        # set joint pos
        joint_pos_des[:, right_arm_idx] = right_arm_action
        joint_pos_des[:, left_arm_idx] = left_arm_action
        joint_pos_des[:, left_hand_idx] = left_hand_action
        joint_pos_des[:, right_hand_idx] = right_hand_action
        #print(joint_pos_des[:, 51])
        #print(right_hand_idx[debug_idx])
        #print(right_hand_action[debug_idx])
        joint_pos_des[:, waist_idx] = waist_action
        #print("right_hand_action", right_hand_action)
        #print(right_arm_idx)
        #print(right_hand_idx)
        # Apply actions
        # Set joint position target
        # tilt head
        joint_pos_des[:, head_pitch_idx] = math.pi/15
        robot.set_joint_position_target(joint_pos_des)
        diff = robot.data.joint_pos.clone() - joint_pos_des
        print(diff[:, right_hand_idx])
        #robot.write_data_to_sim()
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
