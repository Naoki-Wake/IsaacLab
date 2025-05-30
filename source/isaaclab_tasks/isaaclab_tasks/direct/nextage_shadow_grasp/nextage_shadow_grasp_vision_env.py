from __future__ import annotations
import torch
import os
import cv2
import shutil
import glob
from isaaclab.sensors import CameraCfg, Camera
from isaaclab.utils import configclass
from isaaclab.sensors import Camera
from .nextage_shadow_grasp_env import NextageShadowGraspEnv, NextageShadowGraspEnvCfg
from isaaclab_tasks.utils.gpt_video_checker_buffer import ask_gpt
from isaaclab_tasks.utils.hand_utils import ShadowHandUtils, HondaHandUtils, ReferenceTrajInfo
import isaaclab.sim as sim_utils
import numpy as np
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from .robot_cfg import RobotCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from datetime import datetime

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

@configclass
class NextageShadowGraspVisionEnvCfg(NextageShadowGraspEnvCfg):
    decimation = 10
    is_training: bool = True

    episode_length_s = 3
    decimation = 8
    observation_space = 58
    state_space = 0

    # obj parameters
    # obj_half_size = 0.03  # Half size of the obj in meters
    obj_size_half = (0.035, 0.08, 0.08)  # Size of the cuboid obj in meters
    # exploration parameters
    exploration_noise = 0.2  # Noise level for actions to encourage exploration
    joint_reset_noise = 0.1  # Randomization amount for joint positions at reset

    # Memory optimization parameters
    max_envs_per_batch = 8192  # Maximum environments to process in a single batch
    history_length = 5  # Number of historical positions to track (reduced from 10)
    use_memory_efficient_mode = True  # Enable memory optimization


    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        # disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # robot
    robot_name = "shadow" # or "nextage-shadow" or "shadow"
    if "shadow" in robot_name: n_finger_joint = 16
    elif "honda" in robot_name: n_finger_joint = 18
    action_space = 6 + n_finger_joint + 1
    # The code `contact_se` is not a valid Python code snippet. It seems to be incomplete or incorrect. If you provide
    # more context or the full code snippet, I can help you understand what it is trying to do.
    off_contact_sensor = robot_name == "ur10-honda"
    env_spacing = 1.5 if robot_name == "shadow" else 3.0
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=env_spacing, replicate_physics=False)

    robot_cfg = RobotCfg(robot_name)
    robot_cfg.init_joint_pos["HEAD_JOINT1"] = 0.32
    robot = robot_cfg.get_articulation_cfg()
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=1, update_period=0.005, track_air_time=True
    )
    grasp_type = "active"  # or "passive"
    hand_util = ShadowHandUtils(grasp_type=grasp_type) if "shadow" in robot_name else HondaHandUtils(grasp_type=grasp_type)

    off_camera_sensor = True
    camera = CameraCfg(
        prim_path="/World/envs/env_.*/side_cam",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=40, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.3,0.3,1.0), rot=(0.29,0.24,0.55,0.74), convention="opengl"),
    )


class NextageShadowGraspVisionEnv(NextageShadowGraspEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: NextageShadowGraspVisionEnvCfg

    def __init__(self, cfg: NextageShadowGraspVisionEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.frames = [[] for _ in range(self.num_envs)]
        self.episode_ctr = torch.zeros(self.num_envs, dtype=torch.int32)
        self.champion_indices = torch.zeros(self.num_envs, dtype=torch.int32)
        self.gpt_ctr = 0
        self.step_in_episode = 0
        self.camera_skip = 1
        self.experiment_date = datetime.now().strftime("%Y-%m-%d-%H-%M")

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated, truncated = super()._get_dones()
        done_envs = torch.where(terminated | truncated)[0].tolist()
        done_envs_truncated = torch.where(truncated)[0].tolist() # only consider truncated cases because episodes fail if terminated

        if not self.cfg.off_camera_sensor:
            for env_id in done_envs:
                if env_id == 0:  # only write video for the first environment
                    is_truncated = env_id in done_envs_truncated
                    # print(f"debug: num of frames for env {env_id} is {len(self.frames[env_id])}, is_truncated: {is_truncated}")
                    self._write_video(env_id, is_truncated)
                    #print(f"debug: done writing video. num of frames for env {env_id} is {len(self.frames[env_id])}")
        return terminated, truncated



    def _write_video(self, env_id, is_truncated: bool = False):
        if not self.frames[env_id]:
            return
        if is_truncated:
            print(f"Writing video for env {env_id} with {len(self.frames[env_id])} frames.")
            fps = int(1.0 / (self.dt * self.camera_skip))
            ep = int(self.episode_ctr[env_id])
            output_dir = os.path.join("videos", self.experiment_date)
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, f"env{env_id:04d}_ep{ep:05d}.mp4")
            H, W, _ = self.frames[env_id][0].shape
            vw = cv2.VideoWriter(path, fourcc, fps, (W, H))
            for frame in self.frames[env_id]:
                # Move to CPU, convert to numpy, then to BGR
                # if isinstance(frame, torch.Tensor):
                frame_cpu = frame.cpu().numpy()
                bgr = cv2.cvtColor(frame_cpu, cv2.COLOR_RGB2BGR)
                vw.write(bgr)

            vw.release()
            self.episode_ctr[env_id] += 1
            champion_path = os.path.join("videos", self.experiment_date, "champion.mp4")
            result = self._compare_champion(path, champion_path)
            if result:
                print(f"[env {env_id}] new champion video saved → {path}")
                self.champion_indices[env_id] = 1
        self.frames[env_id].clear()

    def _compare_champion(self, candidate_path, champion_path="videos/champion.mp4"):
        experiment_dir = os.path.dirname(candidate_path)
        if os.path.exists(champion_path):
            candidate_win, reason = ask_gpt(
                candidate_path,
                champion_path,
                "grasp and pick up the object",
                creds_path="source/isaaclab_tasks/isaaclab_tasks/utils/auth.env",
                num_frames=5,
            )
            self.gpt_ctr += 1
            if candidate_win:
                existing = glob.glob(os.path.join(experiment_dir,"*_champion.mp4"))
                indices = [int(os.path.basename(p).split("_")[0]) for p in existing if os.path.basename(p).split("_")[0].isdigit()]
                archive_index = max(indices, default=-1) + 1
                archived_path = os.path.join(experiment_dir,f"{archive_index}_champion.mp4")
                shutil.move(champion_path, archived_path)
                with open(os.path.join(experiment_dir,f"{archive_index}_reason.txt"), "w") as f:
                    f.write(reason + f" (GPTcount-{self.gpt_ctr})")
                shutil.move(candidate_path, champion_path)
                return True
            else:
                os.remove(candidate_path)
                return False
        else:
            shutil.move(candidate_path, champion_path)
            return False

#     def _get_rewards(self) -> torch.Tensor:
#         # Refresh the intermediate values after the physics steps
#         self._compute_intermediate_values()
# 
#         d = torch.stack(
#             [torch.norm(self.current_finger_pos[finger] - self.target_finger_pos[finger], # dim=-1) for finger in self.current_finger_pos.keys()], dim=-1
#         ).mean(dim=-1)
#         dist_reward = 1.0 / (1.0 + d**2)
#         dist_reward = dist_reward * dist_reward * self.cfg.dist_reward_scale
# 
#         # check grasp stability
#         vel = torch.norm(self.hand2obj["lin_vel"], dim=-1)
#         ang_vel = torch.norm(self.hand2obj["ang_vel"], dim=-1)
#         vel_penalty = - (vel * self.cfg.vel_penalty_scale + ang_vel * self.cfg.# angvel_penalty_scale)
# 
#         obj_z_pos_reward = torch.clamp(self.obj_pos[:, 2] - self._obj_scales[:, 2], min=0.# 0) * self.cfg.z_pos_reward_scale
# 
#         # contact forces
#         is_contact = torch.max(torch.norm(self.net_contact_forces, dim=-1), dim=1)[0] > 1.0
#         contacts_reward = torch.sum(is_contact, dim=1) * self.cfg.contact_reward_scale
# 
#         # grasp is success if the object is not moving with respect to the hand in the # process of picking
#         is_grasped = torch.logical_and(vel < 0.1, self.reference_traj_info.pick_flg)
# 
#         self.is_grasped_buf[:] = is_grasped
#         self.relation_between_obj_and_hand[:] = torch.norm(self.hand2obj["pos"], dim=-1)
# 
#         grasp_success_bonus = torch.where(
#             is_grasped,
#             torch.ones_like(dist_reward) * self.cfg.grasp_reward_scale,
#             torch.zeros_like(dist_reward)
#         )
# 
#         # rewards = dist_reward + vel_penalty + grasp_success_bonus + obj_z_pos_reward + # contacts_reward
#         rewards = dist_reward + vel_penalty + obj_z_pos_reward
#         rewards_wo_bonus = rewards.clone()
#         # if self.champion_indices has an index of 1, give it a bonus
#         if torch.any(self.champion_indices == 1):
#             index_to_bonus = torch.where(self.champion_indices == 1)[0]
#             rewards[index_to_bonus] += 100.0
#             print(f"Bonus applied to envs: {index_to_bonus.tolist()}")
#             # flash the self.champion_indices
#             self.champion_indices[index_to_bonus] = 0
#             assert torch.all(self.champion_indices[index_to_bonus] == 0), f"Champion # indices should be reset to 0, but got {self.champion_indices[index_to_bonus]}"
# 
#         # print(f"rewards: {rewards}")
#         def safe_mean(x, mask=None):
#             if mask is not None:
#                 return safe_mean(x[mask])
#             if isinstance(x, torch.Tensor) and x.numel() > 0:
#                 return x.float().mean().item()
#             return x if isinstance(x, (int, float)) else 0.0
# 
#         self.extras["log"] = {
#             "rewards": safe_mean(rewards),
#             "rewards_wo_bonus": safe_mean(rewards_wo_bonus),
#             "dist_reward": safe_mean(dist_reward),
#             "grasp_reward": safe_mean(grasp_success_bonus),
#             "vel_penalty": safe_mean(vel_penalty),
#             "num_grasped": safe_mean(is_grasped, mask=self.reference_traj_info.pick_flg),
#             "z_pos_reward": safe_mean(obj_z_pos_reward),
#             "contacts_reward": safe_mean(contacts_reward),
#         }
#         return rewards
    def _get_rewards(self) -> torch.Tensor:
        # Refresh the intermediate values after the physics steps
        self._compute_intermediate_values()

        d = torch.stack(
            [torch.norm(self.current_finger_pos[finger] - self.target_finger_pos[finger], dim=-1) for finger in self.current_finger_pos.keys()], dim=-1
        ).mean(dim=-1)
        dist_reward = 1.0 / (1.0 + d**2)
        dist_reward = dist_reward * dist_reward * self.cfg.dist_reward_scale

        # check grasp stability
        vel = torch.norm(self._obj.data.root_lin_vel_w, dim=-1)
        ang_vel = torch.norm(self._obj.data.root_ang_vel_w, dim=-1)
        vel_penalty = - (vel * self.cfg.vel_penalty_scale + ang_vel * self.cfg.angvel_penalty_scale)
        vel_penalty = torch.where(~self.reference_traj_info.pick_flg, vel_penalty, torch.zeros_like(vel_penalty))

        obj_z_pos = torch.clamp(self.obj_pos[:, 2] - self.cfg.table_height - self._obj_scales[:, 2], min=0.0)
        obj_z_pos_reward = torch.where(
            self.reference_traj_info.pick_flg,
            obj_z_pos * self.cfg.z_pos_reward_scale,
            torch.zeros_like(obj_z_pos)
        )

        # contact forces
        if self.net_contact_forces is not None:
            is_contact = torch.max(torch.norm(self.net_contact_forces, dim=-1), dim=1)[0] > 1.0
        else:
            is_contact = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.bool)
        contacts_reward = torch.sum(is_contact, dim=1) * self.cfg.contact_reward_scale

        rel_vel = torch.norm(self.hand2obj["lin_vel"], dim=-1)
        # grasp is success if the object is not moving with respect to the hand in the process of picking
        is_grasped = torch.logical_and(rel_vel < 0.1, self.reference_traj_info.pick_flg)
        is_grasped_full = torch.logical_and(is_grasped, obj_z_pos > self.cfg.height_bonus_threshold * 0.8)
        is_grasped_half =  torch.logical_and(is_grasped, obj_z_pos > self.cfg.height_bonus_threshold / 2 * 0.8)

        self.is_grasped_buf[:] = is_grasped_full


        grasp_success_bonus = torch.where(
            is_grasped_full,
            torch.ones_like(is_grasped_full) * torch.clamp(obj_z_pos / self.cfg.height_bonus_threshold, max=1) * self.cfg.grasp_reward_scale,
            torch.zeros_like(is_grasped_full)
        )

        rewards = dist_reward + vel_penalty + grasp_success_bonus + obj_z_pos_reward + contacts_reward
        # print(f"rewards: {rewards}")
        def safe_mean(x, mask=None):
            if mask is not None:
                return safe_mean(x[mask])
            if isinstance(x, torch.Tensor) and x.numel() > 0:
                return x.float().mean().item()
            return x if isinstance(x, (int, float)) else 0.0

        self.extras["success"] = is_grasped_full
        self.extras["log"] = {
            "rewards": safe_mean(rewards),
            "dist_reward": safe_mean(dist_reward),
            "grasp_reward": safe_mean(grasp_success_bonus),
            "vel_penalty": safe_mean(vel_penalty),
            "num_grasped": safe_mean(is_grasped_full, mask=self.reference_traj_info.pick_flg),
            "num_grasped_half": safe_mean(is_grasped_half, mask=self.reference_traj_info.pick_flg),
            "z_pos_reward": safe_mean(obj_z_pos_reward, mask=self.reference_traj_info.pick_flg),
            "contacts_reward": safe_mean(contacts_reward),
        }
        return rewards
    def _get_observations(self):
        obs = super()._get_observations()
        # print(f"debug1: episode_length_buf: {self.episode_length_buf[0]}, num of self.frames: {len(self.frames[0])}")
        if not self.cfg.off_camera_sensor:
            # Grab the raw RGB tensor (N, H, W, 4) on CUDA, slice off alpha
            rgb_gpu = self._camera.data.output["rgb"][..., :3]
            rgb_gpu = rgb_gpu.to(torch.uint8)
            for env_id in range(self.num_envs):
                if env_id == 0:
                    self.frames[env_id].append(rgb_gpu[env_id].clone())
        # print(f"debug3: num of self.frames: {len(self.frames[0])}")
        return obs