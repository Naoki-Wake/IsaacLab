from __future__ import annotations
import torch
import os
import cv2
import shutil
import glob
from isaaclab.sensors import CameraCfg, Camera
from isaaclab.sensors import TiledCameraCfg, TiledCamera
from isaaclab.utils import configclass
from .nextage_shadow_grasp_env import NextageShadowGraspEnv, NextageShadowGraspEnvCfg
from isaaclab_tasks.utils.gpt_video_checker_progress import ask_gpt
from isaaclab_tasks.utils.gpt_video_checker_progress_phi4 import ask_phi4
from isaaclab_tasks.utils.hand_utils import ShadowHandUtils, HondaHandUtils, ReferenceTrajInfo
import isaaclab.sim as sim_utils
import numpy as np
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from .robot_cfg import RobotCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from datetime import datetime
from .robot_cfg import get_robot_cfg, RobotCfg
from isaaclab.utils.math import subtract_frame_transforms, quat_error_magnitude, quat_mul, quat_conjugate, quat_from_euler_xyz


fourcc = cv2.VideoWriter_fourcc(*"mp4v")

@configclass
class NextageShadowGraspVisionEnvCfg(NextageShadowGraspEnvCfg):
    robot_name = "shadow-wake"

def load_credentials(env_file: str) -> dict:
    """
    Load credentials from a .env file or environment.
    """
    try:
        from dotenv import dotenv_values
    except ImportError:
        dotenv_values = lambda f: os.environ

    creds = dotenv_values(env_file)
    required = [
        "OPENAI_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_DEPLOYMENT_NAME",
    ]
    for key in required:
        creds.setdefault(key, os.getenv(key, ""))
    return creds


def init_vlm_client(creds: dict):
    from openai import OpenAI, AzureOpenAI
    """
    Initialize the GPT-4 Vision client for Azure or OpenAI.
    """
    if creds.get("AZURE_OPENAI_API_KEY"):
        client = AzureOpenAI(
            api_key=creds["AZURE_OPENAI_API_KEY"],
            azure_endpoint=creds["AZURE_OPENAI_ENDPOINT"],
            api_version="2024-02-01"
        )
        return client, {"model": creds["AZURE_OPENAI_DEPLOYMENT_NAME"]}
    client = OpenAI(api_key=creds["OPENAI_API_KEY"])
    return client, {"model": "gpt-4o"}

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
        self.robot_cfg: RobotCfg = get_robot_cfg(cfg.robot_name, cfg.grasp_type, cfg.is_training)
        cfg.action_space = self.robot_cfg.action_space
        super().__init__(cfg, render_mode, **kwargs)
        self.gpt_frames = [[] for _ in range(self.num_envs)]
        self.gpt_progress = torch.zeros(self.num_envs, dtype=torch.float32)
        self.gpt_ctr = 0
        self.step_in_episode = 0
        self.camera_skip = 1
        self.experiment_date = datetime.now().strftime("%Y-%m-%d-%H-%M")
        self.creds = load_credentials("source/isaaclab_tasks/isaaclab_tasks/utils/auth.env")
        self.client, self.client_params = init_vlm_client(self.creds)
        self.envs_to_save = None

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated, truncated = super()._get_dones()
        done_envs = torch.where(terminated | truncated)[0].tolist()
        done_envs_truncated = torch.where(truncated)[0].tolist() # only consider truncated cases because episodes fail if terminated
        self.envs_to_save = done_envs_truncated
        if not self.robot_cfg.off_camera_sensor:
            for env_id in done_envs:
                # if env_id == 0:  # only write video for the first environment
                is_truncated = env_id in done_envs_truncated
                # print(f"debug: num of frames for env {env_id} is {len(self.gpt_frames[env_id])}, is_truncated: {is_truncated}")
                self._write_video(env_id, is_truncated)
                #print(f"debug: done writing video. num of frames for env {env_id} is {len(self.gpt_frames[env_id])}")
        return terminated, truncated



    def _write_video(self, env_id, is_truncated: bool = False):
        if not self.gpt_frames[env_id]:
            return
        if is_truncated:
            num_frames_to_sample = 3
            total = len(self.gpt_frames[env_id])
            if num_frames_to_sample == 1:
                    indices = [total - 1]
            else:
                if total < num_frames_to_sample:
                    num_frames_to_sample = total
                indices = np.linspace(0, total - 1, num_frames_to_sample, endpoint=True, dtype=int)
                indices = np.unique(indices)  # avoid duplicates in short videos
            sampled_frames = [cv2.cvtColor(self.gpt_frames[env_id][i].cpu().numpy(), cv2.COLOR_BGR2RGB) for i in indices]
            progress = ask_gpt(
                self.client, self.client_params,
                sampled_frames,
            )
            #progress = ask_phi4(
            #    sampled_frames,
            #)
            self.gpt_ctr += 1
            # if progress is high, save the video
            # if progress > 0.0:
            #     fps  = int(1.0 / (self.dt * self.camera_skip))
            #     if not os.path.exists(f"./videos/{self.experiment_date}"):
            #         os.makedirs(f"./videos/{self.experiment_date}")
            #     path = f"./videos/{self.experiment_date}/env{env_id:04d}_counter_{self.gpt_ctr:04d}.mp4"
# 
            #     H, W, _ = self.gpt_frames[env_id][0].shape
            #     vw = cv2.VideoWriter(path, fourcc, fps, (W, H))   # open writer
            #     print(f"writing {len(self.gpt_frames[env_id])} frames to {path}")
            #     for f in self.gpt_frames[env_id]:
            #         f_cpu = f.cpu().numpy()  # move to CPU and convert to numpy
            #         f_cpu = cv2.cvtColor(f_cpu, cv2.COLOR_BGR2RGB)  # convert from RGBA to RGB
            #         vw.write(f_cpu)  # write frame
            #     vw.release()
            #     # import pdb; pdb.set_trace()
            self.gpt_progress[env_id] = progress
        else:
            self.gpt_progress[env_id] = 0.0
        self.gpt_frames[env_id].clear()

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

        obj_rot = quat_error_magnitude(self.obj_initial_rot, self.obj_rot)
        obj_rot_penalty = -self.cfg.obj_rot_penalty_scale * obj_rot

        obj_z_pos = torch.clamp(self.obj_pos[:, 2] - self.cfg.table_height - self._obj_scales[:, 2], min=0.0)
        obj_z_pos_reward = torch.where(
            self.reference_traj_info.pick_flg,
            obj_z_pos * self.cfg.z_pos_reward_scale,
            torch.zeros_like(obj_z_pos)
        )

        # contact forces
        if self.net_contact_forces is not None:
            is_contact = torch.max(torch.norm(self.net_contact_forces, dim=-1), dim=1)[0] > 1.0
            obj_internal_force = torch.sum(-self.net_contact_forces[:, 0], dim=1)  # (N, 4, 3) -> (N, 3)
            obj_internal_torque = torch.sum(
                torch.cross(self.contact_position - self.obj_pos[:, None, :], -self.net_contact_forces[:, 0]), dim=1
            )
            force_penalty = -torch.norm(obj_internal_force, dim=-1) * self.cfg.force_penalty_scale - torch.norm(obj_internal_torque, dim=-1) * self.cfg.force_penalty_scale
            # if self.reference_traj_info.pick_flg.any():
            #     import pdb; pdb.set_trace()
        else:
            is_contact = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.bool)
            force_penalty = torch.zeros((self.num_envs,), device=self.device)

        contacts_reward = torch.sum(is_contact, dim=1) * self.cfg.contact_reward_scale

        rel_vel = torch.norm(self.hand2obj["lin_vel"], dim=-1)
        # grasp is success if the object is not moving with respect to the hand in the process of picking
        # is_grasped = torch.logical_and(
        #     obj_rot < self.cfg.obj_rot_threshold,
        #     torch.logical_and(rel_vel < self.cfg.rel_obj_vel_threshold, self.reference_traj_info.pick_flg)
        # )
        is_grasped = torch.logical_and(rel_vel < self.cfg.rel_obj_vel_threshold, self.reference_traj_info.pick_flg)
        is_grasped_full = torch.logical_and(is_grasped, obj_z_pos > self.cfg.height_bonus_threshold * 0.8)
        is_grasped_half = torch.logical_and(is_grasped, obj_z_pos > self.cfg.height_bonus_threshold / 2 * 0.8)

        self.is_grasped_buf[:] = is_grasped_full

        grasp_success_bonus = torch.where(
            is_grasped_full,
            torch.ones_like(is_grasped_full) * torch.clamp(obj_z_pos / self.cfg.height_bonus_threshold, max=1) * self.cfg.grasp_reward_scale,
            torch.zeros_like(is_grasped_full)
        )

        rewards = dist_reward + vel_penalty + grasp_success_bonus + obj_z_pos_reward + contacts_reward + obj_rot_penalty#  + force_penalty
        rewards_wo_bonus = rewards.clone()
        
        progress_coefficient = 1.0
        # add progress_coefficient*gpt_progress if nonzero
        for env_id in range(self.num_envs):
            additional_bonus = progress_coefficient * self.gpt_progress[env_id]
            if additional_bonus > 0:
                rewards[env_id] += additional_bonus
                # print(f"Bonus applied to env {env_id}: {additional_bonus:.2f}")

        # print(f"rewards: {rewards}")
        def safe_mean(x, mask=None):
            if mask is not None:
                return safe_mean(x[mask])
            if isinstance(x, torch.Tensor) and x.numel() > 0:
                return x.float().mean().item()
            return x if isinstance(x, (int, float)) else 0.0

        self.extras["success"] = is_grasped_full
        self.extras["contact_info"] = {
            "is_contact": is_contact,
            "net_contact_forces": self.net_contact_forces,
            "contact_positions": self.contact_position,
            "obj_pos": self.obj_pos,
        }
        self.extras["log"] = {
            "rewards": safe_mean(rewards),
            "rewards_wo_bonus": safe_mean(rewards_wo_bonus),
            "dist_reward": safe_mean(dist_reward),
            "grasp_reward": safe_mean(grasp_success_bonus),
            "vel_penalty": safe_mean(vel_penalty),
            "obj_rot_penalty": safe_mean(obj_rot_penalty),
            "num_grasped": safe_mean(is_grasped_full, mask=self.reference_traj_info.pick_flg),
            "num_grasped_half": safe_mean(is_grasped_half, mask=self.reference_traj_info.pick_flg),
            "z_pos_reward": safe_mean(obj_z_pos_reward, mask=self.reference_traj_info.pick_flg),
            "contacts_reward": safe_mean(contacts_reward),
            "force_penalty": safe_mean(force_penalty),
        }
        for i in self.envs_to_save:
            save_log = {
                "rewards": rewards[i].item(),
                "rewards_wo_bonus": rewards_wo_bonus[i].item(),
                "GPT_reward": progress_coefficient * self.gpt_progress[i].item(),
                "dist_reward": dist_reward[i].item(),
                "grasp_reward": grasp_success_bonus[i].item(),
                "vel_penalty": vel_penalty[i].item(),
                "obj_rot_penalty": obj_rot_penalty[i].item(),
                "num_grasped": is_grasped_full[i].item(),
                "num_grasped_half": is_grasped_half[i].item(),
                "z_pos_reward": obj_z_pos_reward[i].item(),
                "contacts_reward": contacts_reward[i].item(),
                "force_penalty": force_penalty[i].item(),
                "gpt_progress": self.gpt_progress[i].item(),
            }
            if not os.path.exists(f"./LLM_logs/{self.experiment_date}"):
                os.makedirs(f"./LLM_logs/{self.experiment_date}")
            path = f"./LLM_logs/{self.experiment_date}/env{i:04d}_counter_{self.gpt_ctr:04d}.json"
            with open(path, "w") as f:
                import json
                json.dump(save_log, f, indent=4)
            #import pdb; pdb.set_trace()
        # clear self.gpt_progress[env_id]
        self.gpt_progress[:] = 0.0
        return rewards


    def _get_observations(self):
        obs = super()._get_observations()
        # print(f"debug1: episode_length_buf: {self.episode_length_buf[0]}, num of self.gpt_frames: {len(self.gpt_frames[0])}")
        if not self.robot_cfg.off_camera_sensor:
            # Grab the raw RGB tensor (N, H, W, 4) on CUDA, slice off alpha
            rgb_gpu = self._camera.data.output["rgb"][..., :3]
            rgb_gpu = rgb_gpu.to(torch.uint8)
            for env_id in range(self.num_envs):
                #if env_id == 0:
                self.gpt_frames[env_id].append(rgb_gpu[env_id].clone())
        # print(f"debug3: num of self.gpt_frames: {len(self.gpt_frames[0])}")
        return obs