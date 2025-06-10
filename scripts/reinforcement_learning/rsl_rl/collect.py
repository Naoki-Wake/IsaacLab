# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""
try:
    from set_debugger import set_debugger; set_debugger()
except ImportError:
    pass

from collections import defaultdict
import argparse
import numpy as np

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--collect-data", action="store_true", default=False, help="Collect data for DP.")
parser.add_argument("--store-frames", action="store_true", default=False, help="Store frames in the dataset.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
# if args_cli.video:
args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
import shutil
import pathlib
from datetime import datetime
from collections import defaultdict

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# PLACEHOLDER: Extension template (do not remove this comment)


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env_cfg.is_training = False
    if args_cli.collect_data:
        env_cfg.is_data_collection = True
    # env_cfg.off_camera_sensor = False
    env_cfg.robot_name = "shadow"
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    base_env: DirectRLEnv = env.unwrapped

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = ppo_runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = ppo_runner.alg.actor_critic

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(
        policy_nn, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    dt = env.unwrapped.step_dt

    # reset environmen
    #
    obs, _ = env.get_observations()
    # changed to obtain extras data.
    timestep = 0
    # simulate environment

    def tensor_to_numpy(tensor):
        if isinstance(tensor, dict):
            return {k: tensor_to_numpy(v) for k, v in tensor.items()}
        elif isinstance(tensor, list):
            return np.array([tensor_to_numpy(t) for t in tensor])
        elif isinstance(tensor, torch.Tensor):
            # check invalid tensor
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                raise ValueError("Tensor contains NaN or Inf values.")
            tensor = tensor.cpu().numpy()

        if isinstance(tensor, np.ndarray):
            # replace inf to large number
            tensor = np.nan_to_num(tensor, nan=0.0, posinf=1e6, neginf=-1e6)
            # check invalid numpy array
            if np.isnan(tensor).any() or np.isinf(tensor).any():
                raise ValueError("Numpy array contains NaN or Inf values.")
        return tensor

    def list_of_dict_to_dict_of_list(list_of_dicts):
        result = defaultdict(list)
        for d in list_of_dicts:
            for k, v in d.items():
                result[k].append(v)
        return dict(result)

    results: list = []
    if args_cli.collect_data:
        action_space = base_env.cfg.action_space
        n_finger_joints = base_env.robot_cfg.n_finger_joint
        H, W = base_env.robot_cfg.camera.height, base_env.robot_cfg.camera.width

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        REPO_NAME = f"{base_env.cfg.robot_name}-grasp" # -{timestamp}"
        output_path = pathlib.Path("./data", REPO_NAME)
        if output_path.exists():
            shutil.rmtree(output_path)

        dataset = LeRobotDataset.create(
            repo_id=REPO_NAME,
            robot_type=base_env.cfg.robot_name,
            root=output_path,
            fps=10,
            features={
                "image": {
                    "dtype": "image",
                    "shape": (H, W, 3),
                    "names": ["height", "width", "channel"],
                },
                "depth": {
                    "dtype": "float32",
                    "shape": (H, W, 1),
                    "names": ["height", "width", "channel"],
                },
                "segmentation": {
                    "dtype": "uint8",
                    "shape": (H, W, 1),
                    "names": ["height", "width", "channel"],
                },
                "pointcloud": {
                    "dtype": "float32",
                    "shape": ((H * W) // 4, 3),
                    "names": ["n_points", "channel"],
                },
                "state": {
                    "dtype": "float32",
                    "shape": (n_finger_joints, ),
                    "names": ["state"],
                },
                "actions": {
                    "dtype": "float32",
                    "shape": (action_space,),
                    "names": ["actions"],
                },
            },
            image_writer_threads=0,
            image_writer_processes=0,
        )

        extras = base_env.extras.copy()
        finger_js_prev = extras["dp_ref"]["finger_js"]
        prev_frames = extras["frames"]
        dp_data_buf = dict(
            state=[[] for _ in range(args_cli.num_envs)],
            actions=[[] for _ in range(args_cli.num_envs)],
            image=[[] for _ in range(args_cli.num_envs)],
            depth=[[] for _ in range(args_cli.num_envs)],
            segmentation=[[] for _ in range(args_cli.num_envs)],
            pointcloud=[[] for _ in range(args_cli.num_envs)],
        )

    cnt = 0
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            res = dict(obs=obs, actions=actions)
            # env stepping
            obs, rewards, dones, extras = env.step(actions)
            res.update(dict(rewards=rewards, dones=dones))
            res.update(extras)
            res = tensor_to_numpy(res)

            if args_cli.collect_data:
                dp_ref = extras["dp_ref"]
                delta_eef_pos, delta_eef_rot, finger_js = dp_ref["delta_eef_pos"], dp_ref["delta_eef_rot"], dp_ref["finger_js"]
                _dp_actions = torch.cat(
                    [delta_eef_pos, delta_eef_rot, finger_js, dones[:, None]], dim=-1,
                )
                _dp_state = finger_js_prev
                _dp_data_buf_step = dict(
                    state=_dp_state,
                    actions=_dp_actions,
                    image=prev_frames["rgb"],
                    pointcloud=prev_frames["pointcloud"].astype(np.float32),
                    segmentation=prev_frames["semantic_segmentation"].astype(np.uint8),
                    depth=prev_frames["depth"],
                )
                _dp_data_buf_step: dict = tensor_to_numpy(_dp_data_buf_step)
                for key in _dp_data_buf_step.keys():
                    for env_id in range(args_cli.num_envs):
                        dp_data_buf[key][env_id].append(_dp_data_buf_step[key][env_id])

                finger_js_prev = finger_js
                prev_frames = extras["frames"]
            else:
                if not args_cli.store_frames:
                    extras.pop("frames", None)
                results.append(res)

        if args_cli.collect_data:
            done_envs = torch.nonzero(dones).squeeze(-1)
            if done_envs.numel() > 0:
                # for each fiished env, store dataset
                for env_id in done_envs:
                    for step in range(len(dp_data_buf["actions"][env_id])):
                        dp_data = dict(
                            state=dp_data_buf["state"][env_id][step],
                            actions=dp_data_buf["actions"][env_id][step],
                            image=dp_data_buf["image"][env_id][step],
                            pointcloud=dp_data_buf["pointcloud"][env_id][step],
                            segmentation=dp_data_buf["segmentation"][env_id][step],
                            depth=dp_data_buf["depth"][env_id][step],
                            task=f"grasp {base_env.cfg.grasp_type}",
                        )
                        dataset.add_frame(dp_data)
                    print("[INFO] Added frame to dataset:", env_id, dataset.meta.total_frames, dataset.meta.total_episodes)
                    dataset.save_episode()

                # reset the data buffer for the finished environments
                for env_id in done_envs:
                    for key in dp_data_buf.keys():
                        dp_data_buf[key][env_id] = []

            if dataset.meta.total_episodes > 30:
                break
        else:
            if args_cli.store_frames and cnt > 50:
                # break early as storing frames is heavy
                break
            elif cnt > 300:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)
        cnt += 1

    results_dict: dict = list_of_dict_to_dict_of_list(results)
    # close the simulator
    env.close()

    # save logs as npz file
    logs_path = os.path.join(log_dir, "logs.npz")
    np.savez(logs_path, **results_dict)
    print(f"[INFO] Saved logs to {logs_path}")

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
