# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import statistics
import time
import torch
from typing import List, Union
from collections import deque

import rsl_rl
from rsl_rl.algorithms import PPO, Distillation
from rsl_rl.env import VecEnv
from rsl_rl.modules import (
    ActorCritic,
    ActorCriticRecurrent,
    EmpiricalNormalization,
    StudentTeacher,
    StudentTeacherRecurrent,
)
from rsl_rl.utils import store_code_state
from rsl_rl.runners import OnPolicyRunner

class OnPolicyRunnerMultiPolicy(OnPolicyRunner):
    """On-policy runner for training and evaluation."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        # check if multi-gpu is enabled
        self._configure_multi_gpu()

        # resolve training type depending on the algorithm
        if self.alg_cfg["class_name"] == "PPO":
            self.training_type = "rl"
        elif self.alg_cfg["class_name"] == "Distillation":
            self.training_type = "distillation"
        else:
            raise ValueError(f"Training type not found for algorithm {self.alg_cfg['class_name']}.")

        # resolve dimensions of observations
        obs, extras = self.env.get_observations()

        self.n_policy = 1  # default to single policy
        if isinstance(obs, list):
            num_obs = [o.shape[1] for o in obs]
            self.n_policy = len(obs)
        else:
            num_obs = [obs.shape[1]]

        # resolve type of privileged observations
        if self.training_type == "rl":
            if "critic" in extras["observations"]:
                self.privileged_obs_type = "critic"  # actor-critic reinforcement learnig, e.g., PPO
            else:
                self.privileged_obs_type = None
        if self.training_type == "distillation":
            if "teacher" in extras["observations"]:
                self.privileged_obs_type = "teacher"  # policy distillation
            else:
                self.privileged_obs_type = None

        # resolve dimensions of privileged observations
        if self.privileged_obs_type is not None:
            raise NotImplementedError("This should be implemented in the future.")
            num_privileged_obs = extras["observations"][self.privileged_obs_type].shape[1]
        else:
            num_privileged_obs = num_obs

        # evaluate the policy class
        policy_class = eval(self.policy_cfg.pop("class_name"))
        policy = [
            policy_class(
                num_obs[i], num_privileged_obs[i], self.env.num_actions, **self.policy_cfg
            ).to(self.device) for i in range(self.n_policy)
        ]

        # resolve dimension of rnd gated state
        if "rnd_cfg" in self.alg_cfg and self.alg_cfg["rnd_cfg"] is not None:
            # check if rnd gated state is present
            rnd_state = extras["observations"].get("rnd_state")
            if rnd_state is None:
                raise ValueError("Observations for the key 'rnd_state' not found in infos['observations'].")
            # get dimension of rnd gated state
            num_rnd_state = rnd_state.shape[1]
            # add rnd gated state to config
            self.alg_cfg["rnd_cfg"]["num_states"] = num_rnd_state
            # scale down the rnd weight with timestep (similar to how rewards are scaled down in legged_gym envs)
            self.alg_cfg["rnd_cfg"]["weight"] *= env.unwrapped.step_dt

        # if using symmetry then pass the environment config object
        if "symmetry_cfg" in self.alg_cfg and self.alg_cfg["symmetry_cfg"] is not None:
            # this is used by the symmetry function for handling different observation terms
            self.alg_cfg["symmetry_cfg"]["_env"] = env

        # initialize algorithm
        alg_class = eval(self.alg_cfg.pop("class_name"))
        self.alg: List = [
            alg_class(
                policy[i], device=self.device, **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg
            ) for i in range(self.n_policy)
        ]

        # store training configuration
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.empirical_normalization = self.cfg["empirical_normalization"]
        if self.empirical_normalization:
            self.obs_normalizer = [EmpiricalNormalization(shape=[num_obs[i]], until=1.0e8).to(self.device) for i in range(self.n_policy)]
            self.privileged_obs_normalizer = [EmpiricalNormalization(shape=[num_privileged_obs[i]], until=1.0e8).to(
                self.device
            ) for i in range(self.n_policy)]
        else:
            self.obs_normalizer = [torch.nn.Identity().to(self.device) for i in range(self.n_policy)] # no normalization
            self.privileged_obs_normalizer = [torch.nn.Identity().to(self.device) for i in range(self.n_policy)] # no normalization

        # init storage and model
        for i in range(self.n_policy):
            self.alg[i].init_storage(
                self.training_type,
                self.env.num_envs,
                self.num_steps_per_env,
                [num_obs[i]],
                [num_privileged_obs[i]],
                [self.env.num_actions],
            )

        # Decide whether to disable logging
        # We only log from the process with rank 0 (main process)
        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0
        # Logging
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]

    def load(self, path: List[str], load_optimizer: bool = True):
        assert isinstance(path, list), "Path must be a list of paths for each policy."
        if len(path) == 1:
            path = [path[0]] * self.n_policy  # if only one path is given, use it for all policies
        loaded_dicts = []
        for i in range(self.n_policy):
            loaded_dict = torch.load(path[i], weights_only=False)
            # -- Load model
            resumed_training = self.alg[i].policy.load_state_dict(loaded_dict["model_state_dict"])
            # -- Load RND model if used
            if self.alg[i].rnd:
                self.alg[i].rnd.load_state_dict(loaded_dict["rnd_state_dict"])
            # -- Load observation normalizer if used
            if self.empirical_normalization:
                if resumed_training:
                    # if a previous training is resumed, the actor/student normalizer is loaded for the actor/student
                    # and the critic/teacher normalizer is loaded for the critic/teacher
                    self.obs_normalizer[i].load_state_dict(loaded_dict["obs_norm_state_dict"])
                    self.privileged_obs_normalizer[i].load_state_dict(loaded_dict["privileged_obs_norm_state_dict"])
                else:
                    # if the training is not resumed but a model is loaded, this run must be distillation training following
                    # an rl training. Thus the actor normalizer is loaded for the teacher model. The student's normalizer
                    # is not loaded, as the observation space could differ from the previous rl training.
                    self.privileged_obs_normalizer[i].load_state_dict(loaded_dict["obs_norm_state_dict"])
            # -- load optimizer if used
            if load_optimizer and resumed_training:
                # -- algorithm optimizer
                self.alg[i].optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
                # -- RND optimizer if used
                if self.alg[i].rnd:
                    self.alg[i].rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
            # -- load current learning iteration
            if resumed_training:
                self.current_learning_iteration = loaded_dict["iter"]
            loaded_dicts.append(loaded_dict["infos"])
        return loaded_dicts

    def get_inference_policy(self, device=None):
        def make_policy(policy, normalizer=None):
            if normalizer is not None:
                return lambda x: policy(normalizer(x))
            else:
                return policy

        policy_list = []
        self.eval_mode()  # evaluation mode

        for i in range(self.n_policy):
            alg = self.alg[i]
            policy = alg.policy.act_inference

            if device is not None:
                alg.policy.to(device)

            if self.cfg.get("empirical_normalization", False):
                normalizer = self.obs_normalizer[i]
                if device is not None:
                    normalizer.to(device)
            else:
                normalizer = None

            policy_list.append(make_policy(policy, normalizer))

        return policy_list

    def eval_mode(self):
        # -- PPO
        for i in range(self.n_policy):
            self.alg[i].policy.eval()
            # -- RND
            if self.alg[i].rnd:
                self.alg[i].rnd.eval()
            # -- Normalization
            if self.empirical_normalization:
                self.obs_normalizer[i].eval()
                self.privileged_obs_normalizer[i].eval()
