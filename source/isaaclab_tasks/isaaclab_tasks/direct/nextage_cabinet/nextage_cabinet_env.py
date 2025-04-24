# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from pxr import UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import sample_uniform

def couplingRuleTensor(jv: torch.Tensor) -> torch.Tensor:
    """
    Expand a [N,10] control vector
      [FF2, FF3, MF2, MF3, RF2, RF3, TH1, TH2, TH4, TH5]
    into the 16 physical joint targets expected by the hand URDF/USDA:
    """
    ff2, ff3, mf2, mf3, rf2, rf3, th1, th2, th4, th5 = jv.split(1, dim=1)
    # Coupling rules -----------------------------------------------------------
    ff1, mf1, rf1 = ff2, mf2, rf2          # proximal = middle
    ff4 = torch.zeros_like(ff2)
    mf4 = torch.zeros_like(mf2)
    rf4 = torch.zeros_like(rf2)
    # Assemble in the robot’s DOF order ----------------------------------------
    js = torch.cat(
        (
            ff4, ff3, ff2, ff1,            # rh_FFJ4-1
            mf4, mf3, mf2, mf1,            # rh_MFJ4-1
            rf4, rf3, rf2, rf1,            # rh_RFJ4-1
            th5, th4, th2, th1,            # rh_THJ5-1
        ),
        dim=1,
    )
    return js

@configclass
class NextageCabinetEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 8.3333  # 500 timesteps
    decimation = 2
    # 6 DOF right arm  + 10 controllable right-hand DOF
    action_space = 16
    # 16 pos + 16 vel + 3 (Δ grasp) + 1 (drawer q) + 1 (drawer ẋ)
    observation_space = 37
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=256, env_spacing=3.0, replicate_physics=True)

    # robot
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"scripts/my_models/nextage/nextage_env.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=12, solver_velocity_iteration_count=1
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "CHEST_JOINT0": 0.0,
                "HEAD_JOINT0": 0.0,
                "HEAD_JOINT1": 0.0,
                "LARM_JOINT0": 0.0,
                "LARM_JOINT1": 0.0,
                "LARM_JOINT2": 0.0,
                "LARM_JOINT3": 0.0,
                "LARM_JOINT4": 0.0,
                "LARM_JOINT5": 0.0,
                "lh_FFJ4": 0.0,
                "lh_FFJ3": 0.0,
                "lh_FFJ2": 0.0,
                "lh_FFJ1": 0.0,
                "lh_MFJ4": 0.0,
                "lh_MFJ3": 0.0,
                "lh_MFJ2": 0.0,
                "lh_MFJ1": 0.0,
                "lh_RFJ4": 0.0,
                "lh_RFJ3": 0.0,
                "lh_RFJ2": 0.0,
                "lh_RFJ1": 0.0,
                "lh_THJ5": 0.0,
                "lh_THJ4": 0.0,
                "lh_THJ2": 0.0,
                "lh_THJ1": 0.0,
                "RARM_JOINT0": 0.0,
                "RARM_JOINT1": 0.0,
                "RARM_JOINT2": 0.0,
                "RARM_JOINT3": 0.0,
                "RARM_JOINT4": 0.0,
                "RARM_JOINT5": 0.0,
                "rh_FFJ4": 0.0,
                "rh_FFJ3": 0.0,
                "rh_FFJ2": 0.0,
                "rh_FFJ1": 0.0,
                "rh_MFJ4": 0.0,
                "rh_MFJ3": 0.0,
                "rh_MFJ2": 0.0,
                "rh_MFJ1": 0.0,
                "rh_RFJ4": 0.0,
                "rh_RFJ3": 0.0,
                "rh_RFJ2": 0.0,
                "rh_RFJ1": 0.0,
                "rh_THJ5": 0.0,
                "rh_THJ4": 0.0,
                "rh_THJ2": 0.0,
                "rh_THJ1": 0.0,
            },
            pos=(1.4, 0.0, 0.5),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        actuators={
            "left_arm": ImplicitActuatorCfg(
                joint_names_expr=["LARM_JOINT[0-5]"],
                effort_limit=50.0,
                velocity_limit=1.5,
                stiffness=100.0,
                damping=5.0,
            ),
            "right_arm": ImplicitActuatorCfg(
                joint_names_expr=["RARM_JOINT[0-5]"],
                effort_limit=50.0,
                velocity_limit=1.5,
                stiffness=100.0,
                damping=5.0,
            ),
            # "left_hand": ImplicitActuatorCfg(
            #     joint_names_expr=["lh_.*"],
            #     effort_limit=20.0,
            #     velocity_limit=1.0,
            #     stiffness=50.0,
            #     damping=2.0,
            # ),
            # "right_hand": ImplicitActuatorCfg(
            #     joint_names_expr=["rh_.*"],
            #     effort_limit=20.0,
            #     velocity_limit=1.0,
            #     stiffness=50.0,
            #     damping=2.0,
            # ),
        },
    )

    # cabinet
    cabinet = ArticulationCfg(
        prim_path="/World/envs/env_.*/Cabinet",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd",
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0, 0.4),
            rot=(0.1, 0.0, 0.0, 0.0),
            joint_pos={
                "door_left_joint": 0.0,
                "door_right_joint": 0.0,
                "drawer_bottom_joint": 0.0,
                "drawer_top_joint": 0.0,
            },
        ),
        actuators={
            "drawers": ImplicitActuatorCfg(
                joint_names_expr=["drawer_top_joint", "drawer_bottom_joint"],
                effort_limit=87.0,
                velocity_limit=100.0,
                stiffness=10.0,
                damping=1.0,
            ),
            "doors": ImplicitActuatorCfg(
                joint_names_expr=["door_left_joint", "door_right_joint"],
                effort_limit=87.0,
                velocity_limit=100.0,
                stiffness=10.0,
                damping=2.5,
            ),
        },
    )

    # ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    action_scale = 7.5
    dof_velocity_scale = 0.1

    # reward scales
    dist_reward_scale = 1.5
    rot_reward_scale = 1.5
    open_reward_scale = 10.0
    action_penalty_scale = 0.05
    finger_reward_scale = 2.0


class NextageCabinetEnv(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: NextageCabinetEnvCfg

    def __init__(self, cfg: NextageCabinetEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        def get_env_local_pose(env_pos: torch.Tensor, xformable: UsdGeom.Xformable, device: torch.device):
            """Compute pose in env-local coordinates"""
            world_transform = xformable.ComputeLocalToWorldTransform(0)
            world_pos = world_transform.ExtractTranslation()
            world_quat = world_transform.ExtractRotationQuat()

            px = world_pos[0] - env_pos[0]
            py = world_pos[1] - env_pos[1]
            pz = world_pos[2] - env_pos[2]
            qx = world_quat.imaginary[0]
            qy = world_quat.imaginary[1]
            qz = world_quat.imaginary[2]
            qw = world_quat.real

            return torch.tensor([px, py, pz, qw, qx, qy, qz], device=device)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # create auxiliary variables for computing applied action, observations and rewards
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)

        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)

        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

        stage = get_current_stage()
        print("=== Stage prims ===")
        #for prim in stage.Traverse():
        #    print(prim.GetPath())
        hand_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/joints/RARM_JOINT5")),  # example
            self.device,
        )
        lfinger_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/joints/rh_THJ2")),  # example finger
            self.device,
        )
        rfinger_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/joints/rh_THJ1")),  # example finger
            self.device,
        )

        finger_pose = torch.zeros(7, device=self.device)
        finger_pose[0:3] = (lfinger_pose[0:3] + rfinger_pose[0:3]) / 2.0
        finger_pose[3:7] = lfinger_pose[3:7]
        hand_pose_inv_rot, hand_pose_inv_pos = tf_inverse(hand_pose[3:7], hand_pose[0:3])

        robot_local_grasp_pose_rot, robot_local_pose_pos = tf_combine(
            hand_pose_inv_rot, hand_pose_inv_pos, finger_pose[3:7], finger_pose[0:3]
        )
        robot_local_pose_pos += torch.tensor([0, 0.04, 0], device=self.device)
        self.robot_local_grasp_pos = robot_local_pose_pos.repeat((self.num_envs, 1))
        self.robot_local_grasp_rot = robot_local_grasp_pose_rot.repeat((self.num_envs, 1))

        drawer_local_grasp_pose = torch.tensor([0.3, 0.01, 0.0, 1.0, 0.0, 0.0, 0.0], device=self.device)
        self.drawer_local_grasp_pos = drawer_local_grasp_pose[0:3].repeat((self.num_envs, 1))
        self.drawer_local_grasp_rot = drawer_local_grasp_pose[3:7].repeat((self.num_envs, 1))

        self.gripper_forward_axis = torch.tensor([0, 0, -1], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.drawer_inward_axis = torch.tensor([-1, 0, 0], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.gripper_up_axis = torch.tensor([1, 0, 0], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.drawer_up_axis = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )

        self.hand_link_idx = self._robot.find_bodies("RARM_JOINT5_Link")[0][0]  # or whatever is the palm link
        self.left_finger_link_idx = self._robot.find_bodies("rh_thdistal")[0][0]
        self.right_finger_link_idx = self._robot.find_bodies("rh_mfdistal")[0][0]
        self.drawer_link_idx = self._cabinet.find_bodies("drawer_top")[0][0]

        self.robot_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.robot_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.drawer_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.drawer_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)

        # ── Right-arm DOF indices (6) ──────────────────────────────────────────
        self.arm_indices = torch.tensor(
            self._robot.find_joints("RARM_JOINT[0-5]")[0], device=self.device
        )

        # ── 10 controllable right-hand joints (input to the coupling) ──────────
        hand_ctrl_names = [
            "rh_FFJ2", "rh_FFJ3",
            "rh_MFJ2", "rh_MFJ3",
            "rh_RFJ2", "rh_RFJ3",
            "rh_THJ1", "rh_THJ2", "rh_THJ4", "rh_THJ5",
        ]
        self.hand_ctrl_indices = torch.tensor(
            self._robot.find_joints(hand_ctrl_names)[0], device=self.device
        )

        # ── 16 physical hand joints (output of the coupling) in the *exact*
        #    order returned by `couplingRuleTensor` ────────────────────────────
        hand_full_names = [
            "rh_FFJ4", "rh_FFJ3", "rh_FFJ2", "rh_FFJ1",
            "rh_MFJ4", "rh_MFJ3", "rh_MFJ2", "rh_MFJ1",
            "rh_RFJ4", "rh_RFJ3", "rh_RFJ2", "rh_RFJ1",
            "rh_THJ5", "rh_THJ4", "rh_THJ2", "rh_THJ1",
        ]
        self.hand_full_indices = torch.tensor(
            self._robot.find_joints(hand_full_names)[0], device=self.device
        )

        # ── Vector of *policy* controllable DOF (6 arm + 10 hand) ─────────────
        self.control_indices = torch.cat([self.arm_indices, self.hand_ctrl_indices])

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self._cabinet = Articulation(self.cfg.cabinet)
        self.scene.articulations["robot"] = self._robot
        self.scene.articulations["cabinet"] = self._cabinet

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # pre-physics step calls

    def _pre_physics_step(self, actions: torch.Tensor):
        """
        actions[:, :6]   → incremental velocities for the right arm
        actions[:, 6:]   → incremental velocities for the 10 hand-control DOF
        The 10-D hand signal is first integrated, then expanded to
        16-D via `couplingRuleTensor`, and finally written into the *physical*
        hand joints.
        """
        self.actions = actions.clamp(-1.0, 1.0)

        # ── Right arm (6 DOF) ─────────────────────────────────────────────────
        arm_scaled = (
            self.robot_dof_speed_scales[self.arm_indices]
            * self.dt
            * self.actions[:, :6]
            * self.cfg.action_scale
        )
        self.robot_dof_targets[:, self.arm_indices] = torch.clamp(
            self.robot_dof_targets[:, self.arm_indices] + arm_scaled,
            self.robot_dof_lower_limits[self.arm_indices],
            self.robot_dof_upper_limits[self.arm_indices],
        )

        # ── Hand control DOF (10) ─────────────────────────────────────────────
        hand_scaled_ctrl = (
            self.robot_dof_speed_scales[self.hand_ctrl_indices]
            * self.dt
            * self.actions[:, 6:]
            * self.cfg.action_scale
        )
        self.robot_dof_targets[:, self.hand_ctrl_indices] = torch.clamp(
            self.robot_dof_targets[:, self.hand_ctrl_indices] + hand_scaled_ctrl,
            self.robot_dof_lower_limits[self.hand_ctrl_indices],
            self.robot_dof_upper_limits[self.hand_ctrl_indices],
        )

        # ── Apply coupling to obtain the *16* real finger joints ──────────────
        full_hand_targets = couplingRuleTensor(
            self.robot_dof_targets[:, self.hand_ctrl_indices]
        )
        self.robot_dof_targets[:, self.hand_full_indices] = torch.clamp(
            full_hand_targets,
            self.robot_dof_lower_limits[self.hand_full_indices],
            self.robot_dof_upper_limits[self.hand_full_indices],
        )
    def _apply_action(self):
        self._robot.set_joint_position_target(self.robot_dof_targets)

    # post-physics step calls

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = self._cabinet.data.joint_pos[:, 3] > 0.39
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        # Refresh the intermediate values after the physics steps
        self._compute_intermediate_values()
        robot_left_finger_pos = self._robot.data.body_pos_w[:, self.left_finger_link_idx]
        robot_right_finger_pos = self._robot.data.body_pos_w[:, self.right_finger_link_idx]

        return self._compute_rewards(
            self.actions,
            self._cabinet.data.joint_pos,
            self.robot_grasp_pos,
            self.drawer_grasp_pos,
            self.robot_grasp_rot,
            self.drawer_grasp_rot,
            robot_left_finger_pos,
            robot_right_finger_pos,
            self.gripper_forward_axis,
            self.drawer_inward_axis,
            self.gripper_up_axis,
            self.drawer_up_axis,
            self.num_envs,
            self.cfg.dist_reward_scale,
            self.cfg.rot_reward_scale,
            self.cfg.open_reward_scale,
            self.cfg.action_penalty_scale,
            self.cfg.finger_reward_scale,
            self._robot.data.joint_pos,
        )

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        # robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids] + sample_uniform(
            -0.125,
            0.125,
            (len(env_ids), self._robot.num_joints),
            self.device,
        )
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # cabinet state
        zeros = torch.zeros((len(env_ids), self._cabinet.num_joints), device=self.device)
        self._cabinet.write_joint_state_to_sim(zeros, zeros, env_ids=env_ids)

        # Need to refresh the intermediate values so that _get_observations() can use the latest values
        self._compute_intermediate_values(env_ids)

    def _get_observations(self) -> dict:
        dof_pos_scaled = (
            2.0
            * (self._robot.data.joint_pos[:, self.control_indices] - self.robot_dof_lower_limits[self.control_indices])
            / (self.robot_dof_upper_limits[self.control_indices] - self.robot_dof_lower_limits[self.control_indices])
            - 1.0
        )
        to_target = self.drawer_grasp_pos - self.robot_grasp_pos

        obs = torch.cat(
            (
                dof_pos_scaled,  # 16
                self._robot.data.joint_vel[:, self.control_indices] * self.cfg.dof_velocity_scale,  # 16
                to_target,  # 3
                self._cabinet.data.joint_pos[:, 3].unsqueeze(-1),  # 1
                self._cabinet.data.joint_vel[:, 3].unsqueeze(-1),  # 1
            ),
            dim=-1,
        )
        return {"policy": torch.clamp(obs, -5.0, 5.0)}

    # auxiliary methods

    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        hand_pos = self._robot.data.body_pos_w[env_ids, self.hand_link_idx]
        hand_rot = self._robot.data.body_quat_w[env_ids, self.hand_link_idx]
        drawer_pos = self._cabinet.data.body_pos_w[env_ids, self.drawer_link_idx]
        drawer_rot = self._cabinet.data.body_quat_w[env_ids, self.drawer_link_idx]
        (
            self.robot_grasp_rot[env_ids],
            self.robot_grasp_pos[env_ids],
            self.drawer_grasp_rot[env_ids],
            self.drawer_grasp_pos[env_ids],
        ) = self._compute_grasp_transforms(
            hand_rot,
            hand_pos,
            self.robot_local_grasp_rot[env_ids],
            self.robot_local_grasp_pos[env_ids],
            drawer_rot,
            drawer_pos,
            self.drawer_local_grasp_rot[env_ids],
            self.drawer_local_grasp_pos[env_ids],
        )

    def _compute_rewards(
        self,
        actions,
        cabinet_dof_pos,
        nextage_grasp_pos,
        drawer_grasp_pos,
        nextage_grasp_rot,
        drawer_grasp_rot,
        nextage_lfinger_pos,
        nextage_rfinger_pos,
        gripper_forward_axis,
        drawer_inward_axis,
        gripper_up_axis,
        drawer_up_axis,
        num_envs,
        dist_reward_scale,
        rot_reward_scale,
        open_reward_scale,
        action_penalty_scale,
        finger_reward_scale,
        joint_positions,
    ):
        # distance from hand to the drawer
        d = torch.norm(nextage_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
        dist_reward = 1.0 / (1.0 + d**2)
        dist_reward *= dist_reward
        dist_reward = torch.where(d <= 0.02, dist_reward * 2, dist_reward)

        axis1 = tf_vector(nextage_grasp_rot, gripper_forward_axis)
        axis2 = tf_vector(drawer_grasp_rot, drawer_inward_axis)
        axis3 = tf_vector(nextage_grasp_rot, gripper_up_axis)
        axis4 = tf_vector(drawer_grasp_rot, drawer_up_axis)

        dot1 = (
            torch.bmm(axis1.view(num_envs, 1, 3), axis2.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        )  # alignment of forward axis for gripper
        dot2 = (
            torch.bmm(axis3.view(num_envs, 1, 3), axis4.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        )  # alignment of up axis for gripper
        # reward for matching the orientation of the hand to the drawer (fingers wrapped)
        rot_reward = 0.5 * (torch.sign(dot1) * dot1**2 + torch.sign(dot2) * dot2**2)

        # regularization on the actions (summed for each environment)
        action_penalty = torch.sum(actions**2, dim=-1)

        # how far the cabinet has been opened out
        open_reward = cabinet_dof_pos[:, 3]  # drawer_top_joint

        # penalty for distance of each finger from the drawer handle
        lfinger_dist = nextage_lfinger_pos[:, 2] - drawer_grasp_pos[:, 2]
        rfinger_dist = drawer_grasp_pos[:, 2] - nextage_rfinger_pos[:, 2]
        finger_dist_penalty = torch.zeros_like(lfinger_dist)
        finger_dist_penalty += torch.where(lfinger_dist < 0, lfinger_dist, torch.zeros_like(lfinger_dist))
        finger_dist_penalty += torch.where(rfinger_dist < 0, rfinger_dist, torch.zeros_like(rfinger_dist))

        rewards = (
            dist_reward_scale * dist_reward
            + rot_reward_scale * rot_reward
            + open_reward_scale * open_reward
            + finger_reward_scale * finger_dist_penalty
            - action_penalty_scale * action_penalty
        )

        self.extras["log"] = {
            "dist_reward": (dist_reward_scale * dist_reward).mean(),
            "rot_reward": (rot_reward_scale * rot_reward).mean(),
            "open_reward": (open_reward_scale * open_reward).mean(),
            "action_penalty": (-action_penalty_scale * action_penalty).mean(),
            "left_finger_distance_reward": (finger_reward_scale * lfinger_dist).mean(),
            "right_finger_distance_reward": (finger_reward_scale * rfinger_dist).mean(),
            "finger_dist_penalty": (finger_reward_scale * finger_dist_penalty).mean(),
        }

        # bonus for opening drawer properly
        rewards = torch.where(cabinet_dof_pos[:, 3] > 0.01, rewards + 0.25, rewards)
        rewards = torch.where(cabinet_dof_pos[:, 3] > 0.2, rewards + 0.25, rewards)
        rewards = torch.where(cabinet_dof_pos[:, 3] > 0.35, rewards + 0.25, rewards)

        return rewards

    def _compute_grasp_transforms(
        self,
        hand_rot,
        hand_pos,
        nextage_local_grasp_rot,
        nextage_local_grasp_pos,
        drawer_rot,
        drawer_pos,
        drawer_local_grasp_rot,
        drawer_local_grasp_pos,
    ):
        global_nextage_rot, global_nextage_pos = tf_combine(
            hand_rot, hand_pos, nextage_local_grasp_rot, nextage_local_grasp_pos
        )
        global_drawer_rot, global_drawer_pos = tf_combine(
            drawer_rot, drawer_pos, drawer_local_grasp_rot, drawer_local_grasp_pos
        )

        return global_nextage_rot, global_nextage_pos, global_drawer_rot, global_drawer_pos
