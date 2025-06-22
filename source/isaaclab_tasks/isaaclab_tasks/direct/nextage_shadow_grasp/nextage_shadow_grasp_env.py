# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
#from set_debugger import set_debugger; set_debugger()
import torch
import numpy as np
import math
import glob
from scipy.spatial.transform import Rotation as R
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
# from pxr import Usd, UsdGeom, Ar

from pxr import UsdGeom, Gf, Usd, Ar
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation

from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

from isaaclab.assets import RigidObject, RigidObjectCfg

from isaaclab.managers import SceneEntityCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import subtract_frame_transforms, quat_error_magnitude, quat_mul, quat_conjugate, quat_from_euler_xyz
from isaaclab_tasks.utils._math import euler_xyz_from_quat

#from isaaclab_tasks.utils.hand_utils import ReferenceTrajInfoMulti, CWPPredictor
from isaaclab_tasks.utils.hand_utils import CWPPredictor
from isaaclab_tasks.utils.traj_utils import ReferenceTrajInfoMulti
from isaaclab_tasks.utils.compute_relative_state import compute_object_state_in_hand_frame
from .events import EventCfg, create_grasp_event_cfg
from .robot_cfg import get_robot_cfg, RobotCfg
from .env_cfg import get_obj_cfg, ObjCfg
from isaaclab.sensors import CameraCfg, Camera
from isaaclab.sensors import TiledCamera, TiledCameraCfg, save_images_to_file
from isaaclab.sensors.camera.utils import create_pointcloud_from_depth

import cv2

import omni.usd
import os

torch.autograd.set_detect_anomaly(True)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

@configclass
class NextageShadowGraspEnvCfg(DirectRLEnvCfg):
    # env
    mode = "train"
    # "trian or "eval" or "collect"

    episode_length_s = 3
    decimation = 16
    observation_space = 58
    state_space = 0
    action_space = 0

    # obj parameters
    # obj_half_size = 0.03  # Half size of the obj in meters
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
    env_spacing = 1.5
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=env_spacing, replicate_physics=False)
    grasp_type = "active"  # or "passive"
    object_type = "superquadric"  # or "ycb"

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
    marker = VisualizationMarkersCfg(
        prim_path="/World/Visuals/Markers",
        markers={
            # "target_sphere": sim_utils.SphereCfg(
            #     radius=0.02,
            #     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
            # ),
            "contact_arrow": sim_utils.ConeCfg(
                radius=0.04,
                height=0.04,
                axis="Z",
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0)),
            ),
        }
    )
    # action_scale = [0.0 for _ in range(action_space)] # debug
    dof_velocity_scale = 0.1

    # reward scales - improved for better lifting
    dist_reward_scale = 0.1         # Increased focus on positioning
    action_penalty_scale = 0.05
    grasp_reward_scale = 50.0       # Increased reward for proper grasping
    finger_position_scale = 2.0     # Increased reward for good finger positioning
    preposition_reward_scale = 1.0
    z_velocity_scale = 5.0          # Scale for z-velocity bonus during lifting
    vel_penalty_scale = 1.0         # Penalty for excessive velocity
    angvel_penalty_scale = 0.1      # Penalty for excessive angular velocity
    z_pos_reward_scale = 10.0       # Scale for z-position reward
    contact_reward_scale = 0.5      # Scale for contact reward
    force_penalty_scale = 1.0      # Penalty for excessive force
    height_bonus_threshold = 0.05   # Height threshold for bonus
    obj_rot_penalty_scale = 1.0     # Penalty for object rotation deviation
    obj_rot_threshold = math.radians(15)  # Threshold for object rotation deviation
    obj_rot_threshold = math.radians(180)  # Threshold for object rotation deviation
    rel_obj_vel_threshold = 0.1  # Threshold for relative object velocity


class NextageShadowGraspEnv(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: NextageShadowGraspEnvCfg

    def __init__(self, cfg: NextageShadowGraspEnvCfg, render_mode: str | None = None, **kwargs):
        self.robot_cfg: RobotCfg = get_robot_cfg(cfg.robot_name, cfg.grasp_type, cfg.mode)
        self.env_cfg: ObjCfg = get_obj_cfg(cfg.object_type, cfg.grasp_type)
        cfg.action_space = self.robot_cfg.action_space
        cfg.events = create_grasp_event_cfg(base_obj_size=self.env_cfg.obj_size_half, object_type=cfg.object_type, grasp_type=cfg.grasp_type, mode=cfg.mode)

        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # create auxiliary variables for computing applied action, observations and rewards
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)

        self.action_scale = torch.tensor(self.robot_cfg.action_scale, device=self.device)
        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

        stage = get_current_stage()
        self.hand_util = self.robot_cfg.hand_util

        ### self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(".*THIGH")
        # self.robot_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        # self.robot_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)

        # Initialize obj position and rotation for all environments
        self.obj_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.obj_rot = torch.zeros((self.num_envs, 4), device=self.device)

        # Track initial obj positions
        self.obj_initial_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.obj_initial_rot = torch.zeros((self.num_envs, 4), device=self.device)

        self.gravity = torch.tensor(self.sim.cfg.gravity, device=self.device, dtype=torch.float32)

        if not self.robot_cfg.off_camera_sensor and not self.robot_cfg.first_person_camera:
            # Camera position
            self.camera_pos = torch.tensor(
                self.robot_cfg.camera_pos, device=self.device, dtype=torch.float32
            ).repeat(self.num_envs, 1)

        # Initialize environment-specific buffers
        if self.cfg.use_memory_efficient_mode:
            # Use float32 instead of float64 for better memory efficiency
            self.robot_dof_lower_limits = self.robot_dof_lower_limits.to(dtype=torch.float32)
            self.robot_dof_upper_limits = self.robot_dof_upper_limits.to(dtype=torch.float32)
            self.robot_dof_speed_scales = self.robot_dof_speed_scales.to(dtype=torch.float32)
            self.robot_dof_targets = self.robot_dof_targets.to(dtype=torch.float32)

            # Initialize reward-related buffers with proper tensor types
            self.best_finger_position = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

            # Use a smaller history length for position tracking
            history_len = min(self.cfg.history_length, 5)  # Cap at 5 for memory efficiency
            self.position_history = torch.zeros((self.num_envs, history_len, 3), device=self.device, dtype=torch.float32)
            self.position_history_idx = 0

            # Initialize for vertical tracking
            self.prev_obj_height = None  # Will be initialized during first reward computation

        self.reference_traj_keys = self.robot_cfg.eef_keys
        self.activated_hand = self.robot_cfg.eef_keys[0]

        self.arm_indices, self.hand_full_indices = {}, {}
        self.eef_link_idx, self.position_tip_link_indices, self.force_tip_link_indices = {}, {}, {}
        self.ik_target_pos, self.ik_target_rot, self.finger_targets = {}, {}, {} # Used for store ik targets
        self.ref_target_pos, self.ref_finger_pos = {}, {} # finger and eef target positions
        self._prev_hand_pos, self._prev_hand_rot = {}, {}
        for key in self.reference_traj_keys:
            # map from name to joint index
            # ── Right-arm DOF indices ──────────────────────────────────────────
            self.arm_indices[key] = self._find_all_indices(self.robot_cfg.arm_names[key], mode="joint")

            # ── Right-hand DOF indices ───────────────────────────────────────
            self.hand_full_indices[key] = self._find_all_indices(self.hand_util[key].hand_full_joint_names, mode="joint")
            self.eef_link_idx[key] = self._robot.find_bodies(self.hand_util[key].ik_target_link)[0][0]
            self.position_tip_link_indices[key] = self._find_all_indices(
                self.hand_util[key].position_tip_links, mode="link"
            )
            if not self.robot_cfg.off_contact_sensor:
                self.force_tip_link_indices[key] = self._find_all_indices(
                    self.hand_util[key].force_tip_links, mode="contact"
                )
            else:
                self.force_tip_link_indices[key] = []
            # self.shape_joint = torch.tensor(self.hand_util.shape_joint, device=self.device)
            # self.preshape_joint = torch.tensor(self.hand_util.preshape_joint, device=self.device)

            # store for data collection
            self.ik_target_pos[key] = torch.zeros((self.num_envs, 3), device=self.device)
            self.ik_target_rot[key] = torch.zeros((self.num_envs, 4), device=self.device)
            self.finger_targets[key] = torch.zeros((self.num_envs, len(self.hand_util[key].hand_full_joint_names)), device=self.device)

            # object center, and finger positions from object center
            self.ref_target_pos[key] = torch.zeros((self.num_envs, 3), device=self.device)
            self.ref_finger_pos[key] = torch.zeros((self.num_envs, len(self.hand_util[key].position_tip_links), 3), device=self.device)
            #{key : torch.zeros((self.num_envs, len(self.hand_util[key].hand_full_joint_names)), device=self.device) for key in self.reference_traj_keys}

        self.target_finger_pos, self.current_finger_pos = {}, {}  # Used for storing intermediate values for rewards computation
        self.net_contact_forces, self.contact_position = {}, {}  # Used for storing contact forces and positions
        self.hand2obj = {}

        self.mode = self.cfg.mode

        self.cwp_predictor: dict[str, CWPPredictor] = {
            key: CWPPredictor(
            grasp_type=cfg.grasp_type,
            robot_name=cfg.robot_name,
            hand_laterality=key,
            obj_type=cfg.object_type,
            device=self.device,
        ) for key in self.reference_traj_keys}

        self.reference_traj_info = ReferenceTrajInfoMulti(
            keys=self.reference_traj_keys,
            num_envs=self.num_envs, device=self.device,
            hand_module=self.hand_util,
            cwp_predictor=self.cwp_predictor,
            mode=self.mode,
            pick_height=self.cfg.height_bonus_threshold
        )
        # add variables for Eureka reference
        self.is_grasped_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.relation_between_obj_and_hand = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.camera_data_types = self.robot_cfg.camera_data_types
        if self.robot_cfg.compute_pointcloud:
            self.camera_data_types.append("pointcloud")

        self.store_frames = False
        if self.store_frames:
            self.frames = {
                key: [[] for _ in range(self.num_envs)]
                for key in self.camera_data_types
            }
        else:
            self.frames = {key : None for key in self.camera_data_types}

        self.extras["obj_scale"] = self._obj_scales
        self.extras["obj_sq_params"] = self._obj_sq_params

    def _capture_frame(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        for key in self.camera_data_types:
            if key == "pointcloud": frame = self._camera.data.output["depth"]
            else: frame = self._camera.data.output[key]          # (N, H, W, 4)
            if frame.shape[0] == 0:
                return
            # sensor not ready yet
            if key == "rgb":
                frame = (frame[..., :3]).to(torch.uint8)     # drop alpha, stay GPU

            if key == "pointcloud":
                N, H, W, _ = frame.shape
                n_sampled_points = (H * W) // 4  # Sampled points
                pc = torch.zeros((N, n_sampled_points, 3), device=self.device)
                for env_id in env_ids:
                    _pc = self._get_pointcloud(
                        frame[env_id],
                        cam_pos=self._camera.data.pos_w[env_id],
                        cam_quat=self._camera.data.quat_w_ros[env_id],
                        intrinsic_matrix=self._camera.data.intrinsic_matrices[env_id],
                        n_sampled_points=n_sampled_points
                    )
                    pc[env_id, :len(_pc)] = _pc
                frame = pc  # (N, H*W, 3)
            frame_cpu = frame.contiguous().clone().cpu().numpy()   # deep-copy → CPU

            if self.store_frames:
                for env_id in env_ids:
                    self.frames[key][env_id].append(frame_cpu[env_id])
            else:
                self.frames[key] = frame_cpu

        return self.frames

    def _reset_frames(self, env_ids: torch.Tensor | None = None):
        """Reset the frames for the specified environments."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        for key in self.camera_data_types:
            if self.store_frames:
                for env_id in env_ids:
                    self.frames[key][env_id] = []


    def _get_frames(self, env_ids: torch.Tensor | None = None):
        """Get the captured frames for the specified environments."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        frames = {}
        for key in self.camera_data_types:
            if self.store_frames:
                frames[key] = [self.frames[key][env_id] for env_id in env_ids]
            else:
                frames[key] = self.frames[key][env_ids.cpu().numpy()]
        return frames

    def _get_pointcloud(self, depth, cam_pos, cam_quat, intrinsic_matrix, n_sampled_points=None):
        pointcloud = create_pointcloud_from_depth(
            intrinsic_matrix=intrinsic_matrix,
            depth=depth,
            position=cam_pos,
            orientation=cam_quat,
            device=self.device,
        )
        if n_sampled_points is None:
            n_sampled_points = pointcloud.shape[0]  # Use all points
        idx = torch.randperm(pointcloud.shape[0], device=self.device)[:n_sampled_points]
        pointcloud = pointcloud[idx]  # Sampled points
        return pointcloud.to(torch.float32)

    def _find_all_indices(self, joint_names, mode="link"):
        """Find all indices of joints matching the given names."""
        all_indices = []
        for name in joint_names:
            if mode == "joint":
                indices = self._robot.find_joints([name])[0]
            elif mode == "link":
                indices = self._robot.find_bodies([name])[0]
            elif mode == "contact":
                indices = self._contact_sensor.find_bodies([name])[0]
            else:
                raise ValueError(f"Unknown mode: {mode}")
            all_indices.extend(indices)
        return torch.tensor(all_indices, device=self.device)


    def _setup_scene(self):
        self._robot = Articulation(self.robot_cfg.get_articulation_cfg())
        self._table = RigidObject(self.env_cfg.get_obj_cfg("table"))
        self._obj = RigidObject(self.env_cfg.get_obj_cfg("obj"))

        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["table"] = self._table
        self.scene.rigid_objects["obj"] = self._obj

        if not self.robot_cfg.off_contact_sensor:
            self._contact_sensor = ContactSensor(self.robot_cfg.contact_sensor)
            self.scene.sensors["contact_sensor"] = self._contact_sensor

        if not self.robot_cfg.off_camera_sensor:
            self._camera = Camera(self.robot_cfg.camera)
            self.scene.sensors["camera"] = self._camera

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.markers = VisualizationMarkers(self.cfg.marker)

        # clone and replicate
        # self.scene.clone_environments(copy_from_source=False)

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        stage = get_current_stage()
        self._obj_sq_params = self._infer_sq_params(stage)
        self._obj_scales = self._get_obj_scale(stage)

        self.hand_only_sim = self.cfg.robot_name == "shadow"

    # pre-physics step calls
    def _pre_physics_step(self, actions: torch.Tensor):
        # Clone and store the original actions
        self.actions = actions.clone()

        # Add noise to actions for exploration (only during training)
        if self.mode == "train" and self.cfg.exploration_noise > 0.0:
            # Generate random noise
            noise = torch.randn_like(self.actions) * self.cfg.exploration_noise

            # Apply noise to actions but keep within range [-1, 1]
            self.actions = torch.clamp(self.actions + noise, -1.0, 1.0)
        else:
            # No noise during evaluation
            self.actions = actions.clone().clamp(-1.0, 1.0)

        # Memory-efficient batched processing for many environments
        if self.cfg.use_memory_efficient_mode and self.num_envs > self.cfg.max_envs_per_batch:
            batch_size = self.cfg.max_envs_per_batch
            num_batches = (self.num_envs + batch_size - 1) // batch_size  # Ceiling division

            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, self.num_envs)
                env_slice = slice(start_idx, end_idx)

                # Process this batch of environments
                self._process_batch_actions(env_slice)
        else:
            # Process all environments at once (original behavior)
            self._process_batch_actions(slice(0, self.num_envs))

    def _solve_ik(self, env_ids, ik_target_pos, ik_target_rot, key):
        if self.hand_only_sim:
            return self._solve_ik_(env_ids, ik_target_pos, ik_target_rot, key)
            # # TODO simple IK solver for the base joint
            # return self._solve_base_joint(env_ids, ik_target_pos, ik_target_rot)
        else:
            return self._solve_ik_(env_ids, ik_target_pos, ik_target_rot, key)


    def _solve_ik_(self, env_ids, ik_target_pos, ik_target_rot, key):
        # Create controller
        # convert to tensor indices if type is slice
        if isinstance(env_ids, slice):
            env_ids = torch.arange(env_ids.start, env_ids.stop, device=self.device)
        # diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
        diff_ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",
            ik_params={"lambda_val": 1e-4},
        )

        diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=len(env_ids), device=self.device)
        # reset controller
        diff_ik_controller.reset()
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=self.robot_cfg.arm_names[key],  body_names=[self.hand_util[key].ik_target_link])

        # Resolving the scene entities
        robot_entity_cfg.resolve(self.scene)

        root_pose_w = self._robot.data.root_state_w[env_ids, :7]
        target_pos_b, target_rot_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ik_target_pos, ik_target_rot
        )

        # Set IK target
        target_pose_b = torch.cat([target_pos_b, target_rot_b], dim=-1)
        diff_ik_controller.set_command(target_pose_b)

        # robot base is fixed -> -1
        ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
        ee_pose_w = self._robot.data.body_state_w[env_ids, robot_entity_cfg.body_ids[0], :7]
        jacobian = self._robot.root_physx_view.get_jacobians()[env_ids, ee_jacobi_idx][:, :, robot_entity_cfg.joint_ids]
        joint_pos = self._robot.data.joint_pos[env_ids][:, robot_entity_cfg.joint_ids]
        # compute frame in root frame
        ee_pos_b, ee_quat_b = subtract_frame_transforms(root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        # Compute joint positions and apply limits
        # print("target_pos_b", target_pos_b, "target_rot_b", target_rot_b, "ee_pos_b", ee_pos_b, "ee_quat_b", ee_quat_b)
        arm_targets = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
        ik_fail = torch.norm(target_pos_b - ee_pos_b, p=2, dim=-1) > 0.05

        return arm_targets, ik_fail

    def _process_batch_actions(self, env_slice):
        """Process actions for a batch of environments to reduce memory usage."""
        ### Arm
        timestep_ratio = (self.episode_length_buf / (self.max_episode_length - 1))[env_slice]

        for hand_key in self.reference_traj_keys:
            # Update the current end-effector pose
            cur_eef_pos, cur_eef_rot = self._get_current_eef_pose(hand_key)
            self.ik_target_pos[hand_key], self.ik_target_rot[hand_key], self.finger_targets[hand_key] = self.reference_traj_info.get(
                hand_key,
                env_slice, timestep_ratio,
                current_handP_world=cur_eef_pos[env_slice],
                current_handQ_world=cur_eef_rot[env_slice],
                current_hand_joint=self.robot_dof_targets[env_slice, self.hand_full_indices[hand_key]],
                objectP_world=self.obj_pos[env_slice],
                objectQ_world=self.obj_rot[env_slice],
                objectScale=self._obj_scales[env_slice],
                action_handP=self.actions[env_slice, :3] * self.action_scale[None, :3],
                action_handQ=quat_from_euler_xyz(
                    roll=self.actions[env_slice, 3] * self.action_scale[None, 3],
                    pitch=self.actions[env_slice, 4] * self.action_scale[None, 4],
                    yaw=self.actions[env_slice, 5] * self.action_scale[None, 5]
                ),
                action_hand_joint=self.actions[env_slice, len(self.arm_indices[hand_key]):-1] * self.action_scale[None, len(self.arm_indices[hand_key]):-1],
            )
            arm_targets, self.ik_fail = self._solve_ik(env_slice, self.ik_target_pos[hand_key], self.ik_target_rot[hand_key], hand_key)

            ### Fingers
            # finger_targets = finger_targets + ~self.reference_traj_info.pick_flg[env_slice, None] * self.actions[env_slice, len(self.arm_indices):-1] * self.action_scale[None, len(self.arm_indices):-1]
            self.robot_dof_targets[env_slice, self.arm_indices[hand_key]] = torch.clamp(arm_targets, self.robot_dof_lower_limits[self.arm_indices[hand_key]], self.robot_dof_upper_limits[self.arm_indices[hand_key]])
            self.robot_dof_targets[env_slice, self.hand_full_indices[hand_key]] = self.finger_targets[hand_key] # torch.clamp(finger_targets, self.robot_dof_lower_limits[self.hand_full_indices], self.robot_dof_upper_limits[self.hand_full_indices])

    def _get_current_eef_pose(self, hand_key=None):
        if hand_key is None:
            hand_key = self.activated_hand
        # Get the current end-effector pose
        hand_pos = self._robot.data.body_state_w[:, self.eef_link_idx[hand_key], :3]
        hand_rot = self._robot.data.body_state_w[:, self.eef_link_idx[hand_key], 3:7]
        return hand_pos, hand_rot

    def _apply_action(self):
        # import pdb;pdb.set_trace()
        # self.robot_dof_targets[:, self._robot.joint_names.index("HEAD_JOINT1")] = 0.32
        self._robot.set_joint_position_target(self.robot_dof_targets)

    # post-physics step calls
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Calculate horizontal distance from initial position (only x and y, not z)
        obj_horizontal_displacement = torch.norm(
            self.obj_pos[:, :2] - self.obj_initial_pos[:, :2], p=2, dim=-1
        )
        # Terminate if the obj moves more than 0.15m horizontally from initial position
        out_of_bounds = obj_horizontal_displacement > 0.15

        terminated = out_of_bounds # | self.ik_fail # too_high
        truncated = self.episode_length_buf >= self.max_episode_length - 1

        #camera_images = self._camera.data.output["rgb"]
        #print(camera_images.shape)
        # if torch.any(out_of_bounds):
        #     print(f"obj(s) out of bounds horizontally! Max x-y displacement: {obj_horizontal_displacement.max().item():.3f}m")
        done_envs = torch.where(terminated | truncated)[0].tolist()
        return terminated, truncated

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        # Get environment origins for the selected envs
        self._reset_frames(env_ids)

        env_origins = self.scene.env_origins[env_ids]

        self._obj_scales[env_ids] = self._get_obj_scale(get_current_stage(), env_ids=env_ids)

        # Local obj offset within each env with more randomization
        obj_offset = torch.zeros((len(env_ids), 3), device=self.device)
        obj_offset[:, 2] = self._obj_scales[env_ids, 2] + self.env_cfg.table_height

        # Add random variance to the obj's position with increased randomness
        num_envs_to_reset = len(env_ids) if env_ids is not None else self.num_envs
        # position_variance = torch.rand((num_envs_to_reset, 3), device=self.device) * 0.04 - 0.02  # Increased variance range [-0.02, 0.02]
        obj_pos = env_origins + obj_offset # + position_variance  # shape: (len(env_ids), 3)
        # self._get_obj_scale(get_current_stage(), env_ids=env_ids)  # Ensure obj scales are set

        # Add some random rotation to the obj
        obj_rot_base = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(num_envs_to_reset, 1)
        obj_vel = torch.zeros((num_envs_to_reset, 6), device=self.device)

        # Write to sim
        self._obj.write_root_pose_to_sim(torch.cat((obj_pos, obj_rot_base), dim=-1), env_ids=env_ids)
        self._obj.write_root_velocity_to_sim(obj_vel, env_ids=env_ids)

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = torch.zeros_like(joint_pos)

        # Hack to obtain object location

        self._compute_intermediate_values(env_ids)

        key = self.activated_hand
        # self.reset_trajectory(key, env_ids)
        cur_eef_pos, cur_eef_rot = self._get_current_eef_pose(key)
        ik_target_pos, ik_target_rot, finger_targets = self.reference_traj_info.get(
            key,
            env_ids, torch.Tensor([1e-8] * num_envs_to_reset).to(device=self.device),
            current_handP_world=cur_eef_pos[env_ids],
            current_handQ_world=cur_eef_rot[env_ids],
            current_hand_joint=self.robot_dof_targets[env_ids][:, self.hand_full_indices[key]],
            objectP_world=self.obj_pos[env_ids],
            objectQ_world=self.obj_rot[env_ids],
            objectScale=self._obj_scales[env_ids],
        )
        # ik_target_pos, ik_target_rot, finger_targets = self.reference_traj_info.get(key, env_ids, torch.tensor([0.0] * num_envs_to_reset, device=self.device))
        # ik_target_pos, ik_target_rot, finger_targets = self.reference_traj_info.get(env_ids, torch.tensor([0.0] * num_envs_to_reset, device=self.device))
        arm_targets, ik_fail = self._solve_ik(env_ids, ik_target_pos, ik_target_rot, key)

        joint_pos[:, self.arm_indices[key]] = arm_targets
        joint_pos[:, self.hand_full_indices[key]] = finger_targets

        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)

        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.robot_dof_targets[env_ids] = joint_pos.clone()

        self.obj_initial_pos[env_ids] = obj_pos.clone()
        self.obj_initial_rot[env_ids] = obj_rot_base.clone()

        # Refresh intermediate values for the specified environments
        self._compute_intermediate_values(env_ids)

        if not self.robot_cfg.off_camera_sensor:
            if self.robot_cfg.first_person_camera:
                pass
            else:
                eyes = env_origins + self.camera_pos[env_ids] + torch.randn_like(self.camera_pos[env_ids]) * 0.2
                targets = self.obj_pos[env_ids] + torch.randn_like(self.obj_pos[env_ids]) * 0.05
                self._camera.set_world_poses_from_view(eyes=eyes, targets=targets, env_ids=env_ids)

    def _get_observations(self) -> dict:
        self._compute_intermediate_values()

        key = self.activated_hand
        diff_finger = [
            (self.target_finger_pos[key][finger] - self.current_finger_pos[key][finger]) for finger in self.target_finger_pos[key].keys()
        ]
        dof_pos_scaled = (
            2.0
            * (self._robot.data.joint_pos[:, self.hand_full_indices[key]] - self.robot_dof_lower_limits[self.hand_full_indices[key]])
            / (self.robot_dof_upper_limits[self.hand_full_indices[key]] - self.robot_dof_lower_limits[self.hand_full_indices[key]])
            - 1.0
        )
        finger_effort = self.robot_dof_targets[:, self.hand_full_indices[key]] - self._robot.data.joint_pos[:, self.hand_full_indices[key]]
        def _to_local(pos):
            # Convert world position to local position in the hand frame
            return pos - self.scene.env_origins

        cur_eef_pos, cur_eef_rot = self._get_current_eef_pose(self.activated_hand)
        obs = torch.cat(
            (
                # _to_local(cur_eef_pos),
                # cur_eef_rot,
                # _to_local(self.obj_pos),
                # self.obj_rot,
                self.hand2obj[key]["pos"],
                self.hand2obj[key]["quat"],
                dof_pos_scaled,
                *diff_finger,
                finger_effort,
                self.actions,
            ),
            dim=-1,
        )

        if not self.robot_cfg.off_camera_sensor:
            self._capture_frame()
            self.extras["frames"] = self._get_frames()

        # used for Diffusion Policy
        if self.mode == "collect":
            for key in self.reference_traj_keys:
                hand_pos, hand_rot = self._get_current_eef_pose(key)
                self.extras[f"dp_ref_{key}"] = self._get_dp_ref(hand_pos, hand_rot, key)
                self._prev_hand_pos[key], self._prev_hand_rot[key] = hand_pos, hand_rot

        return {"policy": torch.clamp(obs, -5.0, 5.0)}

    def _get_dp_ref(self, hand_pos, hand_rot, key):
        # Compute the reference for the Diffusion Policy
        # dp_ref = {}
        if hasattr(self, "finger_targets"):
            finger_js = self.finger_targets
        else:
            finger_js = self._robot.data.joint_pos[:, self.hand_full_indices[key]]

        if self._prev_hand_pos.get(key) is not None and self._prev_hand_rot.get(key) is not None:
            delta_hand_pos = hand_pos - self._prev_hand_pos[key]
            delta_hand_rot = quat_mul(quat_conjugate(self._prev_hand_rot[key]), hand_rot)
            delta_hand_rot_rpy = euler_xyz_from_quat(delta_hand_rot)
            # dp_ref.update(
            #     {f"delta_eef_pos_{key}" : delta_hand_pos,
            #      f"delta_eef_rot_{key}" : delta_hand_rot_rpy}
            # )
        else:
            delta_hand_pos, delta_hand_rot_rpy = None, None
        return dict(delta_eef_pos=delta_hand_pos, delta_eef_rot=delta_hand_rot_rpy, finger_js=finger_js)

    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        for hand_key in self.reference_traj_keys:
            self.ref_finger_pos[hand_key][env_ids] = self.reference_traj_info.cwp[hand_key][env_ids]

        self._obj.update(self.dt)
        self.obj_pos[env_ids] = self._obj.data.root_pos_w[env_ids]
        self.obj_rot[env_ids] = self._obj.data.root_quat_w[env_ids]

        # if not self.robot_cfg.off_camera_sensor:
        #     self._camera.update(self.dt)
        # if not self.robot_cfg.off_contact_sensor:
        #     self._contact_sensor.update(self.dt)
        self._compute_intermediate_values_hand(env_ids, key=self.activated_hand)
        # # Update only the specified environments

    def _compute_intermediate_values_hand(self, env_ids: torch.Tensor | None = None, key=None):
        if key is None:
            key = self.activated_hand

        self.current_finger_pos[key] = {
            k: self._robot.data.body_pos_w[:, link_idx] for k, link_idx in zip(self.hand_util[key].position_tip_links, self.position_tip_link_indices[key])
        }

        self.target_finger_pos[key] = {
            k: self.ref_finger_pos[key][:, i] for i, k in enumerate(self.hand_util[key].position_tip_links)
        }

        self.hand2obj[key] = compute_object_state_in_hand_frame(
            self._obj.data.root_pos_w,  self._obj.data.root_quat_w,
            self._obj.data.root_lin_vel_w, self._obj.data.root_ang_vel_w,
            self._robot.data.body_pos_w[:, self.eef_link_idx[key]], self._robot.data.body_quat_w[:, self.eef_link_idx[key]],
            self._robot.data.body_lin_vel_w[:, self.eef_link_idx[key]], self._robot.data.body_ang_vel_w[:, self.eef_link_idx[key]],
            debug=False
        )

        # Contact
        if not self.robot_cfg.off_contact_sensor:
            self.net_contact_forces[key]= self._contact_sensor.data.net_forces_w_history[:, :, self.force_tip_link_indices[key]]
            # self.contact_normal = self._contact_sensor.data.normal_w[:, :, self.force_tip_link_indices]  # (N, T, 4, 3)
            self.contact_position[key] = self._robot.data.body_pos_w[:, self.position_tip_link_indices[key]]
            # self.visualize_contact_forces_as_arrows(
            #     positions=self.contact_position,  # (N, 4, 3)
            #     forces=self.net_contact_forces[:, 0],  # Use the first time step for visualization
            #     scale=1.0
            # )
        else:
            self.net_contact_forces[key] = None
            self.contact_position[key] = torch.zeros((self.num_envs, 4, 3), device=self.device)

    def direction_to_quaternion(self, directions: torch.Tensor) -> torch.Tensor:
        """
        Converts (N, 4, 3) direction vectors to (N, 4, 4) quaternions (w, x, y, z)
        that rotate the +Z axis to each given direction.
        """
        directions = directions.detach().cpu().numpy()  # (N, 4, 3)
        B, K, _ = directions.shape
        directions_flat = directions.reshape(-1, 3)  # (B*K, 3)

        quats = []
        for dir_vec in directions_flat:
            if np.linalg.norm(dir_vec) < 1e-6:
                quat = np.array([1.0, 0.0, 0.0, 0.0])  # identity quaternion
            else:
                rot, _ = R.align_vectors(
                    np.array([dir_vec]),
                    np.array([[0.0, 0.0, 1.0]])
                )
                q_xyzw = rot.as_quat()
                quat = np.roll(q_xyzw, 1)  # convert to (w, x, y, z)
            quats.append(quat)

        quats = np.array(quats, dtype=np.float32)  # (B*K, 4)
        return torch.tensor(quats, dtype=torch.float32).reshape(B, K, 4)

    def visualize_contact_forces_as_arrows(
        self,
        positions: torch.Tensor,   # (N, 4, 3)
        forces: torch.Tensor,      # (N, 4, 3)
        scale: float = 0.05
    ):
        """
        Visualizes contact forces as arrows (cones), with direction and magnitude.

        Args:
            positions: Contact positions in world frame. Shape (N, 4, 3)
            forces: Contact force vectors. Shape (N, 4, 3)
            scale: Float scaling factor for arrow length
        """
        N, K, _ = forces.shape
        device = forces.device

        # Normalize force vectors
        norms = torch.norm(forces, dim=-1, keepdim=True)  # (N, 4, 1)
        directions = torch.where(
            norms > 1e-6,
            forces / (norms + 1e-6),
            torch.tensor([0.0, 0.0, 1.0], device=device)
        )

        # Get quaternions representing rotation from +Z to direction
        orientations = self.direction_to_quaternion(directions)  # (N, 4, 4)

        # Set scales: uniform radius, Z-axis length = force magnitude × scale
        scales = torch.full_like(forces, 0.01)  # (N, 4, 3)
        scales[..., 2] = 0.1 * norms.squeeze(-1) * scale  # set Z-axis scale

        # Flatten for marker visualizer
        translations = positions.reshape(-1, 3)     # (N*4, 3)
        orientations = orientations.reshape(-1, 4)  # (N*4, 4)
        scales = scales.reshape(-1, 3)              # (N*4, 3)

        # Visualize
        self.markers.visualize(
            translations=translations,
            orientations=orientations,
            scales=scales,
            marker_indices=torch.zeros(translations.shape[0], device=device, dtype=torch.int32),  # Cone marker
        )

    def _get_rewards(self) -> torch.Tensor:
        # Refresh the intermediate values after the physics steps
        self._compute_intermediate_values()
        key = self.activated_hand

        d = torch.stack(
            [torch.norm(self.current_finger_pos[key][finger] - self.target_finger_pos[key][finger], dim=-1) for finger in self.current_finger_pos[key].keys()], dim=-1
        ).mean(dim=-1)
        dist_reward = 1.0 / (1.0 + d**2)
        dist_reward = dist_reward * dist_reward * self.cfg.dist_reward_scale

        # check grasp stability
        vel = torch.norm(self._obj.data.root_lin_vel_w, dim=-1)
        ang_vel = torch.norm(self._obj.data.root_ang_vel_w, dim=-1)
        vel_penalty = - (vel * self.cfg.vel_penalty_scale + ang_vel * self.cfg.angvel_penalty_scale)
        vel_penalty = torch.where(~self.reference_traj_info.pick_flg[key], vel_penalty, torch.zeros_like(vel_penalty))

        obj_rot = quat_error_magnitude(self.obj_initial_rot, self.obj_rot)
        obj_rot_penalty = -self.cfg.obj_rot_penalty_scale * obj_rot

        obj_z_pos = torch.clamp(self.obj_pos[:, 2] - self.env_cfg.table_height - self._obj_scales[:, 2], min=0.0)
        obj_z_pos_reward = torch.where(
            self.reference_traj_info.pick_flg[key],
            obj_z_pos * self.cfg.z_pos_reward_scale,
            torch.zeros_like(obj_z_pos)
        )

        # contact forces
        if self.net_contact_forces is not None:
            is_contact = torch.max(torch.norm(self.net_contact_forces[key], dim=-1), dim=1)[0] > 1.0
            obj_internal_force = torch.sum(-self.net_contact_forces[key][:, 0], dim=1)  # (N, 4, 3) -> (N, 3)
            obj_internal_torque = torch.sum(
                torch.cross(self.contact_position[key] - self.obj_pos[:, None, :], -self.net_contact_forces[key][:, 0]), dim=1
            )
            force_penalty = -torch.norm(obj_internal_force, dim=-1) * self.cfg.force_penalty_scale - torch.norm(obj_internal_torque, dim=-1) * self.cfg.force_penalty_scale
            # if self.reference_traj_info.pick_flg.any():
            #     import pdb; pdb.set_trace()
        else:
            is_contact = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.bool)
            force_penalty = torch.zeros((self.num_envs,), device=self.device)

        contacts_reward = torch.sum(is_contact, dim=1) * self.cfg.contact_reward_scale

        rel_vel = torch.norm(self.hand2obj[key]["lin_vel"], dim=-1)
        # grasp is success if the object is not moving with respect to the hand in the process of picking
        # is_grasped = torch.logical_and(
        #     obj_rot < self.cfg.obj_rot_threshold,
        #     torch.logical_and(rel_vel < self.cfg.rel_obj_vel_threshold, self.reference_traj_info.pick_flg)
        # )
        is_grasped = torch.logical_and(rel_vel < self.cfg.rel_obj_vel_threshold, self.reference_traj_info.pick_flg[key])
        is_grasped_full = torch.logical_and(is_grasped, obj_z_pos > self.cfg.height_bonus_threshold * 0.8)
        is_grasped_half = torch.logical_and(is_grasped, obj_z_pos > self.cfg.height_bonus_threshold / 2 * 0.8)

        self.is_grasped_buf[:] = is_grasped_full

        grasp_success_bonus = torch.where(
            is_grasped_full,
            torch.ones_like(is_grasped_full) * torch.clamp(obj_z_pos / self.cfg.height_bonus_threshold, max=1) * self.cfg.grasp_reward_scale,
            torch.zeros_like(is_grasped_full)
        )

        rewards = dist_reward + vel_penalty + grasp_success_bonus + obj_z_pos_reward + contacts_reward + obj_rot_penalty#  + force_penalty

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
            "dist_reward": safe_mean(dist_reward),
            "grasp_reward": safe_mean(grasp_success_bonus),
            "vel_penalty": safe_mean(vel_penalty),
            "obj_rot_penalty": safe_mean(obj_rot_penalty),
            "num_grasped": safe_mean(is_grasped_full, mask=self.reference_traj_info.pick_flg[key]),
            "num_grasped_half": safe_mean(is_grasped_half, mask=self.reference_traj_info.pick_flg[key]),
            "z_pos_reward": safe_mean(obj_z_pos_reward, mask=self.reference_traj_info.pick_flg[key]),
            "contacts_reward": safe_mean(contacts_reward),
            "force_penalty": safe_mean(force_penalty),
        }
        return rewards

    def _solve_base_joint(self, env_ids, pos, rot_quat):
        # pos: (..., 3)
        # rot_quat: (..., 4) (w, x, y, z)
        root_pose_w = self._robot.data.root_state_w[env_ids, :7]
        target_pos_b, target_rot_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], pos, rot_quat
        )

        def quat_to_euler_zyx(quat):
            # quat: shape (..., 4) in (w, x, y, z)
            w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]

            sinr_cosp = 2 * (w * x + y * z)
            cosr_cosp = 1 - 2 * (x * x + y * y)
            roll = torch.atan2(sinr_cosp, cosr_cosp)

            sinp = 2 * (w * y - z * x)
            pitch = torch.where(
                torch.abs(sinp) >= 1,
                torch.sign(sinp) * (math.pi / 2),
                torch.asin(sinp)
            )

            siny_cosp = 2 * (w * z + x * y)
            cosy_cosp = 1 - 2 * (y * y + z * z)
            yaw = torch.atan2(siny_cosp, cosy_cosp)

            return torch.stack([roll, pitch, yaw], dim=-1)  # shape (..., 3)

        dof_pos = torch.zeros((pos.shape[0], 6), device=pos.device)
        dof_pos[:, 0:3] = target_pos_b  # Tx, Ty, Tz
        dof_pos[:, 3:6] = quat_to_euler_zyx(target_rot_b)  # roll, pitch, yaw
        return dof_pos, None

    # def reset_trajectory(self, hand_key: str | None = None, env_ids: torch.Tensor | None = None):
    #     """
    #     Reset the trajectory for the specified environments.
    #     This will reinitialize the reference trajectory information and IK targets.
    #     """
    #     if env_ids is None:
    #         env_ids = self._robot._ALL_INDICES

    #     self.activated_hand = hand_key if hand_key is not None else self.activated_hand
    #     # self.ref_target_pos[hand_key][env_ids] =
    #     # Reset the reference trajectory info for the specified environments
    #     # cwp_config = {
    #     #     "obj_position": self.obj_pos[env_ids],
    #     #     "obj_orientation": torch.tensor([1, 0, 0, 0], device=self.device).repeat(len(env_ids), 1),
    #     #     "obj_scale": self._obj_scales[env_ids],
    #     # }
    #     cwp = self.cwp_predictor.predict(
    #         hand_laterality=self.activated_hand,
    #         obj_position=self.obj_pos[env_ids],
    #         obj_orientation=torch.tensor([1, 0, 0, 0], device=self.device).repeat(len(env_ids), 1),
    #         obj_scale=self._obj_scales[env_ids],
    #     )
    #     ref_traj_cfg = self.hand_util[hand_key].getReferenceTrajInfo(
    #         num_envs=len(env_ids), cwp=cwp, device=self.device
    #     )
    #     # Set the reference target positions and finger positions

    #     self.ref_finger_pos[hand_key][env_ids] = cwp["position"]

    #     self.reference_traj_info.update(hand_key, env_ids, **ref_traj_cfg, reset=True)
    #     if self.mode == "demo":
    #         self._visualize_cwp()

    def _visualize_cwp(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        all_positions = []
        all_orientations = []
        all_marker_indices = []

        for cnt, k in enumerate(self.hand_util[self.activated_hand].position_tip_links):
            pos = self.ref_finger_pos[self.activated_hand][env_ids, cnt]  # (N, 3)
            ori = torch.tensor([[1, 0, 0, 0]], device=self.device).repeat(len(env_ids), 1)  # (N, 4)
            idx = torch.zeros(len(env_ids), dtype=torch.int64, device=self.device)  # (N,)

            all_positions.append(pos)
            all_orientations.append(ori)
            all_marker_indices.append(idx)

        # Concatenate
        positions = torch.cat(all_positions, dim=0)
        orientations = torch.cat(all_orientations, dim=0)
        marker_indices = torch.cat(all_marker_indices, dim=0)
        self.markers.visualize(positions, orientations, marker_indices=marker_indices)

    def _infer_sq_params(self, stage):
        if self.env_cfg.obj_type != "superquadric":
            return torch.zeros((self.num_envs, 2), device=self.device)

        import re

        # Extract e1 and e2 values from the USD file path
        # Example path: '.../sq_005_e1_03_e2_07.usd'
        def extract_e1_e2_from_path(path: str) -> tuple[float, float]:
            """Extract e1 and e2 values from a USD file path like '.../sq_005_e1_03_e2_07.usd'"""
            match = re.search(r"e1_(\d+)_e2_(\d+)", path)
            if not match:
                raise ValueError(f"Could not extract e1/e2 from path: {path}")
            e1 = float(match.group(1)) / 100  # "03" → 0.3
            e2 = float(match.group(2)) / 100
            return e1, e2

        sq_params = []
        for i in range(self.num_envs):
            prim_path = f"/World/envs/env_{i}/Object"
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                print(f"[{i}] Not Found")
                continue
            ref_meta = prim.GetMetadata("references")
            ref_path = ref_meta.GetAddedOrExplicitItems()[0].assetPath
            e1, e2 = extract_e1_e2_from_path(ref_path)
            sq_params.append([e1, e2])

        # Convert to tensor
        sq_params = torch.tensor(sq_params, device=self.device)
        return sq_params


    # def _get_obj_scale(self, stage, env_ids: torch.Tensor | None = None):
    #     """Return the size of the prims in the USD stage."""
    #     from collections import defaultdict
    #     if env_ids is None:
    #         env_ids = torch.arange(self.num_envs, device=self.device)

    #     if self.env_cfg.obj_type != "superquadric":
    #         # temporal fix for non-superquadric objects
    #         return 0.05 * torch.ones((len(env_ids), 3), device=self.device)

    #     scales = []
    #     for i in env_ids:
    #         # Get the prim at the specified path
    #         # Example path: "/World/envs/env_0/Object"
    #         # Note: The path may vary based on your USD structure
    #         prim_path = f"/World/envs/env_{i}/Object"
    #         prim = stage.GetPrimAtPath(prim_path)
    #         xform = UsdGeom.Xformable(prim)
    #         for op in xform.GetOrderedXformOps():
    #             if op.GetOpType() == UsdGeom.XformOp.TypeScale:
    #                 scale = op.Get()
    #                 break
    #         scales.append([scale[0], scale[1], scale[2]])
    #     scales = torch.tensor(scales, device=self.device)
    #     return scales

    def _get_obj_scale(self, stage, env_ids: torch.Tensor | None = None):
        """Return actual size (width, depth, height) of the object prims in each env."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        def get_boundable_geom(prim):
            """Recursively search for a Boundable geom under the given prim."""
            from pxr import UsdGeom
            if prim.IsA(UsdGeom.Boundable):
                return prim
            for child in prim.GetChildren():
                geom = get_boundable_geom(child)
                if geom:
                    return geom
            return None

        sizes = []
        for i in env_ids:
            prim_path = f"/World/envs/env_{i}/Object"
            prim = stage.GetPrimAtPath(prim_path)
            xform = UsdGeom.Xformable(prim)

            # Get scale from XformOp
            scale = Gf.Vec3d(1.0, 1.0, 1.0)
            for op in xform.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                    scale = op.Get()
                    break

            # Compute the bounding box (AABB)
            boundable_prim = get_boundable_geom(prim)
            bound = UsdGeom.Boundable(boundable_prim).ComputeExtent(Usd.TimeCode.Default())
            if not bound:
                sizes.append([0.0, 0.0, 0.0])
                continue

            min_xyz, max_xyz = bound[0], bound[1]
            raw_size = Gf.Vec3d(max_xyz[0] - min_xyz[0],
                                max_xyz[1] - min_xyz[1],
                                max_xyz[2] - min_xyz[2])

            actual_size = [raw_size[i] * scale[i] / 2 for i in range(3)]
            if self.env_cfg.obj_type == "ycb":
                # cm to meters conversion for YCB objects
                actual_size = [size / 100.0 for size in actual_size]
            sizes.append(actual_size)

        return torch.tensor(sizes, device=self.device)
