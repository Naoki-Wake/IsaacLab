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
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from pxr import UsdGeom, Ar

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg

from isaaclab.managers import SceneEntityCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import subtract_frame_transforms

from isaaclab_tasks.utils.hand_utils import ShadowHandUtils, HondaHandUtils, ReferenceTrajInfo
from isaaclab_tasks.utils.compute_relative_state import compute_object_state_in_hand_frame
from .events import EventCfg, create_grasp_event_cfg
from .robot_cfg import RobotCfg
from isaaclab.sensors import CameraCfg, Camera
from isaaclab.sensors import TiledCamera, TiledCameraCfg, save_images_to_file
import cv2                                            # OpenCV-Python

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

import omni.usd
import os

torch.autograd.set_detect_anomaly(True)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

@configclass
class NextageShadowGraspEnvCfg(DirectRLEnvCfg):
    # env
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
    off_camera_sensor = True
    env_spacing = 1.5 if robot_name == "shadow" else 3.0
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=env_spacing, replicate_physics=False)

    robot_cfg = RobotCfg(robot_name)
    robot = robot_cfg.get_articulation_cfg()
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=1, update_period=0.005, track_air_time=True
    )
    grasp_type = "active"  # or "passive"
    hand_util = ShadowHandUtils(grasp_type=grasp_type) if "shadow" in robot_name else HondaHandUtils(grasp_type=grasp_type)
    table_height = 0.8
    table_size = (0.8, 0.8, table_height)
    table = RigidObjectCfg(
        prim_path="/World/envs/env_.*/table",
        spawn=sim_utils.CuboidCfg(
            size=table_size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=10000),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                rest_offset=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.4, 0.4)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, table_height / 2),
            rot=(0.0, 0.0, 0.0, 1.0)
        )
    )

    # first person camera
    camera = CameraCfg(
        prim_path="/World/envs/env_.*/Robot/LEFT_CAMERA/front_cam",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=40, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.0,0.0,0.0), rot=(0.58965,0.39028,-0.39028,-0.58965), convention="opengl"),
    )

    events: EventCfg = create_grasp_event_cfg(base_obj_size=obj_size_half)
    obj_asset_cfgs = [
        sim_utils.UsdFileCfg(
            usd_path=obj_usd_path, scale=(0.1, 0.1, 0.1),
        )
        for obj_usd_path in sorted(glob.glob(os.path.join("source/isaaclab_assets/data/Props/Superquadrics/*", "object.usd")))
    ]

    obj = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",  # Must match env pattern
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=obj_asset_cfgs,
            random_choice=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=1000,
                disable_gravity=False,
                rigid_body_enabled=True,
                kinematic_enabled=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
                enable_gyroscopic_forces=True,
                retain_accelerations=False,
                max_linear_velocity=100,
                max_angular_velocity=100
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, table_height),  # Spawn above table/ground
            rot=(0.0, 0.0, 0.0, 1.0)
        )
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
    marker = VisualizationMarkersCfg(
        prim_path="/World/Visuals/Markers",
        markers={
            "target_sphere": sim_utils.SphereCfg(
                radius=0.02,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
            ),
        }
    )
    deg_to_rad = math.pi / 180
    # action_scale = [0.0 for _ in range(action_space)] # debug
    action_scale = [0.01, 0.01, 0.01] + [5.0 * deg_to_rad] * 3  + [10.0 * deg_to_rad] * n_finger_joint + [1.0]
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
    height_bonus_threshold = 0.05   # Height threshold for bonus


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

        self.action_scale = torch.tensor(self.cfg.action_scale, device=self.device)
        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

        stage = get_current_stage()
        self.eef_link_idx = self._robot.find_bodies(self.cfg.hand_util.ik_target_link)[0][0]
        self.position_tip_link_indices = self._find_all_indices(
            self.cfg.hand_util.position_tip_links, mode="link"
        )
        if not self.cfg.off_contact_sensor:
            self.force_tip_link_indices = self._find_all_indices(
                self.cfg.hand_util.force_tip_links, mode="contact"
            )
        else:
            self.force_tip_link_indices = []

        ### self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(".*THIGH")
        self.robot_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.robot_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)

        # Initialize obj position and rotation for all environments
        self.obj_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.obj_rot = torch.zeros((self.num_envs, 4), device=self.device)

        # Track initial obj positions
        self.obj_initial_pos = torch.zeros((self.num_envs, 3), device=self.device)

        self.ref_target_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.ref_finger_offset = torch.zeros((self.num_envs, len(self.cfg.hand_util.position_tip_links), 3), device=self.device)

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

        self.shape_joint = torch.tensor(self.cfg.hand_util.shape_joint, device=self.device)
        self.preshape_joint = torch.tensor(self.cfg.hand_util.preshape_joint, device=self.device)

        # ── Right-arm DOF indices ──────────────────────────────────────────
        self.arm_indices = self._find_all_indices(self.cfg.robot_cfg.arm_names, mode="joint")

        # ── Right-hand DOF indices ───────────────────────────────────────
        self.hand_full_indices = self._find_all_indices(self.cfg.hand_util.hand_full_joint_names, mode="joint")
        self.is_training = self.cfg.is_training

        # when testing, increase the length of pick-up just for visualization purpose
        if not self.is_training:
            self.cfg.episode_length_s *= 2

        self.reference_traj_info = ReferenceTrajInfo(
            self.num_envs, self.device,
            finger_coupling_rule=self.cfg.hand_util.couplingRuleTensor,
            n_hand_joints=len(self.cfg.hand_util.hand_full_joint_names),
            is_training=self.is_training,
            pick_height=self.cfg.height_bonus_threshold
        )


        # add variables for Eureka reference
        self.is_grasped_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.relation_between_obj_and_hand = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.frames      = [[] for _ in range(self.num_envs)]
        self.episode_ctr = torch.zeros(self.num_envs, dtype=torch.int32)
        self.champion_indices = torch.zeros(self.num_envs, dtype=torch.int32)
        self.extras["obj_scale"] = self._obj_scales
        self.extras["obj_sq_params"] = self._obj_sq_params

    # def _capture_frame(self):
    #     tex = self._camera.data.output["rgb"]          # (N,H,W,4) float32 [0,1]
    #     if tex.shape[0] == 0:
    #         return                                     # sensor not ready yet
    #     rgb = (tex[..., :3] * 255).to(torch.uint8)     # drop alpha, stay GPU
    #     cpu = rgb.contiguous().clone().cpu().numpy()   # deep-copy → CPU
    #     for env_id in range(self.num_envs):
    #         self.frames[env_id].append(cv2.cvtColor(cpu[env_id],
    #                                                 cv2.COLOR_RGB2BGR))

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
        self._robot = Articulation(self.cfg.robot)
        self._table = RigidObject(self.cfg.table)
        self._obj = RigidObject(self.cfg.obj)

        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["table"] = self._table
        self.scene.rigid_objects["obj"] = self._obj

        if not self.cfg.off_contact_sensor:
            self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
            self.scene.sensors["contact_sensor"] = self._contact_sensor

        if not self.cfg.off_camera_sensor:
            self._camera = Camera(self.cfg.camera)
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
        if hasattr(self, 'is_training') and self.is_training:
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

    def _solve_ik(self, env_ids, ik_target_pos, ik_target_rot):
        if self.hand_only_sim:
            return self._solve_ik_(env_ids, ik_target_pos, ik_target_rot)
            # # TODO simple IK solver for the base joint
            # return self._solve_base_joint(env_ids, ik_target_pos, ik_target_rot)
        else:
            return self._solve_ik_(env_ids, ik_target_pos, ik_target_rot)


    def _solve_ik_(self, env_ids, ik_target_pos, ik_target_rot):
        # Create controller
        # convert to tensor indices if type is slice
        if isinstance(env_ids, slice):
            env_ids = torch.arange(env_ids.start, env_ids.stop, device=self.device)
        diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
        diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=len(env_ids), device=self.device)
        # reset controller
        diff_ik_controller.reset()
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=self.cfg.robot_cfg.arm_names,  body_names=[self.cfg.hand_util.ik_target_link])

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
        cur_eef_pos, cur_eef_rot = self._get_current_eef_pose()
        ik_target_pos, ik_target_rot, finger_targets = self.reference_traj_info.get(
            env_slice, timestep_ratio,
            current_handP_world=cur_eef_pos[env_slice],
            current_handQ_world=cur_eef_rot[env_slice],
            current_hand_joint=self.robot_dof_targets[env_slice, self.hand_full_indices],
            action_handP=self.actions[env_slice, :3] * self.action_scale[None, :3],
            action_handQ=None,
            action_hand_joint=self.actions[env_slice, len(self.arm_indices):-1] * self.action_scale[None, len(self.arm_indices):-1],
        )
        arm_targets, self.ik_fail = self._solve_ik(env_slice, ik_target_pos, ik_target_rot)

        ### Fingers
        # finger_targets = finger_targets + ~self.reference_traj_info.pick_flg[env_slice, None] * self.actions[env_slice, len(self.arm_indices):-1] * self.action_scale[None, len(self.arm_indices):-1]

        self.robot_dof_targets[env_slice, self.arm_indices] = torch.clamp(arm_targets, self.robot_dof_lower_limits[self.arm_indices], self.robot_dof_upper_limits[self.arm_indices])
        self.robot_dof_targets[env_slice, self.hand_full_indices] = finger_targets # torch.clamp(finger_targets, self.robot_dof_lower_limits[self.hand_full_indices], self.robot_dof_upper_limits[self.hand_full_indices])

    def _get_current_eef_pose(self):
        # Get the current end-effector pose
        hand_pos = self._robot.data.body_state_w[:, self.eef_link_idx, :3]
        hand_rot = self._robot.data.body_state_w[:, self.eef_link_idx, 3:7]
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
        # robot state with increased randomization

        # Get environment origins for the selected envs
        env_origins = self.scene.env_origins[env_ids]

        self._obj_scales[env_ids] = self._get_obj_scale(get_current_stage(), env_ids=env_ids)

        # Local obj offset within each env with more randomization
        obj_offset = torch.zeros((len(env_ids), 3), device=self.device)
        obj_offset[:, 2] = self._obj_scales[env_ids, 2] + self.cfg.table_height

        # Add random variance to the obj's position with increased randomness
        num_envs_to_reset = len(env_ids) if env_ids is not None else self.num_envs
        position_variance = torch.rand((num_envs_to_reset, 3), device=self.device) * 0.04 - 0.02  # Increased variance range [-0.02, 0.02]
        obj_pos = env_origins + obj_offset # + position_variance  # shape: (len(env_ids), 3)

        # Add some random rotation to the obj
        obj_rot_base = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(num_envs_to_reset, 1)
        obj_vel = torch.zeros((num_envs_to_reset, 6), device=self.device)

        # Write to sim
        self._obj.write_root_pose_to_sim(torch.cat((obj_pos, obj_rot_base), dim=-1), env_ids=env_ids)
        self._obj.write_root_velocity_to_sim(obj_vel, env_ids=env_ids)

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        self._compute_intermediate_values(env_ids)

        cwp_config = {
            "num_envs": num_envs_to_reset,
            "grasp_cweb0_position": self.ref_target_pos[env_ids],
            "grasp_cweb0_orientation": torch.tensor([1, 0, 0, 0], device=self.device).repeat(num_envs_to_reset, 1),
            "grasp_approach_vertical": np.random.uniform(70, 80, size=num_envs_to_reset).tolist(),
            "grasp_approach_horizontal": [0.0 for _ in range(num_envs_to_reset)],
            "back": [0.15 for _ in range(num_envs_to_reset)],
        }
        self.reference_traj_info.update(env_ids, **self.cfg.hand_util.getReferenceTrajInfo(cwp_config, self.device), reset=True)
        ik_target_pos, ik_target_rot, finger_targets = self.reference_traj_info.get(env_ids, torch.tensor([0.0] * num_envs_to_reset, device=self.device))
        arm_targets, ik_fail = self._solve_ik(env_ids, ik_target_pos, ik_target_rot)

        joint_pos[:, self.arm_indices] = arm_targets
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        joint_pos[:, self.hand_full_indices] = finger_targets

        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        if env_ids is None:
            self.obj_initial_pos = obj_pos.clone()
        else:
            self.obj_initial_pos[env_ids] = obj_pos.clone()

        # Refresh intermediate values for the specified environments
        self._compute_intermediate_values(env_ids)
        # self._visualize()

    def _get_observations(self) -> dict:
        self._compute_intermediate_values()
        diff_finger = [
            (self.target_finger_pos[finger] - self.current_finger_pos[finger]) for finger in self.target_finger_pos.keys()
        ]
        dof_pos_scaled = (
            2.0
            * (self._robot.data.joint_pos[:, self.hand_full_indices] - self.robot_dof_lower_limits[self.hand_full_indices])
            / (self.robot_dof_upper_limits[self.hand_full_indices] - self.robot_dof_lower_limits[self.hand_full_indices])
            - 1.0
        )
        finger_effort = self.robot_dof_targets[:, self.hand_full_indices] - self._robot.data.joint_pos[:, self.hand_full_indices]
        # to_target = self.obj_pos - self.robot_grasp_pos

        hand_pos, hand_rot = self._get_current_eef_pose()
        obs = torch.cat(
            (
                dof_pos_scaled,
                hand_pos,
                hand_rot,
                *diff_finger,
                finger_effort,
                self.obj_pos,
                self.obj_rot,
                self.actions,
            ),
            dim=-1,
        )
        return {"policy": torch.clamp(obs, -5.0, 5.0)}

    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        self._obj.update(self.dt)
        if not self.cfg.off_camera_sensor:
            self._camera.update(self.dt)
        if not self.cfg.off_contact_sensor:
            self._contact_sensor.update(self.dt)

        # # Update only the specified environments
        self.obj_pos[env_ids] = self._obj.data.root_pos_w[env_ids]
        self.obj_rot[env_ids] = self._obj.data.root_quat_w[env_ids]

        self.ref_target_pos[env_ids] = self.obj_pos[env_ids]
        self.ref_target_pos[env_ids, 2] += self._obj_scales[env_ids, 2] - 0.02
        # simple x-axis translation
        self.ref_finger_offset[env_ids, :, 0] = torch.tensor([ 1,  1,  1, -1], device='cuda:0').unsqueeze(0) *  self._obj_scales[env_ids, 0].unsqueeze(1)

        self.current_finger_pos = {
            k: self._robot.data.body_pos_w[:, link_idx] for k, link_idx in zip(self.cfg.hand_util.position_tip_links, self.position_tip_link_indices)
        }

        self.target_finger_pos = {
            k: self.ref_target_pos + self.ref_finger_offset[:, i] for i, k in enumerate(self.cfg.hand_util.position_tip_links)
        }

        self.hand2obj = compute_object_state_in_hand_frame(
            self._obj.data.root_pos_w,  self._obj.data.root_quat_w,
            self._obj.data.root_lin_vel_w, self._obj.data.root_ang_vel_w,
            self._robot.data.body_pos_w[:, self.eef_link_idx], self._robot.data.body_quat_w[:, self.eef_link_idx],
            self._robot.data.body_lin_vel_w[:, self.eef_link_idx], self._robot.data.body_ang_vel_w[:, self.eef_link_idx],
            debug=False
        )

        # Contact
        if not self.cfg.off_contact_sensor:
            self.net_contact_forces = self._contact_sensor.data.net_forces_w_history[:, :, self.force_tip_link_indices]
        else:
            self.net_contact_forces = None

        if not self.cfg.off_camera_sensor:
            rgb = self._camera.data.output["rgb"].cpu().numpy()      # (N, H, W, 4)
            rgb = (rgb[..., :3]).astype(np.uint8)              # strip alpha
            for i in range(self.num_envs):
                self.frames[i].append(rgb[i])
            # write a buffer to an image for debugging
            # print('new frame is added')
        else:
            self.frames = None


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

    def _visualize(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

            all_positions = []
            all_orientations = []
            all_marker_indices = []

            for cnt, k in enumerate(self.cfg.hand_util.position_tip_links):
                pos = self.target_finger_pos[k][env_ids]  # (N, 3)
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
        import re

        # Extract e1 and e2 values from the USD file path
        # Example path: '.../sq_005_e1_03_e2_07.usd'
        def extract_e1_e2_from_path(path: str) -> tuple[float, float]:
            """Extract e1 and e2 values from a USD file path like '.../sq_005_e1_03_e2_07.usd'"""
            match = re.search(r"e1_(\d+)_e2_(\d+)", path)
            if not match:
                raise ValueError(f"Could not extract e1/e2 from path: {path}")
            e1 = float(match.group(1)) / 10  # "03" → 0.3
            e2 = float(match.group(2)) / 10
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


    def _get_obj_scale(self, stage, env_ids: torch.Tensor | None = None):
        """Return the size of the prims in the USD stage."""
        from collections import defaultdict
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        scales = []
        for i in env_ids:
            # Get the prim at the specified path
            # Example path: "/World/envs/env_0/Object"
            # Note: The path may vary based on your USD structure
            prim_path = f"/World/envs/env_{i}/Object"
            prim = stage.GetPrimAtPath(prim_path)
            xform = UsdGeom.Xformable(prim)
            for op in xform.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                    scale = op.Get()
                    break
            scales.append([scale[0], scale[1], scale[2]])
        scales = torch.tensor(scales, device=self.device)
        return scales
