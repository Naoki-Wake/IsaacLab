import math
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sensors import CameraCfg, Camera
import isaaclab.sim as sim_utils

from isaaclab_tasks.utils.hand_utils import ShadowHandUtils, HondaHandUtils

DEG_TO_RAD = math.pi / 180

class RobotCfg():
    def __init__(self, grasp_type: str = "active", is_training: bool = True):
        self.action_space = 6 + self.n_finger_joint + 1 # [x, y, z, roll, pitch, yaw] + [n_finger_joint joints] + [terminate]
        self.action_scale = [0.01, 0.01, 0.01] + [0.0, 10.0 * DEG_TO_RAD, 0.0] + [20.0 * DEG_TO_RAD] * self.n_finger_joint + [1.0]
        self.contact_sensor: ContactSensorCfg = ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/.*",
            history_length=1,
            update_period=0.01,
            track_air_time=True,
            # debug_vis=True
        )

    def get_articulation_cfg(self) -> ArticulationCfg:
        return ArticulationCfg(
            prim_path="/World/envs/env_.*/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=self.usd_path,
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    max_depenetration_velocity=1000,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False, solver_position_iteration_count=12, solver_velocity_iteration_count=1
                ),
                semantic_tags=[("class", "robot")]
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos=self.init_joint_pos,
                pos=self.init_pos,
                rot=self.init_rot,
            ),
            actuators=self.actuators,
        )

    def set_camera_pose(self, camera: Camera):
        """Set the camera pose."""
        pass

class NextageShadowRobotCfg(RobotCfg):
    def __init__(self, grasp_type: str = "active", is_training: bool = True):
        self.usd_path = "scripts/my_models/nextage/nextage_env_full_links.usd"
        self.init_joint_pos = {
            "CHEST_JOINT0": 0.0, "HEAD_JOINT0": 0.0, "HEAD_JOINT1": 0.0,
            "LARM_JOINT0": 0.8, "LARM_JOINT1": 0.0, "LARM_JOINT2": 0.0, "LARM_JOINT3": 0.0, "LARM_JOINT4": 0.0, "LARM_JOINT5": 0.0,
            "lh_FFJ4": 0.0, "lh_FFJ3": 0.0, "lh_FFJ2": 0.0,
            "lh_FFJ1": 0.0, "lh_MFJ4": 0.0, "lh_MFJ3": 0.0, "lh_MFJ2": 0.0, "lh_MFJ1": 0.0, "lh_RFJ4": 0.0, "lh_RFJ3": 0.0, "lh_RFJ2": 0.0, "lh_RFJ1": 0.0, "lh_THJ5": 0.0, "lh_THJ4": 0.0, "lh_THJ2": 0.0, "lh_THJ1": 0.0, "RARM_JOINT0": 0.0, "RARM_JOINT1": -0.6, "RARM_JOINT2": -0.6, "RARM_JOINT3": 0.0, "RARM_JOINT4": 0.0, "RARM_JOINT5": 0.0,
            "rh_FFJ4": 0.0, "rh_FFJ3": 0.0, "rh_FFJ2": 0.0, "rh_FFJ1": 0.0, "rh_MFJ4": 0.0, "rh_MFJ3": 0.0, "rh_MFJ2": 0.0, "rh_MFJ1": 0.0, "rh_RFJ4": 0.0, "rh_RFJ3": 0.0, "rh_RFJ2": 0.0, "rh_RFJ1": 0.0, "rh_THJ5": 0.0, "rh_THJ4": 0.0, "rh_THJ2": 0.0, "rh_THJ1": 0.0,
        }
        self.init_pos, self.init_rot = (-0.65, 0.3, 0.8), (1.0, 0.0, 0.0, 0.0)
        self.actuators = {
            "right_arm": ImplicitActuatorCfg(
                joint_names_expr=["RARM_JOINT[0-5]"],
                effort_limit=1e6,         # Allows very high effort
                velocity_limit=1e6,       # Allows very high velocity
                stiffness=1e6,            # Allows very high stiffness
                damping=1e3,              # Enough damping to prevent oscillations
            ),
            "right_hand": ImplicitActuatorCfg(
                joint_names_expr=["rh_.*"],
                effort_limit=1e6,         # Allows very high effort
                velocity_limit=1e6,       # Allows very high velocity
                stiffness=1e6,            # Allows very high stiffness
                damping=1e3,              # Enough damping to prevent oscillations
            )
        }
        self.camera_data_types = ["rgb"] # , "depth", "segmentation"]
        self.compute_pointcloud = False
        # first person camera
        self.camera = CameraCfg(
            prim_path="/World/envs/env_.*/Robot/LEFT_CAMERA/front_cam",
            update_period=0.1,
            height=480,
            width=640,
            data_types=self.camera_data_types,
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=40, clipping_range=(0.1, 1.0e5)
            ),
            offset=CameraCfg.OffsetCfg(pos=(0.0,0.0,0.0), rot=(0.58965,0.39028,-0.39028,-0.58965), convention="opengl"),
        )
        self.arm_names = ["RARM_JOINT0", "RARM_JOINT1", "RARM_JOINT2", "RARM_JOINT3", "RARM_JOINT4", "RARM_JOINT5"]
        self.n_finger_joint = 16
        self.hand_util = ShadowHandUtils(grasp_type=grasp_type)
        self.off_camera_sensor = False
        self.off_contact_sensor = False
        self.first_person_camera = True
        super().__init__(grasp_type=grasp_type, is_training=is_training)

class ShadowRobotCfg(RobotCfg):
    def __init__(self, grasp_type: str = "active", is_training: bool = True):
        self.usd_path = "scripts/my_models/shadow/floating_shadow.usd"
        self.init_joint_pos = {
            "rh_Tx": 0.0, "rh_Ty": 0.0, "rh_Tz": 2.0, "rh_roll": math.pi/2, "rh_pitch": 0.0, "rh_yaw": math.pi,
            "rh_FFJ4": 0.0, "rh_FFJ3": 0.0, "rh_FFJ2": 0.0, "rh_FFJ1": 0.0, "rh_MFJ4": 0.0, "rh_MFJ3": 0.0, "rh_MFJ2": 0.0, "rh_MFJ1": 0.0, "rh_RFJ4": 0.0, "rh_RFJ3": 0.0, "rh_RFJ2": 0.0, "rh_RFJ1": 0.0, "rh_THJ5": 0.0, "rh_THJ4": 0.0, "rh_THJ2": 0.0, "rh_THJ1": 0.0,
        }
        self.init_pos, self.init_rot = (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)
        self.actuators = {
            "right_hand": ImplicitActuatorCfg(
                joint_names_expr=["rh_.*"],
                effort_limit=1e6,         # Allows very high effort
                velocity_limit=1e6,       # Allows very high velocity
                stiffness=1e6,            # Allows very high stiffness
                damping=1e3,              # Enough damping to prevent oscillations
            )
        }
        self.arm_names = ["rh_Tx", "rh_Ty", "rh_Tz", "rh_roll", "rh_pitch", "rh_yaw"]
        self.n_finger_joint = 16
        self.hand_util = ShadowHandUtils(grasp_type=grasp_type)
        self.off_camera_sensor = True
        self.off_contact_sensor = False
        self.camera_data_types = ["rgb", "depth", "semantic_segmentation"] # , "normals"]
        self.compute_pointcloud = True
        if not is_training: self.off_camera_sensor = False
        self.camera = CameraCfg(
            prim_path="/World/envs/env_.*/camera",
            update_period=0.1,
            height=480,
            width=640,
            data_types=self.camera_data_types,
            colorize_semantic_segmentation=False,
            colorize_instance_id_segmentation=False,
            colorize_instance_segmentation=False,
            update_latest_camera_pose=True,
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=40, clipping_range=(0.1, 1.0e5)
            ),
            semantic_segmentation_mapping={"class:robot": 1, "class:object": 2},
        )
        self.first_person_camera = False
        self.camera_pos = (-0.65, 0.3, 1.5)
        super().__init__(grasp_type=grasp_type, is_training=is_training)

class HondaRobotCfg(RobotCfg):
    def __init__(self, grasp_type: str = "active", is_training: bool = True):
        self.usd_path = "scripts/my_models/honda/ur10_honda_fix_physics.usd"
        self.init_joint_pos = {
            "shoulder_pan_joint": 0.0, "shoulder_lift_joint": 0.0, "elbow_joint": 0.0, "wrist_1_joint": 0.0,
            "wrist_2_joint": 0.0, "wrist_3_joint": 0.0,
        }
        self.init_pos, self.init_rot = (-0.65, 0.3, 0.8), (1.0, 0.0, 0.0, 0.0)
        self.actuators = {
            "right_arm": ImplicitActuatorCfg(
                joint_names_expr=["shoulder_.*", "elbow_.*", "wrist_.*"],
                effort_limit=1e6,         # Allows very high effort
                velocity_limit=1e6,       # Allows very high velocity
                stiffness=1e6,            # Allows very high stiffness
                damping=1e3,              # Enough damping to prevent oscillations
            ),
            "right_hand": ImplicitActuatorCfg(
                joint_names_expr=["RHand_.*"],
                effort_limit=1e6,         # Allows very high effort
                velocity_limit=1e6,       # Allows very high velocity
                stiffness=1e6,            # Allows very high stiffness
                damping=1e3,              # Enough damping to prevent oscillations
            )
        }
        self.arm_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        self.n_finger_joint = 18
        self.hand_util = HondaHandUtils(grasp_type=grasp_type)
        self.off_camera_sensor = True
        self.off_contact_sensor = True
        self.first_person_camera = False
        super().__init__(grasp_type=grasp_type, is_training=is_training)


def get_robot_cfg(robot_name: str, grasp_type: str = "active", is_training: bool = True) -> RobotCfg:
    """Get the robot configuration based on the robot name."""
    if robot_name == "nextage-shadow":
        return NextageShadowRobotCfg(grasp_type=grasp_type, is_training=is_training)
    elif robot_name == "shadow":
        return ShadowRobotCfg(grasp_type=grasp_type, is_training=is_training)
    elif robot_name == "ur10-honda":
        return HondaRobotCfg(grasp_type=grasp_type, is_training=is_training)
    else:
        raise ValueError(f"Unknown robot name: {robot_name}")
