import math
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
import isaaclab.sim as sim_utils


class RobotCfg():
    def __init__(self, robot_name: str):
        if robot_name == "nextage-shadow":
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
                ),
                "head": ImplicitActuatorCfg(
                    joint_names_expr=["HEAD_JOINT[01]"],   # match both DoFs
                    effort_limit=200,            # sane torque
                    velocity_limit=10,
                    stiffness=50,                # PD gains; tune to taste
                    damping=5,
                ),
            }
            self.arm_names = ["RARM_JOINT0", "RARM_JOINT1", "RARM_JOINT2", "RARM_JOINT3", "RARM_JOINT4", "RARM_JOINT5"]
        elif robot_name == "shadow":
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
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos=self.init_joint_pos,
                pos=self.init_pos,
                rot=self.init_rot,
            ),
            actuators=self.actuators,
        )
