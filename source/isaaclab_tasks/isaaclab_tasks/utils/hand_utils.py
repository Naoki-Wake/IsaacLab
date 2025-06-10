import os
import sys
import copy
import numpy as np
import torch
# import transformations

from isaaclab_tasks.utils._math import quat_slerp_batch
from isaaclab_tasks.utils.third_party.tf import transformations
from isaaclab_tasks.utils.third_party.urdf_parser_py.urdf import URDF
import isaaclab_tasks.utils.fk_using_urdf as urdf_fk
from source.isaaclab.isaaclab.utils.math import quat_mul

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class ReferenceTrajInfo:
    def __init__(self, num_envs, device, finger_coupling_rule: callable, n_hand_joints: int, eval_mode: bool, pick_height:float=0.05):
        # placeholder for the reference trajectory
        self.num_envs = num_envs
        self.device = device
        self.handP_pybworld = torch.zeros((num_envs, 3), device=device)
        self.handQ_pybworld = torch.zeros((num_envs, 4), device=device)
        self.handP_pybworld_pre = torch.zeros((num_envs, 3), device=device)
        self.handQ_pybworld_pre = torch.zeros((num_envs, 4), device=device)
        self.hand_preshape_joint = torch.zeros((num_envs, n_hand_joints), device=device)
        self.hand_shape_joint = torch.zeros((num_envs, n_hand_joints), device=device)

        self.pick_flg = torch.zeros((num_envs,), device=device).bool()

        # The ratio of the total timestep used for end-effector and finger motion.

        if eval_mode:
            self.subtasks_span = {
                "approach": [0.0, 0.3], "grasp": [0.3, 0.45], "pick": [0.45, 1.0],
            }
        else:
            self.subtasks_span = {
                "approach": [0.0, 0.6], "grasp": [0.6, 0.9], "pick": [0.9, 1.0],
            }
        self._pick_diff = torch.tensor([0., 0., pick_height], device=self.device).unsqueeze(0)
        self.finger_couping_rule = finger_coupling_rule

    def update(self, env_slice, handP_pybworld, handQ_pybworld, handP_pybworld_pre, handQ_pybworld_pre, hand_preshape_joint, hand_shape_joint, reset=False):
        """Update the reference trajectory for the given environment slice"""
        self.handP_pybworld[env_slice] = handP_pybworld.float()
        self.handQ_pybworld[env_slice] = handQ_pybworld.float()
        self.handP_pybworld_pre[env_slice] = handP_pybworld_pre.float()
        self.handQ_pybworld_pre[env_slice] = handQ_pybworld_pre.float()
        self.hand_preshape_joint[env_slice] = hand_preshape_joint.float()
        self.hand_shape_joint[env_slice] = hand_shape_joint.float()
        if reset: self.pick_flg[env_slice] = False

    def _get_eef_reftraj(self, env_slice, interp_ratio, action_handP=None, action_handQ=None):
        """Get the reference trajectory for the given environment slice"""
        # Interpolate the reference trajectory
        if interp_ratio.ndim == 1:
            interp_ratio = interp_ratio[:, None]
        interp_pos = (1 - interp_ratio) * self.handP_pybworld_pre[env_slice] + interp_ratio * self.handP_pybworld[env_slice]
        interp_quat = quat_slerp_batch(self.handQ_pybworld_pre[env_slice], self.handQ_pybworld[env_slice], interp_ratio)

        if action_handP is not None:
            interp_pos = interp_pos + action_handP

        if action_handQ is not None:
            interp_quat = quat_mul(action_handQ, interp_quat)

        return interp_pos.float(), interp_quat.float()

    def _get_finger_reftraj(self, env_slice, interp_ratio, action_hand_joint=None):
        """Get the reference trajectory for the given environment slice"""
        # Interpolate the reference trajectory
        if interp_ratio.ndim == 1:
            interp_ratio = interp_ratio[:, None]
        finger_pos = (1 - interp_ratio) * self.hand_preshape_joint[env_slice] + interp_ratio * self.hand_shape_joint[env_slice]

        if action_hand_joint is not None:
            finger_pos = finger_pos + action_hand_joint

        finger_pos = self.finger_couping_rule(finger_pos)
        return finger_pos.float()

    def _get_timestep_subtasks(self, timestep):
        timestep_subtasks = {}
        for subtask, (start, end) in self.subtasks_span.items():
            timestep_subtask = (timestep - start) / (end - start)
            if subtask == "pick":
                # pick in the first half of the timestep, then stop (i.e., clmaped to 1)
                timestep_subtask *= 2

            timestep_subtasks[subtask] = torch.clamp(timestep_subtask, 0, 1)
        return timestep_subtasks

    def get(self, env_slice, timestep, current_handP_world=None, current_handQ_world=None, current_hand_joint=None, action_handP=None, action_handQ=None, action_hand_joint=None):
        """Get the reference trajectory for the given environment slice"""
        # Get the reference trajectory for the end-effector
        timestep_subtasks = self._get_timestep_subtasks(timestep)
        eef_pos, eef_quat = self._get_eef_reftraj(env_slice, timestep_subtasks["approach"], action_handP, action_handQ)
        finger_pos = self._get_finger_reftraj(env_slice, timestep_subtasks["grasp"], action_hand_joint)

        _pick_mask = timestep_subtasks["pick"] > 0
        if _pick_mask.any():
            assert (current_handP_world is not None) and (current_handQ_world is not None) and (current_hand_joint is not None), "current_handP_world, current_handQ_world, current_hand_joint must be provided for pick subtask"
            # Get the reference trajectory for the pick.
            # Set the pick goal to be X cm above the current hand position while keeping the orientation and finger position the same.
            env_indices = torch.arange(env_slice.start, env_slice.stop, device=self.device) # changed to tensor index
            _pick_first_mask = torch.logical_and(_pick_mask, ~self.pick_flg[env_indices]) # only update the target eef position in the first timestep

            env_indices_pick, env_indices_pick_first = env_indices[_pick_mask], env_indices[_pick_first_mask]

            if env_indices_pick_first.numel() > 0:
                # Safely update reference trajectory
                self.update(
                    env_indices_pick_first,
                    (current_handP_world + self._pick_diff)[_pick_first_mask], current_handQ_world[_pick_first_mask],
                    current_handP_world[_pick_first_mask], current_handQ_world[_pick_first_mask],
                    current_hand_joint[_pick_first_mask], current_hand_joint[_pick_first_mask],
                )
                self.pick_flg[env_indices_pick_first] = True
            eef_pos[env_indices_pick], eef_quat[env_indices_pick] = self._get_eef_reftraj(env_indices_pick, timestep_subtasks["pick"][_pick_mask])
            finger_pos[env_indices_pick] = self._get_finger_reftraj(env_indices_pick, timestep_subtasks["pick"][_pick_mask])

        return eef_pos, eef_quat, finger_pos


class HandUtils:
    def __init__(self, grasp_type):
        raise NotImplementedError("HandUtils is not implemented yet")

    def couplingRule(self, jv):
        return jv

    def couplingRuleTensor(self, jv):
        return jv

    def calcApproachDirectionXfrontZup(self, theta, phi):
        return [np.cos(np.deg2rad(theta))*np.cos(np.deg2rad(phi)),
                np.cos(np.deg2rad(theta))*np.sin(np.deg2rad(phi)),
                np.sin(np.deg2rad(theta))]

    def calcHandQuaternionXfrontZupPassive(self, theta, phi):
        raise NotImplementedError("Passive grasp not implemented yet")

    # def calcHandQuaternionXfrontZupActive(self, theta, phi):
    #     qy = transformations.quaternion_about_axis(np.deg2rad(-(90-theta)), [0, 1, 0])
    #     return transformations.quaternion_multiply(qy, [])
    #     # return qy

    def calcHandQuaternionXfrontZupActive(self, theta, phi):
        # Step 1: base rotation (X: 180, Y: 0, Z: -90)
        q_base = transformations.quaternion_from_euler(np.deg2rad(180), np.deg2rad(0), np.deg2rad(-90), axes='rxyz')
        # import pdb; pdb.set_trace()
        # Step 2: rotation around world Y-axis: -(90 - theta)
        q_y_theta = transformations.quaternion_about_axis(
            np.deg2rad(-(90 - theta)), [0, 1, 0]
        )
        # Step 3: apply q_y_theta * q_base (left rotation applied in world frame)
        q = transformations.quaternion_multiply(q_y_theta, q_base)

        return q  # (w, x, y, z)


    def calculateXfrontZup(self, theta, phi):
        # calculate the hand orientation
        # below handQ calculates when contact-web orientation = identity matrix
        if self.grasp_type == "active":
            handQ_pybworld0 = self.calcHandQuaternionXfrontZupActive(theta, phi)
        elif self.grasp_type == "passive":
            handQ_pybworld0 = self.calcHandQuaternionXfrontZupPassive(theta, phi)
        elif self.grasp_type == "lazy":
            handQ_pybworld0 = self.calcHandQuaternionXfrontZupPassive(theta, phi)

        return handQ_pybworld0

    def getReferenceTrajInfo(self, config, device):
        _n_envs = config["num_envs"]
        ref_traj_config = []

        def _get_single_config(_config, idx):
            # get i-th config and convert to numpy
            _config_res = {}
            for key in _config.keys():
                if isinstance(_config[key], torch.Tensor):
                    _config_res[key] = _config[key][idx].cpu().numpy()
                elif isinstance(_config[key], (list, tuple)):
                    _config_res[key] = np.array(_config[key][idx])
                else:
                    _config_res[key] = _config[key]
            return _config_res

        for i in range(_n_envs):
            # get the contact web position and orientation
            _config = _get_single_config(config, i)
            _ref_traj_config = self._handConfigurationFromContactWeb(_config)
            ref_traj_config.append(_ref_traj_config)

        def _list_dict2dict_tensor(_list, dtype, device):
            result = {}
            keys = _list[0].keys()
            for _key in keys:
                _e_list = [_ref_traj_config[_key] for _ref_traj_config in _list]
                if isinstance(_e_list[0], np.ndarray):
                    _e_list = np.stack(_e_list, axis=0)
                elif isinstance(_e_list[0], (list, tuple)):
                    _e_list = np.array(_e_list, dtype=np.float32)
                elif isinstance(_e_list[0], (int, float)):
                    _e_list = np.array(_e_list, dtype=np.float32)
                else:
                    raise ValueError(f"Unsupported type: {type(_e_list[0])}")

                result[_key] = torch.tensor(_e_list, dtype=dtype, device=device)
            return result

        ref_traj_config = _list_dict2dict_tensor(ref_traj_config, dtype=torch.float32, device=device)
        # ref_traj_config["hand_preshape_joint"] = self.couplingRuleTensor(ref_traj_config["hand_preshape_joint"])
        # ref_traj_config["hand_shape_joint"] = self.couplingRuleTensor(ref_traj_config["hand_shape_joint"])
        return ref_traj_config

    def _handConfigurationFromContactWeb(self, config):
        # Estimate hand configuration from contact web
        # Expected input/output quaternions are in wxyz format

        # calculation utils
        def uround(p, k=4):
            p = copy.deepcopy(p)
            for i in range(len(p)):
                p[i] = round(p[i], k)
                if abs(p[i]) < 0.000001:
                        p[i] = 0.0  # -0.0 -> 0.0
            return list(p)

        def rotateUnitVector(q, vec):
            s = np.linalg.norm(np.array(vec))
            if abs(s) < 0.98 or abs(s) > 1.02:
                print('---------------- not a unit vector !!!!!')
                sys.exit(0)
            return transformations.quaternion_multiply(
                transformations.quaternion_multiply(q, list(vec) + [0.]),
                transformations.quaternion_conjugate(q))[:3]

        def _xyzw2wxyz(q): return [q[3], q[0], q[1], q[2]]
        def _wxyz2xyzw(q): return [q[1], q[2], q[3], q[0]]

        p_pyb2cwebt0 = config["grasp_cweb0_position"]
        q_pyb2cwebt0 = _wxyz2xyzw(config["grasp_cweb0_orientation"])

        vd_theta = config["grasp_approach_vertical"]
        vd_phi = config["grasp_approach_horizontal"]
        vd_theta_pre = config.get("grasp_approach_vertical_pre", vd_theta)
        vd_phi_pre = config.get("grasp_approach_horizontal_pre", vd_phi)

        # calculate the hand orientation
        # below handQ calculates when contact-web orientation = identity matrix
        handQ_pybworld0 = self.calculateXfrontZup(vd_theta, vd_phi)
        handQ_pybworld0_pre = self.calculateXfrontZup(vd_theta_pre, vd_phi_pre)

        # rotate handQ depending on the contact-web orientation
        handQ_pybworld = transformations.quaternion_multiply(q_pyb2cwebt0, handQ_pybworld0)
        handQ_pybworld_pre = transformations.quaternion_multiply(q_pyb2cwebt0, handQ_pybworld0_pre)

        # calculate how much to translate from diff atload and goal toplace
        contactP_atload = self._compute_contactP_atload(handQ_pybworld, skip=False)
        handP_pybworld = p_pyb2cwebt0 - contactP_atload
        vd = np.array(self.calcApproachDirectionXfrontZup(vd_theta, vd_phi))
        vd = rotateUnitVector(q_pyb2cwebt0, vd)

        handP_pybworld_pre = np.array(uround(handP_pybworld)) + config["back"] / np.linalg.norm(vd) * vd

        return {
            "handP_pybworld": handP_pybworld,
            "handQ_pybworld": _xyzw2wxyz(handQ_pybworld),
            "handP_pybworld_pre": handP_pybworld_pre,
            "handQ_pybworld_pre": _xyzw2wxyz(handQ_pybworld_pre),
            "hand_preshape_joint": self.preshape_joint,
            "hand_shape_joint": self.shape_joint,
        }

    def _compute_contactP_atload(self, handQ_pybworld, skip=False):
        # transformation from finger tip to hand root

        contactP_atload = np.array([0.0, 0.0, 0.0])
        if skip: return contactP_atload

        # calculate the hand position
        # below calculation performed in ROS coordinate
        tmp = URDF.from_xml_file(self.urdf_path)
        jvtmp = self.couplingRule(copy.deepcopy(self.shape_joint))

        for localization_num in self.virtual_finger:
            chain_name = tmp.get_chain(tmp.get_root(), self.position_tip_links[localization_num], links=False)
            jointstmp = []
            for c in chain_name:
                j = 0.0
                for jidx, jname in enumerate(self.hand_full_joint_names):
                    if jname == c: j = jvtmp[jidx]
                jointstmp += [j]
            root2tip = urdf_fk.chainname2trans(tmp, chain_name, jointstmp, fixed_excluded=False, get_com=True)
            pyb2root = urdf_fk.Transform()
            pyb2root.R = np.array(transformations.quaternion_matrix(handQ_pybworld))[0:3, 0:3]
            pyb2root.T = np.array([0, 0, 0])
            pyb2tip = pyb2root.dot(root2tip)
            contactP_atload += pyb2tip.T
        contactP_atload /= len(self.virtual_finger)
        return contactP_atload
class ShadowHandUtils(HandUtils):
    def __init__(self, grasp_type):
        self.grasp_type = grasp_type
        self.ik_target_link = "rh_forearm"
        # self.shape_joint = [-0.020622028419300796, 0.791124618965486, 0.7165877145100853, 0.0, 0.9159989384779349, 1.2087578520408695, -0.6628910395680104, 0.15071510414034017]
        # self.preshape_joint = [-0.020622028419300796, 0, 0, 0, 0, 1.2087578520408695, 0, 0]

        # 16 dim input
        self.shape_joint = [
            -0.020622028419300796, 0.791124618965486, 0.7165877145100853, 0.0,
            0.0, 0.791124618965486, 0.7165877145100853, 0.0,
            -0.020622028419300796, 0.791124618965486, 0.7165877145100853, 0.0,
            0.9159989384779349, 1.2087578520408695, -0.6628910395680104, 0.15071510414034017
        ]
        self.preshape_joint = [
            -0.020622028419300796, 0, 0, 0,
            0, 0, 0, 0,
            -0.020622028419300796, 0, 0, 0,
            0, 1.2087578520408695, 0, 0
        ]
        self.hand_full_joint_names = [
            "rh_FFJ4", "rh_FFJ3", "rh_FFJ2", "rh_FFJ1",
            "rh_MFJ4", "rh_MFJ3", "rh_MFJ2", "rh_MFJ1",
            "rh_RFJ4", "rh_RFJ3", "rh_RFJ2", "rh_RFJ1",
            "rh_THJ5", "rh_THJ4", "rh_THJ2", "rh_THJ1",
        ]
        self.urdf_path = os.path.join(THIS_DIR, "urdf", "shadowhand_lite.urdf")

        if self.grasp_type == "active":
            self.position_tip_links = ["rh_ffdistal", "rh_mfdistal", "rh_rfdistal", "rh_thdistal"]
            self.force_tip_links = ["rh_ffdistal", "rh_mfdistal", "rh_rfdistal", "rh_thdistal"]
            self.virtual_finger = [1, 3]
        elif self.grasp_type == "passive":
            self.position_tip_links = ["rh_ffdistal", "rh_mfdistal", "rh_rfdistal", "rh_thdistal", "rh_palm"]
            self.force_tip_links = ["rh_ffdistal", "rh_mfdistal", "rh_rfdistal", "rh_thdistal", "rh_ffproximal", "rh_mfproximal", "rh_rfproximal", "rh_palm"]
            self.virtual_finger = [1, 3, 4]

            # passive-force only parameters
            self.force_root_links = ["rh_thproximal", "rh_ffproximal", "rh_mfproximal", "rh_rfproximal"]
            self.hand_point_pairs = [("rh_mfdistal", "rh_thdistal"), ("rh_thdistal", "rh_palm"), ("rh_palm", "rh_mfdistal")]
            # fingertip direction of pair.first for each pair in hand_point_pairs, direction in fingertip coords
            self.tip_point_direction = [[0, -1, 0], [0, -1, 0], [0, -1/np.sqrt(2), 1/np.sqrt(2)]]
        elif self.grasp_type == "lazy":
            self.load_joint = "rh_FFJ3"

            # other settings for loading robot, calculating rewards, etc.
            self.robot_urdf = 'components/robots/shadow/urdf/shadowhand_lite.urdf'
            self.position_tip_links = ["rh_ffmiddle", "rh_mfmiddle", "rh_rfmiddle"]
            self.force_tip_links = ["rh_ffmiddle", "rh_mfmiddle", "rh_rfmiddle", "rh_thmiddle"]
            self.reward_tip_links = {"rh_ffmiddle": "rh_ffmiddle", "rh_mfmiddle": "rh_mfmiddle", "rh_rfmiddle": "rh_rfmiddle"}
            self.virtual_finger = [0]


    def couplingRule(self, jv):
        if len(jv) == 8:
            assert len(jv) == 8, "input jv should have 8 columns"
            ff4 = copy.deepcopy(jv[0])
            ff3 = copy.deepcopy(jv[1])
            ff2 = copy.deepcopy(jv[2])
            ff1 = copy.deepcopy(jv[3])*ff2
            th5 = copy.deepcopy(jv[-4])
            th4 = copy.deepcopy(jv[-3])
            th2 = copy.deepcopy(jv[-2])
            th1 = copy.deepcopy(jv[-1])
            js = [ff4, ff3, ff2, ff1, 0, ff3, ff2, ff1, ff4, ff3, ff2, ff1, th5, th4, th2, th1]
        if len(jv) == 16:
            ff4 = copy.deepcopy(jv[0])
            ff3 = copy.deepcopy(jv[1])
            ff2 = copy.deepcopy(jv[2])
            ff1 = copy.deepcopy(jv[3])*ff2
            mf4 = copy.deepcopy(jv[4])
            mf3 = copy.deepcopy(jv[5])
            mf2 = copy.deepcopy(jv[6])
            mf1 = copy.deepcopy(jv[7])*mf2
            rf4 = copy.deepcopy(jv[8])
            rf3 = copy.deepcopy(jv[9])
            rf2 = copy.deepcopy(jv[10])
            rf1 = copy.deepcopy(jv[11])*rf2
            th5 = copy.deepcopy(jv[-4])
            th4 = copy.deepcopy(jv[-3])
            th2 = copy.deepcopy(jv[-2])
            th1 = copy.deepcopy(jv[-1])
            js = [ff4, ff3, ff2, ff1, mf4, mf3, mf2, mf1, rf4, rf3, rf2, rf1, th5, th4, th2, th1]
        return js

    def couplingRuleTensor(self, jv):
        if jv.shape[1] == 8:
            ff4 = jv[:, 0]
            ff3 = jv[:, 1]
            ff2 = jv[:, 2]
            ff1 = jv[:, 3] * ff2
            th5 = jv[:, 4]
            th4 = jv[:, 5]
            th2 = jv[:, 6]
            th1 = jv[:, 7]
            zeros = torch.zeros(jv.shape[0], device=jv.device)
            js = torch.stack((ff4, ff3, ff2, ff1, zeros, ff3, ff2, ff1, ff4, ff3, ff2, ff1, th5, th4, th2, th1), dim=1)
        elif jv.shape[1] == 16:
            ff4 = jv[:, 0]
            ff3 = jv[:, 1]
            ff2 = jv[:, 2]
            ff1 = torch.clamp(jv[:, 3], min=torch.zeros_like(jv[:, 3])) * ff2
            mf4 = torch.zeros_like(jv[:, 4])
            mf3 = jv[:, 5]
            mf2 = jv[:, 6]
            mf1 = torch.clamp(jv[:, 7], min=torch.zeros_like(jv[:, 7])) * mf2
            rf4 = jv[:, 8]
            rf3 = jv[:, 9]
            rf2 = jv[:, 10]
            rf1 = torch.clamp(jv[:, 11], min=torch.zeros_like(jv[:, 11])) * rf2
            th5 = jv[:, 12]
            th4 = jv[:, 13]
            th2 = jv[:, 14]
            th1 = jv[:, 15]
            js = torch.stack((ff4, ff3, ff2, ff1, mf4, mf3, mf2, mf1, rf4, rf3, rf2, rf1, th5, th4, th2, th1), dim=1)
        return js

class HondaHandUtils(HandUtils):
    def __init__(self, grasp_type):
        self.grasp_type = grasp_type
        self.ik_target_link = "RHand_FOREARM_ROOT_LINK"

        # 16 dim input
        self.hand_full_joint_names = [
            'RHand_WRZ', 'RHand_WRY', 'RHand_I1Z',
            'RHand_M1Z', 'RHand_R1Z', 'RHand_T1Z',
            'RHand_I1Y', 'RHand_M1Y', 'RHand_R1Y',
            'RHand_T1Y', 'RHand_I2Y', 'RHand_M2Y',
            'RHand_R2Y', 'RHand_T2Y', 'RHand_I3Y',
            'RHand_M3Y', 'RHand_R3Y', 'RHand_T3Y'
        ]
        self.preshape_joint = [0.0 for _ in range(len(self.hand_full_joint_names))]
        self.shape_joint = [3.0 for _ in range(len(self.hand_full_joint_names))]
        self.urdf_path = os.path.join(THIS_DIR, "urdf", "shadowhand_lite.urdf")

        if self.grasp_type == "active":
            self.position_tip_links = ["RHand_I3Y_LINK", "RHand_M3Y_LINK", "RHand_R3Y_LINK", "RHand_T3Y_LINK"]
            self.force_tip_links = ["RHand_I3Y_LINK", "RHand_M3Y_LINK", "RHand_R3Y_LINK", "RHand_T3Y_LINK"]
            self.virtual_finger = [1, 3]
        else:
            raise NotImplementedError("Honda hand only supports active grasp type")

    def _compute_contactP_atload(self, handQ_pybworld, skip=False):
        #TODO should be overridden later!
        return np.array([0.0, 0.0, 0.0])
