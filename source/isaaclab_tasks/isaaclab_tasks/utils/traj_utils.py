import torch
from typing import Callable, List, Optional, Sequence, Tuple, Dict, Any

from isaaclab_tasks.utils._math import quat_slerp_batch
from isaaclab.utils.math import quat_mul

# -----------------------------------------------------------------------------
#  Main class
# -----------------------------------------------------------------------------

class ReferenceTrajInfo:
    """Reference trajectory generator for multi‑fingered hands (IsaacLab).

    Example default timeline (modifiable via `subtasks`):
    ```
    [
      {"name": "init",     "start": 0.00, "end": 0.05},
      {"name": "approach", "start": 0.05, "end": 0.40},
      {"name": "grasp",    "start": 0.40, "end": 0.60},
      {"name": "pick",     "start": 0.60, "end": 0.80,
       "param": {"pick_delta": [0,0,0.05]}},
      {"name": "release",  "start": 0.80, "end": 1.00,
       "param": {"open_pose": [0]*DOF}},
    ]
    ```
    """

    # ------------------------------------------------------------------
    #  construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        hand_module,
        cwp_predictor,
        mode: str,
        pick_height: float,
        env_origins: torch.Tensor | None = None,
    ) -> None:
        self.device = device
        self.num_envs = num_envs
        self.hand_module = hand_module
        self.cwp_predictor = cwp_predictor
        self.finger_coupling_rule = hand_module.couplingRuleTensor
        self.finger_decoupling_rule = hand_module.decouplingRuleTensor
        self.n_hand_joints = len(hand_module.hand_full_joint_names)

        # persistent state -------------------------------------------------
        self.handP_world         = torch.zeros((num_envs, 3), device=device)
        self.handQ_world         = torch.zeros((num_envs, 4), device=device)
        self.handP_world_pre     = torch.zeros((num_envs, 3), device=device)
        self.handQ_world_pre     = torch.zeros((num_envs, 4), device=device)

        self.objectP_world      = torch.zeros((num_envs, 3), device=device)
        self.objectQ_world      = torch.zeros((num_envs, 4), device=device)
        self.objectScale        = torch.ones((num_envs, 3), device=device)  # scale of the object

        self.cwp_pos               = torch.zeros((num_envs, len(self.hand_module.position_tip_links), 3), device=device)  # current workspace position
        self.cwp_quat              = torch.zeros((num_envs, 4), device=device)  # current workspace rotation
        self.contact_center_world = torch.zeros((num_envs, 3), device=device)  # contact center world position

        self.hand_preshape_joint = torch.zeros((num_envs, self.n_hand_joints), device=device)
        self.hand_shape_joint    = torch.zeros((num_envs, self.n_hand_joints), device=device)

        self.env_origins = env_origins

        self.phase_idx = -torch.ones(num_envs, dtype=torch.long , device=device)
        self.pick_flg  = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.approach_flg = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.all_done  = torch.zeros(num_envs, dtype=torch.bool, device=device)

        # default poses ----------------------------------------------------
        self.default_open_pose = torch.Tensor(self.hand_module.preshape_joint).to(device)

        if mode in ["train", "eval", "collect"]:
            subtasks = [
                # {"name": "init",     "start": 0.00, "end": 0.05},
                {"name": "approach", "start": 0.00, "end": 0.40},
                {"name": "grasp",    "start": 0.40, "end": 0.80},
                {"name": "bring",     "start": 0.80, "end": 0.9, "param": {"deltaP": [0.0, 0.0, pick_height]}},
                {"name": "bring",     "start": 0.9, "end": 1.00, "param": {"deltaP": [0.0, 0.0, 0.0]}},
                # {"name": "release",  "start": 0.80, "end": 1.00,
                #  "param": {"open_pose": open_pose.tolist()}},
            ]
        # elif mode in ["test", "collect"]:
        #     subtasks = [
        #         {"name": "approach", "start": 0.00, "end": 0.20},
        #         {"name": "grasp",    "start": 0.20, "end": 0.40},
        #         {"name": "bring",     "start": 0.40, "end": 0.45,
        #          "param": {"delta": [0.0, 0.0, pick_height]}},
        #         {"name": "bring",     "start": 0.9, "end": 1.00,
        #          "param": {"delta": [0.0, 0.0, 0.0]}},
        #     ]
        elif mode == "demo":
            # debug
            if self.hand_module.hand_laterality == "left":
                subtasks = [
                    # {"name": "init",     "start": 0.00, "end": 0.05},
                    {"name": "approach", "start": 0.0, "end": 0.4},
                    {"name": "grasp",    "start": 0.4, "end": 0.7},
                    {"name": "bring",     "start": 0.7, "end": 1.0,  "param": {"deltaP": [0.0, -0.0, 0.1], "deltaQ": [0.7071, -0.7071, 0.0, 0.0]}},
                ]
            else:
                subtasks = [
                    {"name": "init",     "start": 0.00, "end": 1.0, "param": {"init_pose": ([0.0, -0.5, 1.5], [0.1228, 0.6964, 0.6964, 0.1228])}},
                ]

            if self.hand_module.hand_laterality == "left":
                subtasks = [
                    # {"name": "approach", "start": 0.00, "end": 0.1, "param": {"prepare": True}},
                    {"name": "approach", "start": 0.0, "end": 0.2},
                    {"name": "grasp",    "start": 0.2, "end": 0.3},
                    {"name": "bring",     "start": 0.3, "end": 0.4,  "param": {"deltaP": [0.0, 0.0, 0.1],  "deltaQ": [1, 0, 0, 0]}}, # [0.8660, -0.5000, 0.0, 0.0]}}, #"deltaQ": [0.7071, -0.7071, 0.0, 0.0]}},
                    {"name": "bring",     "start": 0.4, "end": 0.8,  "param": {"deltaP": [0.0, 0.0, 0.0]}},
                    {"name": "release",  "start": 0.8, "end": 0.9},
                    {"name": "bring",     "start": 0.9, "end": 1.0,  "param": {"deltaP": [0.0, 0.2, 0.0]}},
                ]
                # subtasks = [
                #     {"name": "init",     "start": 0.00, "end": 1.0, "param": {"init_pose": ([0.0, -0.5, 1.5], [0.1228, 0.6964, 0.6964, 0.1228])}},
                # ]
            else:
                # subtasks = [
                #     # {"name": "approach", "start": 0.00, "end": 0.1, "param": {"prepare": True}},
                #     {"name": "approach", "start": 0.0, "end": 0.2},
                #     {"name": "grasp",    "start": 0.2, "end": 0.3},
                #     {"name": "bring",     "start": 0.3, "end": 0.4,  "param": {"deltaP": [0.0, 0.0, 0.1],  "deltaQ": [1, 0, 0, 0]}}, #"deltaQ": [0.7071, -0.7071, 0.0, 0.0]}},
                #     {"name": "bring",     "start": 0.4, "end": 0.8,  "param": {"deltaP": [0.0, 0.0, 0.0]}},
                #     {"name": "release",  "start": 0.8, "end": 0.9},
                #     {"name": "bring",     "start": 0.9, "end": 1.0,  "param": {"deltaP": [0.0, 0.2, 0.0]}},
                # ]
                subtasks = [
                    {"name": "init",     "start": 0.00, "end": 0.4, "param": {"init_pose": ([0.0, -0.5, 1.5], [0.1228, 0.6964, 0.6964, 0.1228])}},
                    # {"name": "approach", "start": 0.4, "end": 0.5, "param": {"prepare": True}},
                    {"name": "approach", "start": 0.4, "end": 0.6},
                    {"name": "grasp",    "start": 0.6, "end": 0.7},
                    {"name": "bring",     "start": 0.7, "end": 1.0,  "param": {"deltaP": [0.0, 0.0, 0.0]}},
                ]
                # subtasks = [
                #     {"name": "init",     "start": 0.00, "end": 1.0, "param": {"init_pose": ([0.0, -0.5, 1.5], [0.1228, 0.6964, 0.6964, 0.1228])}},
                # ]

        for st in subtasks:
            st.setdefault("param", {})
        self.subtasks: List[Dict[str, Any]] = subtasks
        self.phase_end_times = torch.tensor([st["end"] for st in self.subtasks], device=device)

        # bind handlers -----------------------------------------------------
        handlers = {
            "init":     self._h_init,
            "approach": self._h_approach,
            "grasp":    self._h_grasp,
            "bring":     self._h_bring,
            "release":  self._h_release,
        }
        for st in self.subtasks:
            st["fn"] = handlers[st["name"]]

        # constants ---------------------------------------------------------
        # self._init_pos  = torch.tensor([0.0, 0.0, 0.9], device=device)
        # self._init_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)

    # ------------------------------------------------------------------
    #  helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _lerp(a: torch.Tensor, b: torch.Tensor, t: torch.Tensor):
        return a + (b - a) * (t if t.ndim > 1 else t.unsqueeze(-1))

    def _interp_eef(self, idx, ratio, dP=None, dQ=None):
        if ratio.ndim == 1:
            ratio = ratio[:, None]
        pos  = self._lerp(self.handP_world_pre[idx], self.handP_world[idx], ratio)
        quat = quat_slerp_batch(self.handQ_world_pre[idx], self.handQ_world[idx], ratio)
        if dP is not None:
            pos += dP

        if dQ is not None:
            # quat = quat_mul(dQ, quat)
            # WARNING:
            quat = quat_mul(quat, dQ)  # dQ is in local frame, so we need to apply it after the interpolation
        return pos.float(), quat.float()

    def _interp_fingers(self, idx, ratio, dJ=None):
        if ratio.ndim == 1:
            ratio = ratio[:, None]
        joints = self._lerp(self.hand_preshape_joint[idx], self.hand_shape_joint[idx], ratio)
        if dJ is not None:
            joints += dJ
        # return joints.float()
        return self.finger_coupling_rule(joints).float()

    def _estimate_cwp(self, env_ids: torch.Tensor):
        cwp = self.cwp_predictor.predict(
            obj_position=self.objectP_world[env_ids],
            obj_orientation=self.objectQ_world[env_ids],
            obj_scale=self.objectScale[env_ids],
        )
        ref_traj_cfg = self.hand_module.getReferenceTrajInfo(
            num_envs=len(env_ids), cwp=cwp, device=self.device
        )
        self.cwp_pos[env_ids] = cwp["position"].to(self.device)
        self.cwp_quat[env_ids] = cwp["orientation"].to(self.device)
        self.update(env_ids, **ref_traj_cfg)
    # ------------------------------------------------------------------
    #  phase handlers
    # ------------------------------------------------------------------
    def _h_init(self, idx, ratio, params, **_):
        _init_pos, _init_quat = params["init_pose"]
        _init_pos, _init_quat = torch.tensor(_init_pos, device=self.device), torch.tensor(_init_quat, device=self.device)
        pos  = _init_pos.expand(len(idx), 3) + self.env_origins[idx] # world coordinates
        quat = _init_quat.expand(len(idx), 4)
        fingers = self.default_open_pose.expand(len(idx), -1)
        return pos, quat, fingers

    def _h_approach(self, idx, ratio, params, action_handP=None, action_handQ=None, action_hand_joint=None, **_):
        first = torch.logical_and(ratio > 0, ~self.approach_flg[idx])
        is_prepare = params.get("prepare", False)
        if first.any():
            # detect
            env_ids = idx[first]
            self._estimate_cwp(env_ids)
            self.approach_flg[env_ids] = True
            if action_handP is not None and action_handQ is not None and action_hand_joint is not None:
                # make sure first action is not applied.
                action_handP[env_ids] = torch.zeros_like(action_handP[env_ids])
                action_handQ[env_ids] = torch.zeros_like(action_handQ[env_ids])
                action_hand_joint[env_ids] = torch.zeros_like(action_hand_joint[env_ids])

        if is_prepare:
            pos, quat = self._interp_eef(idx, torch.zeros_like(ratio))
            fingers = self._interp_fingers(idx, torch.zeros_like(ratio))
        else:
            pos, quat = self._interp_eef(idx, ratio, action_handP, action_handQ)
            fingers   = self._interp_fingers(idx, torch.zeros_like(ratio), action_hand_joint)
        return pos, quat, fingers

    def _h_grasp(self, idx, ratio, params, action_handP=None, action_handQ=None, action_hand_joint=None, **_):
        pos, quat = self._interp_eef(idx, torch.ones_like(ratio), action_handP, action_handQ)
        fingers   = self._interp_fingers(idx, ratio, action_hand_joint)
        return pos, quat, fingers

    def _h_bring(self, idx, ratio, params, current_handP_world=None, current_handQ_world=None, current_hand_joint=None, **_):
        deltaP, deltaQ = params["deltaP"], params.get("deltaQ", [1.0, 0.0, 0.0, 0.0])
        deltaP = torch.tensor(deltaP, device=self.device).unsqueeze(0)
        deltaQ = torch.tensor(deltaQ, device=self.device).unsqueeze(0)
        first = torch.logical_and(ratio > 0, ~self.pick_flg[idx])
        if first.any():
            self.handP_world[idx[first]] = current_handP_world[first] + deltaP
            self.handQ_world[idx[first]] = quat_mul(deltaQ.repeat(len(idx[first]), 1), current_handQ_world[first])
            self.hand_shape_joint[idx[first]] = self.finger_decoupling_rule(current_hand_joint[first])
            self.pick_flg[idx[first]]    = True
        pos, quat = self._interp_eef(idx, ratio)
        fingers   = self._interp_fingers(idx, torch.ones_like(ratio))
        return pos, quat, fingers

    def _h_release(self, idx, ratio, params, current_hand_joint, **_):
        pos, quat = self._interp_eef(idx, torch.ones_like(ratio))
        if ratio.ndim == 1:
            ratio = ratio[:, None]
        fingers = self._lerp(current_hand_joint, self.default_open_pose.expand(len(idx), -1), ratio)
        return pos, quat, fingers

    # ------------------------------------------------------------------
    #  main API
    # ------------------------------------------------------------------
    def get(
        self,
        env_slice: slice | torch.Tensor,
        timestep: torch.Tensor,
        *,
        current_handP_world: torch.Tensor,
        current_handQ_world: torch.Tensor,
        current_hand_joint: torch.Tensor,
        objectP_world: torch.Tensor,
        objectQ_world: torch.Tensor,
        objectScale: torch.Tensor,
        action_handP: Optional[torch.Tensor] = None,
        action_handQ: Optional[torch.Tensor] = None,
        action_hand_joint: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute reference trajectory for selected envs at *timestep* (0‑1).
            External code should pass measured pose/joints each frame.
        """
        if isinstance(env_slice, slice):
            env_idx = torch.arange(env_slice.start, env_slice.stop, device=self.device)
        elif isinstance(env_slice, torch.Tensor):
            env_idx = env_slice.to(self.device)
        N = len(env_idx)

        # -------- phase transition & auto‑cache ---------------------------
        new_phase = torch.bucketize(timestep, self.phase_end_times)
        changed   = new_phase != self.phase_idx[env_idx]
        if changed.any():
            idx_chg = env_idx[changed]
            self.handP_world_pre[idx_chg]     = current_handP_world[changed]
            self.handQ_world_pre[idx_chg]     = current_handQ_world[changed]
            self.hand_preshape_joint[idx_chg] = self.finger_decoupling_rule(current_hand_joint[changed])

            self.pick_flg[idx_chg]            = False
            self.approach_flg[idx_chg]        = False
            # self.cwp[idx_chg] = torch.zeros(
            #     (len(idx_chg), len(self.hand_module.position_tip_links), 3), device=self.device
            # )
            self.phase_idx[idx_chg]           = new_phase[changed]

        if objectP_world is not None and objectQ_world is not None and objectScale is not None:
            self.objectP_world[env_idx] = objectP_world
            self.objectQ_world[env_idx] = objectQ_world
            self.objectScale[env_idx] = objectScale

        # -------- allocate outputs ---------------------------------------
        eef_pos  = torch.zeros((N, 3), device=self.device)
        eef_quat = torch.zeros((N, 4), device=self.device)
        fingers  = torch.zeros((N, self.default_open_pose.numel()), device=self.device)

        def _get_kwargs(mask: torch.Tensor) -> Dict[str, torch.Tensor]:
            common_kwargs = dict(
                action_handP=action_handP[mask] if action_handP is not None else None,
                action_handQ=action_handQ[mask] if action_handQ is not None else None,
                action_hand_joint=action_hand_joint[mask] if action_hand_joint is not None else None,
                current_handP_world=current_handP_world[mask],
                current_handQ_world=current_handQ_world[mask],
                current_hand_joint=current_hand_joint[mask],
            )
            return common_kwargs

        # -------- per‑phase computation -----------------------------------
        for p, st in enumerate(self.subtasks):
            mask = new_phase == p
            if not mask.any():
                continue
            idx   = env_idx[mask]
            ratio = torch.clamp((timestep[mask] - st["start"]) / (st["end"] - st["start"]), 0.0, 1.0)
            pos, quat, fing = st["fn"](idx, ratio, st["param"], **_get_kwargs(mask))
            eef_pos[mask]   = pos
            eef_quat[mask]  = quat
            fingers[mask]   = fing

        # -------- completion flag ----------------------------------------
        self.all_done[env_idx] = timestep >= 1.0
        return eef_pos, eef_quat, fingers

    def update(
        self, env_ids: torch.Tensor, handP_world: torch.Tensor, handQ_world: torch.Tensor, handP_world_pre: torch.Tensor,
        handQ_world_pre: torch.Tensor, hand_shape_joint: torch.Tensor, hand_preshape_joint: torch.Tensor, contact_center: torch.Tensor
    ):
        """Update the current hand pose and joint state."""
        self.handP_world[env_ids] = handP_world
        self.handQ_world[env_ids] = handQ_world
        self.handP_world_pre[env_ids] = handP_world_pre
        self.handQ_world_pre[env_ids] = handQ_world_pre
        self.hand_shape_joint[env_ids] = hand_shape_joint
        self.hand_preshape_joint[env_ids] = hand_preshape_joint
        self.contact_center_world[env_ids] = contact_center

class ReferenceTrajInfoMulti():
    def __init__(self, keys, num_envs, device, hand_module, cwp_predictor, mode: str, pick_height:float=0.05, env_origins: torch.Tensor | None = None):
        self._keys = keys
        self.device = device
        self.ref_traj_info = {
            key: ReferenceTrajInfo(
                num_envs, device,
                hand_module[key],
                cwp_predictor[key],
                mode, pick_height,
                env_origins
            ) for key in keys
        }

    def get(self, key, *args, **kwargs):
        return self.ref_traj_info[key].get(*args, **kwargs)

    def update(self, key, *args, **kwargs):
        self.ref_traj_info[key].update(*args, **kwargs)

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(f"{name} is private.")
        return {key: getattr(self.ref_traj_info[key], name) for key in self._keys}
