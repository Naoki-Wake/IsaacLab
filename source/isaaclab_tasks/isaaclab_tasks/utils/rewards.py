import torch


def compute_grasp_reward_active(
    finger_pos, finger_target_pos, th_contact, mf_contact, mf_pos_obj, th_force_direction, mf_force_direction, vf0_direction, vf1_direction,
    before_object_pos, before_object_rot, object_pos, object_rot, tumble_metric, termination,
    reset_buf, progress_buf, max_episode_length, grasp_step, terminate_thresh
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, float, float) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]

    assert len(finger_pos) == len(finger_target_pos)
    # finger position reward
    dist = [torch.norm(finger_pos[] - finger_target_pos[], dim=1)]
    # dist0 = torch.norm(th_pos - vf0_pos, dim=1)
    # dist1 = torch.norm(mf_pos - vf1_pos, dim=1)

    # contact reward
    r_force = (th_contact + mf_contact) / 20

    # # force direction reward
    fdist0 = torch.arccos((th_force_direction*vf0_direction).sum(1))
    fdist1 = torch.arccos((mf_force_direction*vf1_direction).sum(1))
    fdist0[th_contact==0] = 0.
    fdist1[mf_contact==0] = 0.

    # tumble penalty
    is_tumble = torch.logical_or(tumble_metric>torch.pi/6, torch.logical_and(mf_pos_obj[:, 0]<0., mf_pos_obj[:, 2]<0.05))
    tumble = torch.where(is_tumble, -10., 0.)

    # phase miss penalty
    phase = torch.where((progress_buf <= 5) & ((th_contact == 1) | (mf_contact == 1)), -10., 0.)

    # break penalty
    # not implemented

    # termination reward
    position_error = torch.sqrt(((object_pos - before_object_pos) ** 2).sum(-1))
    quat_diff = quat_mul(object_rot, quat_conjugate(before_object_rot))
    rotation_error = torch.abs(-2.0 * torch.rad2deg(torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))))
    is_success = torch.logical_and(termination>terminate_thresh, torch.logical_and(position_error<0.01, rotation_error<20))
    terminationReward = torch.where(is_success, 10., 0.)
    terminationReward[torch.logical_and(termination>terminate_thresh, torch.logical_or(position_error>=0.01, rotation_error>=20))] = -0.
    #terminationReward[torch.logical_and(termination>0.5, torch.logical_or(th_contact==0., mf_contact==0.))] = -20.
    #terminationReward[torch.logical_and(termination<=0.5, progress_buf==max_episode_length-1)] = -(1.-termination[torch.logical_and(termination<=0.5, progress_buf==max_episode_length-1)])
    #terminationReward[torch.logical_and(termination<=0.5, progress_buf==max_episode_length-1)] = 0.

    pos_scale = 10
    reward = torch.log(torch.where(2-pos_scale*dist0<0.1, 0.1, 2-pos_scale*dist0)) \
                + torch.log(torch.where(2-pos_scale*dist1<0.1, 0.1, 2-pos_scale*dist1)) \
                + r_force*10 + tumble + phase
    reward[termination>terminate_thresh]  = terminationReward[termination>terminate_thresh]

    reset = reset_buf
    reset = torch.where(termination > terminate_thresh, torch.ones_like(reset_buf), reset_buf)
    #reset = torch.where((progress_buf <= 5) & ((th_contact == 1) | (mf_contact == 1)), torch.ones_like(reset_buf), reset)
    # reset = torch.where(torch.logical_and(progress_buf < max_episode_length - 1, is_tumble), reset)
    reset = torch.where(torch.logical_and(progress_buf >= grasp_step, position_error>=0.05), torch.ones_like(reset_buf), reset)
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset)

    return reward, dist0, dist1, tumble, phase, terminationReward, reset, is_success

# def compute_grasp_reward_active(
#     th_pos, mf_pos, vf0_pos, vf1_pos, th_contact, mf_contact, mf_pos_obj, th_force_direction, mf_force_direction, vf0_direction, vf1_direction,
#     before_object_pos, before_object_rot, object_pos, object_rot, tumble_metric, termination,
#     reset_buf, progress_buf, max_episode_length, grasp_step, terminate_thresh
# ):
#     # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, float, float) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]

#     # finger position reward
#     dist0 = torch.norm(th_pos - vf0_pos, dim=1)
#     dist1 = torch.norm(mf_pos - vf1_pos, dim=1)

#     # contact reward
#     r_force = (th_contact + mf_contact) / 20

#     # # force direction reward
#     fdist0 = torch.arccos((th_force_direction*vf0_direction).sum(1))
#     fdist1 = torch.arccos((mf_force_direction*vf1_direction).sum(1))
#     fdist0[th_contact==0] = 0.
#     fdist1[mf_contact==0] = 0.

#     # tumble penalty
#     is_tumble = torch.logical_or(tumble_metric>torch.pi/6, torch.logical_and(mf_pos_obj[:, 0]<0., mf_pos_obj[:, 2]<0.05))
#     tumble = torch.where(is_tumble, -10., 0.)

#     # phase miss penalty
#     phase = torch.where((progress_buf <= 5) & ((th_contact == 1) | (mf_contact == 1)), -10., 0.)

#     # break penalty
#     # not implemented

#     # termination reward
#     position_error = torch.sqrt(((object_pos - before_object_pos) ** 2).sum(-1))
#     quat_diff = quat_mul(object_rot, quat_conjugate(before_object_rot))
#     rotation_error = torch.abs(-2.0 * torch.rad2deg(torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))))
#     is_success = torch.logical_and(termination>terminate_thresh, torch.logical_and(position_error<0.01, rotation_error<20))
#     terminationReward = torch.where(is_success, 10., 0.)
#     terminationReward[torch.logical_and(termination>terminate_thresh, torch.logical_or(position_error>=0.01, rotation_error>=20))] = -0.
#     #terminationReward[torch.logical_and(termination>0.5, torch.logical_or(th_contact==0., mf_contact==0.))] = -20.
#     #terminationReward[torch.logical_and(termination<=0.5, progress_buf==max_episode_length-1)] = -(1.-termination[torch.logical_and(termination<=0.5, progress_buf==max_episode_length-1)])
#     #terminationReward[torch.logical_and(termination<=0.5, progress_buf==max_episode_length-1)] = 0.

#     pos_scale = 10
#     reward = torch.log(torch.where(2-pos_scale*dist0<0.1, 0.1, 2-pos_scale*dist0)) \
#                 + torch.log(torch.where(2-pos_scale*dist1<0.1, 0.1, 2-pos_scale*dist1)) \
#                 + r_force*10 + tumble + phase
#     reward[termination>terminate_thresh]  = terminationReward[termination>terminate_thresh]

#     reset = reset_buf
#     reset = torch.where(termination > terminate_thresh, torch.ones_like(reset_buf), reset_buf)
#     #reset = torch.where((progress_buf <= 5) & ((th_contact == 1) | (mf_contact == 1)), torch.ones_like(reset_buf), reset)
#     # reset = torch.where(torch.logical_and(progress_buf < max_episode_length - 1, is_tumble), reset)
#     reset = torch.where(torch.logical_and(progress_buf >= grasp_step, position_error>=0.05), torch.ones_like(reset_buf), reset)
#     reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset)

#     return reward, dist0, dist1, tumble, phase, terminationReward, reset, is_success
