
import isaaclab.envs.mdp as mdp
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm


@configclass
class EventCfg:
    """Configuration for events."""
    randomize_scale: EventTerm = None


def create_grasp_event_cfg(base_obj_size) -> EventCfg:
    # This event term randomizes the scale of the cube.
    # The mode is set to 'prestartup', which means that the scale is randomize on the USD stage before the
    # simulation starts.
    # Note: USD-level randomizations require the flag 'replicate_physics' to be set to False.
    randomize_scale = EventTerm(
        func=mdp.randomize_rigid_body_scale,
        mode="prestartup",
        params={
            # scale relative to the base size x(0.5, 1.25)
            "scale_range": {"x": (0.5 * base_obj_size[0], 1.25 * base_obj_size[0]),
                           "y": (0.5 * base_obj_size[1], 1.25 * base_obj_size[1]),
                           "z": (0.5 * base_obj_size[2], 1.25 * base_obj_size[2])},
            "asset_cfg": SceneEntityCfg("obj"),
        },
    )

    return EventCfg(randomize_scale=randomize_scale)

# This event term resets the base position of the cube.
# The mode is set to 'reset', which means that the base position is reset whenever
# the environment instance is reset (because of terminations defined in 'TerminationCfg').
# reset_base = EventTerm(
#     func=mdp.reset_root_state_uniform,
#     mode="reset",
#     params={
#         "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
#         "velocity_range": {
#             "x": (-0.5, 0.5),
#             "y": (-0.5, 0.5),
#             "z": (-0.5, 0.5),
#         },
#         "asset_cfg": SceneEntityCfg("cube"),
#     },
# )

# This event term randomizes the visual color of the cube.
# Similar to the scale randomization, this is also a USD-level randomization and requires the flag
# 'replicate_physics' to be set to False.
# randomize_color = EventTerm(
#     func=mdp.randomize_visual_color,
#     mode="prestartup",
#     params={
#         "colors": {"r": (0.0, 1.0), "g": (0.0, 1.0), "b": (0.0, 1.0)},
#         "asset_cfg": SceneEntityCfg("obj"),
#         "mesh_name": "geometry/mesh",
#         "event_name": "rep_obj_randomize_color",
#     },
# )
