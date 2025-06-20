import os
import glob

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.assets import RigidObject, RigidObjectCfg

class ObjCfg():
    def __init__(self, grasp_type: str = "active"):
        self.obj_type = "base"
        self.grasp_type = grasp_type
        self.obj_size_half: tuple[float, float, float] = (0.05, 0.05, 0.05)  # Half size for cuboid objects
        self._obj_cfg = {}
        self.table_height = 0.8
        table_size = (0.8, 0.8, self.table_height)
        self._obj_cfg["table"] = RigidObjectCfg(
            prim_path="/World/envs/env_.*/table",
            spawn=sim_utils.CuboidCfg(
                size=table_size,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=10000),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True,
                    rest_offset=0.0
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.4, 0.4)),
                semantic_tags=[("class", "table")]
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.0, 0.0, self.table_height / 2),
                rot=(0.0, 0.0, 0.0, 1.0)
            )
        )

    def get_obj_cfg(self, key: str):
        return self._obj_cfg[key]


class YCBObjCfg(ObjCfg):
    def __init__(self, grasp_type: str = "active"):
        super().__init__(grasp_type=grasp_type)
        self.obj_type = "ycb"

        self._obj_cfg["obj"] = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Object",  # Must match env pattern
            spawn=sim_utils.UsdFileCfg(
                # usd_path=f"source/isaaclab_assets/data/Props/035_power_drill.usd",
                usd_path=f"source/isaaclab_assets/data/Props/021_bleach_cleaner.usd",
                scale=(0.9, 0.9, 1.3),
                # usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/035_power_drill.usd",
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
                mass_props=sim_utils.MassPropertiesCfg(mass=0.01),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True
                ),
                semantic_tags=[("class", "object")]
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.0, 0.0, self.table_height),  # Spawn above table/ground
                rot=(1.0, 0.0, 0.0, 0.0)
            )
        )

class SQObjCfg(ObjCfg):
    def __init__(self, grasp_type: str = "active"):
        super().__init__(grasp_type=grasp_type)
        self.obj_type = "superquadric"
        if self.grasp_type == "active":
            self.obj_size_half = (0.035, 0.08, 0.08)  # Size of the cuboid obj in meters
        else:
            self.obj_size_half = (0.03, 0.03, 0.10)
        obj_asset_cfgs = [
            sim_utils.UsdFileCfg(
                usd_path=obj_usd_path, scale=(0.1, 0.1, 0.1),
            )
            for obj_usd_path in sorted(glob.glob(os.path.join("source/isaaclab_assets/data/Props/Superquadrics/*", "object.usd")))
        ]
        assert len(obj_asset_cfgs) > 0, "No object USD files found in the specified directory."

        self._obj_cfg["obj"] = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Object",  # Must match env pattern
            spawn=sim_utils.MultiAssetSpawnerCfg(
                assets_cfg=obj_asset_cfgs,
                random_choice=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    max_depenetration_velocity=2000,
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
                semantic_tags=[("class", "object")]
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.0, 0.0, self.table_height),  # Spawn above table/ground
                rot=(1.0, 0.0, 0.0, 0.0)
            )
        )

def get_obj_cfg(obj_type: str, grasp_type: str = "active") -> ObjCfg:
    """Get the object configuration based on the type."""
    if obj_type == "ycb":
        return YCBObjCfg(grasp_type=grasp_type)
    elif obj_type == "superquadric":
        return SQObjCfg(grasp_type=grasp_type)
    else:
        raise ValueError(f"Unsupported object type: {obj_type}. Supported types are 'ycb' and 'superquadric'.")
