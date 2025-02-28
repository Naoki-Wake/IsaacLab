#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
このサンプルは、differential inverse kinematics (IK) コントローラーを用いて
UR10ロボットの手先を適当に動かす例です。
Isaac-Reach-UR10-v0 の設定をベースに、複数の環境上でエンドエフェクタの目標姿勢を変更します。

Usage:
    ./isaaclab.sh -p scripts/tutorials/test/ik_control_ur10.py --robot ur10 --num_envs 1
"""

"""Launch Isaac Sim Simulator first."""
import argparse

from isaaclab.app import AppLauncher

# argparseの引数設定
parser = argparse.ArgumentParser(description="Differential IK controller with UR10 (Isaac-Reach-UR10-v0)")
parser.add_argument("--num_envs", type=int, default=32, help="スポーンする環境の数")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Omniverseアプリの起動
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import subtract_frame_transforms

# 事前定義されたロボット設定（Isaac Lab Assets）
from isaaclab_assets import UR10_CFG  # UR10用設定（Isaac-Reach-UR10-v0)

@configclass
class TableTopSceneCfg(InteractiveSceneCfg):
    """テーブルトップシーンの設定（UR10ロボット版）"""

    # 地面
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    # ライト
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", 
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # テーブル（マウント）
    mount = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd", 
            scale=(2.0, 2.0, 2.0)
        ),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd", 
            scale=(1.0, 1.0, 1.0)
        ),
    )
    # ロボット（UR10）
    robot = UR10_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """シミュレーションループの実行"""
    # シーンエンティティの抽出
    robot = scene["robot"]

    # Differential IK コントローラーの作成
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)

    # 可視化用マーカーの設定
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    # エンドエフェクタの目標姿勢（位置＋クォータニオン）のリスト
    ee_goals = [
        [0.6,  0.3, 0.8, 0.707, 0.0, 0.707, 0.0],
        [0.6, -0.3, 0.7, 0.707, 0.707, 0.0, 0.0],
        [0.6,  0.0, 0.6, 0.0,   1.0,   0.0, 0.0],
    ]
    ee_goals = torch.tensor(ee_goals, device=sim.device)
    current_goal_idx = 0

    # コントローラーの入力バッファを初期化
    ik_commands = torch.zeros(scene.num_envs, diff_ik_controller.action_dim, device=robot.device)
    ik_commands[:] = ee_goals[current_goal_idx]

    # UR10用のエンドエフェクタ設定（全ジョイント名とエンドリンク）
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=[".*"], body_names=["ee_link"])
    robot_entity_cfg.resolve(scene)
    # 固定ベースの場合、Jacobianの対象インデックスの調整
    if robot.is_fixed_base:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
    else:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0]

    # シミュレーションステップの設定
    sim_dt = sim.get_physics_dt()
    count = 0

    while simulation_app.is_running():
        if count % 150 == 0:
            # 一定ステップごとにリセットし目標を切り替え
            count = 0
            # ジョイント状態のリセット
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()
            # 現在の目標をコントローラーにセット
            ik_commands[:] = ee_goals[current_goal_idx]
            diff_ik_controller.reset()
            diff_ik_controller.set_command(ik_commands)
            # 目標インデックスの更新
            current_goal_idx = (current_goal_idx + 1) % len(ee_goals)
        else:
            # シミュレーションから各種データを取得
            jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
            ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
            root_pose_w = robot.data.root_state_w[:, 0:7]
            joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
            # エンドエフェクタの座標をルート座標系に変換
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7],
                ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
            # Differential IK で目標に対する関節位置指令を計算
            joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

            # 目標関節位置をロボットに適用
            robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
        scene.write_data_to_sim()
        sim.step()
        count += 1
        scene.update(sim_dt)

        # マーカーの更新
        ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
        ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        # 各環境の原点オフセットを加味して目標を可視化
        goal_marker.visualize(ik_commands[:, 0:3] + scene.env_origins, ik_commands[:, 3:7])

def main():
    """メイン関数"""
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    scene_cfg = TableTopSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene)

if __name__ == "__main__":
    main()
    simulation_app.close()
