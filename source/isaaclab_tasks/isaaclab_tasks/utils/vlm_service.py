import torch
import multiprocessing
import cv2
import numpy as np
from datetime import datetime
import os
import queue # スレッドを好む場合のスレッド用
import threading # スレッドを好む場合のスレッド用

# 既存のセットアップから利用可能であると仮定
#from isaaclab_tasks.utils.gpt_video_checker_progress import ask_gpt
from gpt_video_checker_progress_phi4 import ask_phi4
# from .nextage_shadow_grasp_env import load_credentials, init_vlm_client

# サービス用に認証情報を一度だけロード
# creds = load_credentials("source/isaaclab_tasks/isaaclab_tasks/utils/auth.env")
# client, client_params = init_vlm_client(creds)
# 説明のためのダミークライアント

def vlm_evaluation_service(
    frame_queue: multiprocessing.Queue,
    reward_results_dict: multiprocessing.Manager.dict,
    stop_event: multiprocessing.Event
):
    print("VLM評価サービスが開始されました。")
    while not stop_event.is_set():
        try:
            # 停止イベントのチェックを可能にするため、タイムアウト付きでアイテムを取得
            env_id, step_in_episode, frames = frame_queue.get(timeout=0.1)
            # VLMのためにtorchテンソルのリストをnumpy配列のリストに変換
            np_frames = [cv2.cvtColor(f.cpu().numpy(), cv2.COLOR_BGR2RGB) for f in frames]

            # 実際のVLMクライアントを呼び出す
            #progress = ask_gpt(client, client_params, np_frames)
            progress = ask_phi4(np_frames) # phi4を使用する場合

            # 結果を保存
            key = f"{env_id}_{step_in_episode}"
            reward_results_dict[key] = progress
            print(f"環境{env_id}のステップ{step_in_episode}でVLMが評価されました: 進捗 = {progress:.2f}")

        except queue.Empty:
            # キューが空の場合、ループを続行し、停止イベントを確認
            pass
        except Exception as e:
            print(f"VLM評価サービスのエラー: {e}")
            # オプションで、回復可能なエラーの場合はアイテムをキューに戻すことも可能
            # frame_queue.put((env_id, step_in_episode, frames))
    print("VLM評価サービスが停止されました。")

# 使用例（テスト用であり、メインのRLループの一部ではない）
if __name__ == "__main__":
    manager = multiprocessing.Manager()
    frame_q = manager.Queue()
    results_dict = manager.dict()
    stop = manager.Event()

    p = multiprocessing.Process(target=vlm_evaluation_service, args=(frame_q, results_dict, stop))
    p.start()

    # フレームのプッシュをシミュレート
    dummy_frame = torch.zeros((100, 100, 3), dtype=torch.uint8) # 例のフレーム
    frame_q.put((0, 10, [dummy_frame.clone(), dummy_frame.clone()]))
    frame_q.put((1, 20, [dummy_frame.clone(), dummy_frame.clone()]))
    frame_q.put((0, 30, [dummy_frame.clone()]))

    time.sleep(2) # サービスが処理する時間を与える

    print("結果:", dict(results_dict))

    stop.set()
    p.join()