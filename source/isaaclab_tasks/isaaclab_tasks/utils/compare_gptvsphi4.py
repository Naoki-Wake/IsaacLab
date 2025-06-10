from gpt_video_checker_progress import ask_gpt
from gpt_video_checker_progress_phi4 import ask_phi4


video_dir = "/home/nawake/IsaacLab/videos/2025-05-16_18-00-52"

# iterate all videos in the directory
import argparse
import os
import numpy as np
import cv2
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Video progress check via GPT-4 Vision.")
    parser.add_argument("--creds", default="auth.env", help="Env file path.")
    parser.add_argument("--num_frames", type=int, default=1, help="Frames per video.")
    args = parser.parse_args()

    # save the results to a CSV file
    results = []
    for video_file in os.listdir(video_dir):
        if video_file.endswith(".mp4"):
            video_path = os.path.join(video_dir, video_file)
            frames_candidate = sample_frames(video_path, args.num_frames)
            progress_gpt = ask_gpt(args.creds, frames_candidate)
            progress_phi4 = ask_phi4(frames_candidate)
            print(f"Video: {video_file}, GPT Progress: {progress_gpt}, Phi4 Progress: {progress_phi4}")
            results.append((video_file, progress_gpt, progress_phi4))
            if len(results) > 50:
                print("Stopping after 50 videos for testing purposes.")
                break
    # Save results to CSV
    df = pd.DataFrame(results, columns=["video", "gpt_progress", "phi4_progress"])
    df.to_csv(f"video_progress_comparison_frame_{args.num_frames}_n_{len(results)}.csv", index=False)

def sample_frames(video_path: str, num_frames: int) -> list[np.ndarray]:
    """
    Evenly sample `num_frames` frames from the video at `video_path`.
    Ensures the last frame is included. Returns a list of BGR images as numpy arrays.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise ValueError(f"Video has no frames: {video_path}")
    # randomly change the total within 0 to total
    # total = np.random.randint(0, total + 1)
    # Ensure last frame is included
    if num_frames == 1:
        indices = [total - 1]
    else:
        indices = np.linspace(0, total - 1, num_frames, endpoint=True, dtype=int)
        indices = np.unique(indices)  # avoid duplicates in short videos
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret and frames:
            h, w = frames[-1].shape[:2]
            frame = np.zeros((h, w, 3), dtype=np.uint8)
        elif not ret:
            raise IOError(f"Failed to read frame {idx} from {video_path}")
        frames.append(frame)

    cap.release()
    return frames


if __name__ == "__main__":
    main()