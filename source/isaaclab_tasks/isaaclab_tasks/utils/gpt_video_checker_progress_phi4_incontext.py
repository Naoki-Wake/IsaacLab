import argparse
import base64
import json
import os
import time

import cv2
import numpy as np

try:
    from dotenv import dotenv_values
except ImportError:
    dotenv_values = lambda f: os.environ

from openai import OpenAI, AzureOpenAI
import openai
import re
import logging
import pandas as pd
import matplotlib.pyplot as plt
import requests

# logging.getLogger().setLevel(logging.ERROR)
# compare two videos using GPT-4 Vision
def load_credentials(env_file: str) -> dict:
    """
    Load credentials from a .env file or environment.
    """
    creds = dotenv_values(env_file)
    required = [
        "OPENAI_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_DEPLOYMENT_NAME",
    ]
    for key in required:
        creds.setdefault(key, os.getenv(key, ""))
    return creds


def init_vlm_client(creds: dict):
    """
    Initialize the GPT-4 Vision client for Azure or OpenAI.
    """
    if creds.get("AZURE_OPENAI_API_KEY"):
        client = AzureOpenAI(
            api_key=creds["AZURE_OPENAI_API_KEY"],
            azure_endpoint=creds["AZURE_OPENAI_ENDPOINT"],
            api_version="2024-02-01"
        )
        return client, {"model": creds["AZURE_OPENAI_DEPLOYMENT_NAME"]}
    client = OpenAI(api_key=creds["OPENAI_API_KEY"])
    return client, {"model": "gpt-4o"}


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

    # Ensure last frame is included
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


def resize_and_encode(frame: np.ndarray, max_short: int = 768, max_long: int = 2000) -> str:
    """
    Resize a BGR image to fit within max dimensions, encode to JPEG+base64 URI.
    """
    h, w = frame.shape[:2]
    ar = w / h
    if ar >= 1:
        new_w = min(w, max_long)
        new_h = min(int(new_w / ar), max_short)
    else:
        new_h = min(h, max_long)
        new_w = min(int(new_h * ar), max_short)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    _, buf = cv2.imencode('.jpg', resized)
    b64 = base64.b64encode(buf).decode('utf-8')
    return f"data:image/jpeg;base64,{b64}"

def build_prompt_content_wo_context(
    frames_candidate: list[np.ndarray],
    frames_demo: list[np.ndarray] = None,
    additional_string: str = "",
) -> list[dict]:
    """
    Build the chat content: instruction text and image_url objects.
    """
    print("Building prompt content without context...")
    content = []
    header = (
        "Determine the progress of a robot task involving grasping an object.\n"
        "Consider the sequence of images below."
        "Note: These images may show multiple environments in the background. "
        "Focus only on the robot and object that appear most prominently in the foreground."
    )
    content.append({"type": "text", "text": header})
    assert len(frames_candidate) == 1
    for frame in frames_candidate:
        uri = resize_and_encode(frame)
        content.append({"type": "image_url", "image_url": {"url": uri, "detail": "low"}})
    if len(additional_string) > 0:
        content.append({"type": "text", "text": additional_string})
    footer = (
        "Evaluate the progress in five stages:"
        "(1) The hand is still far from the object and must move closer to interact with it;"
        "(2) The hand is within a few centimeters of the object;"
        "(3) The hand is touching the object;"
        "(4) The hand is securely grasping or holding the object;"
        "(5) The hand is lifting the object."
        "Answer with a single number and return a JSON object in the format: {\"answer\": \"0\"}, {\"answer\": \"1\"}, {\"answer\": \"2\"}, {\"answer\": \"3\"}, or {\"answer\": \"4\"}."
    )
    content.append({"type": "text", "text": footer})
    return content

def build_prompt_content(
    frames_candidate: list[np.ndarray],
    frames_demo: list[np.ndarray] = None,
    additional_string: str = "",
) -> list[dict]:
    """
    Build the chat content: instruction text and image_url objects.
    """
    print("Building prompt content with context...")
    content = []
    header = (
        "The image below shows the stages of a robot picking up an object in a ground-truth demonstration. Each stage is labeled with a number.\n"
        "Your task:\n"
        "Given the ground-truth demonstration and the subsequent image showing the robot's current progress, select the stage number that most closely matches the robot's current progress.\n"
        "Note:\n"
        "- Multiple robots or environments may be visible.\n"
        "- Focus only on the robot and object most prominently featured in the foreground.\n"
    )
    content.append({"type": "text", "text": header})
    uri = resize_and_encode(frames_demo[-1])
    content.append({"type": "image_url", "image_url": {"url": uri, "detail": "high"}})

    middle = (
        "Next image is the current progress of the robot action.\n"
    )
    content.append({"type": "text", "text": middle})
    uri = resize_and_encode(frames_candidate[-1])
    content.append({"type": "image_url", "image_url": {"url": uri, "detail": "high"}})

    if len(additional_string) > 0:
        content.append({"type": "text", "text": additional_string})
    footer = (
        "How to answer:\n"
        "Return your answer as a JSON object in the following format:\n"
        "{\"answer\": \"N\"}\n"
        "where N is the stage number (e.g., 1-5) that best matches the robot's current state."
    )
    content.append({"type": "text", "text": footer})
    return content


def query_vlm(prompt_content: list[dict]) -> dict:
    """
    Send image/text prompt_content to a local HTTP server running the VLM.
    """
    try:
        # Convert prompt content to a simple dict with images and text parts
        texts = []
        images_b64 = []
        for item in prompt_content:
            if item["type"] == "text":
                texts.append(item["text"])
            elif item["type"] == "image_url":
                images_b64.append(item["image_url"]["url"])  # base64 encoded image URI

        payload = {
            "prompt": "\n".join(texts),
            "images": images_b64
        }

        response = requests.post("http://10.137.70.15:8000/vlm_query", json=payload, timeout=60)
        response.raise_for_status()
        return response.json()

    except Exception as e:
        print(f"Local VLM error: {e}")
        return {}


def ask_gpt(
    client, client_params: dict,
    frames_candidate: list[np.ndarray],
    frames_demo: list[np.ndarray],
    additional_string: str = "",
) -> float:
    """
    Sample frames from both videos, query GPT, and return True if 'first' wins.
    """

    #prompt_content = build_prompt_content(frames_candidate, frames_demo, additional_string=additional_string)
    prompt_content = build_prompt_content(frames_candidate, frames_demo, additional_string=additional_string)

    result = query_vlm(prompt_content)

    # print(result)
    answer = result.get("answer", "").lower()
    print(f"GPT-4 Vision response: {answer}")
    # check if the answer is a valid stage
    if answer not in {"1", "2", "3", "4", "5"}:
        print(f"Invalid answer: {answer}. Expected '1', '2', '3', '4', or '5'.")
        return 0.0
    # return success ratio
    answer = (float(answer)-1)/5.0
    # overwrite the answer on the last frame of the image and save it
    # if frames_candidate:
    #     last_frame = frames_candidate[-1]
    #     cv2.putText(last_frame, f"Progress: {answer:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    #     str_answer = str(answer).replace('.', '_')
    #     file_name = f"progress_result-{str_answer}-{int(time.time())}.jpg"
    #     cv2.imwrite(file_name, last_frame)
    #     # print(f"Saved progress result image as '{file_name_random}'.")
    return answer

def read_image(image_path: str) -> np.ndarray:
    """
    Reads an image from `image_path` as a BGR NumPy array.
    Raises an error if loading fails.
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Cannot read image: {image_path}")
    return img

def main2():
    parser = argparse.ArgumentParser(description="Video progress check via GPT-4 Vision.")
    parser.add_argument("--creds", default="auth.env", help="Env file path.")
    parser.add_argument("--num_frames", type=int, default=3, help="Frames per video.")
    parser.add_argument("--ask", default="ask_video.mp4", help="Video A path.")
    args = parser.parse_args()
    demo_image = read_image("video_frame_grid_with_shade.png")
    frames_candidate = sample_frames(args.ask, args.num_frames)
    creds = load_credentials(args.creds)
    client, client_params = init_vlm_client(creds)
    progress = ask_gpt(
        client,
        client_params,
        frames_candidate,
        frames_demo=[demo_image],
    )
    print(f"Video progress stage: {progress}")

def gpt_progress_for_frame(frame, demo_image, credpath, wocontext):
    if wocontext:
        prompt_content = build_prompt_content_wo_context([frame], frames_demo=[demo_image])
    else:
        prompt_content = build_prompt_content([frame], frames_demo=[demo_image])
    result = query_vlm(prompt_content)
    answer = result.get("raw", "").lower()
    if '1' in answer:
        answer = "1"
    elif '2' in answer:
        answer = "2"
    elif '3' in answer:
        answer = "3"
    elif '4' in answer:
        answer = "4"
    elif '5' in answer:
        answer = "5"
    print(f"GPT-4 Vision response: {answer}")
    if answer not in {"1", "2", "3", "4", "5"}:
        print(f"Invalid answer: {answer}. Expected '1', '2', '3', '4', or '5'.")
        return 0.0
    answer = (float(answer)-1)/5.0
    return answer

def main():
    from concurrent.futures import ThreadPoolExecutor, as_completed
    parser = argparse.ArgumentParser(description="Video progress check via GPT-4 Vision.")
    parser.add_argument("--creds", default="auth.env", help="Env file path.")
    parser.add_argument("--num_samples", type=int, default=30, help="Number of random frames to sample.")
    parser.add_argument("--ask", default="grasp_demo_with_shade_testvideo.mp4", help="Test video path.")
    parser.add_argument("--demo_image", default="video_frame_grid_with_shade.png", help="Demo image path.")
    parser.add_argument("--output_csv", default="phi4v_progress_vs_time.csv", help="CSV file for saving results.")
    parser.add_argument("--wocontext", action="store_true", help="Use prompt without context/demo image.")
    args = parser.parse_args()
    desired_times = [0.001, 0.2, 0.4, 0.6, 0.8, 1.0]
    tolerance = 0.1  # how close we have to be to save
    # Load demo/reference image
    demo_image = read_image(args.demo_image)

    # Open the video
    cap = cv2.VideoCapture(args.ask)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 1:
        raise RuntimeError("No frames in video.")

    # Sample N unique random frames (avoid duplicates for short videos)
    #np.random.seed(42)  # for reproducibility
    #indices = np.sort(np.random.choice(total_frames, min(args.num_samples, total_frames), replace=False))
    num_samples = min(args.num_samples, total_frames)
    indices = np.linspace(3, total_frames - 1, num_samples, dtype=int)
    indices = np.unique(indices)

    frame_idx_and_data = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            raise IOError(f"Failed to read frame {idx} from {args.ask}")
        if total_frames > 1 and idx > 0:
            scaled_time = idx / (total_frames - 1)
            frame_idx_and_data.append((idx, scaled_time, frame))

    results = []
    # Parallel querying (max_workers can be tuned based on API rate limit)
    with ThreadPoolExecutor(max_workers=3) as executor:
        # Submit all queries at once
        futures = [
            executor.submit(gpt_progress_for_frame, frame, demo_image, args.creds, args.wocontext)
            for (_, _, frame) in frame_idx_and_data
        ]
        for (fut, (idx, scaled_time, _)) in zip(futures, frame_idx_and_data):
            progress = fut.result()
            print(f"Frame {idx}: time={scaled_time:.3f}, progress={progress:.3f}")
            results.append([scaled_time, progress])
    cap.release()

    # Save to CSV
    df = pd.DataFrame(results, columns=["scaled_time", "gpt_progress"])
    filename = args.output_csv
    if args.wocontext:
        filename = filename.replace(".csv", "_wocontext.csv")
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")


    plt.scatter(df["scaled_time"], df["gpt_progress"], c='b', alpha=0.6)
    plt.xlabel("Normalized Time in Video (0=start, 1=end)")
    plt.ylabel("GPT-4V Progress Value (0=early, 1=late)")
    plt.title("Robot Task Progress vs. Video Time")
    plt.grid(True)
    plt.tight_layout()
    if args.wocontext:
        plt.savefig("phi4v_progress_vs_time_wocontext.png")
    else:
        plt.savefig("phi4v_progress_vs_time.png")
if __name__ == "__main__":
    main()
