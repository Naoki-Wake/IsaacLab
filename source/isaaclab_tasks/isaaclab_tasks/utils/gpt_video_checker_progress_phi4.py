import argparse
import base64
import json
import os
import time
import requests

import cv2
import numpy as np

try:
    from dotenv import dotenv_values
except ImportError:
    dotenv_values = lambda f: os.environ

import re

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


def build_prompt_content(
    frames_candidate: list[np.ndarray],
) -> list[dict]:
    """
    Build the chat content: instruction text and image_url objects.
    """
    content = []
    header = (
        "Determine the progress of a robot task involving grasping an object.\n"
        "Consider the sequence of images below."
        "Note: These images may show multiple environments in the background. "
        "Focus only on the robot and object that appear most prominently in the foreground."
    )
    content.append({"type": "text", "text": header})

    for frame in frames_candidate:
        uri = resize_and_encode(frame)
        content.append({"type": "image_url", "image_url": {"url": uri, "detail": "low"}})

    footer = (
        "Evaluate the progress in four stages: "
        "(0) the hand is not near the object; "
        "(1) the hand approaches the object; "
        "(2) the hand touches the object; "
        "(3) the hand grasps or holds the object; "
        "(4) the hand lifts the object.\n"
        "Answer with a single number and return a JSON object in the format: {\"answer\": \"0\"}, {\"answer\": \"1\"}, {\"answer\": \"2\"}, {\"answer\": \"3\"}, or {\"answer\": \"4\"}."
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


def ask_phi4(
    frames_candidate: list[np.ndarray],
) -> float:
    """
    Sample frames from both videos, query GPT, and return True if 'first' wins.
    """

    prompt_content = build_prompt_content(frames_candidate)
    result = query_vlm(prompt_content)
    # print(result)
    answer = result.get("raw", "").lower()
    if '1' in answer:
        answer = 1
    elif '2' in answer:
        answer = 2
    elif '3' in answer:
        answer = 3
    elif '4' in answer:
        answer = 4
    elif '0' in answer:
        answer = 0
    else:
        print(f"Invalid answer: {answer}. Expected '0', '1', '2', '3', or '4'.")
        return 0.0
    # return success ratio
    answer = float(answer)/4.0
    return answer


def main():
    parser = argparse.ArgumentParser(description="Video progress check via GPT-4 Vision.")
    parser.add_argument("--num_frames", type=int, default=3, help="Frames per video.")
    parser.add_argument("--ask", default="ask_video.mp4", help="Video A path.")
    args = parser.parse_args()

    frames_candidate = sample_frames(args.ask, args.num_frames)
    progress = ask_phi4(
        frames_candidate
    )
    print(f"Video progress stage: {progress}")

if __name__ == "__main__":
    main()
