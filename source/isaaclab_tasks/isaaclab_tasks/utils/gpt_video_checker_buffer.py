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
    Returns a list of BGR images as numpy arrays.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise ValueError(f"Video has no frames: {video_path}")

    indices = np.linspace(0, total - 1, num_frames, dtype=int)
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
    frames_a: list[np.ndarray],
    frames_b: list[np.ndarray],
    task: str
) -> list[dict]:
    """
    Build the chat content: instruction text and image_url objects.
    """
    content = []
    header = (
        f"I provide two sequences of first-person snapshots for task '{task}'.\n"
        "First sequence A below."
    )
    content.append({"type": "text", "text": header})

    for frame in frames_a:
        uri = resize_and_encode(frame)
        content.append({"type": "image_url", "image_url": {"url": uri, "detail": "high"}})

    content.append({"type": "text", "text": "Now sequence B below."})
    for frame in frames_b:
        uri = resize_and_encode(frame)
        content.append({"type": "image_url", "image_url": {"url": uri, "detail": "high"}})

    footer = (
        f"Compare the two sequences and tell me which performs '{task}' better. If the two look similar, say second.\n"
        "Answer in one word and return JSON like {\"answer\": \"first\" or \"second\", \"short_reason\": \"the reason in one sentence.\"}."
    )
    content.append({"type": "text", "text": footer})
    return content


def query_vlm(client, client_params: dict, prompt_content: list[dict]) -> dict:
    """
    Call the Vision LLM with provided content and return parsed JSON.
    """
    messages = [{"role": "user", "content": prompt_content}]
    params = {**client_params, "messages": messages, "max_tokens": 200, "temperature": 0.1, "top_p": 0.5}

    for attempt in range(5):
        try:
            resp = client.chat.completions.create(**params)
            text = resp.choices[0].message.content
            m = re.search(r'\{.*?"answer".*?\}', text, re.DOTALL)
            if not m:
                raise ValueError(f"Couldn't extract JSON from:\n{text}")

            payload = m.group(0)
            return json.loads(payload)
            #compact = ''.join(text.split())
            #start = compact.find('{"answer":')
            #end = compact.find('}', start) + 1
            #return json.loads(compact[start:end])
        except (openai.RateLimitError, openai.APIStatusError) as e:
            print(f"API error: {e} (retrying)")
            time.sleep(1)
        except Exception as e:
            print(f"Error: {e} (retrying)")
            time.sleep(1)

    print(f"Failed after {attempt+1} attempts.")
    return {}


def ask_gpt(
    ask_video: str,
    champ_video: str,
    task: str,
    creds_path: str,
    num_frames: int
) -> tuple[bool, str]:
    """
    Sample frames from both videos, query GPT, and return True if 'first' wins.
    """
    creds = load_credentials(creds_path)
    client, params = init_vlm_client(creds)

    frames_a = sample_frames(ask_video, num_frames)
    frames_b = sample_frames(champ_video, num_frames)

    prompt_content = build_prompt_content(frames_a, frames_b, task)
    result = query_vlm(client, params, prompt_content)
    # print(result)
    answer = result.get("answer", "").lower()
    # print(f"Answer: {result}")
    if answer == "first":
        return True, result.get("short_reason", "")
    elif answer == "second":
        return False, result.get("short_reason", "")
    else:
        print(f"Unexpected answer: {answer}")
        return False, "failed to parse answer"


def main():
    parser = argparse.ArgumentParser(description="Video A/B test via GPT-4 Vision.")
    parser.add_argument("--creds", default="auth.env", help="Env file path.")
    parser.add_argument("--num_frames", type=int, default=9, help="Frames per video.")
    parser.add_argument("--ask", default="ask_video.mp4", help="Video A path.")
    parser.add_argument("--champ", default="champion_video.mp4", help="Video B path.")
    parser.add_argument("--task", default="grasp the object", help="Task name.")
    args = parser.parse_args()

    winner_first = ask_gpt(
        args.ask,
        args.champ,
        args.task,
        args.creds,
        args.num_frames
    )
    print("first" if winner_first else "second")


if __name__ == "__main__":
    main()
