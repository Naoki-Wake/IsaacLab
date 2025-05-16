import argparse
import base64
import json
import os
import time

import cv2
import numpy as np
dotenv = None
try:
    import dotenv
    dotenv = dotenv.dotenv_values
except ImportError:
    dotenv = lambda f: os.environ

from openai import OpenAI, AzureOpenAI
import openai


def load_credentials(env_file: str) -> dict:
    """
    Load credentials from .env file or environment.
    """
    creds = dotenv(env_file)
    required = ["OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT_NAME"]
    for key in required:
        creds.setdefault(key, os.getenv(key, ""))
    return creds


def init_vlm_client(creds: dict):
    """
    Initialize GPT-4 Vision client, choosing OpenAI or Azure.
    """
    if creds.get("AZURE_OPENAI_API_KEY"):
        client = AzureOpenAI(
            api_key=creds["AZURE_OPENAI_API_KEY"],
            azure_endpoint=creds["AZURE_OPENAI_ENDPOINT"],
            api_version="2024-02-01"
        )
        deployment = creds["AZURE_OPENAI_DEPLOYMENT_NAME"]
        return client, {"model": deployment}
    else:
        client = OpenAI(api_key=creds["OPENAI_API_KEY"])
        return client, {"model": "gpt-4o"}


def resize_for_vlm(frame: np.ndarray, max_short: int = 768, max_long: int = 2000) -> np.ndarray:
    h, w = frame.shape[:2]
    ar = w / h
    if ar >= 1:
        new_w = min(w, max_long)
        new_h = min(int(new_w / ar), max_short)
    else:
        new_h = min(h, max_long)
        new_w = min(int(new_h * ar), max_short)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def encode_frame(frame: np.ndarray) -> str:
    """Encode image to base64 data URI."""
    _, buf = cv2.imencode('.jpg', frame)
    b64 = base64.b64encode(buf).decode('utf-8')
    return f"data:image/jpeg;base64,{b64}"


def extract_json(text: str) -> str:
    """Extract JSON substring starting with {\"answer\": ...}."""
    compact = ''.join(text.split())
    start = compact.find('{"answer":')
    if start < 0:
        raise ValueError("JSON not found in response")
    end = compact.find('}', start) + 1
    return compact[start:end]


def call_vlm(client, client_params: dict, prompt: str, image_uri: str) -> dict:
    """Send prompt and image to VLM and return parsed JSON."""
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_uri, "detail": "high"}}
        ]
    }]
    params = {
        **client_params,
        "messages": messages,
        "max_tokens": 200,
        "temperature": 0.1,
        "top_p": 0.5,
    }
    for attempt in range(5):
        try:
            response = client.chat.completions.create(**params)
            content = response.choices[0].message.content
            js = extract_json(content)
            return json.loads(js)
        except (openai.RateLimitError, openai.APIStatusError) as e:
            print(f"Rate limit or API status error: {e}")
            time.sleep(1)
        # catch other errors
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)
    # print the error message
    print(f"VLM request failed after {attempt + 1} attempts.")
    return {}

def create_frame_grid(video_path: str, grid_size: int = 3,
                      tile_width: int = 200, spacer: int = 0,
                      bg_color: tuple = (255, 255, 255)) -> np.ndarray:
    """
    Sample last half of video frames evenly, tile them with numbering.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num = grid_size**2
    start = total // 2
    indices = list(np.linspace(start, total - 1, num, dtype=int))

    tiles = []
    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * tile_width / cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame = np.zeros((h, tile_width, 3), dtype=np.uint8)
        else:
            h, w = frame.shape[:2]
            new_h = int(h * tile_width / w)
            frame = cv2.resize(frame, (tile_width, new_h))

        # draw label background and text
        label = str(i + 1)
        font, scale, th = cv2.FONT_HERSHEY_SIMPLEX, max(0.5, tile_width/200), 2
        (tw, th_), _ = cv2.getTextSize(label, font, scale, th)
        cv2.rectangle(frame, (5,5), (5+tw+4, 5+th_+4), bg_color, -1)
        cv2.putText(frame, label, (5, 5+th_), font, scale, (0,0,0), th, cv2.LINE_AA)
        tiles.append(frame)

    cap.release()
    th, tw = tiles[0].shape[:2]
    H = grid_size*th + spacer*(grid_size-1)
    W = grid_size*tw + spacer*(grid_size-1)
    canvas = np.full((H, W, 3), 255, dtype=np.uint8)

    for y in range(grid_size):
        for x in range(grid_size):
            idx = y*grid_size + x
            cy, cx = y*(th+spacer), x*(tw+spacer)
            canvas[cy:cy+th, cx:cx+tw] = tiles[idx]
    return canvas


def ask_gpt(ask_video: str, champ_video: str, task: str,
                 creds_path: str, grid: int) -> bool:
    """
    Generate grid image, call VLM, return 'red' or 'blue'.
    """
    creds = load_credentials(creds_path)
    client, params = init_vlm_client(creds)
    grid_ask = create_frame_grid(ask_video, grid_size=grid, bg_color=(255,0,0)) # blue
    grid_champ = create_frame_grid(champ_video, grid_size=grid, bg_color=(0,0,255)) # red
    combined = np.hstack((grid_ask, grid_champ))
    uri = encode_frame(resize_for_vlm(combined))
    # save image for debugging
    cv2.imwrite("grid.png", combined)
    prompt = (
        f"I provide two grids of first person camera snapshots showing a robot learning the \"{task}\" task. The left grid (blue numbers) is policy version A, and the right grid (red numbers) is policy version B. Think of it as an A/B test of two training variants. Compare the blue and red versions and tell me which one appears to perform the {task} task more effectively. Answer in one word. Provide your answer at the end in a json file of this format: {{\"answer\": \"red\" or \"blue\"}}"
    )

    result = call_vlm(client, params, prompt, uri)
    if result.get("answer") == "blue":
        return True
    elif result.get("answer") == "red":
        return False
    else:
        return False
        #raise ValueError(f"Unexpected answer: {result.get('answer')}. Expected 'red' or 'blue'.")

def main():
    parser = argparse.ArgumentParser(description="Grid A/B test with GPT-4V.")
    parser.add_argument("--creds", default="auth.env", help="Env file path.")
    parser.add_argument("--grid", type=int, default=3, help="Grid size per side.")
    parser.add_argument("--ask", help="Video for variant A.", default="ask_video.mp4")
    parser.add_argument("--champ", help="Video for variant B.", default="champion_video.mp4")
    parser.add_argument("--task", help="Task name.", default="grasping a cube")
    args = parser.parse_args()
    
    answer = ask_gpt(args.ask, args.champ, args.task, args.creds, args.grid)
    print(answer)


if __name__ == "__main__":
    main()
