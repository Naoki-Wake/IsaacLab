import cv2
import numpy as np
import os
import argparse
from typing import Tuple, List, Optional
import glob
import math


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv2.resize(image, dim, interpolation=inter)


def create_frame_grid(frames: List[np.ndarray],
                      grid_size: int,
                      frame_width: int = 200,
                      render_pos: str = 'topright',
                      spacer: int = 0) -> np.ndarray:
    frame_height, frame_width = frames[0].shape[:2]
    num_cells = grid_size * grid_size

    if len(frames) < num_cells:
        pad_frame = np.ones_like(frames[0]) * 255
        frames += [pad_frame] * (num_cells - len(frames))

    grid_height = grid_size * frame_height + (grid_size - 1) * spacer
    grid_width = grid_size * frame_width + (grid_size - 1) * spacer
    grid_img = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255

    for i in range(grid_size):
        for j in range(grid_size):
            index = i * grid_size + j
            if index >= len(frames):
                break

            frame = frames[index].copy()
            if np.all(frame == 255):
                continue

            cX, cY = frame.shape[1] // 2, frame.shape[0] // 2
            max_dim = int(min(frame.shape[:2]) * 0.5)

            overlay = frame.copy()
            circle_center = (frame.shape[1] - max_dim // 2, max_dim // 2) if render_pos == 'topright' else (cX, cY)
            cv2.circle(overlay, circle_center, max_dim // 2, (255, 255, 255), -1)
            frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
            cv2.circle(frame, circle_center, max_dim // 2, (255, 255, 255), 2)

            font_scale = max_dim / 50
            text = str(index + 1)
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
            text_x = circle_center[0] - text_size[0] // 2
            text_y = circle_center[1] + text_size[1] // 2

            cv2.putText(frame, text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 10)

            y1 = i * (frame_height + spacer)
            y2 = y1 + frame_height
            x1 = j * (frame_width + spacer)
            x2 = x1 + frame_width
            grid_img[y1:y2, x1:x2] = frame

    return grid_img


def load_images_from_folder(folder_path: str, frame_width: int) -> List[np.ndarray]:
    image_paths = sorted(glob.glob(os.path.join(folder_path, "demo_*.jpg")))
    if not image_paths:
        raise ValueError(f"No images found in {folder_path}")

    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"Warning: Failed to load {path}")
            continue
        img = image_resize(img, width=frame_width)
        images.append(img)
    return images


def save_image_grid_from_folder(folder_path: str,
                                output_path: str,
                                frame_width: int = 200,
                                render_pos: str = 'topright') -> None:
    try:
        frames = load_images_from_folder(folder_path, frame_width)
        grid_size = math.ceil(math.sqrt(len(frames)))
        grid_img = create_frame_grid(frames, grid_size, frame_width, render_pos)
        cv2.imwrite(output_path, grid_img)
        print(f"Grid image saved to: {output_path}")
    except Exception as e:
        print(f"Error creating image grid: {e}")


def main():
    parser = argparse.ArgumentParser(description="Create a grid of images from folder")
    parser.add_argument("--folder_path", default="demo_imgs", help="Path to folder containing demo_*.jpg")
    parser.add_argument("--output_path", default="demo_image_grid_manual.png", help="Path to save the output grid image")
    parser.add_argument("--frame_width", 
                       help="Width to resize each frame to", 
                       type=int, default=666)
    parser.add_argument("--render_pos", choices=['center', 'topright'], default='topright',
                        help="Label position on each frame")

    args = parser.parse_args()

    if not os.path.exists(args.folder_path):
        print(f"Error: Folder not found: {args.folder_path}")
        return

    save_image_grid_from_folder(args.folder_path, args.output_path, args.frame_width, args.render_pos)


if __name__ == "__main__":
    main()
