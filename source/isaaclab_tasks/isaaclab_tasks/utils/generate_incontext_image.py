import cv2
import numpy as np
import os
import argparse
from typing import Tuple, List, Optional


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    """Resize image while maintaining aspect ratio."""
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
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

def create_frame_grid(frames: List[np.ndarray],
                     grid_size: int,
                     frame_width: int = 200,
                     render_pos: str = 'topright',
                     spacer: int = 0) -> np.ndarray:
    """
    Create a grid of frames (with padding if needed).
    Args:
        frames: List of frames (length <= grid_size**2)
        grid_size: grid size (N)
        frame_width: width of each frame
        render_pos: label position
        spacer: space between frames
    Returns:
        grid_img: Final grid image
    """
    num_cells = grid_size * grid_size
    frame_height, frame_width = frames[0].shape[:2]

    # Pad frames with white images if needed
    if len(frames) < num_cells:
        pad_frame = np.ones_like(frames[0]) * 255
        frames = frames + [pad_frame] * (num_cells - len(frames))

    # Prepare grid canvas
    grid_height = grid_size * frame_height + (grid_size - 1) * spacer
    grid_width = grid_size * frame_width + (grid_size - 1) * spacer
    grid_img = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255

    # Place frames
    # Place frames in grid with labels
    for i in range(grid_size):
        for j in range(grid_size):
            index = i * grid_size + j
            if index >= len(frames):
                break
                
            frame = frames[index].copy()
            # skip if frame is white (padding)
            if np.all(frame == 255):
                continue
            cX, cY = frame.shape[1] // 2, frame.shape[0] // 2
            max_dim = int(min(frame.shape[:2]) * 0.5)
            
            # Create overlay for label background
            overlay = frame.copy()
            if render_pos == 'center':
                circle_center = (cX, cY)
            else:  # topright
                circle_center = (frame.shape[1] - max_dim // 2, max_dim // 2)
            
            # Draw white circle background
            cv2.circle(overlay, circle_center, max_dim // 2, (255, 255, 255), -1)
            
            # Blend overlay with frame
            alpha = 0.3
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            
            # Draw circle border
            cv2.circle(frame, circle_center, max_dim // 2, (255, 255, 255), 2)
            
            # Add frame number text
            font_scale = max_dim / 50
            text = str(index + 1)
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
            
            if render_pos == 'center':
                text_x = cX - text_size[0] // 2
                text_y = cY + text_size[1] // 2
            else:  # topright
                text_x = frame.shape[1] - text_size[0] // 2 - max_dim // 2
                text_y = text_size[1] // 2 + max_dim // 2
            
            cv2.putText(frame, text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 10)
            
            # Place frame in grid
            y1 = i * (frame_height + spacer)
            y2 = y1 + frame_height
            x1 = j * (frame_width + spacer)
            x2 = x1 + frame_width
            grid_img[y1:y2, x1:x2] = frame
    

    return grid_img


def extract_uniform_frames(video_path: str, 
                          num_frames: int = 16,
                          frame_width: int = 200,
                          render_pos: str = 'topright') -> Tuple[np.ndarray, List[int]]:
    """
    Extract exactly num_frames uniformly distributed across the video.
    """
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    print(f"Video info: {total_frames} frames, {fps:.2f} FPS, {duration:.2f}s duration")

    # Calculate grid size (for layout)
    grid_size = int(np.ceil(np.sqrt(num_frames)))

    # Compute indices for exactly num_frames
    if num_frames > 1:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    else:
        indices = np.array([total_frames // 2], dtype=int)

    frames = []
    actual_indices = []
    for idx in indices:
        video.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = video.read()
        if success:
            frame = image_resize(frame, width=frame_width)
            frames.append(frame)
            actual_indices.append(idx)
        else:
            print(f"Warning: Frame {idx} not found")
            # Pad with a blank frame of same shape later

    video.release()

    # Use first frame's shape or fallback if list is empty
    if not frames:
        raise ValueError("Could not extract any frames from video.")

    # Pad with blank frames if needed (handled in create_frame_grid)
    grid_img = create_frame_grid(frames, grid_size, frame_width, render_pos)
    return grid_img, actual_indices

def save_grid_image(video_path: str, 
                   output_path: str,
                   num_frames: int = 16,
                   frame_width: int = 200,
                   render_pos: str = 'topright') -> None:
    """
    Save a grid image of video frames to file.
    
    Args:
        video_path: Path to the input video file
        output_path: Path to save the grid image
        num_frames: Number of frames to extract
        frame_width: Width to resize each frame to
        render_pos: Position for labels ('center' or 'topright')
    """
    try:
        grid_img, frame_indices = extract_uniform_frames(
            video_path, num_frames, frame_width, render_pos)
        
        cv2.imwrite(output_path, grid_img)
        print(f"Grid image saved to: {output_path}")
        print(f"Frame indices used: {frame_indices}")
        print(f"Grid dimensions: {grid_img.shape[1]}x{grid_img.shape[0]}")
        
    except Exception as e:
        print(f"Error creating grid image: {e}")


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description="Create a grid of labeled video frames")
    parser.add_argument("--video_path", 
                       help="Path to input video file", 
                       default="grasp_demo_with_shade.mp4")
    parser.add_argument("--output_path", 
                       help="Path to save grid image", 
                       default="video_frame_grid_with_shade.png")
    parser.add_argument("--num_frames", 
                       help="Number of frames to extract", 
                       type=int, default=5)
    parser.add_argument("--frame_width", 
                       help="Width to resize each frame to", 
                       type=int, default=666)
    parser.add_argument("--render_pos", 
                       help="Label position (center or topright)", 
                       choices=['center', 'topright'], 
                       default='topright')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        return
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Generate and save grid image
    save_grid_image(
        video_path=args.video_path,
        output_path=args.output_path,
        num_frames=args.num_frames,
        frame_width=args.frame_width,
        render_pos=args.render_pos
    )


if __name__ == "__main__":
    main()


# Example usage as a module
def create_video_frame_grid(video_path: str, 
                           output_path: Optional[str] = None,
                           num_frames: int = 16,
                           frame_width: int = 200,
                           render_pos: str = 'topright') -> np.ndarray:
    """
    Convenience function to create a video frame grid.
    
    Args:
        video_path: Path to the input video file
        output_path: Optional path to save the grid image
        num_frames: Number of frames to extract
        frame_width: Width to resize each frame to
        render_pos: Position for labels ('center' or 'topright')
        
    Returns:
        Grid image as numpy array
    """
    grid_img, frame_indices = extract_uniform_frames(
        video_path, num_frames, frame_width, render_pos)
    
    if output_path:
        cv2.imwrite(output_path, grid_img)
        print(f"Grid image saved to: {output_path}")
    
    return grid_img