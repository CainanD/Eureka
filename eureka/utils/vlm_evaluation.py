import os
import re
from pathlib import Path
import cv2


def find_highest_consecutive_episodes(video_dir):
    """
    Find the two videos with the highest consecutive episode indices.
    
    Args:
        video_dir: Path to directory containing rl-video-episode-{i}.mp4 files
    
    Returns:
        Tuple of (first_episode_num, second_episode_num, first_video_path, second_video_path)
    """
    # Find all video files matching the pattern
    video_pattern = re.compile(r'rl-video-episode-(\d+)\.mp4')
    episodes = []
    
    for filename in os.listdir(video_dir):
        match = video_pattern.match(filename)
        if match:
            episode_num = int(match.group(1))
            episodes.append((episode_num, os.path.join(video_dir, filename)))
    
    if len(episodes) < 2:
        raise ValueError("Need at least 2 videos to find consecutive episodes")
    
    # Sort by episode number
    episodes.sort(key=lambda x: x[0])
    
    # Find the highest consecutive pair
    highest_consecutive = None
    
    for i in range(len(episodes) - 1):
        current_ep, current_path = episodes[i]
        next_ep, next_path = episodes[i + 1]
        
        if next_ep == current_ep + 1:
            highest_consecutive = (current_ep, next_ep, current_path, next_path)
    
    if highest_consecutive is None:
        raise ValueError("No consecutive episodes found")
    
    return highest_consecutive


def merge_videos(first_video_path, second_video_path, output_path):
    """
    Merge two videos: all frames from first except first 2, plus first 2 frames from second.
    
    Args:
        first_video_path: Path to first video
        second_video_path: Path to second video
        output_path: Path where merged video will be saved
    """
    # Open the first video
    cap1 = cv2.VideoCapture(first_video_path)
    if not cap1.isOpened():
        raise ValueError(f"Cannot open video: {first_video_path}")
    
    # Get video properties from first video
    fps = cap1.get(cv2.CAP_PROP_FPS)
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Read and skip first 2 frames from first video
    for _ in range(2):
        ret, _ = cap1.read()
        if not ret:
            print("Warning: First video has fewer than 2 frames")
            break
    
    # Write remaining frames from first video
    frame_count = 0
    while True:
        ret, frame = cap1.read()
        if not ret:
            break
        out.write(frame)
        frame_count += 1
    
    cap1.release()
    print(f"Added {frame_count} frames from first video (skipped first 2)")
    
    # Open second video and write first 2 frames
    cap2 = cv2.VideoCapture(second_video_path)
    if not cap2.isOpened():
        raise ValueError(f"Cannot open video: {second_video_path}")
    
    frames_from_second = 0
    for _ in range(2):
        ret, frame = cap2.read()
        if not ret:
            print(f"Warning: Second video has only {frames_from_second} frames")
            break
        out.write(frame)
        frames_from_second += 1
    
    cap2.release()
    out.release()
    
    print(f"Added {frames_from_second} frames from second video")
    print(f"Total frames in merged video: {frame_count + frames_from_second}")


def process_episode_clips(video_dir, output_path):
    """
    Main function to find consecutive videos and merge them.
    
    Args:
        video_dir: Directory containing rl-video-episode-{i}.mp4 files
        output_dir: Directory where merged video will be saved
    """
    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Find highest consecutive episodes
    ep1, ep2, path1, path2 = find_highest_consecutive_episodes(video_dir)
    print(f"Found consecutive episodes: {ep1} and {ep2}")
    print(f"Video 1: {path1}")
    print(f"Video 2: {path2}")
    # Create output filename
    
    # Merge the videos
    print(f"\nMerging videos to: {output_path}")
    merge_videos(path1, path2, output_path)
    
    print(f"\nSuccessfully created merged video: {output_path}")
    return output_path