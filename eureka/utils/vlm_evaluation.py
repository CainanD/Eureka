from pathlib import Path
import argparse
from google import genai
from google.genai import types
import sys
import re
import os
import cv2


def find_highest_consecutive_episodes(video_dir):
    """
    Find the two videos with the highest consecutive episode indices.
    Args:
        video_dir: Path to directory containing rl-video-episode-{i}.mp4 files
    Returns:
        Tuple of (first_episode_num, second_episode_num, first_video_path, second_video_path)
        OR (episode_num, None, video_path, None) if only one episode exists
    """
    # Find all video files matching the pattern recursively
    video_pattern = re.compile(r'rl-video-episode-(\d+)\.mp4')
    episodes = []

    # Walk through all subdirectories
    for root, dirs, files in os.walk(video_dir):
        for filename in files:
            match = video_pattern.match(filename)
            if match:
                episode_num = int(match.group(1))
                episodes.append((episode_num, os.path.join(root, filename)))

    if len(episodes) == 0:
        raise ValueError("No video episodes found")

    # Sort by episode number
    episodes.sort(key=lambda x: x[0])

    # If only one episode, return it (no merging needed)
    if len(episodes) == 1:
        return (episodes[0][0], None, episodes[0][1], None)

    # Find the highest consecutive pair
    highest_consecutive = None
    for i in range(len(episodes) - 1):
        current_ep, current_path = episodes[i]
        next_ep, next_path = episodes[i + 1]
        if next_ep == current_ep + 1:
            highest_consecutive = (current_ep, next_ep, current_path, next_path)

    # If no consecutive episodes, just return the last two
    if highest_consecutive is None:
        return (episodes[-2][0], episodes[-1][0], episodes[-2][1], episodes[-1][1])

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
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Read and skip first 2 frames from first video
    for _ in range(4):
        ret, _ = cap1.read()
        if not ret:
            #print("Warning: First video has fewer than 2 frames")
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
    #print(f"Added {frame_count} frames from first video (skipped first 2)")
    
    # Open second video and write first 2 frames
    cap2 = cv2.VideoCapture(second_video_path)
    if not cap2.isOpened():
        raise ValueError(f"Cannot open video: {second_video_path}")
    
    frames_from_second = 0
    for _ in range(2):
        ret, frame = cap2.read()
        if not ret:
            #print(f"Warning: Second video has only {frames_from_second} frames")
            break
        out.write(frame)
        frames_from_second += 1
    
    cap2.release()
    out.release()
    
    #print(f"Added {frames_from_second} frames from second video")
    #print(f"Total frames in merged video: {frame_count + frames_from_second}")


def process_episode_clips(video_dir, output_path):
    """
    Main function to find consecutive videos and merge them.

    Args:
        video_dir: Directory containing rl-video-episode-{i}.mp4 files
        output_path: Path where merged/copied video will be saved
    """
    import shutil

    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Find highest consecutive episodes
    ep1, ep2, path1, path2 = find_highest_consecutive_episodes(video_dir)

    if path2 is None:
        # Only one episode - just copy it
        shutil.copy(path1, output_path)
    else:
        # Merge the videos
        merge_videos(path1, path2, str(output_path))

    return str(output_path)

def get_vlm_feedback(task_description: str, policy_video_path: str) -> str:
    # Only for videos of size <20Mb

    policy_video_path = Path(policy_video_path)
    video_bytes = open(policy_video_path, "rb").read()

    client = genai.Client()

    config = types.GenerateContentConfig(
        media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH,
    )

    response = client.models.generate_content(
        model="models/gemini-3-pro-preview",
        contents=types.Content(
            parts=[
                types.Part(
                    inline_data=types.Blob(data=video_bytes, mime_type="video/mp4"),
                    video_metadata=types.VideoMetadata(fps=5),
                ),
                types.Part(
                    text=f"""
                        We trained a robot policy in simulation to execute a task with the description "{task_description}". 
                        Does the robot succeed in completing its task? Rate the performance on a scale from 1-5. 
                        Please give feedback for improvement."""
                ),
            ]
        ),
        config=config
    )

    #print(response)
    with open(policy_video_path.parent/'full_response.txt', 'w') as f:
        f.write(f"Video: {policy_video_path}\n")
        f.write(f"Task: {task_description}\n")
        f.write("="*60 + "\n\n")
        f.write(str(response))

    return response.text


def get_vlm_selection(task_description: str, video_paths: list) -> tuple:
    """
    Ask Gemini to select the best video from multiple policy videos.

    Args:
        task_description: Description of the task
        video_paths: List of paths to video files

    Returns:
        Tuple of (best_index, reasoning_text)
    """
    client = genai.Client()

    config = types.GenerateContentConfig(
        media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH,
    )

    # Build parts list with all videos
    parts = []
    for i, video_path in enumerate(video_paths):
        video_bytes = open(video_path, "rb").read()
        parts.append(types.Part(
            inline_data=types.Blob(data=video_bytes, mime_type="video/mp4"),
            video_metadata=types.VideoMetadata(fps=5),
        ))
        parts.append(types.Part(text=f"[Video {i}]"))

    # Add the selection prompt
    parts.append(types.Part(
        text=f"""
We trained {len(video_paths)} robot policies in simulation to execute a task with the description "{task_description}".

The videos above show each policy's performance (labeled Video 0, Video 1, etc.).

Please:
1. Evaluate each policy's performance on the task
2. Select the BEST policy (the one that best achieves the task description)
3. Explain your reasoning

Your response MUST include a line in this exact format:
BEST_POLICY: <number>

For example, if Video 2 is best, write:
BEST_POLICY: 2

Then provide your detailed reasoning and feedback for improvement.
"""
    ))

    response = client.models.generate_content(
        model="models/gemini-3-pro-preview",
        contents=types.Content(parts=parts),
        config=config
    )

    response_text = response.text

    # Save full response to file in same directory as first video
    output_dir = Path(video_paths[0]).parent
    with open(output_dir / 'selection_response.txt', 'w') as f:
        f.write(f"Task: {task_description}\n")
        f.write("Videos:\n")
        for i, vp in enumerate(video_paths):
            f.write(f"  [{i}] {vp}\n")
        f.write("="*60 + "\n\n")
        f.write(str(response))

    # Parse the best policy index from response
    match = re.search(r'BEST_POLICY:\s*(\d+)', response_text)
    if match:
        best_idx = int(match.group(1))
        # Validate index is in range
        if 0 <= best_idx < len(video_paths):
            return best_idx, response_text

    # If parsing fails, default to first video
    return 0, response_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VLM Evaluation for video analysis')

    parser.add_argument(
        '--mode',
        type=str,
        choices=['feedback', 'selection', 'process'],
        default='feedback',
        help='Mode: feedback (single video), selection (multiple videos), process (merge clips)'
    )

    parser.add_argument(
        '--task',
        type=str,
        default='Complete the task.',
        help='Description of the task being evaluated'
    )

    parser.add_argument(
        '--video',
        type=str,
        help='Path to video file (for feedback mode)'
    )

    parser.add_argument(
        '--videos',
        type=str,
        help='Comma-separated paths to video files (for selection mode)'
    )

    parser.add_argument(
        '--video-dir',
        type=str,
        help='Directory containing video clips (for process mode)'
    )

    parser.add_argument(
        '--output',
        type=str,
        help='Output path for processed video (for process mode)'
    )

    args = parser.parse_args()

    if args.mode == 'process':
        # Process video clips - merge consecutive episodes
        if not args.video_dir or not args.output:
            print("Error: --video-dir and --output required for process mode")
            sys.exit(1)
        process_episode_clips(Path(args.video_dir), Path(args.output))
        print(f"Processed video saved to: {args.output}")

    elif args.mode == 'feedback':
        # Get feedback on single video
        if not args.video:
            print("Error: --video required for feedback mode")
            sys.exit(1)
        result = get_vlm_feedback(args.task, args.video)
        print(result)

    elif args.mode == 'selection':
        # Select best from multiple videos
        if not args.videos:
            print("Error: --videos required for selection mode")
            sys.exit(1)
        video_paths = args.videos.split(',')
        best_idx, reasoning = get_vlm_selection(args.task, video_paths)
        # Output format: first line is index, rest is reasoning
        print(best_idx)
        print(reasoning)
