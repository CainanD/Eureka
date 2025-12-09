import subprocess
import os
import json
import logging
import os
import re
from pathlib import Path
import cv2

from utils.extract_task_code import file_to_string

def set_freest_gpu():
    freest_gpu = get_freest_gpu()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(freest_gpu)

def get_freest_gpu():
    sp = subprocess.Popen(['gpustat', '--json'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_str, _ = sp.communicate()
    gpustats = json.loads(out_str.decode('utf-8'))
    # Find GPU with most free memory
    freest_gpu = min(gpustats['gpus'], key=lambda x: x['memory.used'])

    return freest_gpu['index']

def filter_traceback(s):
    lines = s.split('\n')
    filtered_lines = []
    for i, line in enumerate(lines):
        if line.startswith('Traceback'):
            for j in range(i, len(lines)):
                if "Set the environment variable HYDRA_FULL_ERROR=1" in lines[j]:
                    break
                filtered_lines.append(lines[j])
            return '\n'.join(filtered_lines)
    return ''  # Return an empty string if no Traceback is found

def block_until_training(rl_filepath, log_status=False, iter_num=-1, response_id=-1):
    # Ensure that the RL training has started before moving on
    while True:
        rl_log = file_to_string(rl_filepath)
        if "fps step:" in rl_log or "Traceback" in rl_log:
            if log_status and "fps step:" in rl_log:
                logging.info(f"Iteration {iter_num}: Code Run {response_id} successfully training!")
            if log_status and "Traceback" in rl_log:
                logging.info(f"Iteration {iter_num}: Code Run {response_id} execution error!")
            break


EUREKA_DIR = Path(__file__).resolve().parent.parent


def get_vlm_feedback(task_description: str, video_path: str) -> str:
    """
    Get VLM feedback on a policy video by calling vlm_evaluation.py via subprocess.

    Args:
        task_description: Description of the task
        video_path: Path to the processed video file

    Returns:
        VLM feedback text
    """
    try:
        result = subprocess.run(
            ['conda', 'run', '-n', 'base', 'python', 'utils/vlm_evaluation.py',
             '--mode', 'feedback',
             '--task', str(task_description),
             '--video', str(video_path)],
            cwd=str(EUREKA_DIR),
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        logging.error(f"VLM feedback failed - Exit code: {e.returncode}")
        logging.error(f"Stderr: {e.stderr}")
        return ""
    except subprocess.TimeoutExpired:
        logging.error("VLM feedback timed out")
        return ""


def get_vlm_selection(task_description: str, video_paths: list) -> tuple:
    """
    Get VLM to select the best video from multiple policy videos.

    Args:
        task_description: Description of the task
        video_paths: List of paths to video files

    Returns:
        Tuple of (best_index, reasoning_text)
    """
    try:
        # Pass video paths as comma-separated string
        videos_arg = ",".join(str(p) for p in video_paths)
        result = subprocess.run(
            ['conda', 'run', '-n', 'base', 'python', 'utils/vlm_evaluation.py',
             '--mode', 'selection',
             '--task', str(task_description),
             '--videos', videos_arg],
            cwd=str(EUREKA_DIR),
            capture_output=True,
            text=True,
            check=True,
            timeout=180,
        )
        output = result.stdout.strip()
        # Parse output: first line is index, rest is reasoning
        lines = output.split('\n', 1)
        if len(lines) >= 1:
            try:
                best_idx = int(lines[0])
                reasoning = lines[1] if len(lines) > 1 else ""
                return best_idx, reasoning
            except ValueError:
                return 0, output
        return 0, output
    except subprocess.CalledProcessError as e:
        logging.error(f"VLM selection failed - Exit code: {e.returncode}")
        logging.error(f"Stderr: {e.stderr}")
        return 0, ""
    except subprocess.TimeoutExpired:
        logging.error("VLM selection timed out")
        return 0, ""


def process_episode_clips(video_dir: Path, output_path: Path) -> str:
    """
    Process episode video clips by calling vlm_evaluation.py via subprocess.

    Args:
        video_dir: Directory containing rl-video-episode-*.mp4 files
        output_path: Path where merged video will be saved

    Returns:
        Path to the processed video
    """
    try:
        result = subprocess.run(
            ['conda', 'run', '-n', 'base', 'python', 'utils/vlm_evaluation.py',
             '--mode', 'process',
             '--video-dir', str(video_dir),
             '--output', str(output_path)],
            cwd=str(EUREKA_DIR),
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
        return str(output_path)
    except subprocess.CalledProcessError as e:
        logging.error(f"Video processing failed - Exit code: {e.returncode}")
        logging.error(f"Stderr: {e.stderr}")
        raise RuntimeError(f"Video processing failed: {e.stderr}")
    except subprocess.TimeoutExpired:
        logging.error("Video processing timed out")
        raise RuntimeError("Video processing timed out")


if __name__ == "__main__":
    print(get_freest_gpu())