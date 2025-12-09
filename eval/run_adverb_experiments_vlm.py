#!/usr/bin/env python3
"""
Script to run Eureka experiments testing adverbs across multiple environments.
Uses the new VLM-feedback pipeline with GPT reward generation.

Usage:
    python run_adverb_experiments_vlm.py                    # Run all experiments
    python run_adverb_experiments_vlm.py --debug            # Test with one adverb/env only
    python run_adverb_experiments_vlm.py --resume           # Resume from previous run
    python run_adverb_experiments_vlm.py --dry-run          # Preview commands without running
"""

import subprocess
import pandas as pd
import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
import json
import time
import concurrent.futures
import shutil
import tempfile
import getpass
import signal
import threading
import re
import select

# tqdm for progress bars
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# ============================================================
# Configuration
# ============================================================
# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).resolve().parent
EUREKA_DIR = SCRIPT_DIR.parent  # Eureka root directory

EUREKA_SCRIPT = str(EUREKA_DIR / "eureka" / "eureka.py")
ADVERB_CSV = str(SCRIPT_DIR / "eval_adv_small.csv")
OUTPUT_DIR = str(SCRIPT_DIR / "adverb_experiment_results_vlm")

# Model and training parameters (matching new config.yaml)
MODEL = "gpt-4.1-mini"
SAMPLE = 2  # Number of reward functions GPT generates per iteration
TRAINING_MODE = "individual"  # "individual" or "averaged"
VLM_ITERATIONS = 1  # Number of VLM feedback loops (1 = 2 total GPT generations)
MAX_ITERATIONS = 500  # RL training iterations per policy

# Timeout per experiment (in seconds)
TIMEOUT_SECONDS = 45 * 60  # 45 minutes (increased due to VLM feedback loop)

# Pretrained checkpoints for specific environments (to start training from a good policy)
# These are converted from skrl/HuggingFace pretrained models to rl_games format
PRETRAINED_CHECKPOINTS = {
    "humanoid": str(SCRIPT_DIR / "pretrained_humanoid_converted.pth"),
    "anymal": str(SCRIPT_DIR / "pretrained_anymal_converted.pth"),
    "ant": str(SCRIPT_DIR / "pretrained_ant_converted.pth"),
}

# Environments with their base descriptions (from Eureka cfg/env/*.yaml)
# NOTE: env_name must match the YAML filename in /home/ttr/Eureka/eureka/cfg/env/
ENVIRONMENTS = [
    {"task": "Ant", "env_name": "ant", "base_description": "make the ant walk forward"},
    {"task": "Anymal", "env_name": "anymal", "base_description": "make the anymal walk forward"},
    {"task": "Humanoid", "env_name": "humanoid", "base_description": "make the humanoid walk forward"}
]


def load_adverbs(csv_path: str) -> pd.DataFrame:
    """Load adverbs from CSV file."""
    df = pd.read_csv(csv_path)
    return df


def build_description(base_description: str, adverb: str) -> str:
    """Build task description with adverb inserted."""
    # Remove trailing period if present
    base = base_description.rstrip(".")
    return f"{base} {adverb}"


def build_command(env_name: str, description: str, checkpoint: str = None,
                  sample: int = SAMPLE, training_mode: str = TRAINING_MODE,
                  vlm_iterations: int = VLM_ITERATIONS, max_iterations: int = MAX_ITERATIONS) -> str:
    """Build the Eureka command with new VLM pipeline parameters.

    Args:
        env_name: Environment name (snake_case)
        description: Task description with adverb
        checkpoint: Optional path to pretrained checkpoint to start from
        sample: Number of reward functions GPT generates per iteration
        training_mode: "individual" or "averaged"
        vlm_iterations: Number of VLM feedback loops
        max_iterations: RL training iterations per policy
    """
    # Set LD_LIBRARY_PATH to include eureka env lib for libpython3.8.so
    eureka_lib = "/data/user_data/wenjiel2/miniconda3/envs/eureka/lib"
    cmd = (
        f"LD_LIBRARY_PATH={eureka_lib}:$LD_LIBRARY_PATH "
        f"conda run -n eureka python {EUREKA_SCRIPT} "
        f'model="{MODEL}" '
        f"sample={sample} "
        f"training_mode={training_mode} "
        f"vlm_iterations={vlm_iterations} "
        f"max_iterations={max_iterations} "
        f"env={env_name} "
        f'env.description="{description}" '
    )
    # Add checkpoint if provided (for environments with pretrained policies)
    if checkpoint:
        cmd += f' checkpoint="{checkpoint}"'
    return cmd


def start_mps(run_id: str = None) -> dict:
    """Start NVIDIA MPS control daemon and set env vars for child processes.

    Returns a dict with keys: started(bool), pipe_dir, log_dir, msg(str).
    If `nvidia-cuda-mps-control` is not found, returns started=False with message.
    """
    mps_ctl = shutil.which("nvidia-cuda-mps-control")
    if mps_ctl is None:
        return {"started": False, "msg": "nvidia-cuda-mps-control not found"}

    user = getpass.getuser()
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    pipe_dir = f"/tmp/nvidia-mps-{user}-{run_id}"
    log_dir = f"/tmp/nvidia-mps-log-{user}-{run_id}"
    os.makedirs(pipe_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Export env vars so child processes inherit MPS config
    os.environ["CUDA_MPS_PIPE_DIRECTORY"] = pipe_dir
    os.environ["CUDA_MPS_LOG_DIRECTORY"] = log_dir

    try:
        subprocess.run([mps_ctl, "-d"], check=True)
    except Exception as e:
        return {"started": False, "pipe_dir": pipe_dir, "log_dir": log_dir, "msg": str(e)}

    return {"started": True, "pipe_dir": pipe_dir, "log_dir": log_dir, "msg": "MPS started"}


def stop_mps() -> dict:
    """Stop NVIDIA MPS control daemon if running. Returns dict with result."""
    mps_ctl = shutil.which("nvidia-cuda-mps-control")
    if mps_ctl is None:
        return {"stopped": False, "msg": "nvidia-cuda-mps-control not found"}

    try:
        proc = subprocess.run("echo quit | nvidia-cuda-mps-control", shell=True)
        return {"stopped": True, "returncode": proc.returncode}
    except Exception as e:
        return {"stopped": False, "msg": str(e)}


def _sanitize_filename(s: str) -> str:
    """Sanitize a string to be safe for filenames (simple)."""
    return (
        s.replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("\"", "_")
        .replace("'", "_")
    )


# Patterns that indicate Eureka failed even if exit code is 0
# Note: "execution error!" is NOT included because individual policy failures
# are expected - GPT may generate reward functions using non-existent attributes.
# The pipeline succeeds if at least one policy trains successfully overall.
FAILURE_PATTERNS = [
    "All iterations of code generation failed",
    "All code generation failed",
    "All training runs failed!",  # Note the "!" - this is the critical failure
    "No valid reward functions generated",
]

# Patterns that indicate true success (pipeline completed with a valid policy)
SUCCESS_PATTERNS = [
    "TRAINING COMPLETE",
    "Best policy:",
    "Done!",
]


def _check_stdout_for_failure(stdout_lines: list) -> tuple:
    """Check stdout for failure patterns.

    Returns (is_failure, failure_reason) tuple.

    The logic prioritizes success patterns: if the pipeline completed successfully
    (indicated by SUCCESS_PATTERNS), we don't consider it a failure even if
    some individual policies failed during training.
    """
    stdout_text = "".join(stdout_lines[-100:])  # Check last 100 lines

    # First check for success patterns - if found, the pipeline succeeded
    success_found = any(pattern in stdout_text for pattern in SUCCESS_PATTERNS)
    if success_found:
        return False, None

    # Only check failure patterns if no success pattern was found
    for pattern in FAILURE_PATTERNS:
        if pattern in stdout_text:
            # Find the line containing the pattern for context
            for line in reversed(stdout_lines[-50:]):
                if pattern in line:
                    return True, line.strip()
            return True, f"Found failure pattern: {pattern}"

    return False, None


def _parse_training_progress(line: str) -> tuple:
    """Parse a training output line to extract progress.

    Looks for patterns like:
    - "epoch: 136/200" (rl_games format)
    - "Iteration X/Y" (VLM feedback loop iterations)
    - "Training policy X/Y" (individual policy training)

    Returns (current, total, progress_type) or (None, None, None) if not found.
    """
    # Match rl_games epoch format: "epoch: 136/200"
    epoch_match = re.search(r'epoch:\s*(\d+)/(\d+)', line)
    if epoch_match:
        return int(epoch_match.group(1)), int(epoch_match.group(2)), 'epoch'

    # Match VLM iteration format: "Iteration X/Y"
    iter_match = re.search(r'Iteration\s+(\d+)/(\d+)', line)
    if iter_match:
        return int(iter_match.group(1)), int(iter_match.group(2)), 'vlm_iter'

    # Match policy training: "Training policy X/Y"
    policy_match = re.search(r'Training policy\s+(\d+)/(\d+)', line)
    if policy_match:
        current = int(policy_match.group(1))
        total = int(policy_match.group(2))
        return current, total, 'policy'

    # Match policy_iter format in log files
    policy_iter_match = re.search(r'policy_iter(\d+)_response(\d+)', line)
    if policy_iter_match:
        iter_num = int(policy_iter_match.group(1))
        resp_num = int(policy_iter_match.group(2))
        # Calculate overall progress
        total_iterations = 1 + VLM_ITERATIONS
        current = iter_num * SAMPLE + resp_num
        total = total_iterations * SAMPLE
        return current, total, 'policy'

    return None, None, None


class TrainingProgressBar:
    """Manages a tqdm progress bar for training epochs within a policy."""

    def __init__(self, task: str, adverb: str, total_iterations: int, sample: int):
        self.task = task
        self.adverb = adverb
        self.total_iterations = total_iterations
        self.sample = sample
        self.total_policies = total_iterations * sample
        self.current_iteration = 0
        self.current_policy = 0
        self.current_epoch = 0
        self.max_epochs = MAX_ITERATIONS
        self.pbar = None
        self.last_description = ""

    def _create_bar(self, total: int, desc: str):
        """Create or recreate the progress bar."""
        if self.pbar is not None:
            self.pbar.close()
        if tqdm is not None:
            self.pbar = tqdm(
                total=total,
                desc=desc,
                unit='epoch',
                leave=False,
                ncols=100,
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
            )
        self.last_description = desc

    def update_from_line(self, line: str):
        """Parse a line and update progress bar accordingly."""
        current, total, prog_type = _parse_training_progress(line)

        if prog_type == 'vlm_iter':
            # VLM iteration changed
            if current != self.current_iteration:
                self.current_iteration = current
                self.current_policy = 0
                self.current_epoch = 0

        elif prog_type == 'policy':
            # Policy training progress
            if current != self.current_policy:
                self.current_policy = current
                self.current_epoch = 0
                desc = f"{self.task}+{self.adverb} [iter {self.current_iteration}, policy {current}/{total}]"
                self._create_bar(self.max_epochs, desc)

        elif prog_type == 'epoch':
            # Update epoch progress
            if total != self.max_epochs and self.pbar is not None:
                # Max epochs changed, recreate bar
                self.max_epochs = total
                self._create_bar(total, self.last_description or f"{self.task}+{self.adverb}")

            if self.pbar is None:
                desc = f"{self.task}+{self.adverb} [iter {self.current_iteration}]"
                self._create_bar(total, desc)

            if self.pbar is not None and current > self.current_epoch:
                increment = current - self.current_epoch
                self.pbar.update(increment)
                self.current_epoch = current

    def close(self):
        """Close the progress bar."""
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None


def _find_best_video(per_exp_dir: str) -> str:
    """Find the best video from the VLM analysis directory.

    The new pipeline saves processed videos in vlm_analysis/ directory.
    Falls back to searching in policy directories.

    Returns the path to the best video, or None if not found.
    """
    import glob

    if not per_exp_dir or not os.path.exists(per_exp_dir):
        return None

    # Find the eureka output directory
    eureka_dirs = glob.glob(os.path.join(per_exp_dir, "outputs/eureka/*/"))
    if not eureka_dirs:
        # Try current directory structure
        eureka_dirs = [per_exp_dir]

    for eureka_dir in eureka_dirs:
        # First, look in vlm_analysis directory (new pipeline)
        vlm_videos = glob.glob(os.path.join(eureka_dir, "vlm_analysis", "*.mp4"))
        if vlm_videos:
            # Return the last iteration video (highest iter number)
            vlm_videos.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return vlm_videos[0]

        # Fallback: look in policy directories
        policy_videos = glob.glob(os.path.join(eureka_dir, "policy_iter*", "videos", "*.mp4"), recursive=True)
        if policy_videos:
            # Return the largest mp4 (likely most complete)
            policy_videos.sort(key=lambda p: os.path.getsize(p) if os.path.exists(p) else 0, reverse=True)
            return policy_videos[0]

    return None


def _find_and_copy_best_video(per_exp_dir: str, dest_name: str = "success_video.mp4") -> str:
    """Find the best video and copy it to the experiment directory.

    Args:
        per_exp_dir: The experiment output directory
        dest_name: Destination filename (default: success_video.mp4)

    Returns the path to the copied video, or None if no video found.
    """
    best_video = _find_best_video(per_exp_dir)

    if best_video is None:
        print(f"  Warning: no video found in {per_exp_dir}")
        return None

    # Copy to destination in the experiment directory
    dest_path = os.path.join(per_exp_dir, dest_name)
    try:
        shutil.copy(best_video, dest_path)
        print(f"  Copied best video to: {dest_path}")
        return dest_path
    except Exception as e:
        print(f"  Warning: could not copy best video: {e}")
        return None


def run_experiment(env_name: str, task: str, adverb: str, description: str,
                   dry_run: bool = False, output_dir: str = None, run_id: str = None,
                   checkpoint: str = None, sample: int = SAMPLE,
                   training_mode: str = TRAINING_MODE, vlm_iterations: int = VLM_ITERATIONS,
                   max_iterations: int = MAX_ITERATIONS) -> dict:
    """Run a single experiment and return results.

    Args:
        env_name: Environment name (snake_case)
        task: Task name (PascalCase)
        adverb: The adverb being tested
        description: Full task description with adverb
        dry_run: If True, don't actually run the experiment
        output_dir: Directory for experiment outputs
        run_id: Unique run identifier
        checkpoint: Optional pretrained checkpoint path
        sample: Number of reward functions per iteration
        training_mode: "individual" or "averaged"
        vlm_iterations: Number of VLM feedback loops
        max_iterations: RL training iterations
    """
    cmd = build_command(env_name, description, checkpoint=checkpoint,
                        sample=sample, training_mode=training_mode,
                        vlm_iterations=vlm_iterations, max_iterations=max_iterations)

    result = {
        "env_name": env_name,
        "task": task,
        "adverb": adverb,
        "description": description,
        "command": cmd,
        "sample": sample,
        "training_mode": training_mode,
        "vlm_iterations": vlm_iterations,
        "max_iterations": max_iterations,
        "start_time": datetime.now().isoformat(),
        "status": None,
        "returncode": None,
        "duration_seconds": None,
        "timeout_seconds": TIMEOUT_SECONDS,
        "error_message": None,
    }

    if dry_run:
        print(f"[DRY-RUN] Would execute: {cmd}")
        result["status"] = "dry_run"
        return result

    print(f"\n{'='*70}")
    print(f"Running: {task} + '{adverb}'")
    print(f"Description: {description}")
    print(f"Config: sample={sample}, training_mode={training_mode}, vlm_iterations={vlm_iterations}")
    print(f"Command: {cmd}")
    print(f"{'='*70}")

    start_time = time.time()

    # We'll stream subprocess output and display a per-experiment progress bar
    stdout_lines = []
    proc = None
    progress_bar = None
    # Prepare per-experiment output directory early so child processes
    # (e.g., Eureka -> train.py -> ffmpeg) write their outputs there.
    per_exp_dir = None
    try:
        if output_dir and run_id:
            safe_adverb = _sanitize_filename(adverb)
            per_exp_dir = os.path.join(output_dir, run_id, f"{env_name}_{safe_adverb}")
            os.makedirs(per_exp_dir, exist_ok=True)
    except Exception as e:
        print(f"Warning: could not create per-experiment dir before launch: {e}")

    # Create progress bar for this experiment
    total_iterations = 1 + vlm_iterations
    progress_bar = TrainingProgressBar(task, adverb, total_iterations, sample)

    try:
        proc = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            preexec_fn=None,
            cwd=per_exp_dir if per_exp_dir else None,
        )

        last_update = start_time
        # Read lines until process ends or timeout
        # Use select to avoid blocking on readline
        while True:
            # Check if stdout has data available (with 0.5s timeout)
            if proc.stdout:
                readable, _, _ = select.select([proc.stdout], [], [], 0.5)
                if readable:
                    line = proc.stdout.readline()
                    if line:
                        stdout_lines.append(line)
                        # Update progress bar based on output
                        progress_bar.update_from_line(line)

            ret = proc.poll()
            now = time.time()
            elapsed = now - start_time
            # check termination
            if ret is not None:
                # Drain any remaining output after process exits
                if proc.stdout:
                    for line in proc.stdout:
                        stdout_lines.append(line)
                break
            # check timeout
            if elapsed >= TIMEOUT_SECONDS:
                try:
                    proc.terminate()
                    # give it a short time, then kill
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=2)
                except Exception:
                    pass
                result["status"] = "timeout"
                result["duration_seconds"] = round(elapsed, 2)
                result["error_message"] = f"Timed out after {TIMEOUT_SECONDS}s"
                print(f"TIMEOUT after {TIMEOUT_SECONDS}s")
                # drain remaining output (with timeout to avoid blocking)
                if proc.stdout:
                    try:
                        # Give it a short time to drain
                        drain_start = time.time()
                        while time.time() - drain_start < 5:
                            readable, _, _ = select.select([proc.stdout], [], [], 0.1)
                            if readable:
                                line = proc.stdout.readline()
                                if not line:
                                    break
                                stdout_lines.append(line)
                            else:
                                break
                    except Exception:
                        pass
                break

        # process exited normally (or we broke on timeout)
        end_time = time.time()
        duration = end_time - start_time
        # Don't overwrite duration if already set (e.g., for timeout)
        if "duration_seconds" not in result:
            result["duration_seconds"] = round(duration, 2)
        # capture returncode
        if proc.returncode is None:
            proc.returncode = 0 if result.get("status") != "timeout" else -1
        result["returncode"] = proc.returncode

        # Save collected stdout/stderr if requested
        try:
            if per_exp_dir:
                stdout_path = os.path.join(per_exp_dir, "stdout.txt")
                with open(stdout_path, "w") as f:
                    f.write("".join(stdout_lines))
                result["stdout_path"] = stdout_path
                result["output_path"] = per_exp_dir
        except Exception as e:
            print(f"Warning: could not write per-experiment outputs: {e}")

        # Close progress bar before printing final status
        if progress_bar:
            progress_bar.close()

        if result.get("status") == "timeout":
            # Try to find and copy best available video for timeout case
            if per_exp_dir:
                video_path = _find_and_copy_best_video(per_exp_dir, dest_name="final_video.mp4")
                if video_path:
                    result["final_video"] = video_path
                    print(f"  Saved best available video to: {video_path}")
        elif proc.returncode == 0:
            # Even with exit code 0, check stdout for failure patterns
            # (Eureka may exit(0) even on failure in some code paths)
            is_failure, failure_reason = _check_stdout_for_failure(stdout_lines)
            if is_failure:
                result["status"] = "failed"
                result["error_message"] = failure_reason
                print(f"FAILED (detected in output: {failure_reason[:80]}...)")
            else:
                result["status"] = "success"
                print(f"SUCCESS (took {duration:.1f}s)")
                # Find and copy the best video to success_video.mp4
                if per_exp_dir:
                    video_path = _find_and_copy_best_video(per_exp_dir)
                    if video_path:
                        result["success_video"] = video_path
        else:
            result["status"] = "failed"
            last_output = "".join(stdout_lines[-20:])
            result["error_message"] = last_output[-1000:] if last_output else "Unknown error"
            print(f"FAILED (returncode={proc.returncode})")


    except Exception as e:
        duration = time.time() - start_time
        result["status"] = "error"
        result["duration_seconds"] = round(duration, 2)
        result["error_message"] = str(e)
        print(f"ERROR: {e}")
    finally:
        # Always close progress bar
        if progress_bar:
            progress_bar.close()

    result["end_time"] = datetime.now().isoformat()

    return result


def save_results(results: list, output_dir: str, run_id: str):
    """No longer saves CSV/JSON files - results are in individual experiment directories."""
    # Results are already saved in each experiment's output directory
    # No need to create additional CSV/JSON files
    pass


def load_completed_experiments(output_dir: str) -> set:
    """Load set of completed (env_name, adverb) pairs from previous runs.

    Scans output directories for successful experiments by checking for
    best_policy.pth or eureka.log with success patterns.
    """
    completed = set()

    if not os.path.exists(output_dir):
        return completed

    # Scan run directories (format: YYYYMMDD_HHMMSS)
    for run_dir in Path(output_dir).iterdir():
        if not run_dir.is_dir():
            continue
        # Check experiment subdirectories (format: env_adverb)
        for exp_dir in run_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            # Parse env_adverb from directory name
            dir_name = exp_dir.name
            parts = dir_name.split("_", 1)
            if len(parts) != 2:
                continue
            env_name, adverb = parts

            # Check for success indicators
            eureka_dirs = list(exp_dir.glob("outputs/eureka/*"))
            for eureka_dir in eureka_dirs:
                # Check for best_policy.pth (explicit success)
                if (eureka_dir / "best_policy.pth").exists():
                    completed.add((env_name, adverb))
                    break
                # Check eureka.log for success patterns
                log_file = eureka_dir / "eureka.log"
                if log_file.exists():
                    try:
                        log_text = log_file.read_text()[-2000:]  # Last 2000 chars
                        if "TRAINING COMPLETE" in log_text and "Best policy:" in log_text:
                            completed.add((env_name, adverb))
                            break
                    except Exception:
                        pass

    return completed


def print_summary(results: list):
    """Print experiment summary."""
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)

    df = pd.DataFrame(results)

    if len(df) == 0:
        print("No experiments run.")
        return

    # Overall stats
    total = len(df)
    success = len(df[df["status"] == "success"])
    failed = len(df[df["status"] == "failed"])
    timeout = len(df[df["status"] == "timeout"])
    error = len(df[df["status"] == "error"])

    print(f"\nTotal experiments: {total}")
    print(f"  Success: {success} ({100*success/total:.1f}%)")
    print(f"  Failed:  {failed} ({100*failed/total:.1f}%)")
    print(f"  Timeout: {timeout} ({100*timeout/total:.1f}%)")
    print(f"  Error:   {error} ({100*error/total:.1f}%)")

    # Per-environment breakdown
    print("\nBy environment:")
    for env in df["env_name"].unique():
        env_df = df[df["env_name"] == env]
        env_success = len(env_df[env_df["status"] == "success"])
        print(f"  {env}: {env_success}/{len(env_df)} success")

    # Duration stats
    durations = df[df["duration_seconds"].notna()]["duration_seconds"]
    if len(durations) > 0:
        print(f"\nDuration stats:")
        print(f"  Mean: {durations.mean():.1f}s")
        print(f"  Min:  {durations.min():.1f}s")
        print(f"  Max:  {durations.max():.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Run adverb experiments with VLM feedback pipeline")
    parser.add_argument("--debug", action="store_true", help="Debug mode: run only 1 adverb on 1 environment")
    parser.add_argument("--dry-run", action="store_true", help="Preview commands without executing")
    parser.add_argument("--resume", action="store_true", help="Skip already-completed experiments")
    parser.add_argument("--adverb", type=str, help="Run only this specific adverb")
    parser.add_argument("--env", type=str, help="Run only this specific environment (snake_case)")
    parser.add_argument("--max-concurrent", type=int, default=1, help="Maximum concurrent experiments (default=1)")
    parser.add_argument("--use-mps", action="store_true", help="Attempt to start NVIDIA MPS for concurrent GPU sharing")

    # New pipeline parameters
    parser.add_argument("--sample", type=int, default=SAMPLE, help=f"Number of reward functions per iteration (default={SAMPLE})")
    parser.add_argument("--training-mode", type=str, default=TRAINING_MODE, choices=["individual", "averaged"],
                        help=f"Training mode: individual or averaged (default={TRAINING_MODE})")
    parser.add_argument("--vlm-iterations", type=int, default=VLM_ITERATIONS, help=f"Number of VLM feedback loops (default={VLM_ITERATIONS})")
    parser.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS, help=f"RL training iterations per policy (default={MAX_ITERATIONS})")

    args = parser.parse_args()

    # Load adverbs
    adverb_df = load_adverbs(ADVERB_CSV)
    adverbs = adverb_df["word"].tolist()

    # Filter if specific adverb requested
    if args.adverb:
        if args.adverb not in adverbs:
            print(f"Error: adverb '{args.adverb}' not found in {ADVERB_CSV}")
            sys.exit(1)
        adverbs = [args.adverb]

    # Filter environments if specific one requested
    environments = ENVIRONMENTS
    if args.env:
        environments = [e for e in ENVIRONMENTS if e["env_name"] == args.env]
        if not environments:
            print(f"Error: environment '{args.env}' not found")
            print(f"Available: {[e['env_name'] for e in ENVIRONMENTS]}")
            sys.exit(1)

    # Debug mode: just one experiment
    if args.debug:
        adverbs = adverbs[:1]
        environments = environments[:1]
        print(f"DEBUG MODE: Testing {adverbs[0]} on {environments[0]['env_name']}")

    # Load completed experiments if resuming
    completed = set()
    if args.resume:
        completed = load_completed_experiments(OUTPUT_DIR)
        print(f"Resuming: found {len(completed)} completed experiments")

    # Generate run ID
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Calculate total experiments
    total_experiments = len(environments) * len(adverbs)
    skip_count = 0

    # Calculate total policies per experiment
    total_iterations = 1 + args.vlm_iterations
    policies_per_exp = total_iterations * args.sample if args.training_mode == "individual" else total_iterations

    print(f"\nAdverb Experiment Runner (VLM Pipeline)")
    print(f"========================================")
    print(f"Adverbs:        {len(adverbs)}")
    print(f"Environments:   {len(environments)}")
    print(f"Total:          {total_experiments} experiments")
    print(f"Timeout:        {TIMEOUT_SECONDS}s per experiment")
    print(f"Output dir:     {OUTPUT_DIR}")
    print(f"Run ID:         {run_id}")
    print(f"Max concurrent: {args.max_concurrent}")
    print(f"\nPipeline config:")
    print(f"  sample:         {args.sample}")
    print(f"  training_mode:  {args.training_mode}")
    print(f"  vlm_iterations: {args.vlm_iterations}")
    print(f"  max_iterations: {args.max_iterations}")
    print(f"  policies/exp:   {policies_per_exp}")
    if args.use_mps:
        print("  MPS: enabled (attempting to start)")

    if args.dry_run:
        print("\n*** DRY RUN MODE - No experiments will be executed ***\n")

    # Run experiments
    results = []
    # Build list of work items
    work_items = []
    for env in environments:
        for adverb in adverbs:
            if (env["env_name"], adverb) in completed:
                skip_count += 1
                continue
            description = build_description(env["base_description"], adverb)
            # Check if this environment has a pretrained checkpoint
            checkpoint = PRETRAINED_CHECKPOINTS.get(env["env_name"])
            work_items.append((env["env_name"], env["task"], adverb, description, checkpoint))

    # Optionally start MPS if requested
    mps_started = False
    mps_info = None
    if args.use_mps and args.max_concurrent > 1:
        mps_info = start_mps(run_id)
        mps_started = bool(mps_info.get("started"))
        if not mps_started:
            print(f"Warning: could not start MPS: {mps_info.get('msg')}")
        else:
            print(f"NVIDIA MPS started: pipe_dir={mps_info.get('pipe_dir')}")

    # If max_concurrent == 1, fall back to sequential for clarity
    if args.max_concurrent <= 1:
        # Sequential execution without progress display
        for idx, (env_name, task, adverb, description, checkpoint) in enumerate(work_items, start=1):
            print(f"\n[{idx}/{len(work_items)}] {task} + '{adverb}'")
            if checkpoint:
                print(f"  Using pretrained checkpoint: {checkpoint}")
            result = run_experiment(
                env_name=env_name, task=task, adverb=adverb, description=description,
                dry_run=args.dry_run, output_dir=OUTPUT_DIR, run_id=run_id,
                checkpoint=checkpoint, sample=args.sample, training_mode=args.training_mode,
                vlm_iterations=args.vlm_iterations, max_iterations=args.max_iterations
            )
            results.append(result)

            # Save intermediate results after each experiment
            if not args.dry_run and results:
                save_results(results, OUTPUT_DIR, run_id)
    else:
        # Run with a ThreadPoolExecutor to limit concurrent subprocess launches
        max_workers = max(1, args.max_concurrent)
        print(f"Running up to {max_workers} experiments concurrently")
        futures = []
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
                # Submit all tasks
                for item in work_items:
                    env_name, task, adverb, description, checkpoint = item
                    # pass output_dir and run_id so each worker can save per-experiment logs
                    futures.append(
                        ex.submit(
                            run_experiment,
                            env_name,
                            task,
                            adverb,
                            description,
                            args.dry_run,
                            OUTPUT_DIR,
                            run_id,
                            checkpoint,
                            args.sample,
                            args.training_mode,
                            args.vlm_iterations,
                            args.max_iterations,
                        )
                    )

                # Collect results without progress display
                for fut in concurrent.futures.as_completed(futures):
                    try:
                        res = fut.result()
                        results.append(res)
                    except Exception as e:
                        print(f"Worker error: {e}")
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt received, attempting to cancel running experiments...")
            # Note: subprocesses spawned by run_experiment will continue; user may need to clean up.
        finally:
            if mps_started:
                stopped = stop_mps()
                print(f"Stopped MPS: {stopped}")

    # Final save and summary
    if results and not args.dry_run:
        save_results(results, OUTPUT_DIR, run_id)
        print_summary(results)

    if skip_count > 0:
        print(f"\nSkipped {skip_count} already-completed experiments")

    print("\nDone!")


if __name__ == "__main__":
    main()
