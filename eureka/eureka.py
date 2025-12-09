import json
import logging
import os

import hydra
import numpy as np
from openai import OpenAI
from pathlib import Path

import re
import shutil
import subprocess
import time

from utils.generate_success import transform_code, replace_exterior_compute_reward
from utils.create_task import create_task
from utils.extract_task_code import file_to_string, get_function_signature
from utils.file_utils import load_tensorboard_logs
from utils.misc import (
    set_freest_gpu,
    filter_traceback,
    block_until_training,
    get_vlm_feedback,
    get_vlm_selection,
    process_episode_clips,
)

# Dynamically determine paths based on script location
EUREKA_ROOT_DIR = str(Path(__file__).resolve().parent)
ISAAC_ROOT_DIR = str(Path(EUREKA_ROOT_DIR).parent / "isaacgymenvs" / "isaacgymenvs")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def extract_code_from_response(response_content: str) -> str:
    """Extract Python code from GPT response."""
    patterns = [
        r"```python(.*?)```",
        r"```(.*?)```",
        r'"""(.*?)"""',
        r'""(.*?)""',
        r'"(.*?)"',
    ]
    code_string = None
    for pattern in patterns:
        match = re.search(pattern, response_content, re.DOTALL)
        if match is not None:
            code_string = match.group(1).strip()
            break
    code_string = response_content if not code_string else code_string

    # Remove unnecessary imports, keep only function definition
    lines = code_string.split("\n")
    for i, line in enumerate(lines):
        if line.strip().startswith("def "):
            code_string = "\n".join(lines[i:])
            break

    return code_string


def generate_averaged_reward_function(reward_functions: list) -> str:
    """
    Generate a single reward function that averages n reward functions.

    Args:
        reward_functions: List of (code_string, signature, input_list) tuples

    Returns:
        Combined reward function code string
    """
    if len(reward_functions) == 1:
        return reward_functions[0][0]  # Return single function as-is

    # Extract the input parameters from the first function (assume all have same signature)
    _, first_signature, first_inputs = reward_functions[0]

    # Build individual function definitions with unique names
    individual_functions = []
    for i, (code, sig, inputs) in enumerate(reward_functions):
        # Rename compute_reward to compute_reward_i
        renamed_code = code.replace("def compute_reward(", f"def compute_reward_{i}(")
        individual_functions.append(renamed_code)

    # Build the combined function that averages all rewards
    # Parse the first signature to get parameter list
    param_match = re.search(r'compute_reward\((.*?)\)', first_signature)
    if param_match:
        params = param_match.group(1)
    else:
        params = ", ".join(first_inputs)

    # Build call strings for each individual function
    calls = [f"compute_reward_{i}({params})" for i in range(len(reward_functions))]

    combined_function = f'''
@torch.jit.script
def compute_reward({params}) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Averaged reward from {len(reward_functions)} reward functions."""
    rewards = []
    all_dicts = []

'''

    # Add calls to each function
    for i, call in enumerate(calls):
        combined_function += f"    rew_{i}, dict_{i} = {call}\n"
        combined_function += f"    rewards.append(rew_{i})\n"
        combined_function += f"    all_dicts.append(dict_{i})\n"

    # Average the rewards
    combined_function += f'''
    # Average all rewards
    avg_reward = torch.stack(rewards).mean(dim=0)

    # Combine all reward dicts (prefix with function index)
    combined_dict: Dict[str, torch.Tensor] = {{}}
    for i, d in enumerate(all_dicts):
        for k, v in d.items():
            combined_dict[f"reward_{{i}}_{{k}}"] = v
    combined_dict["averaged_reward"] = avg_reward

    return avg_reward, combined_dict
'''

    # Combine all individual functions + the averaging function
    all_code = "\n\n".join(individual_functions) + "\n\n" + combined_function

    return all_code


def train_policy(
    task: str,
    suffix: str,
    output_file: str,
    iteration: int,
    response_id: int,
    cfg,
    code_string: str,
    task_code_string: str,
    is_first: bool = False,
) -> dict:
    """
    Train a single policy with given reward function.

    Returns dict with training results including video path, tensorboard logs, etc.
    """
    # Get function signature
    try:
        gpt_reward_signature, input_lst = get_function_signature(code_string)
    except Exception as e:
        logging.error(f"Cannot parse function signature: {e}")
        return {"success": False, "error": "signature_parse_error"}

    # Prepare task file with reward function
    if is_first:
        reward_signature_lines = [
            f"self.rew_buf[:], self.rew_dict = {gpt_reward_signature}",
            "self.extras['gpt_reward'] = self.rew_buf.mean()",
            "for rew_state in self.rew_dict: self.extras[rew_state] = self.rew_dict[rew_state].mean()",
        ]
        indent = " " * 8
        reward_signature = "\n".join([indent + line for line in reward_signature_lines])

        if "def compute_reward(self)" in task_code_string:
            task_code_string_iter = task_code_string.replace(
                "def compute_reward(self):",
                "def compute_reward(self):\n" + reward_signature,
            )
        elif "def compute_reward(self, actions)" in task_code_string:
            task_code_string_iter = task_code_string.replace(
                "def compute_reward(self, actions):",
                "def compute_reward(self, actions):\n" + reward_signature,
            )
        else:
            raise NotImplementedError("Unknown compute_reward signature")

        # Write task file
        with open(output_file, "w") as file:
            file.write(task_code_string_iter + "\n")
            file.write("from typing import Tuple, Dict\n")
            file.write("import math\n")
            file.write("import torch\n")
            file.write("from torch import Tensor\n")
            if "@torch.jit.script" not in code_string:
                code_string = "@torch.jit.script\n" + code_string
            file.write(code_string + "\n")

        # Transform code (copy GPT reward to compute_success)
        with open(output_file, 'r') as file:
            task_code = transform_code(file.read())
        with open(output_file, 'w') as file:
            file.write(task_code)
    else:
        # Replace existing compute_reward
        with open(output_file, 'r') as file:
            task_code = file.read()

        if "@torch.jit.script" not in code_string:
            code_string = "@torch.jit.script\n" + code_string
        task_code = replace_exterior_compute_reward(task_code, code_string)

        with open(output_file, 'w') as file:
            file.write(task_code)

    # Save reward-only file for reference
    reward_only_path = f"env_iter{iteration}_response{response_id}_rewardonly.py"
    with open(reward_only_path, "w") as file:
        file.write(code_string + "\n")

    # Copy full env file for bookkeeping
    env_copy_path = f"env_iter{iteration}_response{response_id}.py"
    shutil.copy(output_file, env_copy_path)

    # Find freest GPU
    set_freest_gpu()

    # Run training
    rl_filepath = f"env_iter{iteration}_response{response_id}.txt"
    hydra_output_dir = f"policy_iter{iteration}_response{response_id}"

    with open(rl_filepath, "w") as f:
        process = subprocess.Popen(
            [
                "python", "-u", f"{ISAAC_ROOT_DIR}/train.py",
                "hydra/output=subprocess",
                f"task={task}{suffix}",
                f"wandb_activate={cfg.use_wandb}",
                f"wandb_entity={cfg.wandb_username}",
                f"wandb_project={cfg.wandb_project}",
                f"headless={not cfg.capture_video}",
                f"capture_video={cfg.capture_video}",
                "force_render=False",
                f"max_iterations={cfg.max_iterations}",
                f"hydra.run.dir={hydra_output_dir}",
            ] + ([f"checkpoint={cfg.checkpoint}"] if cfg.checkpoint else []),
            stdout=f,
            stderr=f,
        )

    block_until_training(rl_filepath, log_status=True, iter_num=iteration, response_id=response_id)
    process.communicate()

    # Check for errors
    with open(rl_filepath, "r") as f:
        stdout_str = f.read()

    traceback_msg = filter_traceback(stdout_str)
    if traceback_msg != "":
        logging.error(f"Training failed:\n{traceback_msg}")
        return {
            "success": False,
            "error": "training_error",
            "traceback": traceback_msg,
        }

    # Parse tensorboard logs
    lines = stdout_str.split("\n")
    tensorboard_logdir = None
    for line in lines:
        if line.startswith("Tensorboard Directory:"):
            tensorboard_logdir = line.split(":")[-1].strip()
            break

    if tensorboard_logdir is None:
        return {"success": False, "error": "no_tensorboard"}

    tensorboard_logs = load_tensorboard_logs(tensorboard_logdir)

    # Get max GPT reward
    max_gpt_reward = max(tensorboard_logs.get("gpt_reward", [0]))

    # Find video directory
    video_dir = Path(hydra_output_dir) / "videos"

    return {
        "success": True,
        "iteration": iteration,
        "response_id": response_id,
        "code_string": code_string,
        "reward_only_path": reward_only_path,
        "env_copy_path": env_copy_path,
        "hydra_output_dir": hydra_output_dir,
        "tensorboard_logdir": tensorboard_logdir,
        "tensorboard_logs": tensorboard_logs,
        "max_gpt_reward": max_gpt_reward,
        "video_dir": video_dir,
    }


@hydra.main(config_path="cfg", config_name="config", version_base="1.1")
def main(cfg):
    workspace_dir = Path.cwd()
    logging.info(f"Workspace: {workspace_dir}")
    logging.info(f"Project Root: {EUREKA_ROOT_DIR}")

    task = cfg.env.task
    task_description = cfg.env.description

    # Check if description is a file path
    if task_description.endswith(".txt"):
        if os.path.exists(task_description):
            with open(task_description, "r") as f:
                task_description = f.read()
            logging.info(f"Loaded task description from file: {cfg.env.description}")
        else:
            logging.warning(f"Description file not found: {task_description}, using as literal string")

    suffix = cfg.suffix
    model = cfg.model
    n_samples = cfg.sample  # Number of reward functions to generate
    training_mode = cfg.get("training_mode", "individual")  # "individual" or "averaged"
    vlm_iterations = cfg.get("vlm_iterations", 1)  # Number of VLM feedback loops

    logging.info(f"Using LLM: {model}")
    logging.info(f"Task: {task}")
    logging.info(f"Task description: {task_description}")
    logging.info(f"Number of reward samples: {n_samples}")
    logging.info(f"Training mode: {training_mode}")
    logging.info(f"VLM iterations: {vlm_iterations}")
    if cfg.checkpoint:
        logging.info(f"Loading checkpoint: {cfg.checkpoint}")

    env_name = cfg.env.env_name.lower()
    env_parent = (
        "isaac"
        if f"{env_name}.py" in os.listdir(f"{EUREKA_ROOT_DIR}/envs/isaac")
        else "dexterity"
    )
    task_file = f"{EUREKA_ROOT_DIR}/envs/{env_parent}/{env_name}.py"
    task_obs_file = f"{EUREKA_ROOT_DIR}/envs/{env_parent}/{env_name}_obs.py"
    shutil.copy(task_obs_file, "env_init_obs.py")
    task_code_string = file_to_string(task_file)
    task_obs_code_string = file_to_string(task_obs_file)
    output_file = f"{ISAAC_ROOT_DIR}/tasks/{env_name}{suffix.lower()}.py"

    # Load prompts
    prompt_dir = f"{EUREKA_ROOT_DIR}/utils/prompts"
    initial_system = file_to_string(f"{prompt_dir}/initial_system.txt")
    code_output_tip = file_to_string(f"{prompt_dir}/code_output_tip.txt")
    initial_user = file_to_string(f"{prompt_dir}/initial_user.txt")
    reward_signature = file_to_string(f"{prompt_dir}/reward_signature.txt")

    initial_system = (
        initial_system.format(task_reward_signature_string=reward_signature)
        + code_output_tip
    )
    initial_user = initial_user.format(
        task_obs_code_string=task_obs_code_string, task_description=task_description
    )

    messages = [
        {"role": "system", "content": initial_system},
        {"role": "user", "content": initial_user},
    ]

    task_code_string = task_code_string.replace(task, task + suffix)
    create_task(ISAAC_ROOT_DIR, cfg.env.task, cfg.env.env_name, suffix)

    # Create VLM analysis directory
    os.makedirs("vlm_analysis", exist_ok=True)

    # Track best results across all iterations
    best_overall_result = None
    best_overall_reward = float("-inf")

    # Main loop: initial generation + VLM feedback iterations
    total_iterations = 1 + vlm_iterations  # 1 initial + vlm_iterations refinements

    for iteration in range(total_iterations):
        logging.info(f"\n{'='*60}")
        logging.info(f"Iteration {iteration}/{total_iterations-1}")
        logging.info(f"{'='*60}")

        # Generate n reward functions from GPT
        logging.info(f"Generating {n_samples} reward functions with {model}")

        responses = []
        for attempt in range(100):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=cfg.temperature,
                    n=n_samples,
                )
                responses = response.choices
                break
            except Exception as e:
                logging.info(f"Attempt {attempt + 1} failed: {e}")
                time.sleep(1)

        if not responses:
            logging.error("Failed to get GPT response after 100 attempts")
            exit(1)

        logging.info(f"Received {len(responses)} reward functions")

        # Extract code from each response
        reward_functions = []
        for i, resp in enumerate(responses):
            code_string = extract_code_from_response(resp.message.content)
            try:
                sig, inputs = get_function_signature(code_string)
                reward_functions.append((code_string, sig, inputs))
                logging.info(f"Reward function {i}: parsed successfully")
            except Exception as e:
                logging.warning(f"Reward function {i}: failed to parse - {e}")

        if not reward_functions:
            logging.error("No valid reward functions generated!")
            continue

        # Training phase
        training_results = []

        if training_mode == "individual":
            # Mode A: Train n separate policies
            logging.info(f"Training {len(reward_functions)} individual policies")

            for i, (code_string, sig, inputs) in enumerate(reward_functions):
                logging.info(f"Training policy {i}/{len(reward_functions)-1}")
                result = train_policy(
                    task=task,
                    suffix=suffix,
                    output_file=output_file,
                    iteration=iteration,
                    response_id=i,
                    cfg=cfg,
                    code_string=code_string,
                    task_code_string=task_code_string,
                    is_first=(iteration == 0 and i == 0),
                )
                training_results.append(result)

                if result["success"]:
                    logging.info(f"Policy {i}: Max GPT Reward = {result['max_gpt_reward']:.2f}")
                else:
                    logging.warning(f"Policy {i}: Training failed - {result.get('error')}")

        else:
            # Mode B: Train one policy with averaged reward
            logging.info("Training single policy with averaged reward")

            # Generate averaged reward function
            averaged_code = generate_averaged_reward_function(reward_functions)

            result = train_policy(
                task=task,
                suffix=suffix,
                output_file=output_file,
                iteration=iteration,
                response_id=0,
                cfg=cfg,
                code_string=averaged_code,
                task_code_string=task_code_string,
                is_first=(iteration == 0),
            )
            training_results.append(result)

            if result["success"]:
                logging.info(f"Averaged policy: Max GPT Reward = {result['max_gpt_reward']:.2f}")

        # Filter successful results
        successful_results = [r for r in training_results if r.get("success")]

        if not successful_results:
            logging.error("All training runs failed!")
            continue

        # Process videos for successful policies (only if VLM feedback is needed)
        video_paths = []
        if iteration < total_iterations - 1:
            for result in successful_results:
                video_dir = result["video_dir"].absolute()
                if video_dir.exists():
                    output_video = Path(f"vlm_analysis/iter{iteration}_policy{result['response_id']}.mp4").absolute()
                    try:
                        process_episode_clips(video_dir, output_video)
                        video_paths.append((result, str(output_video)))
                        logging.info(f"Processed video for policy {result['response_id']}: {output_video}")
                    except Exception as e:
                        logging.warning(f"Failed to process video for policy {result['response_id']}: {e}")

        if not video_paths:
            # Select best by reward (no VLM feedback)
            best_result = max(successful_results, key=lambda r: r["max_gpt_reward"])
        else:
            # VLM feedback phase (skip on last iteration)
            if iteration < total_iterations - 1:
                logging.info("Getting VLM feedback")

                if training_mode == "individual" and len(video_paths) > 1:
                    # Mode A: Gemini selects best from n videos
                    logging.info(f"Gemini selecting best from {len(video_paths)} videos")

                    # Get selection and feedback for each video
                    all_feedbacks = []
                    for result, video_path in video_paths:
                        try:
                            feedback = get_vlm_feedback(task_description, video_path)
                            all_feedbacks.append((result, video_path, feedback))
                            logging.info(f"Policy {result['response_id']} feedback received")

                            # Save individual feedback
                            with open(f"vlm_analysis/iter{iteration}_policy{result['response_id']}_feedback.txt", "w") as f:
                                f.write(feedback)
                        except Exception as e:
                            logging.warning(f"Failed to get feedback for policy {result['response_id']}: {e}")

                    # Get Gemini to select the best video
                    if len(all_feedbacks) > 1:
                        video_paths_for_selection = [vp for _, vp, _ in all_feedbacks]
                        try:
                            best_idx, selection_reasoning = get_vlm_selection(
                                task_description, video_paths_for_selection
                            )
                            best_result, best_video, vlm_feedback = all_feedbacks[best_idx]
                            logging.info(f"Gemini selected policy {best_result['response_id']}")

                            with open(f"vlm_analysis/iter{iteration}_selection.txt", "w") as f:
                                f.write(f"Selected: Policy {best_result['response_id']}\n\n")
                                f.write(selection_reasoning)
                        except Exception as e:
                            logging.warning(f"Selection failed, using highest reward: {e}")
                            best_result, best_video, vlm_feedback = max(
                                all_feedbacks, key=lambda x: x[0]["max_gpt_reward"]
                            )
                    else:
                        best_result, best_video, vlm_feedback = all_feedbacks[0]
                else:
                    # Mode B or single video: get feedback on single video
                    best_result, best_video = video_paths[0]
                    try:
                        vlm_feedback = get_vlm_feedback(task_description, best_video)
                        with open(f"vlm_analysis/iter{iteration}_feedback.txt", "w") as f:
                            f.write(vlm_feedback)
                    except Exception as e:
                        logging.warning(f"Failed to get VLM feedback: {e}")
                        vlm_feedback = ""

                # Update messages for next GPT call
                # Include: original task, previous reward function, VLM feedback
                previous_reward = best_result["code_string"]

                refinement_prompt = f"""
The previous reward function was:
```python
{previous_reward}
```

Here is feedback from a VLM that watched a video of the trained policy:
{vlm_feedback}

Please revise the reward function based on this feedback to better achieve the task: "{task_description}"

Provide an improved reward function that addresses the feedback.
"""

                # Update message history
                if len(messages) == 2:
                    # First refinement: add assistant response and user feedback
                    messages.append({
                        "role": "assistant",
                        "content": f"```python\n{previous_reward}\n```"
                    })
                    messages.append({
                        "role": "user",
                        "content": refinement_prompt
                    })
                else:
                    # Subsequent refinements: update last messages
                    messages[-2] = {
                        "role": "assistant",
                        "content": f"```python\n{previous_reward}\n```"
                    }
                    messages[-1] = {
                        "role": "user",
                        "content": refinement_prompt
                    }

                # Save messages
                with open("messages.json", "w") as f:
                    json.dump(messages, f, indent=2)

            else:
                # Last iteration: VLM selects best policy
                logging.info("Final iteration: VLM selecting best policy")

                if training_mode == "individual" and len(video_paths) > 1:
                    # Get VLM to evaluate and select best
                    all_feedbacks = []
                    for result, video_path in video_paths:
                        try:
                            feedback = get_vlm_feedback(task_description, video_path)
                            all_feedbacks.append((result, video_path, feedback))
                            logging.info(f"Policy {result['response_id']} feedback received")

                            # Save individual feedback
                            with open(f"vlm_analysis/iter{iteration}_policy{result['response_id']}_feedback.txt", "w") as f:
                                f.write(feedback)
                        except Exception as e:
                            logging.warning(f"Failed to get feedback for policy {result['response_id']}: {e}")

                    if len(all_feedbacks) > 1:
                        video_paths_for_selection = [vp for _, vp, _ in all_feedbacks]
                        try:
                            best_idx, selection_reasoning = get_vlm_selection(
                                task_description, video_paths_for_selection
                            )
                            best_result, best_video, _ = all_feedbacks[best_idx]
                            logging.info(f"VLM selected policy {best_result['response_id']} as best")

                            with open(f"vlm_analysis/iter{iteration}_final_selection.txt", "w") as f:
                                f.write(f"Selected: Policy {best_result['response_id']}\n\n")
                                f.write(selection_reasoning)
                        except Exception as e:
                            logging.warning(f"VLM selection failed, using highest reward: {e}")
                            best_result = max(successful_results, key=lambda r: r["max_gpt_reward"])
                            # Find corresponding video
                            for result, video_path in video_paths:
                                if result['response_id'] == best_result['response_id']:
                                    best_video = video_path
                                    break
                    elif len(all_feedbacks) == 1:
                        best_result, best_video, _ = all_feedbacks[0]
                    else:
                        best_result = max(successful_results, key=lambda r: r["max_gpt_reward"])
                        best_video = video_paths[0][1] if video_paths else None
                else:
                    # Single video case
                    best_result = video_paths[0][0] if video_paths else successful_results[0]
                    best_video = video_paths[0][1] if video_paths else None

                    if best_video:
                        try:
                            feedback = get_vlm_feedback(task_description, best_video)
                            with open(f"vlm_analysis/iter{iteration}_final_feedback.txt", "w") as f:
                                f.write(feedback)
                        except Exception as e:
                            logging.warning(f"Failed to get final VLM feedback: {e}")

        # Track best overall (now based on VLM selection, not just GPT reward)
        # We update if this is the last iteration (VLM selected) or if reward is higher
        if iteration == total_iterations - 1:
            # Last iteration: VLM selection takes precedence
            best_overall_result = best_result
            best_overall_reward = best_result["max_gpt_reward"]
        elif best_result["max_gpt_reward"] > best_overall_reward:
            best_overall_reward = best_result["max_gpt_reward"]
            best_overall_result = best_result

        logging.info(f"Iteration {iteration} complete. Best reward: {best_result['max_gpt_reward']:.2f}")

    # Final summary
    logging.info(f"\n{'='*60}")
    logging.info("TRAINING COMPLETE")
    logging.info(f"{'='*60}")

    if best_overall_result:
        logging.info(f"Best policy: iteration {best_overall_result['iteration']}, response {best_overall_result['response_id']}")
        logging.info(f"Best GPT reward: {best_overall_reward:.2f}")
        logging.info(f"Policy saved in: {best_overall_result['hydra_output_dir']}")

        # Copy best policy to final location
        shutil.copy(best_overall_result["env_copy_path"], output_file)
        logging.info(f"Best reward function saved to: {output_file}")

        # Save best_policy.pth explicitly
        # Find the trained model file in the hydra output directory
        policy_dir = Path(best_overall_result['hydra_output_dir'])
        nn_dirs = list(policy_dir.glob("runs/*/nn"))
        if nn_dirs:
            # Find the latest .pth file
            pth_files = list(nn_dirs[0].glob("*.pth"))
            if pth_files:
                # Sort by modification time, get latest
                latest_pth = max(pth_files, key=lambda p: p.stat().st_mtime)
                best_policy_path = Path("best_policy.pth")
                shutil.copy(latest_pth, best_policy_path)
                logging.info(f"Best policy weights saved to: {best_policy_path}")

        # Save best_video.mp4 explicitly
        # Look for the processed video in vlm_analysis
        best_iter = best_overall_result['iteration']
        best_resp = best_overall_result['response_id']
        vlm_video = Path(f"vlm_analysis/iter{best_iter}_policy{best_resp}.mp4")
        if vlm_video.exists():
            best_video_path = Path("best_video.mp4")
            shutil.copy(vlm_video, best_video_path)
            logging.info(f"Best video saved to: {best_video_path}")
        else:
            # Try to find any video for this policy
            policy_dir = Path(best_overall_result['hydra_output_dir'])
            video_files = list(policy_dir.glob("**/*.mp4"))
            if video_files:
                best_video_path = Path("best_video.mp4")
                shutil.copy(video_files[0], best_video_path)
                logging.info(f"Best video saved to: {best_video_path}")

    else:
        logging.error("No successful training runs!")

    logging.info("Done!")


if __name__ == "__main__":
    main()
