# Roboly: VLM-Guided Reward Learning for Robot Locomotion

This extension of Eureka adds a Vision-Language Model (VLM) feedback loop for training robot policies with natural language task descriptions that include adverbs (e.g., "walk forward quickly", "move slowly and steadily").

## Overview

The system uses a multi-stage pipeline:
1. **GPT generates reward functions** from natural language task descriptions
2. **RL trains policies** using the generated reward functions
3. **VLM (Gemini) evaluates** policy videos and provides feedback
4. **GPT refines reward functions** based on VLM feedback
5. **Final VLM selection** picks the best policy

## Training Loop (`eureka/eureka.py`)

### Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        VLM Feedback Loop                        │
│  (repeats for vlm_iterations, e.g., 1 iteration = 2 GPT calls)  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 1. GPT Generates Reward Functions                               │
│    - Input: Task description + environment info + VLM feedback  │
│    - Output: N reward functions (controlled by `sample` param)  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. Train Policies (training_mode)                               │
│    - "individual": Train N separate policies, one per reward    │
│    - "averaged": Train 1 policy with averaged reward signal     │
│    - Each policy trains for `max_iterations` RL steps           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. Process Videos                                               │
│    - Merge episode clips into single video per policy           │
│    - Save to vlm_analysis/iter{N}_policy{M}.mp4                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. VLM Evaluation (Gemini)                                      │
│    - Get feedback on each policy video                          │
│    - Select best policy via multi-video comparison              │
│    - Save feedback to vlm_analysis/                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. Refinement (if not final iteration)                          │
│    - Send VLM feedback + previous reward to GPT                 │
│    - GPT generates improved reward functions                    │
│    - Loop back to step 2                                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. Final Output                                                 │
│    - best_policy.pth: Neural network weights                    │
│    - best_video.mp4: Policy rollout video                       │
│    - Best reward function saved to task file                    │
└─────────────────────────────────────────────────────────────────┘
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sample` | 2 | Number of reward functions GPT generates per iteration |
| `training_mode` | "individual" | "individual" (N policies) or "averaged" (1 policy) |
| `vlm_iterations` | 1 | Number of VLM feedback loops (1 = 2 total GPT generations) |
| `max_iterations` | 500 | RL training iterations per policy |
| `checkpoint` | "" | Path to pretrained policy weights (optional) |

### Example Usage

```bash
# Basic usage
cd /path/to/Eureka/eureka
LD_LIBRARY_PATH=/path/to/eureka/lib:$LD_LIBRARY_PATH \
conda run -n eureka python eureka.py \
    env=ant \
    env.description="make the ant walk forward quickly"

# With custom parameters
LD_LIBRARY_PATH=/path/to/eureka/lib:$LD_LIBRARY_PATH \
conda run -n eureka python eureka.py \
    model="gpt-4.1-mini" \
    sample=4 \
    training_mode=individual \
    vlm_iterations=2 \
    max_iterations=100 \
    env=ant \
    env.description="make the ant walk forward slowly and steadily" \
    checkpoint="/path/to/pretrained_ant.pth"
```

### Output Directory Structure

```
outputs/eureka/YYYY-MM-DD_HH-MM-SS/
├── eureka.log                    # Main log file
├── messages.json                 # GPT conversation history
├── best_policy.pth               # Final trained policy weights
├── best_video.mp4                # Video of best policy
├── vlm_analysis/
│   ├── iter0_policy0.mp4         # Processed video for policy 0
│   ├── iter0_policy1.mp4         # Processed video for policy 1
│   ├── iter0_policy0_feedback.txt
│   ├── iter0_policy1_feedback.txt
│   ├── iter0_selection.txt       # VLM's selection reasoning
│   ├── selection_response.txt    # Full VLM response
│   └── full_response.txt         # Full feedback response
├── policy_iter0_response0/       # Policy 0 training outputs
│   ├── reward.py                 # Generated reward function
│   └── runs/                     # RL training logs
├── policy_iter0_response1/       # Policy 1 training outputs
└── ...
```

## Batch Experiments (`eval/run_adverb_experiments_vlm.py`)

This script runs systematic experiments across multiple environments and adverbs.

### Features

- **Batch execution**: Run experiments for all combinations of environments and adverbs
- **Resume capability**: Skip already-completed experiments with `--resume`
- **Dry-run mode**: Preview commands without executing with `--dry-run`
- **Pretrained checkpoints**: Start from pretrained policies for faster convergence
- **Configurable parameters**: Override default pipeline settings via CLI

### Example Usage

```bash
cd /path/to/Eureka/eval

# Run single experiment
python run_adverb_experiments_vlm.py --adverb quickly --env ant

# Dry run to see commands
python run_adverb_experiments_vlm.py --dry-run

# Run with custom parameters
python run_adverb_experiments_vlm.py \
    --adverb slowly \
    --env humanoid \
    --sample 4 \
    --vlm-iterations 2 \
    --max-iterations 100

# Resume previous run (skip completed experiments)
python run_adverb_experiments_vlm.py --resume
```

### CLI Arguments

| Argument | Description |
|----------|-------------|
| `--adverb WORD` | Run only this specific adverb |
| `--env NAME` | Run only this specific environment |
| `--dry-run` | Preview commands without executing |
| `--resume` | Skip already-completed experiments |
| `--sample N` | Number of reward functions per iteration |
| `--training-mode MODE` | "individual" or "averaged" |
| `--vlm-iterations N` | Number of VLM feedback loops |
| `--max-iterations N` | RL training iterations per policy |

### Supported Environments

- `ant` - Quadruped ant robot
- `humanoid` - Bipedal humanoid robot
- `anymal` - ANYmal quadruped robot
- `cartpole` - Cart-pole balancing
- `ball_balance` - Ball balancing on plate
- `franka_cabinet` - Franka arm opening cabinet

### Adverbs

Adverbs are loaded from `eval/eval_adv_small.csv`. Examples:
- quickly, slowly
- carefully, recklessly
- smoothly, jerkily
- steadily, erratically

### Output Structure

```
eval/adverb_experiment_results_vlm/
└── YYYYMMDD_HHMMSS/              # Run timestamp
    ├── ant_quickly/
    │   ├── stdout.txt            # Experiment output
    │   └── outputs/eureka/...    # Eureka output directory
    ├── ant_slowly/
    ├── humanoid_quickly/
    └── ...
```

## VLM Evaluation (`eureka/utils/vlm_evaluation.py`)

Standalone utility for VLM operations, runs in `base` conda environment (Python 3.12+).

### Modes

1. **feedback**: Get VLM feedback on a single policy video
2. **selection**: Select best video from multiple policies
3. **process**: Merge episode clips into single video

### Example Usage

```bash
# Get feedback on a video
conda run -n base python utils/vlm_evaluation.py \
    --mode feedback \
    --task "make the ant walk forward quickly" \
    --video /path/to/video.mp4

# Select best from multiple videos
conda run -n base python utils/vlm_evaluation.py \
    --mode selection \
    --task "make the ant walk forward quickly" \
    --videos "/path/to/video1.mp4,/path/to/video2.mp4"

# Process episode clips
conda run -n base python utils/vlm_evaluation.py \
    --mode process \
    --video-dir /path/to/policy/runs/ \
    --output /path/to/output.mp4
```

## Environment Setup

### Conda Environments

Two conda environments are required:

1. **eureka** (Python 3.8): For RL training with IsaacGym
   ```bash
   conda activate eureka
   ```

2. **base** (Python 3.12+): For VLM calls with google-genai
   ```bash
   conda activate base
   pip install google-genai
   ```

### Required Environment Variables

```bash
# For IsaacGym
export LD_LIBRARY_PATH=/path/to/eureka/lib:$LD_LIBRARY_PATH

# For Gemini API (in base environment)
export GOOGLE_API_KEY=your_api_key
```

## Key Differences from Original Eureka

1. **VLM Feedback Loop**: Uses Gemini to evaluate policy videos and provide natural language feedback
2. **Adverb Support**: Task descriptions include adverbs for nuanced behavior specification
3. **Multi-Policy Selection**: VLM selects best policy from multiple candidates
4. **Explicit Outputs**: Saves `best_policy.pth` and `best_video.mp4` explicitly
5. **Pretrained Checkpoints**: Can start from pretrained policies for faster convergence
6. **Subprocess VLM Calls**: VLM runs in separate conda environment to handle Python version differences
