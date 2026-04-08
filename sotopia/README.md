# Sotopia

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

The Sotopia dataset includes agent profiles, environment profiles, and task splits. Run the setup script to download and prepare the data:

```bash
python scripts/setup_data.py
```

This downloads data from the [Sotopia HuggingFace dataset](https://huggingface.co/datasets/cmu-lti/sotopia) and saves it to `~/.sotopia/data/`.

### 3. Configure

Copy and edit the configuration file:

```bash
cp config.json my_config.json
```

Edit `my_config.json` to set your API key, model name, and other parameters.

### 4. Run

```bash
python run_sotopia_battle.py -c my_config.json
```

In watch mode, non-`direct` agents display intermediate thinking processes in cyan.

## Game Overview

**Sotopia** is an open-ended social simulation environment where two AI agents interact through natural language conversation. Each agent has:
- A **character profile** (name, age, occupation, personality, values)
- A **private social goal** they aim to achieve
- A **secret** they must protect
- A **relationship** with the other agent (strangers, acquaintances, friends, etc.)

Scenarios span diverse social dynamics including negotiation, cooperation, competition, persuasion, and conflict resolution.

## Configuration

```jsonc
{
    "game": {
        "task_split": "hard",                // Task split: "hard" (90) or "testing" (450)
        "language": "english",               // Language
        "mode": "watch",                     // Mode (watch: display thinking process)
        "game_count": 10,                    // Number of episodes to run
        "start_game_idx": 0,                 // Starting episode index
        "exp_name": "battle",                // Experiment name
        "output_dir": "logs/sotopia/battle", // Output directory
        "max_turns": 20,                     // Maximum conversation turns per agent
        "max_stale_turns": 2,                // Max consecutive "none" actions before termination
        "evaluate": true,                    // Whether to run LLM evaluation after each episode
        "evaluator_model": null,             // Evaluator model (null = use default_model)
        "data_dir": null,                    // Data directory (null = ~/.sotopia/data)
        "enable_intent_identification": false // Enable intent identification (for training data)
    },
    "default_model": {
        "model_name": "gpt-5-mini",              // Default backend LLM
        "api_key": "your-api-key-here",      // API key
        "api_base": null,                    // Custom API base URL (optional)
        "temperature": 0.3                   // Temperature
    },
    "agents": [
        {
            "agent_type": "react",           // Agent type for agent 1
            "model": null                    // Per-agent model override (optional)
        },
        {
            "agent_type": "react",           // Agent type for agent 2
            "model": null
        }
    ],
    "evaluator": {
        "model_name": null,                  // null = use default_model
        "api_key": null,
        "api_base": null,
        "temperature": 0
    },
    "extractors": {
        "model_name": null,
        "api_key": null,
        "api_base": null,
        "temperature": 0
    },
    "refiner": {
        "model_path": "/path/to/Qwen2.5-7B-Instruct",
        "lora_path": "/path/to/lora-checkpoint",
        "temperature": 0.7
    }
}
```

## Agent Types

| Agent Type | Description | API Calls |
|-----------|-------------|-----------|
| `direct` | Direct response generation | 1 |
| `react` | ReAct framework (Analysis + Response) | 2 |
| `recon` | ReCon framework (Relationship Analysis + Strategy + Response) | 3 |
| `metamind` | MetaMind framework (Theory of Mind + Strategy + Response) | 3 |
| `refiner+<type>` | Wraps any agent type with a trained Refiner model | +1 |

Examples: `refiner+react`, `refiner+direct`, `refiner+metamind`

## Training Pipeline

### Step 1: Collect Self-Play Data

Enable `enable_intent_identification` to generate training data with intent information:

```json
{
    "game": {
        "enable_intent_identification": true
    }
}
```

Run episodes to collect data:

```bash
python run_sotopia_battle.py -c config.json
```

Each episode generates a directory containing `process.json`. With intent identification enabled, each conversation turn includes:
- `desired_responses`: 3 desired follower responses (K=3)
- `undesired_responses`: 3 undesired follower responses (K=3)

### Step 2: Convert to Training Format

Convert game logs to GRPO training format:

```bash
python scripts/convert_logs_to_grpo_data.py \
    --log_dir logs/sotopia/battle \
    --output grpo_training_data.jsonl \
    --include_intent
```

### Step 3: Start Reward Server

The Reward Server uses a local LLM as the Measurer to compute follower response log probabilities:

```bash
pip install fastapi uvicorn

python scripts/reward_server.py \
    --model_path /path/to/Qwen2.5-72B-Instruct \
    --port 8000 \
    --torch_dtype bfloat16
```

Verify the server is running:

```bash
curl http://localhost:8000/health
```

### Step 4: GRPO Training

We use [ms-swift](https://github.com/modelscope/ms-swift) for GRPO training. Replace `scripts/rewards.py` as the reward function in your training config.

```bash
# Set Reward Server address (if not localhost:8000)
export REWARD_SERVER_HOST=127.0.0.1
export REWARD_SERVER_PORT=8000
```

### Step 5: Inference with Trained Refiner

After training, set the `refiner` config in your game configuration to use the trained checkpoint:

```json
{
    "refiner": {
        "model_path": "/path/to/Qwen2.5-7B-Instruct",
        "lora_path": "/path/to/trained-lora-checkpoint",
        "temperature": 0.7
    }
}
```

Then set any agent's `agent_type` to `refiner+<base_type>` (e.g., `refiner+react`).
