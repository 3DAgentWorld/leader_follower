# Werewolf

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure

Copy and edit the configuration file:

```bash
cp config.json my_config.json
```

Edit `my_config.json` to set your API key, model name, and other parameters.

### 3. Run

```bash
python run_werewolf_battle.py -c my_config.json
```

In watch mode, non-`direct` agents display intermediate thinking processes in cyan.

## Game Rules

**Werewolf** is a classic social deduction game with 7 players:
- **2 Werewolves**: Know each other. Eliminate one player each night.
- **1 Seer**: Investigates one player each night to learn if they are a Werewolf.
- **1 Guardian**: Protects one player each night from elimination.
- **3 Villagers**: No special abilities.

**Night Phase**: Werewolves choose a target → Seer investigates → Guardian protects.
**Day Phase**: Announce night results → Discussion → Vote to eliminate.

**Victory**: Village team wins when both Werewolves are eliminated. Werewolf team wins when Werewolves equal or outnumber Village team members.

## Configuration

```jsonc
{
    "game": {
        "player_nums": 7,                    // Number of players (must be 7)
        "language": "english",               // Language (english)
        "mode": "watch",                     // Mode (watch: display thinking process)
        "game_count": 10,                    // Number of games to run
        "start_game_idx": 0,                 // Starting game index
        "exp_name": "battle",                // Experiment name
        "camp": null,                        // Camp filter (village/werewolf/null for no filter)
        "output_dir": "logs/werewolf/battle",// Output directory
        "enable_intent_identification": false // Enable intent identification (for training data collection)
    },
    "default_model": {
        "model_name": "gpt-5-mini",              // Default backend LLM
        "api_key": "your-api-key-here",      // API key
        "api_base": null,                    // Custom API base URL (optional)
        "temperature": 0.3                   // Temperature
    },
    "players": [
        {
            "name": "player 1",
            "role": null,                    // Role (null = random assignment)
            "agent_type": "react",           // Agent type
            "model": null                    // Per-player model override (optional)
        }
        // ... more players (7 total)
    ],
    "roles": ["Werewolf", "Werewolf", "Seer", "Guardian", "Villager", "Villager", "Villager"],
    "extractors": {
        "model_name": null,                  // null = use default_model
        "api_key": null,
        "api_base": null,
        "temperature": 0
    },
    "refiner": {                             // Refiner configuration (required for refiner+ agents)
        "model_path": "/path/to/Qwen2.5-7B-Instruct",
        "lora_path": "/path/to/lora-checkpoint",  // LoRA adapter path (optional)
        "temperature": 0.7
    }
}
```

## Agent Types

| Agent Type | Description | API Calls |
|-----------|-------------|-----------|
| `direct` | Direct response generation | 1 |
| `react` | ReAct framework (Reasoning + Acting) | 2 |
| `recon` | ReCon framework (cross-player relation analysis) | 3 |
| `lasi` | LASI framework | 4 |
| `refiner+<type>` | Wraps any agent type with a trained Refiner model | +1 |

Examples: `refiner+react`, `refiner+direct`, `refiner+lasi`

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

Run games to collect data:

```bash
python run_werewolf_battle.py -c config.json
```

Each game generates a directory containing `process.json`. With intent identification enabled, each discussion event includes:
- `desired_responses`: 3 desired follower responses (K=3)
- `undesired_responses`: 3 undesired follower responses (K=3)

### Step 2: Convert to Training Format

Convert game logs to GRPO training format:

```bash
python scripts/convert_logs_to_grpo_data.py \
    --log_dir logs/werewolf/battle \
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

Then set any player's `agent_type` to `refiner+<base_type>` (e.g., `refiner+react`).
