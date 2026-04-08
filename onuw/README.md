# ONUW (One Night Ultimate Werewolf)

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
python run_onuw_battle.py -c my_config.json
```

In watch mode, non-`direct` agents display intermediate thinking processes in cyan.

## Game Rules

**One Night Ultimate Werewolf (ONUW)** is a fast-paced social deduction game with 5 players and 7 roles. Each player receives one role, and the remaining 2 roles are placed in the center pool.

**Roles** (7 total):
- **Werewolf** (1): Team Werewolf. Wakes up at night to check for other Werewolves.
- **Villager** (2): Team Village. No special abilities.
- **Seer** (1): Team Village. May examine one player's role or two center pool roles at night.
- **Robber** (1): Team Village. May swap their role with another player and view their new role.
- **Troublemaker** (1): Team Village. May swap two other players' roles without viewing them.
- **Insomniac** (1): Team Village. Views their own final role at night's end.

**Game Flow**: The game has three sequential phases:
1. **Night Phase**: Players with night abilities act in order: Werewolf → Seer → Robber → Troublemaker → Insomniac. Role swaps during the night create uncertainty.
2. **Day Phase**: All players discuss openly for multiple rounds to identify the Werewolf. Concealing and deceiving are encouraged.
3. **Voting Phase**: All players simultaneously vote to eliminate one player. The player(s) with the most votes are eliminated.

**Victory**: Team Village wins if the Werewolf is eliminated. Team Werewolf wins if the Werewolf avoids elimination. If no Werewolf exists among players (both in center) and no one dies, it's a draw.

## Configuration

```jsonc
{
    "game": {
        "player_nums": 5,                    // Number of players (must be 5)
        "language": "english",               // Language (english)
        "mode": "watch",                     // Mode (watch: display thinking process)
        "game_count": 10,                    // Number of games to run
        "start_game_idx": 0,                 // Starting game index
        "exp_name": "battle",                // Experiment name
        "camp": null,                        // Camp filter (village/werewolf/null for no filter)
        "output_dir": "logs/onuw/battle",    // Output directory
        "max_discuss_round": 3,              // Number of discussion rounds
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
            "agent_type": "direct",          // Agent type
            "model": null                    // Per-player model override (optional)
        }
        // ... more players (5 total)
    ],
    "roles": ["Werewolf", "Villager", "Villager", "Seer", "Robber", "Troublemaker", "Insomniac"],
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
| `recon` | Relation Consistency framework (cross-player relation analysis) | 3 |
| `lasi` | LASI framework (Landscape Analysis - Strategy - Implementation) | 4 |
| `belief` | Belief-based agent with belief modeling before response | 2 |
| `llm_ins` | LLM-instructed speaking strategy selection with belief modeling | 3 |
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
python run_onuw_battle.py -c config.json
```

Each game generates a directory containing `process.json`. With intent identification enabled, each discussion event includes:
- `desired_responses`: 3 desired follower responses (K=3)
- `undesired_responses`: 3 undesired follower responses (K=3)

### Step 2: Convert to Training Format

Convert game logs to GRPO training format:

```bash
python scripts/convert_logs_to_grpo_data.py \
    --log_dir logs/onuw/battle \
    --output grpo_training_data.jsonl \
    --include_intent
```

### Step 3: Start Reward Server

The Reward Server uses a local LLM (Qwen2.5-72B-Instruct) as the Measurer to compute follower response log probabilities:

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
