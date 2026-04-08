#!/usr/bin/env python
# encoding: utf-8
"""
Main runner for Sotopia social simulation battles.
Supports running multiple episodes with different agent types and task splits.
"""
import argparse
import json
import logging
import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

from colorama import Fore, Style

from src.agents.llm_agent.chatgpt_agent import SotopiaAgent, RefinerWrapper
from src.games.sotopia.sotopia import (
    SotopiaGame,
    AgentProfile,
    EnvironmentProfile,
)
from src.utils import create_dir

logging.basicConfig(level=logging.WARNING, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_data(data_dir: str) -> Tuple[Dict[str, AgentProfile], Dict[str, EnvironmentProfile]]:
    """
    Load agent and environment profiles from local data directory.

    Args:
        data_dir: Path to the data directory containing AgentProfile/ and EnvironmentProfile/ subdirs.

    Returns:
        (agent_profiles_dict, env_profiles_dict) keyed by pk.
    """
    agents = {}
    envs = {}

    agent_dir = os.path.join(data_dir, "AgentProfile")
    env_dir = os.path.join(data_dir, "EnvironmentProfile")

    if not os.path.exists(agent_dir) or not os.path.exists(env_dir):
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}\n"
            f"Please run 'python scripts/setup_data.py' first to download and prepare the data."
        )

    for fname in os.listdir(agent_dir):
        if fname.endswith(".json"):
            with open(os.path.join(agent_dir, fname), "r") as f:
                data = json.load(f)
            agents[data["pk"]] = AgentProfile(data)

    for fname in os.listdir(env_dir):
        if fname.endswith(".json"):
            with open(os.path.join(env_dir, fname), "r") as f:
                data = json.load(f)
            envs[data["pk"]] = EnvironmentProfile(data)

    return agents, envs


def load_task_split(task_file: str) -> List[Dict]:
    """Load a task split file (hard_tasks_90.json or testing_tasks_450.json)."""
    with open(task_file, "r") as f:
        tasks = json.load(f)
    return tasks


def validate_config(config: dict) -> bool:
    """Validate the configuration."""
    required_keys = ["game", "default_model", "agents"]
    for key in required_keys:
        if key not in config:
            print(f"Missing required config key: {key}")
            return False

    game_cfg = config["game"]
    if "task_split" not in game_cfg and "task_file" not in game_cfg:
        print("Config must specify either 'task_split' or 'task_file'")
        return False

    return True


def create_agent(
    agent_cfg: dict,
    default_model: dict,
    refiner_config: Optional[dict] = None,
    mode: str = "watch",
) -> Tuple:
    """
    Create an agent based on configuration.

    Returns:
        (AgentClass, kwargs) tuple for lazy instantiation.
    """
    agent_type = agent_cfg.get("agent_type", "direct")
    model_cfg = agent_cfg.get("model") or default_model

    name = agent_cfg.get("name", "Agent")

    base_kwargs = {
        "name": name,
        "agent_type": agent_type.replace("refiner+", ""),
        "model": model_cfg.get("model_name", default_model["model_name"]),
        "api_key": model_cfg.get("api_key") or default_model["api_key"],
        "api_base": model_cfg.get("api_base") or default_model.get("api_base"),
        "temperature": model_cfg.get("temperature", default_model.get("temperature", 0.3)),
        "mode": mode,
    }

    if agent_type.startswith("refiner+"):
        if not refiner_config:
            raise ValueError("Refiner config required for refiner+ agent types.")

        class RefinerFactory:
            """Factory to create RefinerWrapper with lazy agent instantiation."""
            @staticmethod
            def init_instance(**override_kwargs):
                merged = {**base_kwargs, **override_kwargs}
                # Override name if provided
                if "name" in override_kwargs:
                    merged["name"] = override_kwargs["name"]
                base_agent = SotopiaAgent(**merged)
                return RefinerWrapper(
                    wrapped_agent=base_agent,
                    refiner_model_path=refiner_config["model_path"],
                    refiner_lora_path=refiner_config.get("lora_path"),
                    refiner_temperature=refiner_config.get("temperature", 0.7),
                )

        return RefinerFactory, base_kwargs
    else:
        return SotopiaAgent, base_kwargs


def run_episode(
    config: dict,
    task: dict,
    agents_dict: Dict[str, AgentProfile],
    envs_dict: Dict[str, EnvironmentProfile],
    episode_idx: int,
) -> Optional[SotopiaGame]:
    """
    Run a single Sotopia episode.

    Args:
        config: Full configuration dict.
        task: Task dict with env_id and agent_ids.
        agents_dict: All available agent profiles.
        envs_dict: All available environment profiles.
        episode_idx: Episode index for logging.

    Returns:
        SotopiaGame instance after completion, or None on error.
    """
    game_cfg = config["game"]
    default_model = config["default_model"]
    refiner_config = config.get("refiner")

    env_id = task["env_id"]
    agent_ids = task["agent_ids"]

    # Look up profiles
    if env_id not in envs_dict:
        print(f"Environment {env_id} not found, skipping.")
        return None

    env_profile = envs_dict[env_id]
    agent_profiles = []
    for aid in agent_ids:
        if aid not in agents_dict:
            print(f"Agent {aid} not found, skipping.")
            return None
        agent_profiles.append(agents_dict[aid])

    # Create output directory
    exp_name = game_cfg.get("exp_name", "battle")
    output_dir = os.path.join(
        game_cfg.get("output_dir", "logs/sotopia"),
        f"{exp_name}-episode_{episode_idx}",
    )
    create_dir(output_dir)

    # Create game instance
    mode = game_cfg.get("mode", "watch")
    game = SotopiaGame(
        mode=mode,
        ai_model=default_model["model_name"],
        output_dir=output_dir,
        max_turns=game_cfg.get("max_turns", 20),
        max_stale_turns=game_cfg.get("max_stale_turns", 2),
        enable_intent_identification=game_cfg.get("enable_intent_identification", False),
        evaluator_model=game_cfg.get("evaluator_model", default_model["model_name"]),
    )

    # Load scenario
    game.load_scenario(env_profile, agent_profiles)

    # Create agents
    agents_config = config.get("agents", [])
    agent_args = []

    for i, profile in enumerate(agent_profiles):
        # Use per-agent config if available, otherwise use default
        if i < len(agents_config):
            agent_cfg = agents_config[i]
        else:
            agent_cfg = agents_config[0] if agents_config else {"agent_type": "direct"}

        agent_cfg_copy = dict(agent_cfg)
        agent_cfg_copy["name"] = profile.full_name

        agent_cls, kwargs = create_agent(
            agent_cfg_copy,
            default_model,
            refiner_config=refiner_config,
            mode=mode,
        )
        kwargs["name"] = profile.full_name
        agent_args.append((agent_cls, kwargs))

    game.add_agents(agent_args)

    # Run the episode
    try:
        game.start()
    except Exception as e:
        print(f"Episode {episode_idx} failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Evaluate
    if game_cfg.get("evaluate", True):
        try:
            eval_model_cfg = config.get("evaluator", default_model)
            game.evaluate(
                evaluator_api_key=eval_model_cfg.get("api_key", default_model["api_key"]),
                evaluator_api_base=eval_model_cfg.get("api_base", default_model.get("api_base")),
            )
        except Exception as e:
            print(f"Evaluation failed for episode {episode_idx}: {e}")

    return game


def run_batch(config: dict):
    """
    Run a batch of Sotopia episodes based on configuration.
    """
    game_cfg = config["game"]
    default_model = config["default_model"]

    # Load data
    data_dir = game_cfg.get("data_dir", os.path.expanduser("~/.sotopia/data"))
    print(f"Loading data from {data_dir}...")
    agents_dict, envs_dict = load_data(data_dir)
    print(f"Loaded {len(agents_dict)} agents, {len(envs_dict)} environments.")

    # Load task split
    task_file = game_cfg.get("task_file")
    task_split = game_cfg.get("task_split", "hard")

    if task_file:
        tasks = load_task_split(task_file)
    else:
        # Look for built-in task splits
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if task_split == "hard":
            task_file = os.path.join(base_dir, "data", "hard_tasks_90.json")
        elif task_split == "testing":
            task_file = os.path.join(base_dir, "data", "testing_tasks_450.json")
        else:
            raise ValueError(f"Unknown task split: {task_split}. Use 'hard' or 'testing'.")

        if not os.path.exists(task_file):
            raise FileNotFoundError(
                f"Task file not found: {task_file}\n"
                f"Please run 'python scripts/setup_data.py' first."
            )
        tasks = load_task_split(task_file)

    print(f"Loaded {len(tasks)} tasks from {task_split} split.")

    # Apply game count and start index
    game_count = game_cfg.get("game_count", len(tasks))
    start_idx = game_cfg.get("start_game_idx", 0)
    tasks = tasks[start_idx:start_idx + game_count]

    print(f"Running episodes {start_idx} to {start_idx + len(tasks) - 1}...")
    print(f"Agent types: {[a.get('agent_type', 'direct') for a in config.get('agents', [])]}")
    print("=" * 60)

    # Run episodes
    results = []
    goal_scores_1 = []
    goal_scores_2 = []
    overall_scores_1 = []
    overall_scores_2 = []

    for i, task in enumerate(tasks):
        episode_idx = start_idx + i
        print(f"\n{'='*60}")
        print(f"Episode {episode_idx}: {task.get('codename', 'unknown')} (env: {task['env_id'][:12]}...)")
        print(f"{'='*60}")

        t_start = time.time()
        game = run_episode(config, task, agents_dict, envs_dict, episode_idx)
        t_elapsed = time.time() - t_start

        if game and game.evaluation_results:
            g1, g2 = game.get_goal_scores()
            o1, o2 = game.get_overall_scores()
            goal_scores_1.append(g1)
            goal_scores_2.append(g2)
            overall_scores_1.append(o1)
            overall_scores_2.append(o2)

            results.append({
                "episode_idx": episode_idx,
                "codename": task.get("codename", ""),
                "env_id": task["env_id"],
                "agent_ids": task["agent_ids"],
                "goal_scores": [g1, g2],
                "overall_scores": [o1, o2],
                "time": t_elapsed,
            })

            print(f"\nGoal Scores: Agent1={g1:.1f}, Agent2={g2:.1f}")
            print(f"Overall Scores: Agent1={o1:.2f}, Agent2={o2:.2f}")
        else:
            results.append({
                "episode_idx": episode_idx,
                "codename": task.get("codename", ""),
                "env_id": task["env_id"],
                "error": True,
                "time": t_elapsed,
            })

        print(f"Time: {t_elapsed:.1f}s")

    # Print summary
    print(f"\n{'='*60}")
    print("BATCH SUMMARY")
    print(f"{'='*60}")
    print(f"Total episodes: {len(tasks)}")
    print(f"Successful: {len(goal_scores_1)}")

    if goal_scores_1:
        avg_goal_1 = sum(goal_scores_1) / len(goal_scores_1)
        avg_goal_2 = sum(goal_scores_2) / len(goal_scores_2)
        avg_overall_1 = sum(overall_scores_1) / len(overall_scores_1)
        avg_overall_2 = sum(overall_scores_2) / len(overall_scores_2)

        print(f"\nAgent 1 - Avg Goal: {avg_goal_1:.2f}, Avg Overall: {avg_overall_1:.2f}")
        print(f"Agent 2 - Avg Goal: {avg_goal_2:.2f}, Avg Overall: {avg_overall_2:.2f}")
        print(f"Combined Avg Goal: {(avg_goal_1 + avg_goal_2) / 2:.2f}")
        print(f"Combined Avg Overall: {(avg_overall_1 + avg_overall_2) / 2:.2f}")

    # Save results summary
    output_dir = game_cfg.get("output_dir", "logs/sotopia")
    exp_name = game_cfg.get("exp_name", "battle")
    summary_file = os.path.join(output_dir, f"{exp_name}-summary.json")
    create_dir(output_dir)

    summary = {
        "config": {
            "task_split": task_split,
            "agent_types": [a.get("agent_type", "direct") for a in config.get("agents", [])],
            "model": default_model["model_name"],
            "max_turns": game_cfg.get("max_turns", 20),
        },
        "results": results,
        "summary": {
            "total_episodes": len(tasks),
            "successful": len(goal_scores_1),
            "avg_goal_score_agent1": sum(goal_scores_1) / len(goal_scores_1) if goal_scores_1 else 0,
            "avg_goal_score_agent2": sum(goal_scores_2) / len(goal_scores_2) if goal_scores_2 else 0,
            "avg_overall_score_agent1": sum(overall_scores_1) / len(overall_scores_1) if overall_scores_1 else 0,
            "avg_overall_score_agent2": sum(overall_scores_2) / len(overall_scores_2) if overall_scores_2 else 0,
        },
    }

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="Run Sotopia social simulation battles.")
    parser.add_argument("-c", "--config", type=str, default="config.json", help="Path to config file.")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = json.load(f)

    if not validate_config(config):
        sys.exit(1)

    run_batch(config)


if __name__ == "__main__":
    main()
