#!/usr/bin/env python
# encoding: utf-8
"""
Quick fix script to patch existing local Sotopia data.
Downloads a small portion of episodes to extract agent_goals for environments
that are missing them.
"""
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

HF_SOTOPIA_URL = "https://huggingface.co/datasets/cmu-lti/sotopia/resolve/main"


def download_episodes():
    """Download the full episodes file."""
    print("Downloading episodes data (~180MB, please wait)...")
    try:
        with urllib.request.urlopen(
            f"{HF_SOTOPIA_URL}/sotopia_episodes_v1.jsonl", timeout=300
        ) as response:
            return response.read().decode("utf-8")
    except Exception as e:
        print(f"Download failed: {e}")
        return None


def clean_goal(goal_text: str) -> str:
    """Remove <extra_info> tags from goal text."""
    return re.sub(r"<extra_info>.*?</extra_info>", "", goal_text, flags=re.DOTALL).strip()


def main():
    data_dir = os.path.expanduser("~/.sotopia/data")
    env_dir = os.path.join(data_dir, "EnvironmentProfile")

    if not os.path.exists(env_dir):
        print(f"Environment directory not found: {env_dir}")
        print("Please run 'python scripts/setup_data.py' first.")
        sys.exit(1)

    # Check how many envs need fixing
    envs_to_fix = {}
    for fname in os.listdir(env_dir):
        if fname.endswith(".json"):
            fpath = os.path.join(env_dir, fname)
            with open(fpath) as f:
                data = json.load(f)
            goals = data.get("agent_goals", [])
            if not any(g.strip() for g in goals if g):
                envs_to_fix[data["pk"]] = fpath

    print(f"Found {len(envs_to_fix)} environments with missing goals.")

    if not envs_to_fix:
        print("All environments already have goals. Nothing to fix.")
        return

    # Download episodes
    content = download_episodes()
    if not content:
        sys.exit(1)

    # Parse episodes and extract goals
    env_goals = {}  # env_id -> {agent_name: goal}
    env_relationships = {}  # env_id -> relationship value

    for line in content.strip().split("\n"):
        if not line.strip():
            continue
        try:
            ep = json.loads(line)
        except json.JSONDecodeError:
            continue

        env_id = ep.get("environment_id", "")
        if env_id not in envs_to_fix:
            continue

        if env_id in env_goals:
            continue  # Already found goals for this env

        social_goals = ep.get("social_goals", {})
        if social_goals and isinstance(social_goals, dict):
            goals = []
            for agent_name, goal in social_goals.items():
                goals.append(clean_goal(goal))
            if any(g.strip() for g in goals):
                env_goals[env_id] = goals[:2]

        # Also extract relationship if available
        rel = ep.get("relationship", None)
        if rel is not None and env_id not in env_relationships:
            if isinstance(rel, int):
                env_relationships[env_id] = rel
            elif isinstance(rel, str):
                # Try to parse relationship string
                rel_map = {
                    "stranger": 0, "strangers": 0,
                    "know_by_name": 1,
                    "acquaintance": 2, "acquaintances": 2,
                    "friend": 3, "friends": 3,
                    "romantic": 4, "romantic_relationship": 4,
                    "family": 5, "family_member": 5, "family_members": 5,
                }
                env_relationships[env_id] = rel_map.get(rel.lower().strip(), 2)

    print(f"Found goals for {len(env_goals)} environments.")

    # Patch the environment files
    patched = 0
    for env_id, fpath in envs_to_fix.items():
        with open(fpath) as f:
            data = json.load(f)

        updated = False

        if env_id in env_goals:
            data["agent_goals"] = env_goals[env_id]
            updated = True

        if env_id in env_relationships:
            data["relationship"] = env_relationships[env_id]
            updated = True

        if updated:
            with open(fpath, "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            patched += 1

    print(f"Patched {patched} environment files.")
    print("Done!")


if __name__ == "__main__":
    main()
