#!/usr/bin/env python
# encoding: utf-8
"""
Data setup script for Sotopia social simulation.
Downloads and prepares agent profiles, environment profiles, and task splits
from the Sotopia HuggingFace dataset.
"""
import json
import os
import re
import signal
import sys
import urllib.request
from pathlib import Path

HF_SOTOPIA_URL = "https://huggingface.co/datasets/cmu-lti/sotopia/resolve/main"

# Graceful exit handler
_interrupted = False


def _signal_handler(sig, frame):
    global _interrupted
    _interrupted = True
    print("\nInterrupted! Saving progress...")


signal.signal(signal.SIGINT, _signal_handler)


def download_file(url: str, desc: str) -> str:
    """Download a file and return its content."""
    print(f"  Downloading {desc}...")
    try:
        with urllib.request.urlopen(url, timeout=300) as response:
            return response.read().decode("utf-8")
    except Exception as e:
        print(f"  WARNING: Download failed: {e}")
        return None


def parse_agent_background(name: str, background: str, agent_id: str) -> dict:
    """Parse an agent profile from background text."""
    profile = {
        "pk": agent_id,
        "first_name": "",
        "last_name": "",
        "age": 0,
        "occupation": "",
        "gender": "",
        "gender_pronoun": "",
        "public_info": "",
        "personality_and_values": "",
        "decision_making_style": "",
        "secret": "",
        "big_five": "",
        "mbti": "",
    }

    # Parse name
    name_parts = name.split()
    if len(name_parts) >= 2:
        profile["first_name"] = name_parts[0]
        profile["last_name"] = " ".join(name_parts[1:])
    elif name_parts:
        profile["first_name"] = name_parts[0]

    # Parse age
    age_match = re.search(r"(\d+)-year-old", background)
    if age_match:
        profile["age"] = int(age_match.group(1))

    # Parse gender
    bg_lower = background.lower()
    if "female" in bg_lower or "woman" in bg_lower:
        profile["gender"] = "Female"
    elif "male" in bg_lower or "man" in bg_lower:
        profile["gender"] = "Male"
    elif "nonbinary" in bg_lower:
        profile["gender"] = "Nonbinary"

    # Parse pronouns
    pronoun_match = re.search(r"(He/him|She/her|They/them)", background, re.IGNORECASE)
    if pronoun_match:
        profile["gender_pronoun"] = pronoun_match.group(1)

    # Parse occupation (after age and gender)
    occ_match = re.search(
        r"\d+-year-old\s+(?:male\s+|female\s+|nonbinary\s+)?(.+?)(?:\.|,)",
        background,
        re.IGNORECASE,
    )
    if occ_match:
        profile["occupation"] = occ_match.group(1).strip()

    # Parse public info (everything before "Personality")
    parts = background.split("Personality and values description:")
    if len(parts) > 0:
        sentences = parts[0].split(".")
        if len(sentences) > 2:
            profile["public_info"] = ".".join(sentences[1:]).strip()
        elif len(sentences) > 1:
            profile["public_info"] = sentences[1].strip()

    # Parse personality
    personality_match = re.search(
        r"Personality and values description:\s*(.+?)(?:(?:secret|Secret)s?:|$)",
        background,
        re.IGNORECASE | re.DOTALL,
    )
    if personality_match:
        profile["personality_and_values"] = personality_match.group(1).strip().rstrip(".")

    # Parse secret
    secret_match = re.search(r"(?:secret|Secret)s?:\s*(.+?)$", background, re.IGNORECASE)
    if secret_match:
        profile["secret"] = secret_match.group(1).strip()

    return profile


def parse_environment_from_episode(episode: dict) -> dict:
    """Extract environment profile from an episode."""
    env_id = episode.get("environment_id", "")

    # Parse social goals (can be a dict {agent_name: goal} or a list of dicts)
    agent_goals = []
    social_goals = episode.get("social_goals", {})
    if social_goals:
        if isinstance(social_goals, dict):
            # Format: {"Agent Name": "goal text with <extra_info>...</extra_info>"}
            for agent_name, goal in social_goals.items():
                clean_goal = re.sub(
                    r"<extra_info>.*?</extra_info>", "", goal, flags=re.DOTALL
                ).strip()
                agent_goals.append(clean_goal)
        elif isinstance(social_goals, list):
            for goal_item in social_goals:
                if isinstance(goal_item, dict):
                    for agent_name, goal in goal_item.items():
                        clean_goal = re.sub(
                            r"<extra_info>.*?</extra_info>", "", goal, flags=re.DOTALL
                        ).strip()
                        agent_goals.append(clean_goal)
                elif isinstance(goal_item, str):
                    clean_goal = re.sub(
                        r"<extra_info>.*?</extra_info>", "", goal_item, flags=re.DOTALL
                    ).strip()
                    agent_goals.append(clean_goal)

    # Pad to 2 goals
    while len(agent_goals) < 2:
        agent_goals.append("")

    return {
        "pk": env_id,
        "codename": episode.get("codename", ""),
        "source": "sotopia",
        "scenario": episode.get("scenario", ""),
        "agent_goals": agent_goals[:2],
        "relationship": episode.get("relationship", 2),
        "age_constraint": episode.get("age_constraint"),
        "occupation_constraint": episode.get("occupation_constraint"),
    }


def build_task_splits(episodes: list) -> tuple:
    """
    Build task split files from episodes.

    Returns:
        (all_tasks_450, hard_tasks_90)
    """
    # Collect all unique (env_id, agent_pair) combos
    seen_combos = set()
    all_tasks = []

    for ep in episodes:
        env_id = ep.get("environment_id", "")
        agent_ids = ep.get("agent_ids", [])
        codename = ep.get("codename", "")

        if not env_id or len(agent_ids) < 2:
            continue

        combo_key = (env_id, tuple(agent_ids[:2]))
        if combo_key not in seen_combos:
            seen_combos.add(combo_key)
            all_tasks.append({
                "env_id": env_id,
                "agent_ids": agent_ids[:2],
                "codename": codename,
            })

    # Hard tasks: from benchmark_agents.json (downloaded separately)
    # For now, return all tasks; hard tasks will be loaded from benchmark_agents.json
    return all_tasks, None


def main():
    output_dir = os.path.expanduser("~/.sotopia/data")
    os.makedirs(output_dir, exist_ok=True)

    # Also prepare the local data directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    local_data_dir = os.path.join(project_dir, "data")
    os.makedirs(local_data_dir, exist_ok=True)

    print("=" * 60)
    print("Sotopia Data Setup")
    print("=" * 60)

    # Step 1: Download episodes
    print("\nStep 1: Downloading episodes data...")
    episodes_content = download_file(
        f"{HF_SOTOPIA_URL}/sotopia_episodes_v1.jsonl",
        "sotopia_episodes_v1.jsonl (~180MB, please wait)",
    )

    if not episodes_content:
        print("ERROR: Failed to download episodes file.")
        sys.exit(1)

    print("  Episodes downloaded successfully.")

    # Step 2: Download benchmark_agents.json
    print("\nStep 2: Downloading benchmark agents...")
    combo_content = download_file(
        f"{HF_SOTOPIA_URL}/benchmark_agents.json",
        "benchmark_agents.json",
    )

    # Step 3: Parse episodes
    print("\nStep 3: Parsing episodes...")
    episodes = []
    for line in episodes_content.strip().split("\n"):
        if _interrupted:
            break
        if line.strip():
            try:
                episodes.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    print(f"  Parsed {len(episodes)} episodes.")

    # Step 4: Extract profiles
    print("\nStep 4: Extracting profiles...")
    agents_dict = {}
    envs_dict = {}

    for ep in episodes:
        if _interrupted:
            break

        # Extract environment
        env_id = ep.get("environment_id", "")
        if env_id and env_id not in envs_dict:
            envs_dict[env_id] = parse_environment_from_episode(ep)

        # Extract agents
        agent_ids = ep.get("agent_ids", [])
        agents_background = ep.get("agents_background", {})

        for i, (agent_name, background) in enumerate(agents_background.items()):
            if i < len(agent_ids):
                agent_id = agent_ids[i]
                if agent_id not in agents_dict:
                    agents_dict[agent_id] = parse_agent_background(
                        agent_name, background, agent_id
                    )

    print(f"  Extracted {len(agents_dict)} unique agents.")
    print(f"  Extracted {len(envs_dict)} unique environments.")

    # Check how many environments have goals
    envs_with_goals = sum(
        1 for e in envs_dict.values()
        if any(g.strip() for g in e.get("agent_goals", []) if g)
    )
    print(f"  Environments with goals: {envs_with_goals}/{len(envs_dict)}")

    # Step 5: Save profiles
    print("\nStep 5: Saving profiles...")

    for model_name, records in [
        ("AgentProfile", list(agents_dict.values())),
        ("EnvironmentProfile", list(envs_dict.values())),
    ]:
        model_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        saved = 0
        for record in records:
            pk = record.get("pk", "")
            if not pk:
                continue
            file_path = os.path.join(model_dir, f"{pk}.json")
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2, ensure_ascii=False)
            saved += 1

        print(f"  {model_name}: {saved} records saved.")

    # Step 6: Build task splits
    print("\nStep 6: Building task splits...")

    all_tasks, _ = build_task_splits(episodes)
    print(f"  Total unique task combos: {len(all_tasks)}")

    # Save testing_tasks_450.json (all tasks, limited to 450)
    testing_tasks = all_tasks[:450]
    testing_file = os.path.join(local_data_dir, "testing_tasks_450.json")
    with open(testing_file, "w") as f:
        json.dump(testing_tasks, f, indent=2)
    print(f"  Saved {len(testing_tasks)} testing tasks to {testing_file}")

    # Build hard tasks from benchmark_agents.json
    if combo_content:
        try:
            combos = json.loads(combo_content)
            hard_tasks = []
            seen = set()

            for combo in combos:
                env_id = combo.get("env_id", "")
                agent_ids = combo.get("agent_ids", [])
                if not env_id or len(agent_ids) < 2:
                    continue

                combo_key = (env_id, tuple(agent_ids[:2]))
                if combo_key not in seen:
                    seen.add(combo_key)
                    # Determine source
                    codename = ""
                    if env_id in envs_dict:
                        codename = envs_dict[env_id].get("codename", "")

                    source = "hard"
                    if "craigslist" in codename.lower():
                        source = "competitive"

                    hard_tasks.append({
                        "env_id": env_id,
                        "agent_ids": agent_ids[:2],
                        "source": source,
                        "codename": codename,
                    })

            hard_file = os.path.join(local_data_dir, "hard_tasks_90.json")
            with open(hard_file, "w") as f:
                json.dump(hard_tasks, f, indent=2)
            print(f"  Saved {len(hard_tasks)} hard tasks to {hard_file}")

        except json.JSONDecodeError:
            print("  WARNING: Failed to parse benchmark_agents.json")

    # Step 7: Save EnvAgentComboStorage and EnvironmentList
    if combo_content:
        print("\nStep 7: Saving combo storage...")
        combo_dir = os.path.join(output_dir, "EnvAgentComboStorage")
        os.makedirs(combo_dir, exist_ok=True)

        try:
            combos = json.loads(combo_content)
            for combo in combos:
                pk = combo.get("pk", combo.get("env_id", ""))
                if pk:
                    file_path = os.path.join(combo_dir, f"{pk}.json")
                    with open(file_path, "w") as f:
                        json.dump(combo, f, indent=2)
            print(f"  Saved {len(combos)} combo records.")
        except json.JSONDecodeError:
            print("  WARNING: Failed to save combo storage.")

    print("\n" + "=" * 60)
    print("Data setup complete!")
    print("=" * 60)
    print(f"\nData directory: {output_dir}")
    print(f"Task splits: {local_data_dir}")
    print(f"\nYou can now run: python run_sotopia_battle.py -c config.json")


if __name__ == "__main__":
    main()
