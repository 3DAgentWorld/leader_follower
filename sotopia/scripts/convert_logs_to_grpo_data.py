#!/usr/bin/env python
# encoding: utf-8
"""
Convert Sotopia game logs to GRPO training data format.

Reads process.json files from game logs and converts them into
training examples for the Refiner model.
"""
import argparse
import glob
import json
import os
import re
import sys
from typing import Dict, List, Optional

from prompt.sotopia_prompts import (
    refine_prompt,
    measurer_system_prompt,
    measurer_user_prompt,
)


def build_refiner_prompt(
    player_name: str,
    player_role: str,
    game_state: str,
    dialog_history: str,
    base_utterance: str,
) -> str:
    """
    Build the Refiner's input prompt (consistent with RefinerWrapper._refine_utterance at inference).
    """
    return refine_prompt.format(
        player_name=player_name,
        player_role=player_role,
        game_state=game_state,
        dialog_history=dialog_history,
        base_utterance=base_utterance,
    )


def build_measurer_prompt(
    name: str,
    character_profile: str,
    scenario: str,
    goal: str,
    relationship: str,
    strategy: str,
    summary: str,
    question: str,
) -> Dict[str, str]:
    """
    Build the Measurer's prompt (consistent with the game's Direct agent prompt).

    Returns:
        Dict with 'system' and 'user' keys.
    """
    sys_msg = measurer_system_prompt.format(
        name=name,
        character_profile=character_profile,
        scenario=scenario,
        goal=goal,
        relationship=relationship,
        strategy=strategy,
    )

    user_msg = measurer_user_prompt.format(
        name=name,
        phase="conversation",
        character_summary=character_profile,
        goal=goal,
        strategy=strategy,
        scenario=scenario,
        relationship=relationship,
        summary=summary,
        question=question,
    )

    return {"system": sys_msg, "user": user_msg}


def process_episode(
    process_data: dict,
    include_intent: bool = False,
) -> List[dict]:
    """
    Convert a single episode's process.json into training examples.

    Args:
        process_data: Parsed process.json data.
        include_intent: Whether to include intent identification data.

    Returns:
        List of training examples.
    """
    examples = []

    scenario = process_data.get("scenario", {})
    agents = process_data.get("agents", [])
    agent_goals = process_data.get("agent_goals", [])
    conversation = process_data.get("conversation", [])

    if not agents or not conversation:
        return examples

    # Build agent info lookup
    agent_info = {}
    for i, agent in enumerate(agents):
        name = f"{agent.get('first_name', '')} {agent.get('last_name', '')}".strip()
        agent_info[name] = {
            "profile": (
                f"{name}, {agent.get('age', 0)}-year-old "
                f"{agent.get('gender', '').lower()} {agent.get('occupation', '')}. "
                f"{agent.get('personality_and_values', '')}"
            ),
            "goal": agent_goals[i] if i < len(agent_goals) else "",
            "secret": agent.get("secret", ""),
        }

    scenario_text = scenario.get("scenario", "")
    relationship = scenario.get("relationship", 0)
    from src.games.sotopia.sotopia import RELATIONSHIP_TYPES
    relationship_str = RELATIONSHIP_TYPES.get(relationship, "strangers")

    # Process each conversation turn
    conv_so_far = []
    for entry_idx, entry in enumerate(conversation):
        speaker = entry.get("speaker", "")
        message = entry.get("message", "")
        action_type = entry.get("action_type", "speak")

        if action_type != "speak" or speaker not in agent_info:
            conv_so_far.append(entry)
            continue

        # Build conversation summary up to this point
        summary_lines = []
        for prev in conv_so_far:
            prev_speaker = prev.get("speaker", "")
            prev_msg = prev.get("message", "")
            prev_action = prev.get("action_type", "speak")
            if prev_action == "leave":
                summary_lines.append(f"[{prev_speaker} left the conversation]")
            elif prev_action == "none":
                summary_lines.append(f"[{prev_speaker} did nothing]")
            else:
                summary_lines.append(f"{prev_speaker}: {prev_msg}")
        summary = "\n".join(summary_lines)

        # Build question
        if entry_idx == 0:
            question = f"The conversation is starting. You are {speaker}. Please begin the interaction naturally."
        else:
            prev_entry = conversation[entry_idx - 1]
            question = (
                f"{prev_entry['speaker']} just said: \"{prev_entry['message']}\"\n"
                f"It's your turn to respond as {speaker}."
            )

        info = agent_info[speaker]

        # Build Refiner prompt
        refiner_input = build_refiner_prompt(
            player_name=speaker,
            player_role=info["profile"],
            game_state=f"Scenario: {scenario_text}",
            dialog_history=f"Goal: {info['goal']}\nRelationship: {relationship_str}\n\n{summary}",
            base_utterance=message,
        )

        # Build Measurer prompt
        measurer_prompts = build_measurer_prompt(
            name=speaker,
            character_profile=info["profile"],
            scenario=scenario_text,
            goal=info["goal"],
            relationship=relationship_str,
            strategy="Engage naturally in the conversation.",
            summary=summary if summary else "The conversation has not started yet.",
            question=question,
        )

        example = {
            "episode_idx": process_data.get("episode_idx", 0),
            "turn_idx": entry_idx,
            "speaker": speaker,
            "utterance": message,
            "refiner_prompt": refiner_input,
            "measurer_system_prompt": measurer_prompts["system"],
            "measurer_user_prompt": measurer_prompts["user"],
        }

        # Add intent identification data if available
        if include_intent and "intent_identification" in entry:
            intent_data = entry["intent_identification"]
            example["desired_responses"] = intent_data.get("desired_responses", [])
            example["undesired_responses"] = intent_data.get("undesired_responses", [])

        examples.append(example)
        conv_so_far.append(entry)

    return examples


def convert_to_grpo_format(examples: List[dict]) -> List[dict]:
    """
    Convert training examples to GRPO format.

    Each GRPO example contains:
    - prompt: The Refiner's input prompt
    - measurer_system_prompt: System prompt for the Measurer
    - measurer_user_prompt: User prompt for the Measurer
    - desired_responses: List of desired follower responses (if available)
    - undesired_responses: List of undesired follower responses (if available)
    """
    grpo_examples = []

    for ex in examples:
        grpo_ex = {
            "prompt": ex["refiner_prompt"],
            "measurer_system_prompt": ex["measurer_system_prompt"],
            "measurer_user_prompt": ex["measurer_user_prompt"],
        }

        if "desired_responses" in ex:
            grpo_ex["desired_responses"] = ex["desired_responses"]
        if "undesired_responses" in ex:
            grpo_ex["undesired_responses"] = ex["undesired_responses"]

        grpo_examples.append(grpo_ex)

    return grpo_examples


def main():
    parser = argparse.ArgumentParser(
        description="Convert Sotopia game logs to GRPO training data."
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        required=True,
        help="Directory containing game log subdirectories.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL file path.",
    )
    parser.add_argument(
        "--include_intent",
        action="store_true",
        help="Include intent identification data (desired/undesired responses).",
    )
    args = parser.parse_args()

    # Find all process.json files
    pattern = os.path.join(args.log_dir, "*/process.json")
    process_files = sorted(glob.glob(pattern))

    if not process_files:
        print(f"No process.json files found in {args.log_dir}")
        sys.exit(1)

    print(f"Found {len(process_files)} episode logs.")

    all_examples = []
    for pf in process_files:
        try:
            with open(pf, "r") as f:
                data = json.load(f)

            # Extract episode index from directory name
            dir_name = os.path.basename(os.path.dirname(pf))
            match = re.search(r"episode_(\d+)", dir_name)
            if match:
                data["episode_idx"] = int(match.group(1))

            examples = process_episode(data, include_intent=args.include_intent)
            all_examples.extend(examples)

        except Exception as e:
            print(f"Error processing {pf}: {e}")

    # Convert to GRPO format
    grpo_data = convert_to_grpo_format(all_examples)

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        for ex in grpo_data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Saved {len(grpo_data)} training examples to {args.output}")

    # Print statistics
    with_intent = sum(1 for ex in grpo_data if "desired_responses" in ex)
    print(f"  Examples with intent data: {with_intent}")
    print(f"  Examples without intent data: {len(grpo_data) - with_intent}")


if __name__ == "__main__":
    main()
