#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert Werewolf game logs to verl GRPO training data format.

Usage:
    python scripts/convert_logs_to_grpo_data.py --log_dir logs/werewolf/battle --output output.jsonl

Output format:
    Each line is a JSON object containing:
    - prompt: Refiner's input prompt (messages format, consistent with inference)
    - extra_info: Additional info including follower_prompt_template, intent info, etc.
    - reward_model: Reward model configuration
    - data_source: Data source identifier
    - ability: Ability tag
"""

import os
import sys
import json
import argparse
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from prompt.werewolf_prompts import (
    system_prompt as werewolf_system_prompt_template,
    refine_prompt as werewolf_refine_prompt,
    measurer_system_prompt as werewolf_measurer_system_prompt,
    measurer_user_prompt as werewolf_measurer_user_prompt,
    role_introduction,
    role_target,
    init_strategies
)


def extract_player_info(player_key: str) -> Tuple[str, str]:
    """
    Extract player name and role from a log key.
    
    Args:
        player_key: Key in format "player 2(Werewolf)".
    
    Returns:
        Tuple[player_name, role]: e.g. ("player 2", "Werewolf").
    """
    match = re.match(r"(player \d+)\(([^)]+)\)", player_key)
    if match:
        return match.group(1), match.group(2)
    return None, None


def build_system_prompt(name: str, role: str, strategy: str = None,
                        suggestion: str = "", other_strategy: str = "") -> str:
    """
    Build the game system prompt for a player.
    
    Args:
        name: Player name.
        role: Player role.
        strategy: Player strategy.
        suggestion: Suggestion from previous game experience.
        other_strategy: Strategy for other roles.
    
    Returns:
        Formatted system prompt.
    """
    if strategy is None:
        strategy = init_strategies.get(role, "Play strategically to help your team win.")
    
    return werewolf_system_prompt_template.format(
        name=name,
        role=role,
        strategy=strategy,
        suggestion=suggestion,
        other_strategy=other_strategy
    )


def build_role_intro_prompt(role: str) -> str:
    """Get the role introduction text."""
    return role_introduction.get(role.lower(), "")


def build_game_goal(role: str) -> str:
    """Get the game goal for a role."""
    return role_target.get(role, "Win the game for your team.")


def parse_process_json(process_file: str) -> Dict[str, List[Dict]]:
    """
    Parse a process.json file.
    
    Args:
        process_file: Path to the process.json file.
    
    Returns:
        Parsed game log data, organized by rounds.
    """
    with open(process_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_player_mapping(process_data: Dict) -> Dict[str, str]:
    """
    Extract player name to role mapping from the log data.
    
    Returns:
        Dict[player_name, role]: e.g. {"player 1": "Villager", ...}
    """
    player_mapping = {}
    
    for round_key, events in process_data.items():
        for event in events:
            for key in event.keys():
                if key.startswith("player ") and "(" in key:
                    player_name, role = extract_player_info(key)
                    if player_name and role:
                        player_mapping[player_name] = role
    
    return player_mapping


def build_dialogue_summary(process_data: Dict, current_round: str,
                           current_event_idx: int) -> str:
    """
    Build a dialogue history summary (text format) up to the current event.
    
    Format: "host: xxx" and "player X: xxx"
    
    Args:
        process_data: Full game log data.
        current_round: Current round key.
        current_event_idx: Index of the current event in the round.
    
    Returns:
        Dialogue history summary text.
    """
    summary_lines = []
    rounds = list(process_data.keys())
    current_round_idx = rounds.index(current_round)
    
    # Iterate over all previous rounds
    for i in range(current_round_idx + 1):
        round_key = rounds[i]
        events = process_data[round_key]
        
        # If current round, only process up to the current event
        max_idx = current_event_idx if round_key == current_round else len(events)
        
        for j in range(max_idx):
            event = events[j]
            
            # Extract Host messages
            if "Host" in event:
                summary_lines.append(f"host: {event['Host']}")
            
            # Extract player messages
            for key, value in event.items():
                if key.startswith("player ") and "(" in key:
                    player_name, role = extract_player_info(key)
                    if player_name and value:  # Skip empty responses
                        summary_lines.append(f"{player_name}: {value}")
    
    return "\n".join(summary_lines)


# Leader response placeholder
LEADER_RESPONSE_PLACEHOLDER = "{{LEADER_RESPONSE}}"


def build_refiner_prompt(
    player_name: str,
    player_role: str,
    game_rules: str,
    game_state: str,
    dialog_history: str,
    base_utterance: str
) -> List[Dict[str, str]]:
    """
    Build the Refiner's input prompt (consistent with RefinerWrapper._refine_utterance at inference).
    
    The Refiner takes u_base and context as input, and outputs a refined utterance u_t.
    This prompt format must match the inference-time format exactly.
    
    Args:
        player_name: Player name.
        player_role: Player role.
        game_rules: Game rules text (the system_prompt of the game).
        game_state: Current game state description.
        dialog_history: Dialogue history text.
        base_utterance: The base utterance u_base generated by the backend LLM.
    
    Returns:
        Messages list in the Refiner's prompt format.
    """
    refine_input = werewolf_refine_prompt.format(
        game_rules=game_rules,
        player_name=player_name,
        player_role=player_role,
        game_state=game_state,
        dialog_history=dialog_history,
        base_utterance=base_utterance
    )

    messages = [
        {"role": "system", "content": "You are a communication expert specializing in persuasive dialogue refinement for social deduction games."},
        {"role": "user", "content": refine_input}
    ]

    return messages


def build_measurer_prompt_template(
    process_data: Dict,
    round_key: str,
    current_event_idx: int,
    leader_name: str,
    follower_name: str,
    follower_role: str,
    follower_strategy: str,
    current_host_instruction: str = "",
    next_host_instruction: str = ""
) -> List[Dict[str, str]]:
    """
    Build the Measurer's prompt template for computing follower response log probabilities.
    
    The Measurer simulates a real game player's perspective, so its prompt format
    must be consistent with the actual game agent's prompt:
    - system: system_prompt (game rules + role + strategy, same as normal gameplay)
    - user: response_prompt_without_action (same as normal gameplay)
    
    The leader's utterance is represented by {{LEADER_RESPONSE}} placeholder,
    which is substituted during reward computation.
    
    Args:
        process_data: Full game log data.
        round_key: Current round key.
        current_event_idx: Index of the leader's event.
        leader_name: Leader player name.
        follower_name: Follower player name.
        follower_role: Follower's role.
        follower_strategy: Follower's playing strategy.
        current_host_instruction: Host instruction for the leader.
        next_host_instruction: Host instruction for the follower.
    
    Returns:
        Measurer's prompt template (messages format) with {{LEADER_RESPONSE}} placeholder.
    """
    # 1. Build dialogue history (summary) up to the leader's utterance
    summary = build_dialogue_summary(
        process_data, round_key, current_event_idx
    )
    
    # 2. Append the leader's Host instruction and leader utterance placeholder
    if current_host_instruction:
        summary += f"\nhost: {current_host_instruction}"
    summary += f"\n{leader_name}: {LEADER_RESPONSE_PLACEHOLDER}"
    
    # 3. Build system prompt (same as normal gameplay agent)
    system_content = werewolf_measurer_system_prompt.format(
        name=follower_name,
        role=follower_role,
        strategy=follower_strategy,
        suggestion="",
        other_strategy=""
    )
    
    # 4. Build user prompt (same as normal gameplay agent: response_prompt_without_action)
    follower_introduction = role_introduction.get(follower_role.lower(), "")
    question = next_host_instruction if next_host_instruction else "Please share your thoughts on the current situation."
    
    user_content = werewolf_measurer_user_prompt.format(
        name=follower_name,
        phase=f"discussion, Round: {round_key}",
        role=follower_role,
        introduction=follower_introduction,
        strategy=follower_strategy,
        summary=summary,
        plan="None",
        question=question
    )
    
    # 5. Build complete messages
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]
    
    return messages


def convert_discuss_event_to_training_sample(
    event: Dict,
    process_data: Dict,
    round_key: str,
    event_idx: int,
    player_mapping: Dict[str, str],
    game_dir: str
) -> Optional[Dict[str, Any]]:
    """
    Convert a discussion event into a training sample.
    
    The training sample contains:
    - prompt: Refiner's input (consistent with inference-time RefinerWrapper)
    - extra_info: Measurer's prompt template (consistent with reward_server)
    
    Args:
        event: The current event.
        process_data: Full game log data.
        round_key: Current round key.
        event_idx: Event index.
        player_mapping: Player name to role mapping.
        game_dir: Game log directory.
    
    Returns:
        Training sample dict or None.
    """
    # Find the current speaker (leader)
    speaker_key = None
    speaker_response = None
    
    for key, value in event.items():
        if key.startswith("player ") and "(" in key:
            speaker_key = key
            speaker_response = value
            break
    
    if not speaker_key or not speaker_response:
        return None
    
    # Skip empty responses
    if not speaker_response.strip():
        return None
    
    player_name, role = extract_player_info(speaker_key)
    if not player_name or not role:
        return None
    
    # Build game rules (the system prompt serves as game rules context)
    game_rules = build_system_prompt(
        name=player_name,
        role=role,
        strategy=init_strategies.get(role)
    )
    
    # Build dialogue history summary
    dialog_history = build_dialogue_summary(
        process_data, round_key, event_idx
    )
    
    # Add current Host instruction to dialog history
    host_instruction = event.get("Host", "")
    if host_instruction:
        dialog_history += f"\nhost: {host_instruction}"
    
    # Build game state
    game_state = f"Phase: discussion, Round: {round_key}"
    
    # The speaker's actual response serves as the base utterance (u_base)
    # During self-play data collection, u_base = u_t (no Refiner applied)
    base_utterance = speaker_response
    
    # ===== Build Refiner's prompt (consistent with inference) =====
    prompt = build_refiner_prompt(
        player_name=player_name,
        player_role=role,
        game_rules=game_rules,
        game_state=game_state,
        dialog_history=dialog_history,
        base_utterance=base_utterance
    )
    
    # Find the next speaker (follower)
    round_events = process_data[round_key]
    
    next_speaker_key = None
    next_speaker_name = None
    next_speaker_role = None
    next_response = None
    next_event_idx = None
    
    for i in range(event_idx + 1, len(round_events)):
        next_event = round_events[i]
        for key, value in next_event.items():
            if key.startswith("player ") and "(" in key:
                _next_speaker_name, _next_speaker_role = extract_player_info(key)
                if _next_speaker_name and value and value.strip():
                    next_speaker_key = key
                    next_speaker_name = _next_speaker_name
                    next_speaker_role = _next_speaker_role
                    next_response = value
                    next_event_idx = i
                    break
        if next_speaker_key:
            break
    
    # If no next player found, skip this sample (cannot compute reward)
    if not next_speaker_key:
        return None
    
    # ===== Build Measurer's prompt template for reward computation =====
    # This is the follower's prompt used by the Measurer to compute P_F(response | context)
    # The Measurer uses the same prompt format as a real game player
    
    # 1. Get follower's strategy
    follower_strategy = init_strategies.get(next_speaker_role, "Play strategically to help your team win.")
    
    # 2. Find Host instruction in next_event
    next_event = round_events[next_event_idx]
    next_host_instruction = next_event.get("Host", "")
    
    # 3. Build Measurer's prompt template (with placeholder)
    follower_prompt_template = build_measurer_prompt_template(
        process_data=process_data,
        round_key=round_key,
        current_event_idx=event_idx,
        leader_name=player_name,
        follower_name=next_speaker_name,
        follower_role=next_speaker_role,
        follower_strategy=follower_strategy,
        current_host_instruction=host_instruction,
        next_host_instruction=next_host_instruction
    )
    
    # Build extra_info
    extra_info = {
        "follower_prompt_template": follower_prompt_template,
        "base_utterance": base_utterance,
    }
    
    # Add intent info to extra_info if available
    if "intent_identification" in event:
        extra_info["intent_identification"] = event["intent_identification"]
    
    # reward_model contains style and ground_truth
    reward_model = {
        "style": "rule",
        "ground_truth": next_response,  # Follower's actual response
    }
    
    return {
        "prompt": prompt,
        "extra_info": extra_info,
        "reward_model": reward_model,
        "data_source": "custom_api",
        "ability": "strategic_dialogue"
    }


def convert_game_logs_to_grpo_data(
    log_dir: str,
    output_file: str,
    only_discuss: bool = True,
    include_intent: bool = True
):
    """
    Convert game logs to GRPO training data.
    
    Args:
        log_dir: Log directory (containing one or more game directories).
        output_file: Output file path.
        only_discuss: Whether to only process discussion phase data.
        include_intent: Whether to include intent_identification info.
    """
    samples = []
    
    # Find all game directories
    log_path = Path(log_dir)
    game_dirs = []
    
    # Support two directory structures:
    # 1. log_dir directly contains process.json
    # 2. log_dir has multiple game subdirectories
    if (log_path / "process.json").exists():
        game_dirs = [log_path]
    else:
        for item in log_path.iterdir():
            if item.is_dir() and (item / "process.json").exists():
                game_dirs.append(item)
    
    print(f"Found {len(game_dirs)} game directories")
    
    for game_dir in game_dirs:
        process_file = game_dir / "process.json"
        print(f"Processing: {process_file}")
        
        try:
            process_data = parse_process_json(str(process_file))
        except Exception as e:
            print(f"  Warning: Failed to parse {process_file}: {e}")
            continue
        
        # Extract player mapping
        player_mapping = extract_player_mapping(process_data)
        print(f"  Player mapping: {player_mapping}")
        
        # Iterate over all rounds and events
        for round_key, events in process_data.items():
            for event_idx, event in enumerate(events):
                # Check if this is a discussion event (Day phase discussion)
                host_msg = event.get("Host", "").lower()
                is_discuss_event = "discuss" in host_msg or \
                                   "speak" in host_msg or \
                                   "share your thoughts" in host_msg

                if only_discuss and not is_discuss_event:
                    continue
                
                # Check if intent_identification is present (if required)
                if include_intent and "intent_identification" not in event:
                    pass
                
                # Convert to training sample
                sample = convert_discuss_event_to_training_sample(
                    event=event,
                    process_data=process_data,
                    round_key=round_key,
                    event_idx=event_idx,
                    player_mapping=player_mapping,
                    game_dir=str(game_dir)
                )
                
                if sample:
                    samples.append(sample)
    
    print(f"Generated {len(samples)} training samples")
    
    # Write output file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"Output saved to: {output_file}")
    
    return samples


def convert_single_game(
    game_dir: str,
    output_file: str = None,
    only_discuss: bool = True
) -> List[Dict]:
    """
    Convert a single game's logs.
    
    Args:
        game_dir: Single game's log directory.
        output_file: Output file path (optional).
        only_discuss: Whether to only process discussion phase.
    
    Returns:
        List of training samples.
    """
    game_path = Path(game_dir)
    process_file = game_path / "process.json"
    
    if not process_file.exists():
        raise FileNotFoundError(f"process.json not found: {process_file}")
    
    process_data = parse_process_json(str(process_file))
    player_mapping = extract_player_mapping(process_data)
    
    samples = []
    
    for round_key, events in process_data.items():
        for event_idx, event in enumerate(events):
            host_msg = event.get("Host", "").lower()
            is_discuss_event = "discuss" in host_msg or \
                               "speak" in host_msg or \
                               "share your thoughts" in host_msg
            
            if only_discuss and not is_discuss_event:
                continue
            
            sample = convert_discuss_event_to_training_sample(
                event=event,
                process_data=process_data,
                round_key=round_key,
                event_idx=event_idx,
                player_mapping=player_mapping,
                game_dir=str(game_path)
            )
            
            if sample:
                samples.append(sample)
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Convert Werewolf game logs to verl GRPO training data format"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        required=True,
        help="Game log directory (single game dir or parent dir containing multiple games)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="grpo_training_data.jsonl",
        help="Output file path (default: grpo_training_data.jsonl)"
    )
    parser.add_argument(
        "--only_discuss",
        action="store_true",
        default=True,
        help="Only process discussion phase data (default: True)"
    )
    parser.add_argument(
        "--all_phases",
        action="store_true",
        help="Process all phases (overrides --only_discuss)"
    )
    parser.add_argument(
        "--include_intent",
        action="store_true",
        default=True,
        help="Include intent_identification info (default: True)"
    )
    
    args = parser.parse_args()
    
    only_discuss = not args.all_phases
    
    convert_game_logs_to_grpo_data(
        log_dir=args.log_dir,
        output_file=args.output,
        only_discuss=only_discuss,
        include_intent=args.include_intent
    )


if __name__ == "__main__":
    main()
