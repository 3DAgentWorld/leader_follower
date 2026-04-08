#!/usr/bin/env python
# encoding: utf-8
"""
Avalon Battle Game Runner.
Supports launching games via JSON configuration files.

Agent types:
- direct: Direct response generation (1 API call, fastest)
- react: ReAct framework (think-act loop, 2 API calls)
- recon: Relation Consistency framework (cross-player relation analysis, 3 API calls)
- lasi: LASI framework (landscape analysis - strategy - implementation, 4 API calls)
- refiner+<type>: Wraps any agent type with a trained Refiner model for persuasive refinement
  e.g., refiner+react, refiner+direct, refiner+lasi

Usage:
    python run_avalon_battle.py -c config.json
"""
import argparse
import json
import random
import os
import sys

from colorama import Fore, Style

# Add project path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.extractor.llm_extractor.chatgpt_extractor import ChatGPTBasedExtractor
from src.games.avalon.avalon import Avalon
from src.agents import DirectAgent, ReActAgent, ReConAgent, LASIAgent, RefinerWrapper
from prompt.avalon_prompts import (
    summary_prompt, plan_prompt, response_prompt, system_prompt,
    action_prompt, suggestion_prompt, update_prompt, analysis_prompt,
    strategy_prompt, candidate_actions, init_strategies, role_introduction, role_target
)
from src.games.avalon.extract_demos import (
    number_extract_prompt, player_extractor_demos, vote_extractor_demos,
    quest_extractor_demos, choose_identify_extractor_demos, select_merlin_extractor_demos,
    bool_extract_prompt, quest_extract_prompt
)
from src.utils import create_dir, read_json, write_json, print_text_animated


# Default role configuration
DEFAULT_ROLES = ["Merlin", "Percival", "Loyal Servant", "Loyal Servant", "Morgana", "Assassin"]

# Thinking process color (for watch mode)
THINKING_COLOR = Fore.CYAN + Style.DIM


def create_thinking_callback(player_name: str, mode: str):
    """Create a callback function for displaying thinking process."""
    if mode != 'watch':
        return None
    
    def callback(stage: str, content: str):
        """Display thinking process."""
        print_text_animated(
            THINKING_COLOR + 
            f"    💭 [{player_name}] {stage}:\n" +
            f"    {content[:500]}{'...' if len(content) > 500 else ''}\n" +
            Style.RESET_ALL, 
            delay=0.001
        )
    
    return callback


def load_config(config_path: str) -> dict:
    """Load configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


def get_model_config(player_model: dict, default_model: dict) -> dict:
    """Get model config, prioritizing player-specific config over defaults."""
    if player_model is None:
        return default_model.copy()
    
    # Merge configs: player config takes priority
    merged = default_model.copy()
    for key, value in player_model.items():
        if value is not None:
            merged[key] = value
    return merged


def assign_roles(players: list, roles: list) -> list:
    """
    Assign roles to players.
    If a player config specifies a role, use that role.
    Otherwise, randomly assign from available roles.
    """
    available_roles = roles.copy()
    random.shuffle(available_roles)
    
    assigned_players = []
    
    # First pass: handle players with pre-assigned roles
    for player in players:
        if player.get('role') is not None:
            role = player['role']
            if role in available_roles:
                available_roles.remove(role)
                assigned_players.append({**player, 'role': role})
            else:
                raise ValueError(f"Role {role} is not in the available role list or has already been assigned")
        else:
            assigned_players.append(player.copy())
    
    # Second pass: randomly assign roles to remaining players
    for i, player in enumerate(assigned_players):
        if player.get('role') is None:
            if not available_roles:
                raise ValueError("Not enough available roles for all players")
            player['role'] = available_roles.pop(0)
    
    return assigned_players


def get_base_agent_args(player: dict, model_config: dict, log_dir: str, 
                        game_idx: int, output_dir: str, mode: str,
                        enable_intent_identification: bool = False) -> dict:
    """Get base arguments shared by all agent types."""
    name = player['name']
    role = player['role']
    
    # Load historical experience
    if game_idx == 0:
        role_strategy = init_strategies.get(role, "Play strategically.")
        other_strategy = "None"
        suggestion = "None"
    else:
        load_file = os.path.join(output_dir.format(game_idx - 1), f"{role}_reflection.json")
        if os.path.exists(load_file):
            experience = read_json(load_file)
            role_strategy = experience.get("strategy", init_strategies.get(role, "Play strategically."))
            other_strategy = experience.get("other_strategy", "None")
            suggestion = experience.get("suggestion", "None")
        else:
            role_strategy = init_strategies.get(role, "Play strategically.")
            other_strategy = "None"
            suggestion = "None"
    
    role_system_prompt = system_prompt.format(
        name=name, role=role, strategy=role_strategy,
        suggestion=suggestion, other_strategy=other_strategy
    )
    
    return {
        "name": name,
        "role": role,
        "role_intro": role_introduction.get(role.lower(), ""),
        "game_goal": role_target.get(role, "Win the game."),
        "strategy": role_strategy,
        "system_prompt": role_system_prompt,
        "model": model_config['model_name'],
        "temperature": model_config.get('temperature', 0.3),
        "api_key": model_config['api_key'],
        "api_base": model_config.get('api_base'),
        "output_dir": log_dir,
        "thinking_callback": create_thinking_callback(name, mode),
        "enable_intent_identification": enable_intent_identification,
        "extra_body": model_config.get('extra_body'),
        # Used for LASI reflection
        "suggestion": suggestion,
        "other_strategy": other_strategy,
    }


def create_direct_agent_args(player: dict, model_config: dict, log_dir: str,
                              game_idx: int, output_dir: str, mode: str,
                              enable_intent_identification: bool = False) -> tuple:
    """Create Direct Agent arguments."""
    base_args = get_base_agent_args(player, model_config, log_dir, game_idx, output_dir, mode, enable_intent_identification)
    
    return (
        DirectAgent,
        {
            **base_args,
            "response_prompt": response_prompt,
        }
    )


def create_react_agent_args(player: dict, model_config: dict, log_dir: str,
                             game_idx: int, output_dir: str, mode: str,
                             enable_intent_identification: bool = False) -> tuple:
    """Create ReAct Agent arguments."""
    base_args = get_base_agent_args(player, model_config, log_dir, game_idx, output_dir, mode, enable_intent_identification)
    
    return (
        ReActAgent,
        {
            **base_args,
            "response_prompt": response_prompt,
        }
    )


def create_recon_agent_args(player: dict, model_config: dict, log_dir: str,
                             game_idx: int, output_dir: str, mode: str,
                             enable_intent_identification: bool = False) -> tuple:
    """Create ReCon Agent arguments."""
    base_args = get_base_agent_args(player, model_config, log_dir, game_idx, output_dir, mode, enable_intent_identification)
    
    return (
        ReConAgent,
        {
            **base_args,
            "response_prompt": response_prompt,
        }
    )


def create_lasi_agent_args(player: dict, model_config: dict, log_dir: str,
                            game_idx: int, output_dir: str, mode: str,
                            enable_intent_identification: bool = False) -> tuple:
    """Create LASI Agent arguments."""
    base_args = get_base_agent_args(player, model_config, log_dir, game_idx, output_dir, mode, enable_intent_identification)
    
    return (
        LASIAgent,
        {
            **base_args,
            "analysis_prompt": analysis_prompt,
            "plan_prompt": plan_prompt,
            "action_prompt": action_prompt,
            "response_prompt": response_prompt,
            "suggestion_prompt": suggestion_prompt,
            "strategy_prompt": strategy_prompt,
            "update_prompt": update_prompt,
            "candidate_actions": candidate_actions,
        }
    )


# Agent type to creator function mapping
AGENT_CREATORS = {
    'direct': create_direct_agent_args,
    'react': create_react_agent_args,
    'recon': create_recon_agent_args,
    'lasi': create_lasi_agent_args,
}


def create_agent_with_refiner(base_agent_tuple: tuple, refiner_config: dict) -> tuple:
    """
    Wrap a base agent with the RefinerWrapper.

    Args:
        base_agent_tuple: (AgentClass, agent_kwargs) from a creator function.
        refiner_config: Configuration dict with keys:
            - model_path: Path to the Refiner base model (e.g., Qwen2.5-7B-Instruct)
            - lora_path: Path to the LoRA adapter checkpoint (optional)
            - temperature: Refiner generation temperature (default: 0.7)

    Returns:
        A tuple that, when instantiated, creates a RefinerWrapper around the base agent.
    """
    base_cls, base_kwargs = base_agent_tuple

    class _RefinerAgentFactory:
        """Factory class to create RefinerWrapper with lazy agent instantiation."""

        @classmethod
        def init_instance(cls, **kwargs):
            # Create the base agent first
            base_agent = base_cls.init_instance(**base_kwargs)
            # Wrap it with RefinerWrapper
            return RefinerWrapper(
                wrapped_agent=base_agent,
                refiner_model_path=refiner_config.get('model_path', ''),
                refiner_lora_path=refiner_config.get('lora_path'),
                refiner_temperature=refiner_config.get('temperature', 0.7),
            )

    return (_RefinerAgentFactory, {})


def run_game(config: dict, game_idx: int):
    """Run a single game."""
    game_config = config['game']
    default_model = config['default_model']
    players_config = config['players']
    roles = config.get('roles', DEFAULT_ROLES)
    extractor_config = config.get('extractors', {})
    refiner_config = config.get('refiner', None)
    
    # Build output directory
    output_dir = os.path.join(
        game_config.get('output_dir', 'logs/avalon/battle'),
        f"{game_config.get('exp_name', 'battle')}-{game_config.get('camp', 'good')}-game_{{}}"
    )
    game_output_dir = output_dir.format(game_idx)
    create_dir(game_output_dir)
    
    # Assign roles
    assigned_players = assign_roles(players_config, roles)
    
    # Filter by camp (if specified)
    camp = game_config.get('camp')
    if camp == "good":
        camp_roles = ["Merlin", "Percival", "Loyal Servant"]
    elif camp == "evil":
        camp_roles = ["Morgana", "Assassin"]
    else:
        camp_roles = None  # No filtering
    
    # Create game instance
    player_nums = game_config.get('player_nums', 6)
    language = game_config.get('language', 'english')
    mode = game_config.get('mode', 'watch')
    ai_model = default_model.get('model_name', 'gpt-3.5-turbo-16k')
    enable_intent_identification = game_config.get('enable_intent_identification', False)
    
    game = Avalon(player_nums, language, mode, ai_model, game_output_dir,
                  enable_intent_identification=enable_intent_identification)
    
    # Create player arguments
    player_args = []
    player_mapping = {}
    
    for i, player in enumerate(assigned_players):
        log_dir = os.path.join(game_output_dir, player['name'])
        create_dir(log_dir)
        
        model_config = get_model_config(player.get('model'), default_model)
        agent_type = player.get('agent_type', 'direct').lower()
        role = player['role']
        
        player_mapping[player['name']] = role
        
        # Determine which agent type to use based on camp configuration
        if camp_roles is not None:
            if role in camp_roles:
                use_type = agent_type  # Target camp uses configured agent_type
            else:
                use_type = 'direct'  # Non-target camp uses simplest direct agent
        else:
            use_type = agent_type
        
        # Check if this is a refiner-wrapped agent type (e.g., "refiner+react")
        use_refiner = False
        if use_type.startswith('refiner+'):
            use_refiner = True
            use_type = use_type[len('refiner+'):]  # Extract base type
        
        # Get agent creator function
        creator = AGENT_CREATORS.get(use_type)
        if creator is None:
            print(f"Warning: Unknown agent_type '{use_type}', falling back to 'direct'")
            creator = AGENT_CREATORS['direct']
        
        args = creator(player, model_config, log_dir, game_idx, output_dir, mode, enable_intent_identification)
        
        # Wrap with Refiner if requested
        if use_refiner and refiner_config:
            args = create_agent_with_refiner(args, refiner_config)
        elif use_refiner and not refiner_config:
            print(f"Warning: agent_type uses 'refiner+' but no 'refiner' config found. Skipping refinement.")
        
        player_args.append(args)
    
    game.add_players(player_args)
    
    # Create Extractor configuration
    ext_model = extractor_config.get('model_name') or default_model.get('model_name', 'gpt-4o')
    ext_api_key = extractor_config.get('api_key') or default_model.get('api_key')
    ext_api_base = extractor_config.get('api_base') or default_model.get('api_base')
    ext_temp = extractor_config.get('temperature', 0)
    ext_extra_body = extractor_config.get('extra_body') or default_model.get('extra_body')
    
    extractor_args = [
        (ChatGPTBasedExtractor, {
            "extractor_name": "player extractor",
            "model_name": ext_model,
            "extract_prompt": number_extract_prompt,
            "system_prompt": "You are a helpful assistant.",
            "temperature": ext_temp,
            "few_shot_demos": player_extractor_demos,
            "output_dir": game_output_dir,
            "api_key": ext_api_key,
            "api_base": ext_api_base,
            "extra_body": ext_extra_body
        }),
        (ChatGPTBasedExtractor, {
            "extractor_name": "vote extractor",
            "model_name": ext_model,
            "extract_prompt": bool_extract_prompt,
            "system_prompt": "You are a helpful assistant.",
            "temperature": ext_temp,
            "few_shot_demos": vote_extractor_demos,
            "output_dir": game_output_dir,
            "api_key": ext_api_key,
            "api_base": ext_api_base,
            "extra_body": ext_extra_body
        }),
        (ChatGPTBasedExtractor, {
            "extractor_name": "quest extractor",
            "model_name": ext_model,
            "extract_prompt": quest_extract_prompt,
            "system_prompt": "You are a helpful assistant.",
            "temperature": ext_temp,
            "few_shot_demos": quest_extractor_demos,
            "output_dir": game_output_dir,
            "api_key": ext_api_key,
            "api_base": ext_api_base,
            "extra_body": ext_extra_body
        }),
        (ChatGPTBasedExtractor, {
            "extractor_name": "identify extractor",
            "model_name": ext_model,
            "extract_prompt": bool_extract_prompt,
            "system_prompt": "You are a helpful assistant.",
            "temperature": ext_temp,
            "few_shot_demos": choose_identify_extractor_demos,
            "output_dir": game_output_dir,
            "api_key": ext_api_key,
            "api_base": ext_api_base,
            "extra_body": ext_extra_body
        }),
        (ChatGPTBasedExtractor, {
            "extractor_name": "merlin extractor",
            "model_name": ext_model,
            "extract_prompt": number_extract_prompt,
            "system_prompt": "You are a helpful assistant.",
            "temperature": ext_temp,
            "few_shot_demos": select_merlin_extractor_demos,
            "output_dir": game_output_dir,
            "api_key": ext_api_key,
            "api_base": ext_api_base,
            "extra_body": ext_extra_body
        })
    ]
    
    game.init_extractor(
        player_extractor=extractor_args[0],
        vote_extractor=extractor_args[1],
        quest_extractor=extractor_args[2],
        choose_identify_extractor=extractor_args[3],
        select_merlin_extractor=extractor_args[4]
    )
    
    # Start game
    game.start()
    
    # Save reflection results (only LASI agents perform actual reflection)
    for player_name, agent in game.players.items():
        agent.reflection(
            player_mapping,
            file_name=os.path.join(game_output_dir, f"{player_mapping.get(player_name)}_reflection.json"),
            winners=game.winners,
            duration=game.game_round
        )
    
    return game


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Avalon Battle Game Runner')
    parser.add_argument(
        '-c', '--config',
        type=str,
        required=True,
        help='Path to config file (JSON format)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate config only, do not run the game'
    )
    return parser.parse_args()


def validate_config(config: dict) -> bool:
    """Validate the configuration file."""
    errors = []
    
    # Check required fields
    if 'game' not in config:
        errors.append("Missing 'game' config section")
    
    if 'default_model' not in config:
        errors.append("Missing 'default_model' config section")
    elif not config['default_model'].get('api_key'):
        errors.append("Missing default_model.api_key (set it in config or via environment variable)")
    
    if 'players' not in config:
        errors.append("Missing 'players' config section")
    else:
        player_nums = config.get('game', {}).get('player_nums', 6)
        if len(config['players']) != player_nums:
            errors.append(f"Player count mismatch: configured {len(config['players'])} players, but game.player_nums is {player_nums}")
        
        # Check agent_type validity
        valid_types = list(AGENT_CREATORS.keys())
        refiner_prefix = 'refiner+'
        for i, player in enumerate(config['players']):
            agent_type = player.get('agent_type', 'direct').lower()
            base_type = agent_type[len(refiner_prefix):] if agent_type.startswith(refiner_prefix) else agent_type
            if base_type not in valid_types:
                errors.append(f"Player {i+1} agent_type '{agent_type}' is invalid. Valid base types: {valid_types}")
    
    # Check refiner config if any player uses refiner
    has_refiner_agent = any(
        p.get('agent_type', '').lower().startswith('refiner+')
        for p in config.get('players', [])
    )
    if has_refiner_agent and 'refiner' not in config:
        errors.append("Some players use 'refiner+' agent type but no 'refiner' config section found")
    
    if errors:
        print("Config validation failed:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    print("Config validation passed!")
    print(f"Supported agent types: {list(AGENT_CREATORS.keys())}")
    print(f"  (Prefix with 'refiner+' to enable Refiner, e.g., 'refiner+react')")
    return True


def main():
    """Main entry point."""
    args = parse_args()
    
    # Load config
    print(f"Loading config: {args.config}")
    config = load_config(args.config)
    
    # Validate config
    if not validate_config(config):
        sys.exit(1)
    
    if args.dry_run:
        print("Dry run mode: config validated, not running the game")
        sys.exit(0)
    
    # Run games
    game_config = config['game']
    start_idx = game_config.get('start_game_idx', 0)
    game_count = game_config.get('game_count', 10)
    
    for game_round in range(start_idx, game_count):
        print(f"\n{'='*50}")
        print(f"Starting game {game_round + 1}/{game_count}")
        print(f"{'='*50}")
        
        try:
            game = run_game(config, game_round)
            print(f"\nGame {game_round} completed!")
            print(f"Winners: {game.winners}")
        except Exception as e:
            print(f"Game {game_round} encountered an error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\nAll games completed!")


if __name__ == '__main__':
    main()
