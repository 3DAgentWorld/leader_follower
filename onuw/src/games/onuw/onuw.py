#!/usr/bin/env python
# encoding: utf-8
"""
One Night Ultimate Werewolf (ONUW) game engine.
Implements the ONUW game with 5 players and 7 roles:
1 Werewolf, 2 Villagers, 1 Seer, 1 Robber, 1 Troublemaker, 1 Insomniac.
Two roles remain in the center pool.
"""

import copy
import json
import os.path
import random
import re
from typing import Dict, List, Tuple, Type

from colorama import Fore

from ..abs_game import Game
from ...agents.abs_agent import Agent
from ...extractor.abs_extractor import Extractor
from src.utils import print_text_animated, write_json, create_dir


# Color mapping for 5 players
COLOR_5 = {
    "player 1": Fore.BLUE,
    "player 2": Fore.GREEN,
    "player 3": Fore.YELLOW,
    "player 4": Fore.RED,
    "player 5": Fore.LIGHTGREEN_EX,
}


class ONUW(Game):
    def __init__(self, player_nums: int, language: str, mode: str, ai_model,
                 output_dir, **kwargs):
        """
        Initialize the ONUW game.

        Args:
            player_nums: Number of players (must be 5).
            language: Language setting (english/chinese).
            mode: Game mode (watch: all AI, play: one human).
            ai_model: AI model name.
            output_dir: Output directory for logs.
        """
        config_file = kwargs.get("config_file")
        if not config_file:
            config_file = os.path.join(os.path.dirname(__file__), 'config.json')
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data: dict = json.load(f)

        support_nums = config_data.get("player_nums", [])
        assert player_nums in support_nums, \
            f"Game with {player_nums} players is not supported. Supported: {support_nums}"

        game_config = config_data.get(language, {})
        if not game_config:
            raise NotImplementedError(f'{language} is not supported.')

        host_instruction: dict = game_config.get('host_instruction')
        response_rule: dict = game_config.get('response_rule')
        role_introduce = game_config.get('role_introduce')
        config_for_num = game_config.get(f'config_{player_nums}', {})
        game_introduce = config_for_num.get('game_introduce', '')
        role_pool = config_for_num.get('role_pool', [])
        werewolf_team = config_for_num.get('werewolf_team', [])
        village_team = config_for_num.get('village_team', [])
        role_mapping = game_config.get('role_mapping', {})
        create_dir(output_dir)

        # Instruction templates
        self.language = language
        self.host_instruction = host_instruction
        self.role_introduce = role_introduce
        self.game_introduce = game_introduce

        # Player management
        self.player_nums = player_nums
        self.player_list = []
        self.players: Dict[str, Agent] = {}
        self.player_mapping = {}  # name -> initial role

        # Role configuration
        self.role_pool = role_pool  # All 7 roles
        self.center_pool = []  # 2 roles in center
        self.werewolf_team = werewolf_team
        self.village_team = village_team
        self.role_mapping = role_mapping

        # Ground truth roles (may change during night)
        self.roles_ground_truth = {}  # name -> current actual role

        # Game state
        self.mode = mode
        self.ai_model = ai_model
        self.process_list = []
        self.output_dir = output_dir
        self.max_discuss_round = kwargs.get('max_discuss_round', 3)
        self.winners = []

        self.response_rule = response_rule

        # AI extractors
        self.player_extractor = None
        self.vote_extractor = None

        # Whether to enable intent identification
        self.enable_intent_identification = kwargs.get('enable_intent_identification', False)

    def init_extractor(self, player_extractor: Tuple[Type[Extractor], dict],
                       vote_extractor: Tuple[Type[Extractor], dict],
                       **kwargs):
        """Initialize extractors for parsing LLM outputs."""
        self.player_extractor = player_extractor[0].init_instance(**player_extractor[1])
        self.vote_extractor = vote_extractor[0].init_instance(**vote_extractor[1])

    def add_players(self, players: List[Tuple[Type[Agent], dict]]):
        """Add players to the game and assign roles."""
        assert self.player_nums == len(players), \
            f"Required {self.player_nums} players, got {len(players)}."

        # Determine role assignment
        available_roles = self.role_pool.copy()
        random.shuffle(available_roles)

        # Ensure exactly one Werewolf is among the players
        # First, check if any player has a pre-assigned role
        pre_assigned = {}
        unassigned_indices = []
        for idx, (agent_type, player_params) in enumerate(players):
            if player_params.get('role') is not None:
                pre_assigned[idx] = player_params['role']
                available_roles.remove(player_params['role'])
            else:
                unassigned_indices.append(idx)

        # If no Werewolf is pre-assigned, ensure one is assigned to a player
        has_werewolf = any(r == "Werewolf" for r in pre_assigned.values())
        if not has_werewolf and unassigned_indices:
            # Find a Werewolf in available roles and assign it
            werewolf_idx = None
            for i, r in enumerate(available_roles):
                if r == "Werewolf":
                    werewolf_idx = i
                    break
            if werewolf_idx is not None:
                # Assign Werewolf to a random unassigned player
                target_player = random.choice(unassigned_indices)
                pre_assigned[target_player] = available_roles.pop(werewolf_idx)
                unassigned_indices.remove(target_player)

        # Assign remaining roles
        random.shuffle(available_roles)
        for idx in unassigned_indices:
            pre_assigned[idx] = available_roles.pop(0)

        # The remaining roles go to center pool
        self.center_pool = available_roles[:2] if len(available_roles) >= 2 else available_roles

        # Create player instances
        idx = 0
        for agent_type, player_params in players:
            if player_params.get('name') is None:
                player_params['name'] = f'player {idx + 1}'
            player_params['role'] = pre_assigned[idx]
            player_i = agent_type.init_instance(**player_params)
            self.player_list.append(player_i)
            self.players[player_i.name] = player_i
            self.player_mapping[player_i.name] = player_i.role
            idx += 1

        # Initialize ground truth (same as initial assignment at start)
        self.roles_ground_truth = self.player_mapping.copy()

    def init_game(self):
        """Initialize game state."""
        self.winners = []
        self.roles_ground_truth = self.player_mapping.copy()

    def _get_players_by_initial_role(self, role: str) -> List[str]:
        """Get player names with a specific initial role."""
        return [name for name, r in self.player_mapping.items() if r == role]

    def _extract_player_number(self, output: str, instruction: str) -> str:
        """Extract a player number from LLM output using the extractor."""
        if self.player_extractor is not None:
            s = self.player_extractor.step(
                f"Question: {instruction}\nAnswer: {output}")
        else:
            s = output
        numbers = re.findall(r'\d+', s)
        valid = [n for n in numbers if 1 <= int(n) <= self.player_nums]
        return valid[0] if valid else None

    # ==================== Night Phase ====================

    def night_phase(self):
        """
        Execute the night phase.
        Night actions occur in order: Werewolf, Seer, Robber, Troublemaker, Insomniac.
        """
        if self.language == 'chinese':
            print_text_animated(
                Fore.WHITE + f"\n{'='*60}\n"
                f"[夜间阶段] 夜幕降临...\n"
                f"{'='*60}\n\n")
        else:
            print_text_animated(
                Fore.WHITE + f"\n{'='*60}\n"
                f"[Night Phase] Night falls...\n"
                f"{'='*60}\n\n")

        self.process_list.append({'Host': self.host_instruction['game_start']})

        # 1. Werewolf
        self._werewolf_night_action()

        # 2. Seer
        self._seer_night_action()

        # 3. Robber
        self._robber_night_action()

        # 4. Troublemaker
        self._troublemaker_night_action()

        # 5. Insomniac
        self._insomniac_night_action()

        if self.language == 'chinese':
            print_text_animated(
                Fore.WHITE + f"\n[夜间阶段] 夜间阶段结束。"
                f"实际角色：{self.roles_ground_truth}\n"
                f"中央池：{self.center_pool}\n\n")
        else:
            print_text_animated(
                Fore.WHITE + f"\n[Night Phase] Night phase complete. "
                f"Ground truth roles: {self.roles_ground_truth}\n"
                f"Center pool: {self.center_pool}\n\n")

    def _werewolf_night_action(self):
        """Werewolf wakes up to check for other Werewolves."""
        phase_tag = "night phase, Werewolf"
        werewolves = self._get_players_by_initial_role("Werewolf")

        instruction = self.host_instruction['werewolf_wake']
        self.process_list.append({'Host': instruction})

        if len(werewolves) == 0:
            return

        if len(werewolves) == 1:
            wolf = werewolves[0]
            info = self.host_instruction['werewolf_info_solo']
            self.players[wolf].receive("Host", f"{phase_tag}|{info}")
            self.players[wolf].set_night_info(
                f"[Night Info - Werewolf] {info}")
            if self.language == 'chinese':
                print_text_animated(
                    Fore.WHITE + f"  [狼人] {wolf}：{info}\n")
            else:
                print_text_animated(
                    Fore.WHITE + f"  [Werewolf] {wolf}: {info}\n")
            self.process_list.append({f'Host (private to {wolf})': info})
        else:
            wolf_names = ', '.join(werewolves)
            info = self.host_instruction['werewolf_info_team'].format(
                werewolves=wolf_names)
            for wolf in werewolves:
                self.players[wolf].receive("Host", f"{phase_tag}|{info}")
                self.players[wolf].set_night_info(
                    f"[Night Info - Werewolf] {info}")
            if self.language == 'chinese':
                print_text_animated(
                    Fore.WHITE + f"  [狼人] {wolf_names}：{info}\n")
            else:
                print_text_animated(
                    Fore.WHITE + f"  [Werewolf] {wolf_names}: {info}\n")
            self.process_list.append({f'Host (private to Werewolves)': info})

    def _seer_night_action(self):
        """Seer checks one player's role or two center pool roles."""
        phase_tag = "night phase, Seer"
        seers = self._get_players_by_initial_role("Seer")

        instruction = self.host_instruction['seer_wake']
        self.process_list.append({'Host': instruction})

        if len(seers) == 0:
            return

        seer = seers[0]
        seer_instruction = (
            f"You are the Seer. Choose one of the following:\n"
            f"1. Check one other player's role. Available players: "
            f"{', '.join([p for p in self.players if p != seer])}\n"
            f"2. Check two roles in the center pool.\n"
            f"State your choice clearly."
        )

        max_retry = 3
        for attempt in range(max_retry):
            output = self.players[seer].step(f"{phase_tag}|{seer_instruction}")
            print_text_animated(
                COLOR_5.get(seer, Fore.WHITE) +
                f"{seer}({self.player_mapping[seer]}):\n\n{output}\n\n")

            self.process_list.append({
                'Host': seer_instruction,
                f"{seer}({self.player_mapping[seer]})": output
            })

            # Parse: check if they want center pool or a specific player
            output_lower = output.lower()
            if "center" in output_lower or "pool" in output_lower:
                # Check two center pool roles
                if len(self.center_pool) >= 2:
                    roles_checked = random.sample(self.center_pool, 2)
                else:
                    roles_checked = self.center_pool.copy()
                result = self.host_instruction['seer_check_center'].format(
                    role1=roles_checked[0] if len(roles_checked) > 0 else "None",
                    role2=roles_checked[1] if len(roles_checked) > 1 else "None")
                self.players[seer].receive("Host", f"{phase_tag}|{result}")
                self.players[seer].set_night_info(
                    f"[Night Info - Seer] You checked the center pool. {result}")
                if self.language == 'chinese':
                    print_text_animated(Fore.WHITE + f"  [预言家结果] {result}\n\n")
                else:
                    print_text_animated(Fore.WHITE + f"  [Seer Result] {result}\n\n")
                self.process_list.append({f'Host (private to {seer})': result})
                return
            else:
                # Check a specific player
                target_num = self._extract_player_number(output, seer_instruction)
                if target_num:
                    target = f"player {target_num}"
                    if target in self.players and target != seer:
                        target_role = self.roles_ground_truth[target]
                        result = self.host_instruction['seer_check_player'].format(
                            target=target, role=target_role)
                        self.players[seer].receive("Host", f"{phase_tag}|{result}")
                        self.players[seer].set_night_info(
                            f"[Night Info - Seer] You checked {target}. {result}")
                        if self.language == 'chinese':
                            print_text_animated(
                                Fore.WHITE + f"  [预言家结果] {result}\n\n")
                        else:
                            print_text_animated(
                                Fore.WHITE + f"  [Seer Result] {result}\n\n")
                        self.process_list.append(
                            {f'Host (private to {seer})': result})
                        return

        # Fallback: check a random player
        valid_targets = [p for p in self.players if p != seer]
        target = random.choice(valid_targets)
        target_role = self.roles_ground_truth[target]
        result = self.host_instruction['seer_check_player'].format(
            target=target, role=target_role)
        self.players[seer].receive("Host", f"{phase_tag}|{result}")
        self.players[seer].set_night_info(
            f"[Night Info - Seer] You checked {target}. {result}")
        if self.language == 'chinese':
            print_text_animated(
                Fore.WHITE + f"  [预言家结果（备选）] {result}\n\n")
        else:
            print_text_animated(
                Fore.WHITE + f"  [Seer Result (fallback)] {result}\n\n")

    def _robber_night_action(self):
        """Robber may swap their role with another player and view new role."""
        phase_tag = "night phase, Robber"
        robbers = self._get_players_by_initial_role("Robber")

        instruction = self.host_instruction['robber_wake']
        self.process_list.append({'Host': instruction})

        if len(robbers) == 0:
            return

        robber = robbers[0]
        robber_instruction = (
            f"You are the Robber. You may swap your role with another player "
            f"and then view your new role. Available players to swap with: "
            f"{', '.join([p for p in self.players if p != robber])}\n"
            f"You can also choose not to swap. State your choice clearly."
        )

        max_retry = 3
        for attempt in range(max_retry):
            output = self.players[robber].step(
                f"{phase_tag}|{robber_instruction}")
            print_text_animated(
                COLOR_5.get(robber, Fore.WHITE) +
                f"{robber}({self.player_mapping[robber]}):\n\n{output}\n\n")

            self.process_list.append({
                'Host': robber_instruction,
                f"{robber}({self.player_mapping[robber]})": output
            })

            output_lower = output.lower()
            if "not" in output_lower and "swap" in output_lower or \
               "no swap" in output_lower or "don't swap" in output_lower or \
               "do nothing" in output_lower:
                result = self.host_instruction['robber_no_swap']
                self.players[robber].receive("Host", f"{phase_tag}|{result}")
                self.players[robber].set_night_info(
                    f"[Night Info - Robber] {result}")
                if self.language == 'chinese':
                    print_text_animated(
                        Fore.WHITE + f"  [强盗] 未交换。\n\n")
                else:
                    print_text_animated(
                        Fore.WHITE + f"  [Robber] No swap.\n\n")
                self.process_list.append(
                    {f'Host (private to {robber})': result})
                return

            # Try to extract swap target
            target_num = self._extract_player_number(output, robber_instruction)
            if target_num:
                target = f"player {target_num}"
                if target in self.players and target != robber:
                    # Perform swap
                    new_role = self.roles_ground_truth[target]
                    self.roles_ground_truth[robber], self.roles_ground_truth[target] = \
                        self.roles_ground_truth[target], self.roles_ground_truth[robber]
                    result = self.host_instruction['robber_swap_result'].format(
                        target=target, new_role=new_role)
                    self.players[robber].receive("Host", f"{phase_tag}|{result}")
                    self.players[robber].set_night_info(
                        f"[Night Info - Robber] You swapped with {target}. {result}")
                    if self.language == 'chinese':
                        print_text_animated(
                            Fore.WHITE + f"  [强盗] 与{target}交换。"
                            f"新角色：{new_role}\n\n")
                    else:
                        print_text_animated(
                            Fore.WHITE + f"  [Robber] Swapped with {target}. "
                            f"New role: {new_role}\n\n")
                    self.process_list.append(
                        {f'Host (private to {robber})': result})
                    return

        # Fallback: no swap
        result = self.host_instruction['robber_no_swap']
        self.players[robber].receive("Host", f"{phase_tag}|{result}")
        self.players[robber].set_night_info(
            f"[Night Info - Robber] {result}")
        if self.language == 'chinese':
            print_text_animated(
                Fore.WHITE + f"  [强盗（备选）] 未交换。\n\n")
        else:
            print_text_animated(
                Fore.WHITE + f"  [Robber (fallback)] No swap.\n\n")

    def _troublemaker_night_action(self):
        """Troublemaker may swap two other players' roles."""
        phase_tag = "night phase, Troublemaker"
        troublemakers = self._get_players_by_initial_role("Troublemaker")

        instruction = self.host_instruction['troublemaker_wake']
        self.process_list.append({'Host': instruction})

        if len(troublemakers) == 0:
            return

        tm = troublemakers[0]
        other_players = [p for p in self.players if p != tm]
        tm_instruction = (
            f"You are the Troublemaker. You may swap the roles of two other "
            f"players without viewing them. Available players: "
            f"{', '.join(other_players)}\n"
            f"You can also choose not to swap. State your choice clearly, "
            f"naming the two players you want to swap."
        )

        max_retry = 3
        for attempt in range(max_retry):
            output = self.players[tm].step(f"{phase_tag}|{tm_instruction}")
            print_text_animated(
                COLOR_5.get(tm, Fore.WHITE) +
                f"{tm}({self.player_mapping[tm]}):\n\n{output}\n\n")

            self.process_list.append({
                'Host': tm_instruction,
                f"{tm}({self.player_mapping[tm]})": output
            })

            output_lower = output.lower()
            if "not" in output_lower and "swap" in output_lower or \
               "no swap" in output_lower or "don't swap" in output_lower or \
               "do nothing" in output_lower:
                result = self.host_instruction['troublemaker_no_swap']
                self.players[tm].receive("Host", f"{phase_tag}|{result}")
                self.players[tm].set_night_info(
                    f"[Night Info - Troublemaker] {result}")
                if self.language == 'chinese':
                    print_text_animated(
                        Fore.WHITE + f"  [捣蛋鬼] 未交换。\n\n")
                else:
                    print_text_animated(
                        Fore.WHITE + f"  [Troublemaker] No swap.\n\n")
                self.process_list.append(
                    {f'Host (private to {tm})': result})
                return

            # Try to extract two player numbers
            numbers = re.findall(r'\d+', output)
            valid_nums = [n for n in numbers
                         if 1 <= int(n) <= self.player_nums
                         and f"player {n}" != tm
                         and f"player {n}" in self.players]
            # Remove duplicates while preserving order
            seen = set()
            unique_nums = []
            for n in valid_nums:
                if n not in seen:
                    seen.add(n)
                    unique_nums.append(n)

            if len(unique_nums) >= 2:
                p1 = f"player {unique_nums[0]}"
                p2 = f"player {unique_nums[1]}"
                # Perform swap
                self.roles_ground_truth[p1], self.roles_ground_truth[p2] = \
                    self.roles_ground_truth[p2], self.roles_ground_truth[p1]
                result = self.host_instruction['troublemaker_swap_result'].format(
                    player1=p1, player2=p2)
                self.players[tm].receive("Host", f"{phase_tag}|{result}")
                self.players[tm].set_night_info(
                    f"[Night Info - Troublemaker] {result}")
                if self.language == 'chinese':
                    print_text_animated(
                        Fore.WHITE + f"  [捣蛋鬼] 交换了{p1}和{p2}的角色。\n\n")
                else:
                    print_text_animated(
                        Fore.WHITE + f"  [Troublemaker] Swapped {p1} and {p2}.\n\n")
                self.process_list.append(
                    {f'Host (private to {tm})': result})
                return

        # Fallback: no swap
        result = self.host_instruction['troublemaker_no_swap']
        self.players[tm].receive("Host", f"{phase_tag}|{result}")
        self.players[tm].set_night_info(
            f"[Night Info - Troublemaker] {result}")
        if self.language == 'chinese':
            print_text_animated(
                Fore.WHITE + f"  [捣蛋鬼（备选）] 未交换。\n\n")
        else:
            print_text_animated(
                Fore.WHITE + f"  [Troublemaker (fallback)] No swap.\n\n")

    def _insomniac_night_action(self):
        """Insomniac checks their final role."""
        phase_tag = "night phase, Insomniac"
        insomniacs = self._get_players_by_initial_role("Insomniac")

        instruction = self.host_instruction['insomniac_wake']
        self.process_list.append({'Host': instruction})

        if len(insomniacs) == 0:
            return

        insomniac = insomniacs[0]
        final_role = self.roles_ground_truth[insomniac]
        result = self.host_instruction['insomniac_result'].format(role=final_role)
        self.players[insomniac].receive("Host", f"{phase_tag}|{result}")
        self.players[insomniac].set_night_info(
            f"[Night Info - Insomniac] {result}")
        if self.language == 'chinese':
            print_text_animated(
                Fore.WHITE + f"  [失眠者] {insomniac}：最终角色是{final_role}\n\n")
        else:
            print_text_animated(
                Fore.WHITE + f"  [Insomniac] {insomniac}: Final role is {final_role}\n\n")
        self.process_list.append({f'Host (private to {insomniac})': result})

    # ==================== Day Phase ====================

    def day_phase(self):
        """Execute the day discussion phase."""
        if self.language == 'chinese':
            print_text_animated(
                Fore.WHITE + f"\n{'='*60}\n"
                f"[白天阶段] 天亮了，讨论开始！\n"
                f"{'='*60}\n\n")
        else:
            print_text_animated(
                Fore.WHITE + f"\n{'='*60}\n"
                f"[Day Phase] Dawn breaks. Discussion begins!\n"
                f"{'='*60}\n\n")

        player_names = [p.name for p in self.player_list]
        speak_order = copy.deepcopy(player_names)
        random.shuffle(speak_order)

        day_instruction = self.host_instruction['day_start'].format(
            speak_order=', '.join(speak_order))
        print_text_animated(Fore.WHITE + f"Host:\n\n{day_instruction}\n\n")
        self.process_list.append({'Host': day_instruction})

        # Broadcast day start to all players
        for player_i in player_names:
            self.players[player_i].receive("Host", f"day phase|{day_instruction}")

        for discuss_round in range(self.max_discuss_round):
            if discuss_round > 0:
                new_round_msg = self.host_instruction['discuss_new_round'].format(
                    round_num=discuss_round,
                    rounds_left=self.max_discuss_round - discuss_round,
                    first_player=speak_order[0])
                print_text_animated(
                    Fore.WHITE + f"Host:\n\n{new_round_msg}\n\n")
                self.process_list.append({'Host': new_round_msg})
                for player_i in player_names:
                    self.players[player_i].receive(
                        "Host", f"day phase|{new_round_msg}")

            for idx, player_i in enumerate(speak_order):
                speak_instruction = self.host_instruction['discuss_speak'].format(
                    player=player_i)
                full_instruction = speak_instruction

                # Intent Identification
                intent_info = None
                if self.enable_intent_identification and idx < len(speak_order) - 1:
                    next_player = speak_order[idx + 1]
                    intent_info = self.players[player_i].identify_intent(next_player)

                output = self.players[player_i].step(
                    f"day phase, discussion round {discuss_round + 1}|{full_instruction}")
                print_text_animated(
                    COLOR_5.get(player_i, Fore.WHITE) +
                    f"{player_i}({self.player_mapping[player_i]}):\n\n{output}\n\n")

                # Build log entry
                log_entry = {
                    'Host': full_instruction,
                    f"{player_i}({self.player_mapping[player_i]})": output
                }
                if intent_info is not None:
                    log_entry["intent_identification"] = intent_info
                self.process_list.append(log_entry)

                # Broadcast to all other players
                for player_j in player_names:
                    if player_j != player_i:
                        self.players[player_j].receive(
                            "Host", f"day phase|{speak_instruction}")
                        self.players[player_j].receive(
                            player_i, f"day phase|{output}")

    # ==================== Voting Phase ====================

    def voting_phase(self):
        """Execute the voting phase."""
        if self.language == 'chinese':
            print_text_animated(
                Fore.WHITE + f"\n{'='*60}\n"
                f"[投票阶段] 开始投票！\n"
                f"{'='*60}\n\n")
        else:
            print_text_animated(
                Fore.WHITE + f"\n{'='*60}\n"
                f"[Voting Phase] Time to vote!\n"
                f"{'='*60}\n\n")

        player_names = [p.name for p in self.player_list]
        vote_instruction = self.host_instruction['vote_prompt']
        print_text_animated(Fore.WHITE + f"Host:\n\n{vote_instruction}\n\n")
        self.process_list.append({'Host': vote_instruction})

        # Broadcast vote instruction
        for player_i in player_names:
            self.players[player_i].receive(
                "Host", f"voting phase|{vote_instruction}")

        votes = {}  # voter -> target
        vote_counts = {}  # target -> count

        for player_i in player_names:
            output = self.players[player_i].step(
                f"voting phase|{vote_instruction}")
            print_text_animated(
                COLOR_5.get(player_i, Fore.WHITE) +
                f"{player_i}({self.player_mapping[player_i]}):\n\n{output}\n\n")

            self.process_list.append({
                'Host': vote_instruction,
                f"{player_i}({self.player_mapping[player_i]})": output
            })

            # Extract vote target
            if self.vote_extractor is not None:
                s = self.vote_extractor.step(
                    f"Question: {vote_instruction}\nAnswer: {output}")
                nums = re.findall(r'\d+', s)
                valid = [n for n in nums
                         if 1 <= int(n) <= self.player_nums
                         and f"player {n}" in player_names
                         and f"player {n}" != player_i]
                if valid:
                    votes[player_i] = f"player {valid[0]}"
                else:
                    votes[player_i] = 'abstain'
            else:
                nums = re.findall(r'\d+', output)
                valid = [n for n in nums
                         if 1 <= int(n) <= self.player_nums
                         and f"player {n}" in player_names
                         and f"player {n}" != player_i]
                if valid:
                    votes[player_i] = f"player {valid[0]}"
                else:
                    votes[player_i] = 'abstain'

        # Count votes
        for voter, target in votes.items():
            if target != 'abstain':
                vote_counts[target] = vote_counts.get(target, 0) + 1

        vote_summary = ', '.join(
            [f"{voter}: {target}" for voter, target in votes.items()])

        # Determine result using ONUW rules
        self._resolve_votes(votes, vote_counts, vote_summary)

    def _resolve_votes(self, votes: dict, vote_counts: dict, vote_summary: str):
        """
        Resolve voting results according to ONUW rules.
        - If max votes > 1: player(s) with most votes are eliminated
        - If max votes == 1: no one is eliminated
        - Team Village wins if Werewolf is eliminated
        - Team Werewolf wins if Werewolf avoids elimination
        """
        # Check if any Werewolf exists among players
        werewolf_exists = any(
            role == "Werewolf" for role in self.roles_ground_truth.values())

        if not vote_counts:
            # Everyone abstained
            if werewolf_exists:
                if self.language == 'chinese':
                    result_msg = f"没有人投票。狼人存活。狼人阵营获胜！"
                else:
                    result_msg = f"No votes were cast. The Werewolf survives. Team Werewolf wins!"
                self.winners = ["Werewolf"]
            else:
                if self.language == 'chinese':
                    result_msg = "没有人投票。玩家中没有狼人。平局。"
                else:
                    result_msg = "No votes were cast. No Werewolf exists among players. It's a draw."
                self.winners = ["Draw"]
            print_text_animated(Fore.YELLOW + f"\n{result_msg}\n\n")
            self.process_list.append({'Host': result_msg})
            return

        max_votes = max(vote_counts.values())

        if max_votes == 1:
            # No one gets more than 1 vote - no elimination
            result_msg = self.host_instruction['vote_result_no_majority'].format(
                vote_summary=vote_summary)
            print_text_animated(Fore.YELLOW + f"\n{result_msg}\n\n")
            self.process_list.append({'Host': result_msg})

            if werewolf_exists:
                end_msg = self.host_instruction['game_over_werewolf_wins']
                self.winners = ["Werewolf"]
            else:
                end_msg = self.host_instruction['game_over_draw']
                self.winners = ["Draw"]
            print_text_animated(Fore.YELLOW + f"\n{end_msg}\n\n")
            self.process_list.append({'Host': end_msg})
            return

        # Find all players with max votes
        eliminated = [t for t, c in vote_counts.items() if c == max_votes]

        # Check if any eliminated player is a Werewolf
        werewolf_eliminated = any(
            self.roles_ground_truth.get(p) == "Werewolf" for p in eliminated)

        if len(eliminated) == 1:
            target = eliminated[0]
            target_role = self.roles_ground_truth[target]
            result_msg = self.host_instruction['vote_result_eliminate'].format(
                vote_summary=vote_summary, target=target, role=target_role)
        else:
            targets_str = ', '.join(eliminated)
            result_msg = self.host_instruction['vote_result_tie'].format(
                vote_summary=vote_summary, targets=targets_str)
            # Show each eliminated player's role
            for p in eliminated:
                if self.language == 'chinese':
                    result_msg += f" {p}的最终角色是{self.roles_ground_truth[p]}。"
                else:
                    result_msg += f" {p}'s final role is {self.roles_ground_truth[p]}."

        print_text_animated(Fore.YELLOW + f"\n{result_msg}\n\n")
        self.process_list.append({'Host': result_msg})

        if werewolf_eliminated:
            end_msg = self.host_instruction['game_over_village_wins']
            self.winners = ["Village"]
        else:
            if werewolf_exists:
                end_msg = self.host_instruction['game_over_werewolf_wins']
                self.winners = ["Werewolf"]
            else:
                end_msg = self.host_instruction['game_over_draw']
                self.winners = ["Draw"]

        print_text_animated(Fore.YELLOW + f"\n{end_msg}\n\n")
        self.process_list.append({'Host': end_msg})

        # Broadcast result to all players
        for p in self.players:
            self.players[p].receive("Host", f"game over|{end_msg}")

    # ==================== Night Info Distribution ====================

    def distribute_night_info(self):
        """
        Distribute initial role information to players at game start.
        Each player learns their initial role assignment.
        """
        if self.language == 'chinese':
            print_text_animated(
                Fore.WHITE + f"\n{'='*60}\n"
                f"[游戏开始] 分发角色信息...\n"
                f"{'='*60}\n\n")
        else:
            print_text_animated(
                Fore.WHITE + f"\n{'='*60}\n"
                f"[Game Start] Distributing role information...\n"
                f"{'='*60}\n\n")

        for player_i in self.players:
            role = self.player_mapping[player_i]
            intro = self.role_introduce.get(role.lower(), "")
            if self.language == 'chinese':
                night_info = f"[角色信息] 你是{role}。{intro}"
            else:
                night_info = f"[Role Info] You are the {role}. {intro}"
            self.players[player_i].set_night_info(night_info)
            if self.language == 'chinese':
                print_text_animated(
                    Fore.WHITE + f"  {player_i}({role})：角色信息已设置。\n")
            else:
                print_text_animated(
                    Fore.WHITE + f"  {player_i}({role}): Role info set.\n")

        if self.language == 'chinese':
            print_text_animated(
                Fore.WHITE + f"\n[游戏开始] 所有玩家已收到角色信息。"
                f"游戏开始！\n\n")
        else:
            print_text_animated(
                Fore.WHITE + f"\n[Game Start] All players have received their roles. "
                f"The game begins!\n\n")

    # ==================== Main Game Loop ====================

    def start(self):
        """Start the ONUW game."""
        self.init_game()
        self.distribute_night_info()

        process_json = {}

        try:
            # Night phase
            self.night_phase()
            process_json["night"] = self.process_list
            self.process_list = []

            # Day phase
            self.day_phase()
            process_json["day"] = self.process_list
            self.process_list = []

            # Voting phase
            self.voting_phase()
            process_json["voting"] = self.process_list
            self.process_list = []

            # Save final state
            process_json["game_result"] = {
                "initial_roles": self.player_mapping,
                "final_roles": self.roles_ground_truth,
                "center_pool": self.center_pool,
                "winners": self.winners
            }

        except Exception as e:
            process_json["error"] = self.process_list
            write_json(process_json, f'{self.output_dir}/process.json')
            raise e

        write_json(process_json, f'{self.output_dir}/process.json')
