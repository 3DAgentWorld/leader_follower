#!/usr/bin/env python
# encoding: utf-8
"""
Werewolf game engine.
Implements the standard Werewolf (Mafia) game with 7 players:
2 Werewolves, 1 Seer, 1 Guardian, 3 Villagers.
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
from src.utils import print_text_animated, write_json, create_dir, COLOR


# Extended color mapping for 7 players
COLOR_7 = {
    "player 1": Fore.BLUE,
    "player 2": Fore.GREEN,
    "player 3": Fore.YELLOW,
    "player 4": Fore.RED,
    "player 5": Fore.LIGHTGREEN_EX,
    "player 6": Fore.CYAN,
    "player 7": Fore.MAGENTA,
}


class Werewolf(Game):
    def __init__(self, player_nums: int, language: str, mode: str, ai_model,
                 output_dir, **kwargs):
        """
        Initialize the Werewolf game.

        Args:
            player_nums: Number of players (must be 7).
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
        roles = config_for_num.get('role', [])
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
        self.alive_players = []
        self.dead_players = []
        self.player_mapping = {}  # name -> role

        # Role configuration
        self.roles = roles
        self.werewolf_team = werewolf_team
        self.village_team = village_team
        self.role_mapping = role_mapping

        # Game state
        self.mode = mode
        self.ai_model = ai_model
        self.process_list = []
        self.output_dir = output_dir
        self.day_num = 0
        self.winners = []

        self.response_rule = response_rule

        # AI extractors
        self.player_extractor = None
        self.vote_extractor = None
        self.confirm_extractor = None

        # Whether to enable intent identification
        self.enable_intent_identification = kwargs.get('enable_intent_identification', False)

        # Seer's investigation history: {player_name: True/False (is_werewolf)}
        self.seer_history = {}
        # Guardian's last protection target (to allow repeated protection)
        self.guardian_last_target = None

    def init_extractor(self, player_extractor: Tuple[Type[Extractor], dict],
                       vote_extractor: Tuple[Type[Extractor], dict],
                       confirm_extractor: Tuple[Type[Extractor], dict]):
        """Initialize extractors for parsing LLM outputs."""
        self.player_extractor = player_extractor[0].init_instance(**player_extractor[1])
        self.vote_extractor = vote_extractor[0].init_instance(**vote_extractor[1])
        self.confirm_extractor = confirm_extractor[0].init_instance(**confirm_extractor[1])

    def add_players(self, players: List[Tuple[Type[Agent], dict]]):
        """Add players to the game and assign roles."""
        assert self.player_nums == len(players), \
            f"Required {self.player_nums} players, got {len(players)}."

        need_random_role = any(p[1].get("role") is None for p in players)
        if need_random_role:
            random.shuffle(self.roles)

        idx = 0
        for agent_type, player_params in players:
            if player_params.get('name') is None:
                player_params['name'] = f'player {idx + 1}'
            if player_params.get('role') is None:
                player_params['role'] = self.roles[idx]
            player_i = agent_type.init_instance(**player_params)
            self.player_list.append(player_i)
            self.players[player_i.name] = player_i
            self.player_mapping[player_i.name] = player_i.role
            idx += 1
        self.roles = [p.role for p in self.player_list]

    def init_game(self):
        """Initialize game state."""
        self.alive_players = [p.name for p in self.player_list]
        self.dead_players = []
        self.day_num = 0
        self.winners = []
        self.seer_history = {}
        self.guardian_last_target = None

    def _get_werewolf_players(self) -> List[str]:
        """Get list of living Werewolf player names."""
        return [p for p in self.alive_players
                if self.player_mapping[p] in self.werewolf_team]

    def _get_village_players(self) -> List[str]:
        """Get list of living Village team player names."""
        return [p for p in self.alive_players
                if self.player_mapping[p] in self.village_team]

    def _get_seer_player(self) -> str:
        """Get the Seer player name (if alive)."""
        for p in self.alive_players:
            if self.player_mapping[p] == self.role_mapping['seer']:
                return p
        return None

    def _get_guardian_player(self) -> str:
        """Get the Guardian player name (if alive)."""
        for p in self.alive_players:
            if self.player_mapping[p] == self.role_mapping['guardian']:
                return p
        return None

    def _extract_player_number(self, output: str, instruction: str) -> str:
        """Extract a player number from LLM output using the extractor."""
        if self.player_extractor is not None:
            s = self.player_extractor.step(
                f"Question: {instruction}\nAnswer: {output}")
        else:
            s = output
        numbers = re.findall(r'\d+', s)
        # Filter out invalid player numbers
        valid = [n for n in numbers if 1 <= int(n) <= self.player_nums]
        return valid[0] if valid else None

    def _eliminate_player(self, player_name: str):
        """Remove a player from the game."""
        if player_name in self.alive_players:
            self.alive_players.remove(player_name)
            self.dead_players.append(player_name)

    def check_game_end(self) -> bool:
        """
        Check if the game has ended.
        Village wins: all Werewolves eliminated.
        Werewolf wins: Werewolves >= Village team members.
        """
        werewolves = self._get_werewolf_players()
        villagers = self._get_village_players()

        if len(werewolves) == 0:
            self.winners = ["Village"]
            return True
        if len(werewolves) >= len(villagers):
            self.winners = ["Werewolf"]
            return True
        return False

    # ==================== Night Phase ====================

    def night_phase(self) -> str:
        """
        Execute the night phase.
        Returns the name of the eliminated player, or None if protected.
        """
        if self.language == 'chinese':
            print_text_animated(
                Fore.WHITE + f"\n{'='*60}\n"
                f"[第 {self.day_num + 1} 夜] 夜幕降临...\n"
                f"{'='*60}\n\n")
        else:
            print_text_animated(
                Fore.WHITE + f"\n{'='*60}\n"
                f"[Night {self.day_num + 1}] Night falls...\n"
                f"{'='*60}\n\n")

        # Step 1: Werewolves choose a target
        kill_target = self._werewolf_action()

        # Step 2: Seer investigates
        self._seer_action()

        # Step 3: Guardian protects
        protect_target = self._guardian_action()

        # Resolve night: check if kill target was protected
        if kill_target and kill_target == protect_target:
            if self.language == 'chinese':
                print_text_animated(
                    Fore.YELLOW + f"  [夜间结算] 守卫成功保护了目标！\n\n")
            else:
                print_text_animated(
                    Fore.YELLOW + f"  [Night Resolution] Guardian protected the target!\n\n")
            return None
        else:
            return kill_target

    def _werewolf_action(self) -> str:
        """
        Werewolves collectively choose a target.
        If 2 Werewolves alive: lower ID proposes, other confirms/overrides.
        If 1 Werewolf alive: solo choice.
        """
        werewolves = self._get_werewolf_players()
        if not werewolves:
            return None

        phase_tag = f"night phase, day {self.day_num + 1}"

        if len(werewolves) == 1:
            # Solo Werewolf
            wolf = werewolves[0]
            instruction = self.host_instruction['werewolf_kill_solo']
            output = self.players[wolf].step(f"{phase_tag}|{instruction}")
            print_text_animated(
                COLOR_7.get(wolf, Fore.WHITE) +
                f"{wolf}({self.player_mapping[wolf]}):\n\n{output}\n\n")

            target_num = self._extract_player_number(output, instruction)
            self.process_list.append({
                'Host': instruction,
                f"{wolf}({self.player_mapping[wolf]})": output
            })

            if target_num:
                target = f"player {target_num}"
                # Validate target
                if target in self.alive_players and target != wolf:
                    return target
            # Fallback: random valid target
            valid_targets = [p for p in self.alive_players
                           if p != wolf and self.player_mapping[p] not in self.werewolf_team]
            return random.choice(valid_targets) if valid_targets else None

        else:
            # Two Werewolves: lower ID proposes
            werewolves_sorted = sorted(werewolves,
                                       key=lambda x: int(re.findall(r'\d+', x)[0]))
            proposer = werewolves_sorted[0]
            confirmer = werewolves_sorted[1]

            # Proposer chooses target
            instruction = self.host_instruction['werewolf_kill_prompt']
            output = self.players[proposer].step(f"{phase_tag}|{instruction}")
            print_text_animated(
                COLOR_7.get(proposer, Fore.WHITE) +
                f"{proposer}({self.player_mapping[proposer]}):\n\n{output}\n\n")

            target_num = self._extract_player_number(output, instruction)
            self.process_list.append({
                'Host': instruction,
                f"{proposer}({self.player_mapping[proposer]})": output
            })

            proposed_target = None
            if target_num:
                proposed_target = f"player {target_num}"
                # Validate
                if (proposed_target not in self.alive_players or
                        proposed_target in werewolves):
                    proposed_target = None

            if proposed_target is None:
                # Fallback: random valid target
                valid = [p for p in self.alive_players if p not in werewolves]
                proposed_target = random.choice(valid) if valid else None

            if proposed_target is None:
                return None

            # Confirmer decides
            confirm_instruction = self.host_instruction['werewolf_kill_confirm'].format(
                proposer=proposer, target=proposed_target)
            output = self.players[confirmer].step(
                f"{phase_tag}|{confirm_instruction}")
            print_text_animated(
                COLOR_7.get(confirmer, Fore.WHITE) +
                f"{confirmer}({self.player_mapping[confirmer]}):\n\n{output}\n\n")

            self.process_list.append({
                'Host': confirm_instruction,
                f"{confirmer}({self.player_mapping[confirmer]})": output
            })

            # Parse confirmation
            if self.confirm_extractor is not None:
                s = self.confirm_extractor.step(
                    f"Question: {confirm_instruction}\nAnswer: {output}")
                agreed = 'true' in s.lower()
            else:
                agreed = not bool(re.findall(r'no|disagree|reject', output.lower()))

            if agreed:
                return proposed_target
            else:
                # Confirmer suggests alternative - extract from their response
                alt_num = self._extract_player_number(output, confirm_instruction)
                if alt_num:
                    alt_target = f"player {alt_num}"
                    if alt_target in self.alive_players and alt_target not in werewolves:
                        return alt_target
                # If no valid alternative, use the original proposal
                return proposed_target

    def _seer_action(self):
        """Seer investigates one player."""
        seer = self._get_seer_player()
        if seer is None:
            return

        phase_tag = f"night phase, day {self.day_num + 1}"
        instruction = self.host_instruction['seer_investigate']

        max_retry = 3
        for attempt in range(max_retry):
            output = self.players[seer].step(f"{phase_tag}|{instruction}")
            print_text_animated(
                COLOR_7.get(seer, Fore.WHITE) +
                f"{seer}({self.player_mapping[seer]}):\n\n{output}\n\n")

            target_num = self._extract_player_number(output, instruction)
            self.process_list.append({
                'Host': instruction,
                f"{seer}({self.player_mapping[seer]})": output
            })

            if target_num:
                target = f"player {target_num}"
                # Validate: cannot investigate self or dead players
                if target in self.alive_players and target != seer:
                    is_werewolf = self.player_mapping[target] in self.werewolf_team
                    self.seer_history[target] = is_werewolf

                    if is_werewolf:
                        result = self.host_instruction['seer_result_werewolf'].format(
                            target=target)
                    else:
                        result = self.host_instruction['seer_result_not_werewolf'].format(
                            target=target)

                    # Send result privately to Seer
                    self.players[seer].receive(
                        "Host", f"{phase_tag}|{result}")
                    if self.language == 'chinese':
                        print_text_animated(
                            Fore.WHITE + f"  [预言家结果] {result}\n\n")
                    else:
                        print_text_animated(
                            Fore.WHITE + f"  [Seer Result] {result}\n\n")
                    self.process_list.append({'Host (private to Seer)': result})
                    return

        # Fallback: investigate a random valid target
        valid = [p for p in self.alive_players if p != seer and p not in self.seer_history]
        if valid:
            target = random.choice(valid)
            is_werewolf = self.player_mapping[target] in self.werewolf_team
            self.seer_history[target] = is_werewolf
            if is_werewolf:
                result = self.host_instruction['seer_result_werewolf'].format(target=target)
            else:
                result = self.host_instruction['seer_result_not_werewolf'].format(target=target)
            self.players[seer].receive("Host", f"{phase_tag}|{result}")
            if self.language == 'chinese':
                print_text_animated(Fore.WHITE + f"  [预言家结果（备选）] {result}\n\n")
            else:
                print_text_animated(Fore.WHITE + f"  [Seer Result (fallback)] {result}\n\n")

    def _guardian_action(self) -> str:
        """Guardian chooses a player to protect. Returns the protected player name."""
        guardian = self._get_guardian_player()
        if guardian is None:
            return None

        phase_tag = f"night phase, day {self.day_num + 1}"
        instruction = self.host_instruction['guardian_protect']

        max_retry = 3
        for attempt in range(max_retry):
            output = self.players[guardian].step(f"{phase_tag}|{instruction}")
            print_text_animated(
                COLOR_7.get(guardian, Fore.WHITE) +
                f"{guardian}({self.player_mapping[guardian]}):\n\n{output}\n\n")

            target_num = self._extract_player_number(output, instruction)
            self.process_list.append({
                'Host': instruction,
                f"{guardian}({self.player_mapping[guardian]})": output
            })

            if target_num:
                target = f"player {target_num}"
                # Guardian can protect any living player including themselves
                if target in self.alive_players:
                    self.guardian_last_target = target
                    if self.language == 'chinese':
                        print_text_animated(
                            Fore.WHITE + f"  [守卫] 保护{target}\n\n")
                    else:
                        print_text_animated(
                            Fore.WHITE + f"  [Guardian] Protecting {target}\n\n")
                    return target

        # Fallback: protect a random player
        target = random.choice(self.alive_players)
        self.guardian_last_target = target
        if self.language == 'chinese':
            print_text_animated(
                Fore.WHITE + f"  [守卫（备选）] 保护{target}\n\n")
        else:
            print_text_animated(
                Fore.WHITE + f"  [Guardian (fallback)] Protecting {target}\n\n")
        return target

    # ==================== Day Phase ====================

    def day_phase(self, killed_player: str) -> str:
        """
        Execute the day phase.
        Returns the name of the voted-out player, or None.
        """
        self.day_num += 1
        phase_tag = f"day phase, day {self.day_num}"

        if self.language == 'chinese':
            print_text_animated(
                Fore.WHITE + f"\n{'='*60}\n"
                f"[第 {self.day_num} 天] 天亮了...\n"
                f"{'='*60}\n\n")
        else:
            print_text_animated(
                Fore.WHITE + f"\n{'='*60}\n"
                f"[Day {self.day_num}] Dawn breaks...\n"
                f"{'='*60}\n\n")

        # Step 1: Announce night results
        self._announce_night_result(killed_player, phase_tag)

        # Check game end after night elimination
        if self.check_game_end():
            return None

        # Step 2: Discussion
        self._discussion(phase_tag)

        # Step 3: Voting
        voted_out = self._voting(phase_tag)

        return voted_out

    def _announce_night_result(self, killed_player: str, phase_tag: str):
        """Announce who was eliminated during the night."""
        if killed_player:
            announcement = self.host_instruction['day_announce_kill'].format(
                target=killed_player)
            self._eliminate_player(killed_player)
        else:
            announcement = self.host_instruction['day_announce_no_kill']

        print_text_animated(Fore.WHITE + f"Host:\n\n{announcement}\n\n")
        self.process_list.append({'Host': announcement})

        # Broadcast to all living players
        for player_i in self.alive_players:
            self.players[player_i].receive("Host", f"{phase_tag}|{announcement}")

    def _discussion(self, phase_tag: str):
        """All surviving players discuss in order."""
        speak_order = copy.deepcopy(self.alive_players)
        # Randomize starting speaker each day
        random.shuffle(speak_order)

        discuss_instruction = self.host_instruction['discuss_prompt'].format(
            day_num=self.day_num,
            speak_order=', '.join(speak_order))

        print_text_animated(
            Fore.WHITE + f"Host:\n\n{discuss_instruction}\n\n")

        for idx, player_i in enumerate(speak_order):
            speak_instruction = self.host_instruction['discuss_speak'].format(
                player=player_i)
            full_instruction = discuss_instruction + " " + speak_instruction

            # Intent Identification
            intent_info = None
            if self.enable_intent_identification and idx < len(speak_order) - 1:
                next_player = speak_order[idx + 1]
                intent_info = self.players[player_i].identify_intent(next_player)

            output = self.players[player_i].step(
                f"{phase_tag}|{full_instruction}")
            print_text_animated(
                COLOR_7.get(player_i, Fore.WHITE) +
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
            for player_j in self.alive_players:
                if player_j != player_i:
                    self.players[player_j].receive(
                        "Host", f"{phase_tag}|{speak_instruction}")
                    self.players[player_j].receive(
                        player_i, f"{phase_tag}|{output}")

    def _voting(self, phase_tag: str) -> str:
        """All surviving players vote to eliminate one player."""
        vote_instruction = self.host_instruction['vote_prompt']
        print_text_animated(Fore.WHITE + f"Host:\n\n{vote_instruction}\n\n")

        votes = {}  # voter -> target
        vote_counts = {}  # target -> count

        for player_i in self.alive_players:
            output = self.players[player_i].step(
                f"{phase_tag}|{vote_instruction}")
            print_text_animated(
                COLOR_7.get(player_i, Fore.WHITE) +
                f"{player_i}({self.player_mapping[player_i]}):\n\n{output}\n\n")

            self.process_list.append({
                'Host': vote_instruction,
                f"{player_i}({self.player_mapping[player_i]})": output
            })

            # Extract vote target
            if self.vote_extractor is not None:
                s = self.vote_extractor.step(
                    f"Question: {vote_instruction}\nAnswer: {output}")
                if 'abstain' in s.lower():
                    votes[player_i] = 'abstain'
                else:
                    nums = re.findall(r'\d+', s)
                    valid = [n for n in nums
                             if 1 <= int(n) <= self.player_nums
                             and f"player {n}" in self.alive_players
                             and f"player {n}" != player_i]
                    if valid:
                        votes[player_i] = f"player {valid[0]}"
                    else:
                        votes[player_i] = 'abstain'
            else:
                nums = re.findall(r'\d+', output)
                valid = [n for n in nums
                         if 1 <= int(n) <= self.player_nums
                         and f"player {n}" in self.alive_players
                         and f"player {n}" != player_i]
                if valid:
                    votes[player_i] = f"player {valid[0]}"
                else:
                    votes[player_i] = 'abstain'

        # Count votes
        for voter, target in votes.items():
            if target != 'abstain':
                vote_counts[target] = vote_counts.get(target, 0) + 1

        # Determine result
        vote_summary = ', '.join(
            [f"{voter}: {target}" for voter, target in votes.items()])

        if not vote_counts:
            # Everyone abstained
            result_msg = self.host_instruction['vote_result_no_eliminate'].format(
                vote_summary=vote_summary)
            print_text_animated(Fore.WHITE + f"Host:\n\n{result_msg}\n\n")
            self.process_list.append({'Host': result_msg})
            for p in self.alive_players:
                self.players[p].receive("Host", f"{phase_tag}|{result_msg}")
            return None

        max_votes = max(vote_counts.values())
        top_targets = [t for t, c in vote_counts.items() if c == max_votes]

        if len(top_targets) == 1:
            eliminated = top_targets[0]
            result_msg = self.host_instruction['vote_result_eliminate'].format(
                vote_summary=vote_summary, target=eliminated)
        else:
            # Tie: random elimination
            eliminated = random.choice(top_targets)
            result_msg = self.host_instruction['vote_result_tie'].format(
                vote_summary=vote_summary, target=eliminated)

        if self.language == 'chinese':
            print_text_animated(
                Fore.YELLOW + f"\n[投票结果] {eliminated} 被淘汰。"
                f"({vote_summary})\n\n")
        else:
            print_text_animated(
                Fore.YELLOW + f"\n[Vote Result] {eliminated} is eliminated. "
                f"({vote_summary})\n\n")
        print_text_animated(Fore.WHITE + f"Host:\n\n{result_msg}\n\n")
        self.process_list.append({'Host': result_msg})

        # Broadcast result
        for p in self.alive_players:
            self.players[p].receive("Host", f"{phase_tag}|{result_msg}")

        self._eliminate_player(eliminated)
        return eliminated

    # ==================== Night Info Distribution ====================

    def distribute_night_info(self):
        """
        Distribute initial role information to players at game start.
        Similar to Avalon's night_process.
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

        for player_i in self.alive_players:
            role = self.player_mapping[player_i]
            night_info = ""

            if role in self.werewolf_team:
                # Werewolves know each other
                other_wolves = [p for p in self.alive_players
                               if self.player_mapping[p] in self.werewolf_team
                               and p != player_i]
                if self.language == 'chinese':
                    night_info = (
                        f"[角色信息] 你是狼人。"
                        f"你的狼人队友是：{', '.join(other_wolves)}。"
                        f"合作淘汰村庄阵营成员，同时避免在白天被发现。")
                else:
                    night_info = (
                        f"[Role Info] You are a Werewolf. "
                        f"Your Werewolf teammate is: {', '.join(other_wolves)}. "
                        f"Work together to eliminate Village team members while "
                        f"avoiding detection during the day.")
            elif role == self.role_mapping['seer']:
                if self.language == 'chinese':
                    night_info = (
                        "[角色信息] 你是预言家。"
                        "每晚你可以查验一名玩家，了解其是否为狼人。"
                        "明智地使用这些信息引导村庄阵营，但不要暴露自己。")
                else:
                    night_info = (
                        "[Role Info] You are the Seer. "
                        "Each night, you can investigate one player to learn if "
                        "they are a Werewolf. Use this information wisely to guide "
                        "the Village team without exposing yourself to the Werewolves.")
            elif role == self.role_mapping['guardian']:
                if self.language == 'chinese':
                    night_info = (
                        "[角色信息] 你是守卫。"
                        "每晚你可以保护一名玩家（包括你自己）"
                        "免受狼人淘汰。尝试预测狼人的目标并保护关键玩家。")
                else:
                    night_info = (
                        "[Role Info] You are the Guardian. "
                        "Each night, you can protect one player (including yourself) "
                        "from Werewolf elimination. Try to predict who the Werewolves "
                        "will target and protect key players.")
            else:
                if self.language == 'chinese':
                    night_info = (
                        "[角色信息] 你是村民。"
                        "你没有特殊能力。通过讨论和投票"
                        "来识别狼人并帮助你的队伍获胜。")
                else:
                    night_info = (
                        "[Role Info] You are a Villager. "
                        "You have no special abilities. Use discussion and voting "
                        "patterns to identify the Werewolves and help your team win.")

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
        """Start the Werewolf game."""
        self.init_game()
        self.distribute_night_info()

        process_json = {}
        game_continue = True

        try:
            while game_continue:
                round_key = f"day {self.day_num + 1}"
                if self.language == 'chinese':
                    print_text_animated(
                        Fore.WHITE + f"\n{'='*60}\n"
                        f"[第 {self.day_num + 1} 轮] 开始...\n"
                        f"存活玩家：{', '.join(self.alive_players)}\n"
                        f"{'='*60}\n\n")
                else:
                    print_text_animated(
                        Fore.WHITE + f"\n{'='*60}\n"
                        f"[Round {self.day_num + 1}] Starting...\n"
                        f"Alive: {', '.join(self.alive_players)}\n"
                        f"{'='*60}\n\n")

                # Night phase
                killed = self.night_phase()

                # Day phase
                voted_out = self.day_phase(killed)

                # Check game end
                if self.check_game_end():
                    game_continue = False
                    if self.winners == ["Village"]:
                        end_msg = self.host_instruction['game_over_village_wins']
                    else:
                        end_msg = self.host_instruction['game_over_werewolf_wins']
                    if self.language == 'chinese':
                        print_text_animated(
                            Fore.YELLOW + f"\n{'='*60}\n"
                            f"[游戏结束] {end_msg}\n"
                            f"{'='*60}\n\n")
                    else:
                        print_text_animated(
                            Fore.YELLOW + f"\n{'='*60}\n"
                            f"[GAME OVER] {end_msg}\n"
                            f"{'='*60}\n\n")
                    self.process_list.append({'Host': end_msg})
                    for p in self.alive_players:
                        self.players[p].receive(
                            "Host",
                            f"game over|{end_msg}")

                # Save process
                process_json[round_key] = self.process_list
                write_json(process_json, f'{self.output_dir}/process.json')
                self.process_list = []

                # Safety: prevent infinite games
                if self.day_num >= 10:
                    if self.language == 'chinese':
                        print_text_animated(
                            Fore.RED + "\n[警告] 游戏超过10天。"
                            "强制结束。\n\n")
                    else:
                        print_text_animated(
                            Fore.RED + "\n[WARNING] Game exceeded 10 days. "
                            "Forcing end.\n\n")
                    game_continue = False

        except Exception as e:
            round_key = f"day {self.day_num + 1}"
            process_json[round_key] = self.process_list
            write_json(process_json, f'{self.output_dir}/process.json')
            raise e
