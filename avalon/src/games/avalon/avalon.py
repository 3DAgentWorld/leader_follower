#!/usr/bin/env python 
# encoding: utf-8
# @author: yihuai lan
# @fileName: avalon.py
# @date: 2023/6/30 13:44
#
# describe:
#

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


class Avalon(Game):
    def __init__(self, player_nums: int, language: str, mode: str, ai_model, output_dir, **kwargs):
        """
        :param player_nums: number of players.
        :param language: English or Chinese version of Werewolf game.
        :param mode: game mode. watch: all players are AI agents. play: one player is human and others are AI agent.
        :param ai_model:
        """
        config_file = kwargs.get("config_file")
        if not config_file:
            config_file = os.path.join(os.path.dirname(__file__), 'config.json')
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data: dict = json.load(f)

        support_nums = config_data.get("player_nums", [])
        assert player_nums in support_nums, f"game includes {player_nums} players is not supported"

        game_config = config_data.get(language, {})
        if not game_config:
            raise NotImplementedError(f'{language} is not supported.')

        host_instruction: dict = game_config.get('host_instruction')
        response_rule: dict = game_config.get('response_rule')
        role_introduce = game_config.get('role_introduce')
        config_for_num = game_config.get(f'config_{player_nums}', {})
        game_introduce = config_for_num.get('game_introduce', '')
        roles = config_for_num.get('role', [])
        blue_camp = config_for_num.get('blue_camp', [])
        red_camp = config_for_num.get('red_camp', [])
        task_member = config_for_num.get('task_member', {})
        role_mapping = game_config.get('role_mapping', {})
        create_dir(output_dir)

        # instruction
        self.language = language
        self.host_instruction = host_instruction
        self.role_introduce = role_introduce
        self.game_introduce = game_introduce

        # player
        self.player_nums = player_nums
        self.player_list = []
        self.players: Dict[str, Agent] = {}
        self.alive_players = []
        self.player_mapping = {}

        # role
        self.alive_roles = []
        self.good_side = blue_camp
        self.evil_side = red_camp
        self.task_member = task_member
        self.roles = roles
        self.role_mapping = role_mapping

        self.merlin_player = None
        self.percival_player = None
        self.assassin_player = None
        self.morgana_player = None
        self.task_leader = None

        self.mode = mode
        self.ai_model = ai_model
        self.process_list = []
        self.output_dir = output_dir
        # skill
        self.assassin_kill = True

        # AI extractor
        self.player_extractor = None
        self.select_merlin_extractor = None
        self.choose_identify_extractor = None
        self.quest_extractor = None
        self.vote_extractor = None

        self.game_round = 0
        self.good_score = 0
        self.evil_score = 0
        self.winners = []

        self.response_rule = response_rule
        
        # Whether to enable intent identification
        self.enable_intent_identification = kwargs.get('enable_intent_identification', False)

    def init_extractor(self, player_extractor: Tuple[Type[Extractor], dict], vote_extractor: Tuple[Type[Extractor], dict],
                       quest_extractor: Tuple[Type[Extractor], dict], choose_identify_extractor: Tuple[Type[Extractor], dict],
                       select_merlin_extractor: Tuple[Type[Extractor], dict]):
        self.player_extractor = player_extractor[0].init_instance(**player_extractor[1])
        self.vote_extractor = vote_extractor[0].init_instance(**vote_extractor[1])
        self.quest_extractor = quest_extractor[0].init_instance(**quest_extractor[1])
        self.choose_identify_extractor = choose_identify_extractor[0].init_instance(**choose_identify_extractor[1])
        self.select_merlin_extractor = select_merlin_extractor[0].init_instance(**select_merlin_extractor[1])

    def add_players(self, players: List[Tuple[Type[Agent], dict]]):
        """

        :param players:
        :return:
        """
        # match number of players
        assert self.player_nums == len(players), \
            f"Required {self.player_nums} players, got {len(self.player_list)} players. Please add more players."
        need_random_role = False
        need_random_name = False
        for idx, player_params in enumerate(players):
            name = player_params[1].get("name")
            role = player_params[1].get("role")
            if name is None:
                need_random_name = True
            if role is None:
                need_random_role = True
        self.alive_roles = copy.deepcopy(self.roles)
        random.shuffle(self.roles)
        idx = 0
        for agent_type, player_params in players:
            if need_random_name:
                name = f'player {idx + 1}'
                player_params['name'] = name
            if need_random_role:
                role = self.roles[idx]
                player_params['role'] = role
            player_i = agent_type.init_instance(**player_params)
            self.player_list.append(player_i)
            self.players[player_i.name] = player_i
            self.player_mapping[player_i.name] = player_i.role
            idx += 1
        self.roles = [player_i.role for player_i in self.player_list]
        idx = self.roles.index(self.role_mapping['merlin'])
        self.merlin_player = self.player_list[idx].name
        idx = self.roles.index(self.role_mapping['percival'])
        self.percival_player = self.player_list[idx].name
        idx = self.roles.index(self.role_mapping['assassin'])
        self.assassin_player = self.player_list[idx].name
        idx = self.roles.index(self.role_mapping['morgana'])
        self.morgana_player = self.player_list[idx].name

    def init_game(self):
        self.alive_players = [player_i.name for player_i in self.player_list]
        # reset
        self.assassin_kill = True

        return

    def run_round(self, game_round):
        task_confirm = False
        task_members_idx = []
        retry = 0
        max_retry = 5
        if self.language == 'chinese':
            print_text_animated(
                Fore.YELLOW + f"\n{'='*60}\n[第 {game_round} 轮] 开始组队阶段（最多 {max_retry} 次尝试）\n{'='*60}\n\n")
        else:
            print_text_animated(
                Fore.YELLOW + f"\n{'='*60}\n[Round {game_round}] Starting team selection phase (max {max_retry} attempts)\n{'='*60}\n\n")
        while not task_confirm and retry < max_retry:
            if self.language == 'chinese':
                print_text_animated(
                    Fore.YELLOW + f"\n--- [第 {game_round} 轮] 第 {retry + 1}/{max_retry} 次投票 ---\n\n")
            else:
                print_text_animated(
                    Fore.YELLOW + f"\n--- [Round {game_round}] Vote attempt {retry + 1}/{max_retry} ---\n\n")
    # Public discussion
            self.discuss()

    # Select quest team members
            task_members_idx = self.select()
            if len(task_members_idx) < self.task_member.get(str(self.game_round)):
                retry += 1

    # Public vote
            task_confirm = self.vote(task_members_idx)
            if not task_confirm:
                if self.language == 'chinese':
                    print_text_animated(
                        Fore.YELLOW + f"\n[第 {game_round} 轮] 第 {retry + 1}/{max_retry} 次投票被否决。"
                        f"{'重新投票...' if retry + 1 < max_retry else '已达最大尝试次数，强制执行任务。'}\n\n")
                else:
                    print_text_animated(
                        Fore.YELLOW + f"\n[Round {game_round}] Vote attempt {retry + 1}/{max_retry} REJECTED. "
                        f"{'Retrying...' if retry + 1 < max_retry else 'Max attempts reached, forcing quest execution.'}\n\n")
            retry += 1

    # Execute quest
        # If team size is insufficient, randomly select to keep the game going
        if len(task_members_idx) < self.task_member.get(str(self.game_round)):
            if self.language == 'chinese':
                print_text_animated(
                    Fore.YELLOW + f"\n[第 {game_round} 轮] 队伍人数不足，随机选择成员。\n\n")
            else:
                print_text_animated(
                    Fore.YELLOW + f"\n[Round {game_round}] Team size insufficient, randomly selecting members.\n\n")
            task_members_idx = random.sample(
                ['1', '2', '3', '4', '5', '6'], k=self.task_member.get(str(self.game_round))
            )
        if self.language == 'chinese':
            print_text_animated(
                Fore.YELLOW + f"\n{'='*60}\n[第 {game_round} 轮] 执行任务，队伍成员："
                f"{['玩家 ' + str(idx) for idx in task_members_idx]}\n{'='*60}\n\n")
        else:
            print_text_animated(
                Fore.YELLOW + f"\n{'='*60}\n[Round {game_round}] Executing quest with team: "
                f"{['player ' + str(idx) for idx in task_members_idx]}\n{'='*60}\n\n")
        task_success = self.execute(task_members_idx)

    # Assassin identifies Merlin
        if self.assassin_kill:
            kill, kill_success = self.assassinate()
            if kill:
                self.assassin_kill = False

            if kill and kill_success:
                if self.language == 'chinese':
                    win_msg = "红色阵营获胜。"
                else:
                    win_msg = "Evil Side wins."
                self.process_list.append(
                    {"Host": win_msg}
                )
                for player_i in self.alive_players:
                    self.players[player_i].receive(
                        name="Host", message=f"quest phase, round {self.game_round}|{win_msg}")
                self.winners = ["Morgana", "Assassin"]
                # Game ends
                return False

        self.handle_round_ending(task_success)
        if self.check_game_end():
            if self.good_score >= 3:
                if self.language == 'chinese':
                    win_msg = "蓝色阵营获胜。"
                else:
                    win_msg = "Good Side wins."
                self.process_list.append(
                    {"Host": win_msg}
                )
                for player_i in self.alive_players:
                    self.players[player_i].receive(
                        name="Host", message=f"quest phase, round {self.game_round}|{win_msg}")
                self.winners = ["Merlin", "Loyal Servant", "Percival"]
            else:
                if self.language == 'chinese':
                    win_msg = "红色阵营获胜。"
                else:
                    win_msg = "Evil Side wins."
                self.process_list.append(
                    {"Host": win_msg}
                )
                for player_i in self.alive_players:
                    self.players[player_i].receive(
                        name="Host", message=f"quest phase, round {self.game_round}|{win_msg}")
                self.winners = ["Morgana", "Assassin"]
        return not self.check_game_end()

    def night_process(self):
        """
        Simplified night process:
        - Does not call receive(), directly sets night info into player's system prompt
        - Completely avoids API calls; info is passed via system prompt on first step()
        """
        # Select print messages based on language setting
        if self.language == 'chinese':
            night_start_msg = "[夜晚阶段] 分发角色信息..."
            night_end_msg = "[夜晚阶段结束] 所有玩家睁眼，游戏开始！"
        else:
            night_start_msg = "[Night Phase] Distributing role information..."
            night_end_msg = "[Night Phase Ends] All players open their eyes, the game begins!"
        
        print_text_animated(Fore.WHITE + f"Host:\n\n{night_start_msg}\n\n")
        
        # Set night info for each player
        for player_i in self.alive_players:
            role = self.player_mapping[player_i]
            night_info = ""
            
            if role in self.evil_side:
            # Evil side: knows teammates
                other_evil_players = []
                for player_j, role_j in self.player_mapping.items():
                    if role_j in self.evil_side and player_j != player_i:
                        other_evil_players.append(player_j)
                night_info = f"[Night Phase Info] You are on the EVIL side. Your evil teammate(s): {', '.join(other_evil_players)}. Work together to sabotage quests and identify Merlin."
                
            elif role == self.role_mapping['merlin']:
                # Merlin: sees all evil players
                evil_players = [player_j for player_j, role_j in self.player_mapping.items() if role_j in self.evil_side]
                night_info = f"[Night Phase Info] As Merlin, you can see the evil players: {', '.join(evil_players)}. You don't know their specific roles. Guide your team subtly without revealing yourself to the Assassin."
                
            elif role == self.role_mapping['percival']:
                # Percival: sees Merlin and Morgana (but doesn't know which is which)
                morgana_player = None
                for player_x, role_x in self.player_mapping.items():
                    if role_x == self.role_mapping['morgana']:
                        morgana_player = player_x
                        break
                candidates = [self.merlin_player, morgana_player]
                random.shuffle(candidates)  # Shuffle order
                night_info = f"[Night Phase Info] As Percival, you can see {candidates[0]} and {candidates[1]} - one is Merlin, one is Morgana, but you don't know which is which. Try to identify the real Merlin and protect them."
                
            else:
                # Loyal Servant: no special information
                night_info = "[Night Phase Info] As a Loyal Servant, you have no special information from the night phase. Use discussion and voting patterns to identify evil players."
            
            # Set night info for the player
            self.players[player_i].set_night_info(night_info)
            print_text_animated(Fore.WHITE + f"  {player_i}({role}): Night info set.\n")
        
        print_text_animated(Fore.WHITE + f"Host:\n\n{night_end_msg}\n\n")

    def start(self):
        """
        start werewolf game
        :return:
        """
        self.init_game()
        self.night_process()
        game_continue = True
        game_round = 1
        process_json = {}
        try:
            while game_continue:
                self.game_round = game_round
                instruction = f'round {game_round} starts:'
                if self.language == 'chinese':
                    print_text_animated(Fore.WHITE + f"系统:\n\n第 {game_round} 轮开始：\n\n")
                else:
                    print_text_animated(Fore.WHITE + f"System:\n\n{instruction}\n\n")
                game_continue = self.run_round(game_round)
                # if game_continue:
                #     self.agents_summary()
                process_json[instruction] = self.process_list
                write_json(process_json, f'{self.output_dir}/process.json')
                self.process_list = []
                if game_round >= 5:
                    game_continue = False
                game_round += 1
        except Exception as e:
            instruction = f'round {game_round} starts:'
            process_json[instruction] = self.process_list
            write_json(process_json, f'{self.output_dir}/process.json')
            raise e

    def discuss(self):
        discuss_prompt = self.host_instruction.get('discuss_prompt', '')
        discuss_order = copy.deepcopy(self.alive_players)
        # Determine speaking order
        if self.task_leader is None:
            self.task_leader = random.choice(discuss_order)
        else:
            idx = discuss_order.index(self.task_leader)
            idx = (idx + 1) % len(discuss_order)
            self.task_leader = discuss_order[idx]
        while discuss_order[0] != self.task_leader:
            player_i = discuss_order.pop(0)
            discuss_order.append(player_i)

        discuss_prompt2 = self.host_instruction.get('discuss_prompt2', '')
        res_rule = copy.deepcopy(self.response_rule.get('discuss_prompt2', {}))
        res_rule['count'] = res_rule.get('count', '').format(self.task_member.get(str(self.game_round)))
        print_text_animated(
            Fore.WHITE + f"Host:\n\n{discuss_prompt.format(self.game_round, self.task_member.get(str(self.game_round)), '、'.join(discuss_order), discuss_order[0])}\n\n")
        for idx, player_i in enumerate(discuss_order):
            instruction = discuss_prompt.format(self.game_round, self.task_member.get(str(self.game_round)),
                                                '、'.join(discuss_order),
                                                discuss_order[0]) + discuss_prompt2.format(player_i)
            
            # Intent Identification: identify desired and undesired responses from the next player
            intent_info = None
            if self.enable_intent_identification and idx < len(discuss_order) - 1:
                # Next player to speak
                next_player = discuss_order[idx + 1]
                # Call agent's intent identification method
                intent_info = self.players[player_i].identify_intent(next_player)
            
            output = self.players[player_i].step(message=f"quest phase, round {self.game_round}|" + instruction)
            print_text_animated(COLOR[player_i] + f"{player_i}({self.player_mapping[player_i]}):\n\n{output}\n\n")

            # Build log entry
            log_entry = {
                'Host': instruction,
                f"{player_i}({self.player_mapping[player_i]})": output,
                "response_rule": res_rule
            }
            
            # If intent identification is enabled, add to log
            if intent_info is not None:
                log_entry["intent_identification"] = intent_info
            
            self.process_list.append(log_entry)
            for player_j in self.alive_players:
                if player_i == player_j:
                    continue
                else:
                    self.players[player_j].receive("host", f"quest phase, round {self.game_round}|" + instruction)
                    self.players[player_j].receive(player_i, f"quest phase, round {self.game_round}|" + output)

    def select(self):
        task_members_idx = []
        max_ask = 3
        retry = 0
        task_member_decision = self.host_instruction.get('task_member_decision', '')
        res_rule = self.response_rule.get('task_member_decision', {})
        res_rule['count'] = res_rule.get('count', '').format(self.task_member.get(str(self.game_round)))

        instruction = task_member_decision.format(self.task_member.get(str(self.game_round)))
        while len(task_members_idx) < self.task_member.get(str(self.game_round)) and retry < max_ask:
            output = self.players[self.task_leader].step(f"quest phase, round {self.game_round}|" + instruction)
            # player extractor
            if self.player_extractor is not None:
                s = self.player_extractor.step(
                    f"Question：{instruction}\nAnswer：{output}")
            else:
                s = output
            pattern = r'\d+'
            task_members_idx = re.findall(pattern, s)
            # Check for out-of-range player indices
            illegal_idx = []
            for idx in task_members_idx:
                if int(idx) > 6:
                    illegal_idx.append(idx)
            for idx in illegal_idx:
                task_members_idx.remove(idx)

            if len(task_members_idx) >= self.task_member.get(str(self.game_round)):
                print_text_animated(COLOR[
                                        self.task_leader] + f"{self.task_leader}({self.player_mapping[self.task_leader]}):\n\n{output}\n\n")
                self.process_list.append(
                    {'Host': instruction,
                     f"{self.task_leader}({self.player_mapping[self.task_leader]})": output,
                     "response_rule": res_rule})
            retry += 1
        if len(task_members_idx) > self.task_member.get(str(self.game_round)):
            task_members_idx = task_members_idx[:self.task_member.get(str(self.game_round))]
        return task_members_idx

    def vote(self, task_members_idx):
        vote_step = self.host_instruction.get('vote_step', '')
        res_rule = self.response_rule.get('vote_step', {})
        task_member = [f"player {idx}" for idx in task_members_idx]
        instruction = vote_step.format('、'.join(task_member))
        print_text_animated(
            Fore.WHITE + f"Host:\n\n{instruction}\n\n")

        all_votes = []
        all_vote_mapping = {}
        for player_i in self.alive_players:
            output = self.players[player_i].step(f"quest phase, round {self.game_round}|" + instruction)
            if self.vote_extractor is not None:
                s = self.vote_extractor.step(
                    f"Question: {instruction}Answer: {output}"
                )
                # If no clear stance, assume agreement to keep the game progressing
                vote = 'false' in s or 'False' in s
            else:
                pattern = '反对'
                vote = re.findall(pattern, output)
            if vote:
                all_votes.append(False)
                all_vote_mapping[player_i] = "disagree"
            else:
                all_votes.append(True)
                all_vote_mapping[player_i] = "agree"
            self.process_list.append(
                {'Host': instruction,
                 f"{player_i}({self.player_mapping[player_i]})": output,
                 "response_rule": res_rule})

            print_text_animated(COLOR[player_i] + f"{player_i}({self.player_mapping[player_i]}):\n\n{output}\n\n")
            # for player_j in self.alive_players:
            #     if player_i == player_j:
            #         continue
            #     else:
            #         self.players[player_j].receive("host", f"mission phase, round {self.game_round}|" + instruction)
            #         self.players[player_j].receive(player_i, f"mission phase, round {self.game_round}|" + output)
        vote_summary = self.host_instruction.get("vote_summary", "")
        instruction = vote_summary.format(','.join([f"{player_i}: {v}" for player_i, v in all_vote_mapping.items()]))
        for player_i in self.alive_players:
            self.players[player_i].receive("host", f"quest phase, round {self.game_round}|" + instruction)
        agree_count = all_votes.count(True)
        disagree_count = all_votes.count(False)
        task_confirm = agree_count / len(all_votes) >= 1 / 2
        if self.language == 'chinese':
            vote_status = "通过" if task_confirm else "否决"
            print_text_animated(
                Fore.YELLOW + f"\n[投票结果] {vote_status} — 同意：{agree_count}，反对：{disagree_count} "
                f"({', '.join(f'{p}:{v}' for p, v in all_vote_mapping.items())})\n\n")
        else:
            vote_status = "APPROVED" if task_confirm else "REJECTED"
            print_text_animated(
                Fore.YELLOW + f"\n[Vote Result] {vote_status} — Agree: {agree_count}, Disagree: {disagree_count} "
                f"({', '.join(f'{p}:{v}' for p, v in all_vote_mapping.items())})\n\n")
        if not task_confirm:
            instruction = self.host_instruction.get("vote_again", "")
            for player_i in self.alive_players:
                self.players[player_i].receive("host", f"quest phase, round {self.game_round}|" + instruction)
        return task_confirm

    def execute(self, task_members_idx):
        execute_step = self.host_instruction.get('execute_step', '')
        res_rule = self.response_rule.get('execute_step', {})
        instruction = execute_step

        task_success = True
        all_vote = []
        for player_idx in task_members_idx:
            player_i = f"player {player_idx}"
            output = self.players[f"player {player_idx}"].step(f"quest phase, round {self.game_round}|" + instruction)
            print_text_animated(COLOR[player_i] + f"{player_i}({self.player_mapping[player_i]}):\n\n{output}\n\n")

            if self.quest_extractor is not None:
                s = self.quest_extractor.step(
                    f"Question: {instruction}\nAnswer: {output}"
                )
                # If no clear stance, assume quest failure (favors evil side)
                vote = 'false' in s or 'False' in s
            else:
                pattern = '失败'
                vote = re.findall(pattern, output)
            if vote:
                task_success = False
                all_vote.append(False)
            else:
                all_vote.append(True)
            self.process_list.append(
                {'Host': instruction,
                 f"player {player_idx}({self.player_mapping[f'player {player_idx}']})": output,
                 "response_rule": res_rule})
        if task_success:
            task_step = self.host_instruction.get('task_success', '')
        else:
            task_step = self.host_instruction.get('task_fail', '').format(
                f"{all_vote.count(False)} players" if all_vote.count(False) > 1 else f"{all_vote.count(False)} player")
        instruction = task_step
        print_text_animated(Fore.WHITE + f"Host:\n\n{instruction}\n\n")
        for player_i in self.alive_players:
            self.players[player_i].receive(name='host',
                                           message=f"quest phase, round {self.game_round}|" + instruction)
        self.process_list.append({'Host': instruction})
        return task_success

    def assassinate(self):
        kill_step = self.host_instruction.get('kill_step1', '')
        res_rule = self.response_rule.get('kill_step1', {})
        instruction = kill_step
        output = self.players[self.assassin_player].step(f"quest phase, round {self.game_round}|" + instruction)
        self.process_list.append(
            {'Host': instruction,
             f"{self.assassin_player}({self.player_mapping[self.assassin_player]})": output,
             "response_rule": res_rule})
        # bool extractor
        if self.choose_identify_extractor is not None:
            s = self.choose_identify_extractor.step(
                f"Question: {instruction}Answer: {output}"
            )
            kill = 'true' in s or 'True' in s
        else:
            match = re.search('是|否|yes|no', output)
            ans = match.group() if match else ''
            kill = True if ans in ['是', 'yes'] else False
        if kill:
            kill_step = self.host_instruction.get('kill_step2', '')
            res_rule = self.response_rule.get('kill_step2', {})
            instruction = kill_step
            output = self.players[self.assassin_player].step(f"quest phase, round {self.game_round}|" + instruction)
            self.process_list.append(
                {'Host': instruction,
                 f"{self.assassin_player}({self.player_mapping[self.assassin_player]})": output,
                 "response_rule": res_rule})
            # player extractor
            if self.select_merlin_extractor is not None:
                s = self.select_merlin_extractor.step(
                    f"Question: {instruction}\nAnswer: {output}")
            else:
                s = output
            kill_player_idx = re.findall(r'\d+', s)
            if not kill_player_idx:
                # Treat as no assassination attempt
                return False, False
            else:
                kill_success = True if self.merlin_player == f"player {kill_player_idx[0]}" else False
                if kill_success:
                    instruction = self.host_instruction.get('kill_success', '').format(self.assassin_player,
                                                                                       f"player {kill_player_idx[0]}")
                else:
                    instruction = self.host_instruction.get('kill_fail', '').format(self.assassin_player,
                                                                                    f"player {kill_player_idx[0]}")
                print_text_animated(Fore.WHITE + f"Host:\n\n{instruction}\n\n")
                for player_i in self.alive_players:
                    self.players[player_i].receive(name='host',
                                                   message=f"quest phase, round {self.game_round}|" + instruction)
                self.process_list.append({'Host': instruction})
                return True, kill_success
        else:
            return False, False

    def handle_round_ending(self, task_success):
        if task_success:
            self.good_score += 1
        else:
            self.evil_score += 1
        if self.language == 'chinese':
            result_str = "成功（蓝方 +1）" if task_success else "失败（红方 +1）"
            print_text_animated(
                Fore.YELLOW + f"\n{'='*60}\n"
                f"[第 {self.game_round} 轮结果] 任务{result_str}\n"
                f"[比分] 蓝方：{self.good_score} | 红方：{self.evil_score}\n"
                f"{'='*60}\n\n")
        else:
            result_str = "SUCCESS (Good +1)" if task_success else "FAILED (Evil +1)"
            print_text_animated(
                Fore.YELLOW + f"\n{'='*60}\n"
                f"[Round {self.game_round} Result] Quest {result_str}\n"
                f"[Score] Good: {self.good_score} | Evil: {self.evil_score}\n"
                f"{'='*60}\n\n")

    def check_game_end(self):
        return max(self.good_score, self.evil_score) >= 3
