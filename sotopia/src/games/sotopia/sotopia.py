#!/usr/bin/env python
# encoding: utf-8
"""
Sotopia Social Simulation Engine.
Implements a lightweight, self-contained version of the Sotopia environment
for two-agent social interaction scenarios.
"""
import json
import os
import random
import re
import time
from typing import Dict, List, Optional, Tuple, Any

from colorama import Fore, Style

from src.utils import print_text_animated, create_dir, write_json


# Evaluation dimension score ranges
DIMENSION_RANGES = {
    "believability": (0, 10),
    "relationship": (-5, 5),
    "knowledge": (0, 10),
    "secret": (-10, 0),
    "social_rules": (-10, 0),
    "financial_and_material_benefits": (-5, 5),
    "goal": (0, 10),
}

RELATIONSHIP_TYPES = {
    0: "strangers",
    1: "know each other by name",
    2: "acquaintances",
    3: "friends",
    4: "in a romantic relationship",
    5: "family members",
}


class AgentProfile:
    """Represents a character profile for an agent."""

    def __init__(self, data: dict):
        self.pk = data.get("pk", "")
        self.first_name = data.get("first_name", "")
        self.last_name = data.get("last_name", "")
        self.age = data.get("age", 0)
        self.occupation = data.get("occupation", "")
        self.gender = data.get("gender", "")
        self.gender_pronoun = data.get("gender_pronoun", "")
        self.public_info = data.get("public_info", "")
        self.personality_and_values = data.get("personality_and_values", "")
        self.secret = data.get("secret", "")
        self.big_five = data.get("big_five", "")
        self.mbti = data.get("mbti", "")
        self.decision_making_style = data.get("decision_making_style", "")

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}".strip()

    def to_natural_language(self) -> str:
        """Convert profile to a natural language description."""
        parts = []
        parts.append(f"{self.full_name} is a {self.age}-year-old {self.gender.lower()} {self.occupation}.")
        if self.gender_pronoun:
            parts.append(f"Pronouns: {self.gender_pronoun}.")
        if self.public_info:
            parts.append(self.public_info)
        if self.personality_and_values:
            parts.append(f"Personality: {self.personality_and_values}")
        return " ".join(parts)

    def to_character_summary(self) -> str:
        """Short character summary for prompts."""
        return (
            f"{self.full_name}, {self.age}-year-old {self.gender.lower()} {self.occupation}. "
            f"{self.personality_and_values}"
        )

    def to_dict(self) -> dict:
        return {
            "pk": self.pk,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "age": self.age,
            "occupation": self.occupation,
            "gender": self.gender,
            "gender_pronoun": self.gender_pronoun,
            "public_info": self.public_info,
            "personality_and_values": self.personality_and_values,
            "secret": self.secret,
        }


class EnvironmentProfile:
    """Represents a social interaction scenario."""

    def __init__(self, data: dict):
        self.pk = data.get("pk", "")
        self.codename = data.get("codename", "")
        self.source = data.get("source", "")
        self.scenario = data.get("scenario", "")
        self.agent_goals = data.get("agent_goals", ["", ""])
        self.relationship = data.get("relationship", 0)
        self.age_constraint = data.get("age_constraint")
        self.occupation_constraint = data.get("occupation_constraint")

    @property
    def relationship_str(self) -> str:
        return RELATIONSHIP_TYPES.get(self.relationship, "strangers")

    def to_dict(self) -> dict:
        return {
            "pk": self.pk,
            "codename": self.codename,
            "scenario": self.scenario,
            "agent_goals": self.agent_goals,
            "relationship": self.relationship,
        }


class SotopiaGame:
    """
    Sotopia Social Simulation Engine.

    Manages a two-agent social interaction:
    1. Load environment (scenario) and agent profiles
    2. Run multi-turn dialogue
    3. Evaluate using LLM judge
    """

    def __init__(
        self,
        mode: str = "watch",
        ai_model: str = "gpt-4o",
        output_dir: str = "logs/sotopia",
        max_turns: int = 20,
        max_stale_turns: int = 2,
        enable_intent_identification: bool = False,
        evaluator_model: str = None,
    ):
        self.mode = mode
        self.ai_model = ai_model
        self.output_dir = output_dir
        self.max_turns = max_turns
        self.max_stale_turns = max_stale_turns
        self.enable_intent_identification = enable_intent_identification
        self.evaluator_model = evaluator_model or ai_model

        # Game state
        self.env_profile: Optional[EnvironmentProfile] = None
        self.agent_profiles: List[AgentProfile] = []
        self.agents: Dict[str, Any] = {}  # name -> agent instance
        self.agent_names: List[str] = []
        self.agent_goals: List[str] = []

        # Conversation history
        self.conversation: List[Dict[str, str]] = []
        self.process_log: Dict[str, Any] = {}

        # Results
        self.evaluation_results: Optional[Dict] = None

        create_dir(output_dir)

    def load_scenario(
        self,
        env_profile: EnvironmentProfile,
        agent_profiles: List[AgentProfile],
    ):
        """Load a scenario with environment and agent profiles."""
        self.env_profile = env_profile
        self.agent_profiles = agent_profiles
        self.agent_names = [p.full_name for p in agent_profiles]
        self.agent_goals = env_profile.agent_goals[:len(agent_profiles)]

        # Pad goals if needed
        while len(self.agent_goals) < len(agent_profiles):
            self.agent_goals.append("Engage naturally in the conversation.")

    def add_agents(self, agent_args: List[Tuple]):
        """
        Add agent instances.

        Args:
            agent_args: List of (AgentClass, kwargs) tuples.
        """
        for i, (agent_cls, kwargs) in enumerate(agent_args):
            agent = agent_cls.init_instance(**kwargs)
            name = self.agent_names[i] if i < len(self.agent_names) else f"Agent {i+1}"
            self.agents[name] = agent

    def _format_conversation_for_display(self) -> str:
        """Format conversation history for display."""
        lines = []
        for entry in self.conversation:
            speaker = entry.get("speaker", "")
            message = entry.get("message", "")
            action_type = entry.get("action_type", "speak")
            if action_type == "leave":
                lines.append(f"[{speaker} left the conversation]")
            elif action_type == "none":
                lines.append(f"[{speaker} did nothing]")
            else:
                lines.append(f"{speaker}: {message}")
        return "\n".join(lines)

    def _format_conversation_for_agent(self, agent_idx: int) -> str:
        """Format conversation history from an agent's perspective."""
        lines = []
        for entry in self.conversation:
            speaker = entry.get("speaker", "")
            message = entry.get("message", "")
            action_type = entry.get("action_type", "speak")
            if action_type == "leave":
                lines.append(f"[{speaker} left the conversation]")
            elif action_type == "none":
                lines.append(f"[{speaker} did nothing]")
            else:
                lines.append(f"{speaker}: {message}")
        return "\n".join(lines)

    def _check_termination(self) -> Tuple[bool, str]:
        """Check if the conversation should terminate."""
        # Rule 1: Max turns reached
        if len(self.conversation) >= self.max_turns * 2:  # 2 agents per turn
            return True, "Maximum turns reached."

        # Rule 2: Agent left
        for entry in self.conversation[-2:]:
            if entry.get("action_type") == "leave":
                return True, f"{entry['speaker']} left the conversation."

        # Rule 3: Stale conversation (consecutive "none" actions)
        stale_count = 0
        for entry in reversed(self.conversation):
            if entry.get("action_type") == "none":
                stale_count += 1
            else:
                break
        if stale_count >= self.max_stale_turns * 2:
            return True, "Conversation became stale."

        return False, ""

    def _parse_agent_response(self, response: str) -> Tuple[str, str]:
        """
        Parse agent response to extract action type and message.

        Returns:
            (action_type, message)
        """
        response = response.strip()

        # Check for leave action
        leave_patterns = [
            r"\[leave\]",
            r"\[left the conversation\]",
            r"I need to go",
            r"goodbye",
            r"I have to leave",
        ]
        for pattern in leave_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                # Only mark as leave if it's clearly an exit
                if any(kw in response.lower() for kw in ["leave", "goodbye", "have to go"]):
                    return "leave", response

        # Default: speak action
        return "speak", response

    def start(self):
        """Run the social interaction simulation."""
        if not self.env_profile or not self.agent_profiles or not self.agents:
            raise ValueError("Must load scenario and add agents before starting.")

        agent_list = list(self.agents.values())
        agent_names = list(self.agents.keys())

        # Initialize process log
        self.process_log = {
            "scenario": self.env_profile.to_dict(),
            "agents": [p.to_dict() for p in self.agent_profiles],
            "agent_goals": self.agent_goals,
            "conversation": [],
            "evaluation": None,
        }

        if self.mode == "watch":
            print(f"\n{Fore.YELLOW}{'='*60}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Scenario: {self.env_profile.codename}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}{'='*60}{Style.RESET_ALL}")
            print(f"\n{Fore.CYAN}{self.env_profile.scenario}{Style.RESET_ALL}\n")
            for i, (name, profile) in enumerate(zip(agent_names, self.agent_profiles)):
                print(f"{Fore.GREEN}{name}: {profile.to_character_summary()}{Style.RESET_ALL}")
                print(f"  Goal: {self.agent_goals[i][:100]}...")
            print(f"\nRelationship: {self.env_profile.relationship_str}")
            print(f"{Fore.YELLOW}{'='*60}{Style.RESET_ALL}\n")

        # Set agent context (scenario info, goals, etc.)
        for i, (name, agent) in enumerate(self.agents.items()):
            agent.set_scenario_info(
                scenario=self.env_profile.scenario,
                relationship=self.env_profile.relationship_str,
                goal=self.agent_goals[i],
                character_summary=self.agent_profiles[i].to_character_summary(),
                character_profile=self.agent_profiles[i].to_natural_language(),
                other_name=agent_names[1 - i] if len(agent_names) == 2 else "the other person",
            )

        # Opening message from the environment
        opening = (
            f"The scenario begins. {self.env_profile.scenario} "
            f"The two participants are {agent_names[0]} and {agent_names[1]}. "
            f"They are {self.env_profile.relationship_str}."
        )

        if self.mode == "watch":
            print(f"{Fore.MAGENTA}[Narrator]: {opening}{Style.RESET_ALL}\n")

        # Main conversation loop (round-robin)
        turn = 0
        speaking_order = list(range(len(agent_list)))

        while True:
            for agent_idx in speaking_order:
                agent = agent_list[agent_idx]
                name = agent_names[agent_idx]

                # Build the conversation summary for this agent
                conv_summary = self._format_conversation_for_agent(agent_idx)

                # Build the question/prompt for the agent
                if len(self.conversation) == 0:
                    question = (
                        f"The conversation is starting. You are {name}. "
                        f"Please begin the interaction naturally based on the scenario."
                    )
                else:
                    last_speaker = self.conversation[-1]["speaker"]
                    last_msg = self.conversation[-1]["message"]
                    question = (
                        f"{last_speaker} just said: \"{last_msg}\"\n"
                        f"It's your turn to respond as {name}."
                    )

                # Get agent response
                try:
                    response = agent.step(
                        summary=conv_summary,
                        question=question,
                        turn=turn,
                    )
                except Exception as e:
                    print(f"Error getting response from {name}: {e}")
                    response = "I'm not sure what to say."

                # Parse response
                action_type, message = self._parse_agent_response(response)

                # Record conversation entry
                entry = {
                    "speaker": name,
                    "message": message,
                    "action_type": action_type,
                    "turn": turn,
                    "agent_idx": agent_idx,
                }

                # Intent identification (for training data collection)
                if self.enable_intent_identification and action_type == "speak":
                    intent_info = self._identify_intent(
                        agent_idx, name, message, conv_summary, turn
                    )
                    if intent_info:
                        entry["intent_identification"] = intent_info

                self.conversation.append(entry)
                self.process_log["conversation"].append(entry)

                # Display
                if self.mode == "watch":
                    color = Fore.BLUE if agent_idx == 0 else Fore.RED
                    if action_type == "leave":
                        print_text_animated(
                            f"{color}[{name} left the conversation]{Style.RESET_ALL}\n",
                            delay=0.005,
                        )
                    else:
                        print_text_animated(
                            f"{color}{name}: {message}{Style.RESET_ALL}\n",
                            delay=0.005,
                        )

                # Check termination
                terminated, reason = self._check_termination()
                if terminated:
                    if self.mode == "watch":
                        print(f"\n{Fore.YELLOW}[Conversation ended: {reason}]{Style.RESET_ALL}")
                    break

            if terminated:
                break
            turn += 1

        # Save process log
        self._save_process_log()

    def _identify_intent(
        self,
        speaker_idx: int,
        speaker_name: str,
        utterance: str,
        conv_summary: str,
        turn: int,
    ) -> Optional[Dict]:
        """Generate desired/undesired follower responses for intent identification."""
        agent_list = list(self.agents.values())
        agent_names = list(self.agents.keys())

        # The follower is the next speaker
        follower_idx = 1 - speaker_idx  # For 2-agent setup
        follower = agent_list[follower_idx]
        follower_name = agent_names[follower_idx]

        try:
            intent_info = follower.identify_intent(
                speaker_name=speaker_name,
                speaker_utterance=utterance,
                follower_name=follower_name,
                follower_goal=self.agent_goals[follower_idx],
                follower_profile=self.agent_profiles[follower_idx].to_natural_language(),
                scenario=self.env_profile.scenario,
                conv_summary=conv_summary,
                k=3,
            )
            return intent_info
        except Exception as e:
            print(f"Intent identification failed: {e}")
            return None

    def evaluate(self, evaluator_api_key: str, evaluator_api_base: str = None):
        """
        Evaluate the conversation using an LLM judge.

        Returns evaluation scores for each agent across 7 dimensions.
        """
        from prompt.sotopia_prompts import evaluation_prompt
        from src.apis.chatgpt_api import chatgpt as _chatgpt_raw
        def chatgpt_api(messages, model, temperature, api_key=None, api_base=None):
            return _chatgpt_raw(model, messages, temperature, api_key=api_key, api_base=api_base)

        conversation_text = self._format_conversation_for_display()

        eval_input = evaluation_prompt.format(
            scenario=self.env_profile.scenario,
            agent1_name=self.agent_names[0],
            agent1_profile=self.agent_profiles[0].to_natural_language(),
            agent1_goal=self.agent_goals[0],
            agent2_name=self.agent_names[1],
            agent2_profile=self.agent_profiles[1].to_natural_language(),
            agent2_goal=self.agent_goals[1],
            conversation=conversation_text,
        )

        messages = [
            {"role": "system", "content": "You are an expert evaluator for social interactions. Always respond with valid JSON."},
            {"role": "user", "content": eval_input},
        ]

        response = chatgpt_api(
            messages,
            model=self.evaluator_model,
            temperature=0,
            api_key=evaluator_api_key,
            api_base=evaluator_api_base,
        )

        # Parse evaluation response
        try:
            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                eval_data = json.loads(json_match.group())
            else:
                eval_data = json.loads(response)

            self.evaluation_results = eval_data
            self.process_log["evaluation"] = eval_data

            # Compute overall scores
            for agent_key in ["agent_1", "agent_2"]:
                if agent_key in eval_data:
                    scores = []
                    for dim, (low, high) in DIMENSION_RANGES.items():
                        if dim in eval_data[agent_key]:
                            raw_score = eval_data[agent_key][dim].get("score", 0)
                            # Normalize to 0-10 range for overall computation
                            normalized = (raw_score - low) / (high - low) * 10
                            scores.append(normalized)
                    if scores:
                        eval_data[agent_key]["overall_score"] = sum(scores) / len(scores)

            # Save updated process log
            self._save_process_log()

        except (json.JSONDecodeError, Exception) as e:
            print(f"Failed to parse evaluation response: {e}")
            self.evaluation_results = {"error": str(e), "raw_response": response}

        return self.evaluation_results

    def get_goal_scores(self) -> Tuple[float, float]:
        """Get goal scores for both agents."""
        if not self.evaluation_results:
            return 0.0, 0.0

        score1 = 0.0
        score2 = 0.0

        if "agent_1" in self.evaluation_results:
            goal_data = self.evaluation_results["agent_1"].get("goal", {})
            score1 = goal_data.get("score", 0) if isinstance(goal_data, dict) else 0

        if "agent_2" in self.evaluation_results:
            goal_data = self.evaluation_results["agent_2"].get("goal", {})
            score2 = goal_data.get("score", 0) if isinstance(goal_data, dict) else 0

        return float(score1), float(score2)

    def get_overall_scores(self) -> Tuple[float, float]:
        """Get overall scores for both agents."""
        if not self.evaluation_results:
            return 0.0, 0.0

        score1 = self.evaluation_results.get("agent_1", {}).get("overall_score", 0.0)
        score2 = self.evaluation_results.get("agent_2", {}).get("overall_score", 0.0)

        return float(score1), float(score2)

    def _save_process_log(self):
        """Save the process log to disk."""
        process_file = os.path.join(self.output_dir, "process.json")
        write_json(self.process_log, process_file)
