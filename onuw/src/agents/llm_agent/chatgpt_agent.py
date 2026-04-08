#!/usr/bin/env python
# encoding: utf-8
"""
LLM Agent implementations for ONUW (One Night Ultimate Werewolf).

Supported agent types:
- DirectAgent: Direct response generation (1 API call)
- ReActAgent: ReAct framework (Reasoning + Acting, 2 API calls)
- ReConAgent: Relation Consistency framework (cross-player relation analysis, 3 API calls)
- LASIAgent: LASI framework (Landscape Analysis - Strategy - Implementation, 4 API calls)
- BeliefAgent: Belief-based agent with belief modeling before response (2 API calls)
- LLMInsAgent: LLM-instructed speaking strategy selection with belief modeling (3 API calls)
- RefinerWrapper: Wraps any agent type with a trained Refiner model for persuasive utterance refinement
"""
import json
import re
import time
import logging
from typing import List, Optional, Callable

import openai

from ..abs_agent import Agent
from ..utils import write_json
from ...apis.chatgpt_api import chatgpt

logger = logging.getLogger(__name__)

try:
    OPENAI_MAX_TOKENS_ERROR = openai.error.InvalidRequestError
except AttributeError:
    OPENAI_MAX_TOKENS_ERROR = openai.BadRequestError


def extract_response(output: str) -> str:
    """
    Extract response content from LLM output.
    Supports multiple formats:
    1. <response>...</response> (complete closing tag)
    2. <response>... (unclosed tag, truncated output)
    3. my response is <...>
    4. Return raw text if no match found
    """
    # Try complete <response>...</response> tag first
    match = re.search(r"(?<=<response>).*?(?=</response>)", output, re.S)
    if match:
        return match.group().strip()
    
    # If no closing tag, extract all content after <response> (handles truncated output)
    match = re.search(r"<response>(.+)", output, re.S)
    if match:
        return match.group(1).strip()
    
    # Try "my response is <...>" format
    match = re.search(r"my response is\s*<(.+?)>", output, re.S | re.I)
    if match:
        return match.group(1).strip()
    
    # Try "my response is ..." format (without angle brackets)
    match = re.search(r"my response is\s*[:\-]?\s*(.+)", output, re.S | re.I)
    if match:
        return match.group(1).strip()
    
    # If nothing matched, return raw text
    return output.strip()


class BaseONUWAgent(Agent):
    """
    Base class for ONUW game agents.
    Provides common functionality: night info management, dialogue history, intent identification, etc.
    """
    
    # Intent identification prompt template
    INTENT_IDENTIFICATION_PROMPT = """You are {name}, playing as {role} in ONUW (One Night Ultimate Werewolf).
Your goal: {goal}

Current game state and dialogue history:
{context}

The next player to speak after you is {next_player}.

Based on your role and the current situation, identify:
1. THREE responses you would DESIRE the next player ({next_player}) to say (responses that would benefit your goals)
2. THREE responses you would NOT DESIRE the next player ({next_player}) to say (responses that would harm your goals)

Think strategically about what the next player might say and how it could affect the game.

Output your response in the following format:
<desired_responses>
1. [First desired response from the next player]
2. [Second desired response from the next player]
3. [Third desired response from the next player]
</desired_responses>
<undesired_responses>
1. [First undesired response from the next player]
2. [Second undesired response from the next player]
3. [Third undesired response from the next player]
</undesired_responses>"""

    def __init__(self, name: str, role: str, role_intro: str, game_goal: str, 
                 strategy: str, system_prompt: str, model: str, temperature: float,
                 api_key: str, output_dir: str, api_base: Optional[str] = None,
                 thinking_callback: Optional[Callable[[str, str], None]] = None,
                 enable_intent_identification: bool = False,
                 extra_body: Optional[dict] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.role = role
        self.introduction = role_intro
        self.game_goal = game_goal
        self.strategy = strategy
        self.system_prompt = system_prompt
        self.model = model
        self.temperature = temperature
        self.api_key = api_key
        self.api_base = api_base
        self.output_dir = output_dir
        self.extra_body = extra_body
        
        # Night phase information, merged into system prompt
        self.night_info = ""
        
        # Simplified dialogue history
        self.conversation_history = []
        
        # Current game phase
        self.phase = 0
        
        # Thinking process callback (for watch mode display)
        self.thinking_callback = thinking_callback
        
        # Whether to enable intent identification
        self.enable_intent_identification = enable_intent_identification
        
        # Store the most recent intent identification result
        self.last_intent = None

    def get_system_prompt_with_night_info(self) -> str:
        """Get system prompt with night phase information appended."""
        if self.night_info:
            return f"{self.system_prompt}\n\n{self.night_info}"
        return self.system_prompt

    def get_conversation_context(self) -> str:
        """Get a brief context summary of the current dialogue history."""
        if not self.conversation_history:
            return "None"
        recent = self.conversation_history[-20:]
        return "\n".join([f"{item['name']}: {item['message']}" for item in recent])

    def send_messages(self, messages: List[dict]) -> str:
        """Send messages to the LLM and return the response."""
        output = chatgpt(self.model, messages, self.temperature, 
                        api_key=self.api_key, api_base=self.api_base,
                        extra_body=self.extra_body)
        return output

    def receive(self, name: str, message: str) -> None:
        """Receive a message from another player."""
        temp_phase = message.split("|")[0]
        self.phase = temp_phase
        message = message.split("|")[1]
        self.conversation_history.append({"name": name, "message": message})

    def emit_thinking(self, stage: str, content: str):
        """Emit thinking process (for watch mode display)."""
        if self.thinking_callback:
            self.thinking_callback(stage, content)

    def reflection(self, player_role_mapping: dict, file_name: str, winners: list, duration: int):
        """Post-game reflection (no-op by default)."""
        pass

    @staticmethod
    def log(file, data):
        with open(file, mode='a+', encoding='utf-8') as f:
            f.write(data)

    def identify_intent(self, next_player: str) -> dict:
        """
        Intent Identification: identify desired and undesired responses from the next player.
        
        Args:
            next_player: Name of the next player to speak.
            
        Returns:
            dict: Contains desired_responses and undesired_responses.
        """
        if not self.enable_intent_identification:
            return None
        
        context = self.get_conversation_context()
        
        prompt = self.INTENT_IDENTIFICATION_PROMPT.format(
            name=self.name,
            role=self.role,
            goal=self.game_goal,
            context=context,
            next_player=next_player
        )
        
        messages = [
            {"role": 'system', "content": self.get_system_prompt_with_night_info()},
            {"role": 'user', "content": prompt}
        ]
        
        output = self.send_messages(messages)
        
        # Parse desired responses
        desired_responses = []
        desired_match = re.search(r"<desired_responses>(.+?)</desired_responses>", output, re.S)
        if desired_match:
            desired_text = desired_match.group(1)
            for line in desired_text.strip().split('\n'):
                line = line.strip()
                if line:
                    cleaned = re.sub(r'^\d+\.\s*', '', line)
                    if cleaned:
                        desired_responses.append(cleaned)
        
        # Parse undesired responses
        undesired_responses = []
        undesired_match = re.search(r"<undesired_responses>(.+?)</undesired_responses>", output, re.S)
        if undesired_match:
            undesired_text = undesired_match.group(1)
            for line in undesired_text.strip().split('\n'):
                line = line.strip()
                if line:
                    cleaned = re.sub(r'^\d+\.\s*', '', line)
                    if cleaned:
                        undesired_responses.append(cleaned)
        
        # Limit to first 3 (K=3)
        desired_responses = desired_responses[:3]
        undesired_responses = undesired_responses[:3]
        
        intent_result = {
            "desired_responses": desired_responses,
            "undesired_responses": undesired_responses
        }
        
        # Save to log
        self.log(f"{self.output_dir}/intent_identification.txt",
                 f"phase:{self.phase}\nnext_player:{next_player}\ninput:{prompt}\noutput:\n{output}\n" +
                 f"parsed_intent:{json.dumps(intent_result, ensure_ascii=False)}\n--------------------\n")
        
        self.emit_thinking("Intent Identification", 
                          f"Desired: {desired_responses}\nUndesired: {undesired_responses}")
        
        self.last_intent = intent_result
        return intent_result
    
    def get_last_intent(self) -> dict:
        """Get the most recent intent identification result."""
        return self.last_intent


class DirectAgent(BaseONUWAgent):
    """
    DirectAgent: Direct response generation.
    Simplest implementation with only 1 API call.
    """

    def __init__(self, response_prompt: str, **kwargs):
        super().__init__(**kwargs)
        self.response_prompt = response_prompt

    def step(self, message: str) -> str:
        temp_phase = message.split("|")[0]
        self.phase = temp_phase
        message = message.split("|")[1]

        context = self.get_conversation_context()
        
        # Generate response directly
        prompt = self.response_prompt.format(
            name=self.name, phase=self.phase, role=self.role, 
            introduction=self.introduction, strategy=self.strategy,
            summary=context, plan="None", question=message, actions="None"
        )
        
        messages = [
            {"role": 'system', "content": self.get_system_prompt_with_night_info()},
            {"role": 'user', "content": prompt}
        ]

        output = self.send_messages(messages)
        
        response = extract_response(output)
        
        self.log(f"{self.output_dir}/response.txt",
                 f"phase:{self.phase}\ninput:{prompt}\noutput:\n{output}\n--------------------")
        
        # Record dialogue history
        self.conversation_history.append({"name": "Host", "message": message})
        self.conversation_history.append({"name": self.name, "message": response})
        
        return response


class ReActAgent(BaseONUWAgent):
    """
    ReActAgent: ReAct framework (Reasoning + Acting).
    Think-act loop with 2 API calls.
    """

    def __init__(self, response_prompt: str, 
                 react_prompt: str = None, **kwargs):
        super().__init__(**kwargs)
        self.response_prompt = response_prompt
        # ReAct thinking prompt
        self.react_prompt = react_prompt or """You are {name}, playing as {role} in ONUW.
Current phase: {phase}
Your goal: {goal}
Your strategy: {strategy}

Conversation history:
{summary}

Current question/situation: {question}

Think step by step about:
1. Thought: What do I observe? What does this mean? Consider role swaps that may have occurred during the night.
2. Action: What should I do? (options: claim a role, accuse someone, defend, share information, deflect suspicion, stay neutral)
3. Observation: What response might this action get?

Provide your reasoning in this format:
<thinking>
Thought: [your analysis]
Action: [your chosen action]
Observation: [expected outcome]
</thinking>"""

    def step(self, message: str) -> str:
        temp_phase = message.split("|")[0]
        self.phase = temp_phase
        message = message.split("|")[1]

        context = self.get_conversation_context()
        
        # Step 1: ReAct thinking
        t_think_start = time.time()
        react_prompt = self.react_prompt.format(
            name=self.name, phase=self.phase, role=self.role,
            goal=self.game_goal, strategy=self.strategy,
            summary=context, question=message
        )
        
        messages = [
            {"role": 'system', "content": self.get_system_prompt_with_night_info()},
            {"role": 'user', "content": react_prompt}
        ]
        
        thinking = self.send_messages(messages)
        t_think = time.time() - t_think_start
        
        # Extract thinking content and display
        think_match = re.search("(?<=<thinking>).*?(?=</thinking>)", thinking, re.S)
        think_content = think_match.group().strip() if think_match else thinking
        self.emit_thinking("ReAct Thinking", think_content)
        
        self.log(f"{self.output_dir}/react_thinking.txt",
                 f"phase:{self.phase}\ninput:{react_prompt}\noutput:\n{thinking}\n--------------------")

        # Step 2: Generate response
        t_response_start = time.time()
        prompt = self.response_prompt.format(
            name=self.name, phase=self.phase, role=self.role,
            introduction=self.introduction, strategy=self.strategy,
            summary=context, plan=think_content, question=message, actions="Based on ReAct thinking"
        )
        
        messages = [
            {"role": 'system', "content": self.get_system_prompt_with_night_info()},
            {"role": 'user', "content": prompt}
        ]

        output = self.send_messages(messages)
        t_response = time.time() - t_response_start
        
        response = extract_response(output)
        
        self.log(f"{self.output_dir}/response.txt",
                 f"phase:{self.phase}\ninput:{prompt}\noutput:\n{output}\n--------------------")
        self.log(f"{self.output_dir}/time_cost.txt",
                 f"Think: {t_think}\nResponse: {t_response}\n")
        
        # Record dialogue history
        self.conversation_history.append({"name": "Host", "message": message})
        self.conversation_history.append({"name": self.name, "message": response})
        
        return response


class ReConAgent(BaseONUWAgent):
    """
    ReConAgent: Relation Consistency framework.
    Cross-player relation analysis with 3 API calls.
    
    Core idea:
    1. Analyze inter-player relationship consistency (who supports/suspects whom)
    2. Identify likely factions through relationship networks
    3. Make decisions based on relation analysis
    """

    def __init__(self, response_prompt: str,
                 relation_prompt: str = None,
                 consistency_prompt: str = None, **kwargs):
        super().__init__(**kwargs)
        self.response_prompt = response_prompt
        
        # Relation analysis prompt
        self.relation_prompt = relation_prompt or """You are {name}, playing as {role} in ONUW.
Current phase: {phase}

Analyze the relationships between players based on the conversation:
{summary}

For each player pair, identify:
1. Support relationships (who defended/agreed with whom)
2. Suspicion relationships (who accused/doubted whom)
3. Neutral relationships

Also consider role claims and night action claims. Note that roles may have been swapped during the night.

Output in format:
<relations>
- [Player A] -> [Player B]: [support/suspicion/neutral] (reason: [brief reason])
...
</relations>"""

        # Consistency analysis prompt
        self.consistency_prompt = consistency_prompt or """Based on the relationship analysis:
{relations}

Now check for consistency in the context of ONUW:
1. Role claim consistency: Do the claimed roles add up? (Only 1 Werewolf among players, 2 in center)
2. Night action consistency: Do the claimed night actions (Seer checks, Robber swaps, Troublemaker swaps) conflict?
3. Behavioral consistency: Are there contradictions in a player's statements?
4. Camp prediction: Based on relationship patterns and role claims, who is most likely the Werewolf?

Your role: {role}
Your goal: {goal}

<analysis>
Consistency check: [your analysis]
Suspected Werewolf: [player with confidence]
Recommended action: [what to do]
</analysis>"""

    def step(self, message: str) -> str:
        temp_phase = message.split("|")[0]
        self.phase = temp_phase
        message = message.split("|")[1]

        context = self.get_conversation_context()
        
        # Step 1: Relation analysis
        t_relation_start = time.time()
        relation_prompt = self.relation_prompt.format(
            name=self.name, phase=self.phase, role=self.role,
            summary=context
        )
        
        messages = [
            {"role": 'system', "content": self.get_system_prompt_with_night_info()},
            {"role": 'user', "content": relation_prompt}
        ]
        
        relations = self.send_messages(messages)
        t_relation = time.time() - t_relation_start
        
        # Extract relations and display
        rel_match = re.search("(?<=<relations>).*?(?=</relations>)", relations, re.S)
        rel_content = rel_match.group().strip() if rel_match else relations
        self.emit_thinking("Relation Analysis", rel_content)
        
        self.log(f"{self.output_dir}/relation_analysis.txt",
                 f"phase:{self.phase}\noutput:\n{relations}\n--------------------")

        # Step 2: Consistency analysis
        t_consist_start = time.time()
        consist_prompt = self.consistency_prompt.format(
            relations=rel_content, role=self.role, goal=self.game_goal
        )
        
        messages = [
            {"role": 'system', "content": self.get_system_prompt_with_night_info()},
            {"role": 'user', "content": consist_prompt}
        ]
        
        analysis = self.send_messages(messages)
        t_consist = time.time() - t_consist_start
        
        # Extract analysis and display
        analysis_match = re.search("(?<=<analysis>).*?(?=</analysis>)", analysis, re.S)
        analysis_content = analysis_match.group().strip() if analysis_match else analysis
        self.emit_thinking("Consistency Analysis", analysis_content)
        
        self.log(f"{self.output_dir}/consistency_analysis.txt",
                 f"phase:{self.phase}\noutput:\n{analysis}\n--------------------")

        # Step 3: Generate response
        t_response_start = time.time()
        prompt = self.response_prompt.format(
            name=self.name, phase=self.phase, role=self.role,
            introduction=self.introduction, strategy=self.strategy,
            summary=context, plan=analysis_content, question=message, 
            actions="Based on relation analysis"
        )
        
        messages = [
            {"role": 'system', "content": self.get_system_prompt_with_night_info()},
            {"role": 'user', "content": prompt}
        ]

        output = self.send_messages(messages)
        t_response = time.time() - t_response_start
        
        response = extract_response(output)
        
        self.log(f"{self.output_dir}/response.txt",
                 f"phase:{self.phase}\noutput:\n{output}\n--------------------")
        self.log(f"{self.output_dir}/time_cost.txt",
                 f"Relation: {t_relation}\nConsistency: {t_consist}\nResponse: {t_response}\n")
        
        # Record dialogue history
        self.conversation_history.append({"name": "Host", "message": message})
        self.conversation_history.append({"name": self.name, "message": response})
        
        return response


class LASIAgent(BaseONUWAgent):
    """
    LASIAgent: LASI framework (Landscape Analysis - Strategy - Implementation).
    Full pipeline with 4 API calls.
    
    Pipeline:
    1. Analysis: Landscape analysis (analyze player behaviors and role claims)
    2. Plan: Strategy formulation (plan for upcoming phases)
    3. Action: Action selection (choose specific action)
    4. Response: Response generation (generate natural language response)
    """

    def __init__(self, analysis_prompt: str, plan_prompt: str, 
                 action_prompt: str, response_prompt: str,
                 suggestion_prompt: str, strategy_prompt: str, update_prompt: str,
                 suggestion: str, other_strategy: str, candidate_actions: list,
                 use_analysis: bool = True, use_plan: bool = True, use_action: bool = True,
                 reflection_other: bool = True, improve_strategy: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.analysis_prompt = analysis_prompt
        self.plan_prompt = plan_prompt
        self.action_prompt = action_prompt
        self.response_prompt = response_prompt
        self.suggestion_prompt = suggestion_prompt
        self.strategy_prompt = strategy_prompt
        self.update_prompt = update_prompt
        self.previous_suggestion = suggestion
        self.previous_other_strategy = other_strategy
        self.candidate_actions = candidate_actions
        
        self.use_analysis = use_analysis
        self.use_plan = use_plan
        self.use_action = use_action
        self.reflection_other = reflection_other
        self.improve_strategy = improve_strategy
        
        self.plan = {}

    def step(self, message: str) -> str:
        temp_phase = message.split("|")[0]
        self.phase = temp_phase
        message = message.split("|")[1]

        pattern = r"\d+"
        matches = re.findall(pattern, temp_phase)
        phase = matches[-1] if matches else "0"
        
        context = self.get_conversation_context()
        
        # Step 1: Analysis
        t_analysis_start = time.time()
        if self.use_analysis and context != "None":
            analysis = self.make_analysis(phase, context)
            self.emit_thinking("Analysis", analysis)
        else:
            analysis = "None"
        t_analysis = time.time() - t_analysis_start
        
        # Step 2: Plan
        t_plan_start = time.time()
        if self.use_plan:
            format_plan = self.make_plan(phase, context, analysis)
            self.emit_thinking("Plan", format_plan)
        else:
            format_plan = "None"
        t_plan = time.time() - t_plan_start

        # Step 3: Action
        t_action_start = time.time()
        if self.use_action:
            action = self.make_action(phase, context, format_plan, analysis, message)
            action_str = str(action) if action else "None"
            self.emit_thinking("Action", action_str)
        else:
            action = None
        t_action = time.time() - t_action_start

        # Step 4: Response
        t_response_start = time.time()
        response = self.make_response(phase, context, format_plan, action, message)
        t_response = time.time() - t_response_start
        
        # Record conversation history
        self.conversation_history.append({"name": "Host", "message": message})
        self.conversation_history.append({"name": self.name, "message": response})

        self.log(f"{self.output_dir}/time_cost.txt",
                 f"Analysis: {t_analysis}\nPlan: {t_plan}\nAction: {t_action}\nResponse: {t_response}\n")
        return response

    def make_analysis(self, phase, context):
        prompt = self.analysis_prompt.format(
            name=self.name, phase=self.phase, role=self.role, summary=context
        )
        messages = [
            {"role": 'system', "content": self.get_system_prompt_with_night_info()},
            {"role": 'user', "content": prompt}
        ]
        output = self.send_messages(messages)
        self.log(f"{self.output_dir}/step_reflection.txt",
                 f"phase:{phase}\ninput:{prompt}\noutput:\n{output}\n--------------------")
        return output

    def make_plan(self, phase, context, analysis):
        if self.plan:
            format_previous_plan = '\n'.join(
                [f"Discussion Round {i}: {self.plan.get(str(i), 'None')}" if i != 0
                 else f"Night Phase: {self.plan.get(str(i), 'None')}"
                 for i in range(int(phase) + 1)]
            )
        else:
            format_previous_plan = "None"

        following_format = '\n'.join(
            [f"Discussion Round {i}: <your_plan_{i}>" if i != 0
             else f"Night Phase: <your_plan_0>"
             for i in range(int(phase), 4)]
        )
        prompt = self.plan_prompt.format(
            name=self.name, phase=self.phase, role=self.role, introduction=self.introduction,
            goal=self.game_goal, strategy=self.strategy,
            previous_plan=format_previous_plan, summary=context, analysis=analysis, plan=following_format)
        messages = [
            {"role": 'system', "content": self.get_system_prompt_with_night_info()},
            {"role": 'user', "content": prompt}
        ]
        output = self.send_messages(messages)
        match = re.search('<plan>(.*?)</plan>', output, re.S)
        format_plans = match.group().split('\n') if match else output.split('\n')
        plans = [f.split(":", 1) for f in format_plans]
        dict_plans = {}
        for plan in plans:
            if len(plan) == 2:
                match = re.search(r'\d+', plan[0])
                c_phase = match.group() if match else None
                if c_phase is None and "night" in plan[0].lower():
                    c_phase = "0"
                c_plan = plan[1].strip()
                if c_phase:
                    dict_plans[c_phase] = c_plan
        self.plan.update(dict_plans)
        self.log(f"{self.output_dir}/plan.txt",
                 f"phase:{phase}\ninput:{prompt}\noutput:\n{output}\n--------------------")
        format_plan = '\n'.join([
            f"Discussion Round {str(c_phase)}: {self.plan.get(str(c_phase))}" if str(c_phase) != "0"
            else f"Night Phase: {self.plan.get(str(c_phase))}"
            for c_phase in range(int(phase), 4)])
        return format_plan

    def make_action(self, phase, context, format_plan, analysis, message):
        prompt = self.action_prompt.format(name=self.name, phase=self.phase, role=self.role,
                                           introduction=self.introduction, goal=self.game_goal,
                                           strategy=self.strategy, candidate_actions=self.candidate_actions,
                                           summary=context, analysis=analysis, plan=format_plan,
                                           question=message)

        messages = [
            {"role": 'system', "content": self.get_system_prompt_with_night_info()},
            {"role": 'user', "content": prompt}
        ]
        output = self.send_messages(messages)
        self.log(f"{self.output_dir}/actions.txt", f"input:{prompt}\noutput:\n{output}\n--------------------")
        actions = re.findall("(?<=<actions>).*?(?=</actions>)", output, re.S)
        if not actions:
            actions = re.findall("(?<=<output>).*?(?=</output>)", output, re.S)
            if not actions:
                return output
        return actions

    def make_response(self, phase, context, format_plan, actions, message):
        if self.use_action:
            prompt = self.response_prompt.format(
                name=self.name, phase=self.phase, role=self.role, introduction=self.introduction,
                strategy=self.strategy,
                summary=context, plan=format_plan, question=message, actions=actions)
        else:
            prompt = self.response_prompt.format(
                name=self.name, phase=self.phase, role=self.role, introduction=self.introduction,
                strategy=self.strategy,
                summary=context, plan=format_plan, question=message, actions="None")
        messages = [
            {"role": 'system', "content": self.get_system_prompt_with_night_info()},
            {"role": 'user', "content": prompt}
        ]

        output = self.send_messages(messages)

        self.log(f"{self.output_dir}/response.txt", f"input:{prompt}\noutput:\n{output}\n--------------------")
        response = extract_response(output)
        return response

    def reflection(self, player_role_mapping: dict, file_name: str, winners: list, duration: int):
        """Post-game reflection: analyze strategies and update for future games."""
        p_r_mapping = '\n'.join([f"{k}:{v}" for k, v in player_role_mapping.items()])
        context = self.get_conversation_context()
        
        if self.reflection_other:
            prompt = self.strategy_prompt.format(
                name=self.name, roles=p_r_mapping, summaries=context, strategies=self.previous_other_strategy
            )
            messages = [
                {"role": 'system', "content": ""},
                {"role": 'user', "content": prompt}
            ]
            role_strategy = self.send_messages(messages)
            self.log(f"{self.output_dir}/round_reflection.txt",
                     f"input:{prompt}\noutput:\n{role_strategy}\n--------------------")
        else:
            role_strategy = "None"

        if self.improve_strategy:
            prompt = self.suggestion_prompt.format(
                name=self.name, role=self.role, roles=p_r_mapping, summaries=context, goal=self.game_goal,
                strategy=self.strategy, previous_suggestions=self.previous_suggestion
            )
            messages = [
                {"role": 'system', "content": ""},
                {"role": 'user', "content": prompt}
            ]
            suggestion = self.send_messages(messages)
            self.log(f"{self.output_dir}/round_reflection.txt",
                     f"input:{prompt}\noutput:\n{suggestion}\n--------------------")

            prompt = self.update_prompt.format(
                name=self.name, role=self.role, strategy=self.strategy, suggestions=suggestion
            )
            messages = [
                {"role": 'system', "content": ""},
                {"role": 'user', "content": prompt}
            ]
            output = self.send_messages(messages)
            self.log(f"{self.output_dir}/round_reflection.txt",
                     f"input:{prompt}\noutput:\n{output}\n--------------------")
            match = re.search("(?<=<strategy>).*?(?=</strategy>)", output)
            strategy = match.group() if match else output
        else:
            suggestion = "None"
            strategy = self.strategy

        write_json(
            data={"strategy": strategy, "suggestion": suggestion, "other_strategy": role_strategy},
            path=file_name
        )


class BeliefAgent(BaseONUWAgent):
    """
    BeliefAgent: Belief-based agent (corresponds to dpins:no baseline).
    
    Pipeline (2 API calls):
    1. Belief Modeling: Analyze conversation history to form beliefs about each player's role
    2. Response Generation: Generate response based on beliefs and game context
    
    The agent first reasons about the most likely role distribution among players,
    then uses this belief to inform its speech or voting decision.
    """

    # Belief modeling prompt
    BELIEF_PROMPT = """You are {name} in an ONUW game.
Here are some conversation history you can refer to: {conversation}
Notice that you are {name} in the conversation. You should carefully analyze the conversation history since some players might deceive during the conversation.
Based on the game rules, role descriptions and messages, think about what roles all players (including yourself) can most probably be now. You can reason each player's role step by step, based on the real or highly credible information you know.
Remember, you must give out the most likely role of each player (including yourself) in your concise response. And the number of each role have to be less or equal to the number of it in the candidate roles.
Give your step-by-step thought process and your derived concise result (no more than 2 sentences) at the end with following Response Format:
```
My step-by-step thought process: ...
My concise result: ...
```"""

    # Day phase response prompt (with belief)
    DAY_PROMPT = """Now it is the Day phase. Here are some conversation history you can refer to: {conversation}
Notice that you are {name} in the conversation. You should carefully analyze the conversation history since some players might deceive during the conversation.
And here is your belief about possible roles of all players: {belief}
Based on the game rules, role descriptions, messages and your belief, think about what insights you can summarize from the conversation and your speaking strategy next.
After that, give a concise but informative and specific public speech based on your insights and strategy. Your speech should be no more than 100 words.

Output your response directly within <response></response> tags, nothing else outside the tags."""

    # Voting phase response prompt (with belief)
    VOTING_PROMPT = """Now it is the Voting phase. Here are some conversation history you can refer to: {conversation}
Notice that you are {name} in the conversation. You should carefully analyze the conversation history since some players might deceive during the conversation.
And here is your belief about possible roles of all players: {belief}
Based on the game rules, role descriptions, messages and your belief, think about who is most likely a Werewolf and then vote for this player.

Output your response directly within <response></response> tags, nothing else outside the tags."""

    def __init__(self, response_prompt: str, **kwargs):
        super().__init__(**kwargs)
        self.response_prompt = response_prompt

    def _model_belief(self, context: str) -> str:
        """Model beliefs about player roles based on conversation history."""
        prompt = self.BELIEF_PROMPT.format(
            name=self.name,
            conversation=context
        )
        messages = [
            {"role": 'system', "content": self.get_system_prompt_with_night_info()},
            {"role": 'user', "content": prompt}
        ]
        belief = self.send_messages(messages)
        self.log(f"{self.output_dir}/belief.txt",
                 f"phase:{self.phase}\ninput:{prompt}\noutput:\n{belief}\n--------------------")
        self.emit_thinking("Belief Modeling", belief)
        return belief

    def step(self, message: str) -> str:
        temp_phase = message.split("|")[0]
        self.phase = temp_phase
        message = message.split("|")[1]

        context = self.get_conversation_context()

        # For night phase actions, use direct response (no belief needed)
        if "night" in self.phase.lower():
            prompt = self.response_prompt.format(
                name=self.name, phase=self.phase, role=self.role,
                introduction=self.introduction, strategy=self.strategy,
                summary=context, plan="None", question=message, actions="None"
            )
            messages = [
                {"role": 'system', "content": self.get_system_prompt_with_night_info()},
                {"role": 'user', "content": prompt}
            ]
            output = self.send_messages(messages)
            response = extract_response(output)
            self.log(f"{self.output_dir}/response.txt",
                     f"phase:{self.phase}\noutput:\n{output}\n--------------------")
            self.conversation_history.append({"name": "Host", "message": message})
            self.conversation_history.append({"name": self.name, "message": response})
            return response

        # Step 1: Belief modeling
        t_belief_start = time.time()
        belief = self._model_belief(context)
        t_belief = time.time() - t_belief_start

        # Step 2: Generate response based on belief
        t_response_start = time.time()
        if "voting" in self.phase.lower() or "vote" in self.phase.lower():
            prompt = self.VOTING_PROMPT.format(
                name=self.name, conversation=context, belief=belief
            )
        else:
            prompt = self.DAY_PROMPT.format(
                name=self.name, conversation=context, belief=belief
            )

        messages = [
            {"role": 'system', "content": self.get_system_prompt_with_night_info()},
            {"role": 'user', "content": prompt}
        ]
        output = self.send_messages(messages)
        t_response = time.time() - t_response_start

        response = extract_response(output)

        self.log(f"{self.output_dir}/response.txt",
                 f"phase:{self.phase}\ninput:{prompt}\noutput:\n{output}\n--------------------")
        self.log(f"{self.output_dir}/time_cost.txt",
                 f"Belief: {t_belief}\nResponse: {t_response}\n")

        # Record dialogue history
        self.conversation_history.append({"name": "Host", "message": message})
        self.conversation_history.append({"name": self.name, "message": response})

        return response


class LLMInsAgent(BaseONUWAgent):
    """
    LLMInsAgent: LLM-instructed speaking strategy agent (corresponds to dpins:llm baseline).
    
    Pipeline (3 API calls):
    1. Belief Modeling: Analyze conversation to form beliefs about player roles
    2. Strategy Selection: LLM chooses a speaking strategy from predefined options
    3. Response Generation: Generate response following the chosen strategy
    
    Available speaking strategies:
    - honest_evidence: Provide honest evidence/information
    - deceptive_evidence: Provide misleading evidence/information
    - honest_accusation: Accuse someone honestly
    - deceptive_accusation: Accuse someone deceptively
    - honest_defense: Defend honestly
    - deceptive_defense: Defend deceptively
    """

    # Belief modeling prompt (same as BeliefAgent)
    BELIEF_PROMPT = BeliefAgent.BELIEF_PROMPT

    # Strategy selection prompt
    STRATEGY_PROMPT = """Now it is the Day phase. Here are some conversation history you can refer to: {conversation}
Notice that you are {name} in the conversation. You should carefully analyze the conversation history since some players might deceive during the conversation.
And here is your belief about possible roles of all players: {belief}
Based on the game rules, role descriptions, messages and your belief, think about what kind of speaking strategy you are going to use for your upcoming speech in this turn.

There are 6 speaking strategies you can choose:
{strategies}

Please choose the most appropriate speaking strategy for your upcoming speech in this turn. Your speaking goal is to convince other players to believe what you are going to say and induce them to reveal their true night actions.
You can only choose one speaking strategy from the options: [{strategy_names}].
You must return your response in a JSON format that can be parsed by Python `json.loads`. Here is the Response Format:
{{
    "thought": "<the reason why you choose this speaking strategy>",
    "strategy": "<the speaking strategy you choose>"
}}"""

    # Day phase response prompt (with belief and strategy instruction)
    DAY_PROMPT_WITH_STRATEGY = """Now it is the Day phase. Here are some conversation history you can refer to: {conversation}
Notice that you are {name} in the conversation. You should carefully analyze the conversation history since some players might deceive during the conversation.
And here is your belief about possible roles of all players: {belief}
In this turn, your speaking strategy is: {speaking_strategy}
Based on the game rules, role descriptions, messages, your belief and your speaking strategy, think about what insights you can summarize from the conversation.
After that, give a concise but informative and specific public speech based on your insights and following your speaking strategy. Your speaking goal is to convince other players to believe what you are going to say. Your speech should be no more than 100 words.

Output your response directly within <response></response> tags, nothing else outside the tags."""

    # Voting phase prompt (same as BeliefAgent)
    VOTING_PROMPT = BeliefAgent.VOTING_PROMPT

    def __init__(self, response_prompt: str, speaking_strategies: dict = None, **kwargs):
        super().__init__(**kwargs)
        self.response_prompt = response_prompt
        # Import speaking strategies from prompts module
        if speaking_strategies is None:
            from prompt.onuw_prompts import SPEAKING_STRATEGY
            self.speaking_strategies = SPEAKING_STRATEGY
        else:
            self.speaking_strategies = speaking_strategies

    def _model_belief(self, context: str) -> str:
        """Model beliefs about player roles based on conversation history."""
        prompt = self.BELIEF_PROMPT.format(
            name=self.name,
            conversation=context
        )
        messages = [
            {"role": 'system', "content": self.get_system_prompt_with_night_info()},
            {"role": 'user', "content": prompt}
        ]
        belief = self.send_messages(messages)
        self.log(f"{self.output_dir}/belief.txt",
                 f"phase:{self.phase}\ninput:{prompt}\noutput:\n{belief}\n--------------------")
        self.emit_thinking("Belief Modeling", belief)
        return belief

    def _choose_strategy(self, context: str, belief: str) -> str:
        """Use LLM to choose a speaking strategy."""
        # Build strategy descriptions
        strategies_text = ""
        for idx, (name, desc) in enumerate(self.speaking_strategies.items()):
            strategies_text += f"\n{idx+1}. {name}: {desc}"
        strategy_names = ', '.join(self.speaking_strategies.keys())

        prompt = self.STRATEGY_PROMPT.format(
            name=self.name,
            conversation=context,
            belief=belief,
            strategies=strategies_text,
            strategy_names=strategy_names
        )
        messages = [
            {"role": 'system', "content": self.get_system_prompt_with_night_info()},
            {"role": 'user', "content": prompt}
        ]
        output = self.send_messages(messages)
        self.log(f"{self.output_dir}/strategy_selection.txt",
                 f"phase:{self.phase}\ninput:{prompt}\noutput:\n{output}\n--------------------")

        # Parse JSON response to extract chosen strategy
        try:
            # Try to find JSON in the output
            json_match = re.search(r'\{[^{}]*\}', output, re.S)
            if json_match:
                result = json.loads(json_match.group())
                chosen = result.get("strategy", "")
            else:
                chosen = ""
        except (json.JSONDecodeError, ValueError):
            chosen = ""

        # Fuzzy match to find the best matching strategy
        if chosen:
            try:
                from fuzzywuzzy import process
                best_match, score = process.extractOne(chosen, list(self.speaking_strategies.keys()))
                if score >= 50:
                    chosen = best_match
                else:
                    chosen = list(self.speaking_strategies.keys())[0]
            except ImportError:
                # Fallback: exact match or first strategy
                if chosen not in self.speaking_strategies:
                    # Try partial match
                    for key in self.speaking_strategies:
                        if chosen.lower() in key.lower() or key.lower() in chosen.lower():
                            chosen = key
                            break
                    else:
                        chosen = list(self.speaking_strategies.keys())[0]
        else:
            chosen = list(self.speaking_strategies.keys())[0]

        speaking_strategy = self.speaking_strategies.get(chosen, "")
        self.emit_thinking("Strategy Selection", f"Chosen: {chosen}\n{speaking_strategy}")
        return speaking_strategy

    def step(self, message: str) -> str:
        temp_phase = message.split("|")[0]
        self.phase = temp_phase
        message = message.split("|")[1]

        context = self.get_conversation_context()

        # For night phase actions, use direct response (no belief/strategy needed)
        if "night" in self.phase.lower():
            prompt = self.response_prompt.format(
                name=self.name, phase=self.phase, role=self.role,
                introduction=self.introduction, strategy=self.strategy,
                summary=context, plan="None", question=message, actions="None"
            )
            messages = [
                {"role": 'system', "content": self.get_system_prompt_with_night_info()},
                {"role": 'user', "content": prompt}
            ]
            output = self.send_messages(messages)
            response = extract_response(output)
            self.log(f"{self.output_dir}/response.txt",
                     f"phase:{self.phase}\noutput:\n{output}\n--------------------")
            self.conversation_history.append({"name": "Host", "message": message})
            self.conversation_history.append({"name": self.name, "message": response})
            return response

        # Step 1: Belief modeling
        t_belief_start = time.time()
        belief = self._model_belief(context)
        t_belief = time.time() - t_belief_start

        if "voting" in self.phase.lower() or "vote" in self.phase.lower():
            # For voting, use belief-based voting (no strategy selection needed)
            t_response_start = time.time()
            prompt = self.VOTING_PROMPT.format(
                name=self.name, conversation=context, belief=belief
            )
            messages = [
                {"role": 'system', "content": self.get_system_prompt_with_night_info()},
                {"role": 'user', "content": prompt}
            ]
            output = self.send_messages(messages)
            t_response = time.time() - t_response_start
            response = extract_response(output)
            self.log(f"{self.output_dir}/response.txt",
                     f"phase:{self.phase}\noutput:\n{output}\n--------------------")
            self.log(f"{self.output_dir}/time_cost.txt",
                     f"Belief: {t_belief}\nResponse: {t_response}\n")
            self.conversation_history.append({"name": "Host", "message": message})
            self.conversation_history.append({"name": self.name, "message": response})
            return response

        # Step 2: Strategy selection (day phase only)
        t_strategy_start = time.time()
        speaking_strategy = self._choose_strategy(context, belief)
        t_strategy = time.time() - t_strategy_start

        # Step 3: Generate response with strategy instruction
        t_response_start = time.time()
        prompt = self.DAY_PROMPT_WITH_STRATEGY.format(
            name=self.name, conversation=context, belief=belief,
            speaking_strategy=speaking_strategy
        )
        messages = [
            {"role": 'system', "content": self.get_system_prompt_with_night_info()},
            {"role": 'user', "content": prompt}
        ]
        output = self.send_messages(messages)
        t_response = time.time() - t_response_start

        response = extract_response(output)

        self.log(f"{self.output_dir}/response.txt",
                 f"phase:{self.phase}\ninput:{prompt}\noutput:\n{output}\n--------------------")
        self.log(f"{self.output_dir}/time_cost.txt",
                 f"Belief: {t_belief}\nStrategy: {t_strategy}\nResponse: {t_response}\n")

        # Record dialogue history
        self.conversation_history.append({"name": "Host", "message": message})
        self.conversation_history.append({"name": self.name, "message": response})

        return response


class RefinerWrapper(Agent):
    """
    RefinerWrapper: Wraps any existing agent type with a trained Refiner model.

    Implements a two-stage utterance refinement pipeline:
    1. The wrapped (backend) agent generates a base utterance u_base
    2. The Refiner model (a LoRA-finetuned open-source LLM) refines u_base
       into a more persuasive version u_t

    The Refiner can be seamlessly integrated with any agent type
    (e.g., "refiner+react", "refiner+direct", "refiner+lasi").
    """

    def __init__(self, wrapped_agent: BaseONUWAgent,
                 refiner_model_path: str,
                 refiner_lora_path: Optional[str] = None,
                 refiner_temperature: float = 0.7,
                 refine_prompt_template: Optional[str] = None,
                 **kwargs):
        """
        Initialize the RefinerWrapper.

        Args:
            wrapped_agent: The backend agent that generates base utterances.
            refiner_model_path: Path to the base model for the Refiner.
            refiner_lora_path: Path to the LoRA adapter checkpoint (optional).
            refiner_temperature: Temperature for Refiner generation.
            refine_prompt_template: Custom refine prompt template (uses built-in default if None).
        """
        super().__init__(name=wrapped_agent.name, role=wrapped_agent.role)
        self.wrapped_agent = wrapped_agent
        self.refiner_model_path = refiner_model_path
        self.refiner_lora_path = refiner_lora_path
        self.refiner_temperature = refiner_temperature
        self.refine_prompt_template = refine_prompt_template

        # Lazy-load the Refiner model (loaded on first use)
        self._refiner_model = None
        self._refiner_tokenizer = None

    def _load_refiner(self):
        """Lazy-load the Refiner model and tokenizer."""
        if self._refiner_model is not None:
            return

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
            import torch
        except ImportError as e:
            raise ImportError(
                "RefinerWrapper requires 'transformers', 'peft', and 'torch'. "
                "Install them with: pip install transformers peft torch"
            ) from e

        logger.info(f"Loading Refiner base model from {self.refiner_model_path}...")
        self._refiner_tokenizer = AutoTokenizer.from_pretrained(
            self.refiner_model_path,
            trust_remote_code=True,
            padding_side='left'
        )
        if self._refiner_tokenizer.pad_token is None:
            self._refiner_tokenizer.pad_token = self._refiner_tokenizer.eos_token

        self._refiner_model = AutoModelForCausalLM.from_pretrained(
            self.refiner_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        # Apply LoRA adapter if provided
        if self.refiner_lora_path:
            logger.info(f"Loading LoRA adapter from {self.refiner_lora_path}...")
            self._refiner_model = PeftModel.from_pretrained(
                self._refiner_model,
                self.refiner_lora_path,
            )
            self._refiner_model = self._refiner_model.merge_and_unload()

        self._refiner_model.eval()
        logger.info("Refiner model loaded successfully.")

    def _refine_utterance(self, base_utterance: str) -> str:
        """
        Refine a base utterance using the trained Refiner model.

        Args:
            base_utterance: The base utterance u_base from the backend agent.

        Returns:
            The refined utterance u_t.
        """
        self._load_refiner()

        import torch

        # Build the refine prompt
        if self.refine_prompt_template:
            prompt_template = self.refine_prompt_template
        else:
            from prompt.onuw_prompts import refine_prompt as default_refine_prompt
            prompt_template = default_refine_prompt

        # Get game context from the wrapped agent
        agent = self.wrapped_agent
        game_rules = agent.system_prompt
        game_state = f"Phase: {agent.phase}"
        dialog_history = agent.get_conversation_context()

        refine_input = prompt_template.format(
            game_rules=game_rules,
            player_name=agent.name,
            player_role=agent.role,
            game_state=game_state,
            dialog_history=dialog_history,
            base_utterance=base_utterance
        )

        messages = [
            {"role": "system", "content": "You are a communication expert specializing in persuasive dialogue refinement for social deduction games."},
            {"role": "user", "content": refine_input}
        ]

        # Tokenize and generate
        input_text = self._refiner_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        input_ids = self._refiner_tokenizer.encode(
            input_text, return_tensors='pt', add_special_tokens=False
        )
        input_ids = input_ids.to(self._refiner_model.device)

        with torch.no_grad():
            outputs = self._refiner_model.generate(
                input_ids,
                max_new_tokens=512,
                temperature=self.refiner_temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self._refiner_tokenizer.pad_token_id,
            )

        # Decode only the generated tokens
        generated_ids = outputs[0][input_ids.shape[1]:]
        refined_output = self._refiner_tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Extract the response from the refined output
        response_match = re.search(r"Response:\s*(.+)", refined_output, re.S)
        if response_match:
            refined = response_match.group(1).strip()
            refined = re.sub(r'\s*```\s*$', '', refined).strip()
            if refined:
                return refined

        # Fallback: try <response>...</response> tags
        tag_match = re.search(r"<response>(.+?)(?:</response>|$)", refined_output, re.S)
        if tag_match:
            return tag_match.group(1).strip()

        # If parsing fails, return the raw output (trimmed)
        refined = refined_output.strip()
        if refined:
            return refined

        # If Refiner output is empty, fall back to base utterance
        logger.warning("Refiner produced empty output, falling back to base utterance.")
        return base_utterance

    def step(self, message: str) -> str:
        """
        Generate a response using the two-stage pipeline:
        1. Backend agent generates base utterance u_base
        2. Refiner refines u_base into persuasive u_t
        """
        # Step 1: Get base utterance from the wrapped agent
        base_utterance = self.wrapped_agent.step(message)

        # Step 2: Refine the base utterance
        t_refine_start = time.time()
        refined_utterance = self._refine_utterance(base_utterance)
        t_refine = time.time() - t_refine_start

        # Log the refinement
        self.wrapped_agent.log(
            f"{self.wrapped_agent.output_dir}/refiner.txt",
            f"phase:{self.wrapped_agent.phase}\n"
            f"base_utterance:\n{base_utterance}\n"
            f"refined_utterance:\n{refined_utterance}\n"
            f"refine_time: {t_refine:.2f}s\n"
            f"--------------------\n"
        )

        self.wrapped_agent.emit_thinking(
            "Refiner",
            f"Base: {base_utterance[:200]}...\nRefined: {refined_utterance[:200]}..."
        )

        # Update the wrapped agent's conversation history with the refined utterance
        if self.wrapped_agent.conversation_history:
            last_entry = self.wrapped_agent.conversation_history[-1]
            if last_entry.get("name") == self.wrapped_agent.name:
                last_entry["message"] = refined_utterance

        return refined_utterance

    def receive(self, name: str, message: str) -> None:
        """Delegate to the wrapped agent."""
        self.wrapped_agent.receive(name, message)

    def set_night_info(self, info: str) -> None:
        """Delegate to the wrapped agent."""
        self.wrapped_agent.set_night_info(info)

    def identify_intent(self, next_player: str) -> dict:
        """Delegate to the wrapped agent."""
        return self.wrapped_agent.identify_intent(next_player)

    def reflection(self, player_role_mapping: dict, file_name: str, winners: list, duration: int):
        """Delegate to the wrapped agent."""
        self.wrapped_agent.reflection(player_role_mapping, file_name, winners, duration)
