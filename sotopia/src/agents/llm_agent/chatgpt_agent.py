#!/usr/bin/env python
# encoding: utf-8
"""
LLM-based agents for Sotopia social simulation.
Implements Direct, ReAct, ReCon, and MetaMind agent types.
"""
import re
from typing import Dict, List, Optional, Tuple

from colorama import Fore, Style

from prompt.sotopia_prompts import (
    system_prompt,
    response_prompt,
    response_prompt_without_action,
    analysis_prompt,
    plan_prompt,
    recon_analysis_prompt,
    recon_strategy_prompt,
    metamind_tom_prompt,
    metamind_strategy_prompt,
    metamind_memory_update_prompt,
    intent_identification_prompt,
    init_strategies,
    candidate_actions,
)
from src.agents.abs_agent import AbsAgent
from src.apis.chatgpt_api import chatgpt as _chatgpt_raw


def chatgpt_api(messages, model, temperature, api_key=None, api_base=None):
    """Adapter: reorder args to match the underlying chatgpt(model, messages, temperature, ...) signature."""
    return _chatgpt_raw(model, messages, temperature, api_key=api_key, api_base=api_base)


class SotopiaAgent(AbsAgent):
    """
    LLM-based agent for Sotopia social simulation.

    Supports multiple reasoning frameworks:
    - direct: Single-step response generation (1 API call)
    - react: ReAct framework with analysis + response (2 API calls)
    - recon: ReCon framework with relationship analysis + strategy + response (3 API calls)
    - metamind: MetaMind framework with ToM + strategy + response (3 API calls)
    """

    VALID_AGENT_TYPES = ["direct", "react", "recon", "metamind"]

    def __init__(
        self,
        name: str,
        agent_type: str,
        model: str,
        api_key: str,
        api_base: str = None,
        temperature: float = 0.3,
        mode: str = "watch",
    ):
        super().__init__(name, agent_type, model, api_key, api_base, temperature, mode)

        # Validate agent type (strip refiner+ prefix if present)
        base_type = agent_type.replace("refiner+", "")
        if base_type not in self.VALID_AGENT_TYPES:
            raise ValueError(
                f"Invalid agent type: {agent_type}. "
                f"Valid types: {self.VALID_AGENT_TYPES}"
            )

        # Scenario context (set by game engine)
        self.scenario: str = ""
        self.relationship: str = ""
        self.goal: str = ""
        self.character_summary: str = ""
        self.character_profile: str = ""
        self.other_name: str = ""
        self.strategy: str = init_strategies["default"]

        # MetaMind social memory
        self.social_memory: str = "No prior interactions."

    @classmethod
    def init_instance(cls, **kwargs) -> "SotopiaAgent":
        """Factory method to create an agent instance."""
        return cls(**kwargs)

    def set_scenario_info(
        self,
        scenario: str,
        relationship: str,
        goal: str,
        character_summary: str,
        character_profile: str,
        other_name: str,
    ):
        """Set the scenario context for this agent."""
        self.scenario = scenario
        self.relationship = relationship
        self.goal = goal
        self.character_summary = character_summary
        self.character_profile = character_profile
        self.other_name = other_name

    def step(self, summary: str, question: str, turn: int = 0) -> str:
        """
        Generate a response based on the agent type.

        Args:
            summary: Conversation history summary.
            question: Current prompt/question for the agent.
            turn: Current turn number.

        Returns:
            Agent's response string.
        """
        base_type = self.agent_type.replace("refiner+", "")

        if base_type == "direct":
            return self._step_direct(summary, question)
        elif base_type == "react":
            return self._step_react(summary, question)
        elif base_type == "recon":
            return self._step_recon(summary, question)
        elif base_type == "metamind":
            return self._step_metamind(summary, question)
        else:
            return self._step_direct(summary, question)

    def _step_direct(self, summary: str, question: str) -> str:
        """Direct agent: single API call. (1 call)"""
        sys_msg = system_prompt.format(
            name=self.name,
            character_profile=self.character_profile,
            scenario=self.scenario,
            goal=self.goal,
            relationship=self.relationship,
            strategy=self.strategy,
        )

        user_msg = response_prompt_without_action.format(
            name=self.name,
            phase="conversation",
            character_summary=self.character_summary,
            goal=self.goal,
            strategy=self.strategy,
            scenario=self.scenario,
            relationship=self.relationship,
            summary=summary if summary else "The conversation has not started yet.",
            question=question,
        )

        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ]

        response = chatgpt_api(
            messages,
            model=self.model,
            temperature=self.temperature,
            api_key=self.api_key,
            api_base=self.api_base,
        )

        return self._clean_response(response)

    def _step_react(self, summary: str, question: str) -> str:
        """ReAct agent: analysis + response. (2 calls)"""
        # Step 1: Analyze the situation
        analysis_msg = analysis_prompt.format(
            name=self.name,
            character_summary=self.character_summary,
            scenario=self.scenario,
            goal=self.goal,
            summary=summary if summary else "The conversation has not started yet.",
            question=question,
        )

        analysis = chatgpt_api(
            [{"role": "user", "content": analysis_msg}],
            model=self.model,
            temperature=self.temperature,
            api_key=self.api_key,
            api_base=self.api_base,
        )

        if self.mode == "watch":
            print(f"{Fore.CYAN}[{self.name} Analysis]: {analysis[:200]}...{Style.RESET_ALL}")

        # Step 2: Generate response based on analysis
        sys_msg = system_prompt.format(
            name=self.name,
            character_profile=self.character_profile,
            scenario=self.scenario,
            goal=self.goal,
            relationship=self.relationship,
            strategy=self.strategy,
        )

        user_msg = response_prompt.format(
            name=self.name,
            phase="conversation",
            character_summary=self.character_summary,
            goal=self.goal,
            strategy=self.strategy,
            scenario=self.scenario,
            relationship=self.relationship,
            summary=summary if summary else "The conversation has not started yet.",
            action=f"Based on your analysis:\n{analysis}",
            question=question,
        )

        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ]

        response = chatgpt_api(
            messages,
            model=self.model,
            temperature=self.temperature,
            api_key=self.api_key,
            api_base=self.api_base,
        )

        return self._clean_response(response)

    def _step_recon(self, summary: str, question: str) -> str:
        """ReCon agent: relationship analysis + strategy + response. (3 calls)"""
        # Step 1: Relationship analysis
        rel_msg = recon_analysis_prompt.format(
            name=self.name,
            character_summary=self.character_summary,
            scenario=self.scenario,
            goal=self.goal,
            summary=summary if summary else "The conversation has not started yet.",
        )

        rel_analysis = chatgpt_api(
            [{"role": "user", "content": rel_msg}],
            model=self.model,
            temperature=self.temperature,
            api_key=self.api_key,
            api_base=self.api_base,
        )

        if self.mode == "watch":
            print(f"{Fore.CYAN}[{self.name} Relationship Analysis]: {rel_analysis[:200]}...{Style.RESET_ALL}")

        # Step 2: Strategy formulation
        strat_msg = recon_strategy_prompt.format(
            analysis=rel_analysis,
            goal=self.goal,
        )

        strategy = chatgpt_api(
            [{"role": "user", "content": strat_msg}],
            model=self.model,
            temperature=self.temperature,
            api_key=self.api_key,
            api_base=self.api_base,
        )

        if self.mode == "watch":
            print(f"{Fore.CYAN}[{self.name} Strategy]: {strategy[:200]}...{Style.RESET_ALL}")

        # Step 3: Generate response
        sys_msg = system_prompt.format(
            name=self.name,
            character_profile=self.character_profile,
            scenario=self.scenario,
            goal=self.goal,
            relationship=self.relationship,
            strategy=self.strategy,
        )

        user_msg = response_prompt.format(
            name=self.name,
            phase="conversation",
            character_summary=self.character_summary,
            goal=self.goal,
            strategy=strategy,
            scenario=self.scenario,
            relationship=self.relationship,
            summary=summary if summary else "The conversation has not started yet.",
            action=f"Relationship analysis:\n{rel_analysis}\n\nStrategy:\n{strategy}",
            question=question,
        )

        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ]

        response = chatgpt_api(
            messages,
            model=self.model,
            temperature=self.temperature,
            api_key=self.api_key,
            api_base=self.api_base,
        )

        return self._clean_response(response)

    def _step_metamind(self, summary: str, question: str) -> str:
        """
        MetaMind agent: ToM analysis + strategy selection + response. (3 calls)

        Implements a Theory of Mind based approach:
        1. Analyze the other person's mental state (beliefs, desires, intentions)
        2. Select optimal strategy based on ToM analysis and social constraints
        3. Generate response informed by the strategy
        """
        # Step 1: Theory of Mind analysis
        tom_msg = metamind_tom_prompt.format(
            name=self.name,
            character_summary=self.character_summary,
            scenario=self.scenario,
            goal=self.goal,
            summary=summary if summary else "The conversation has not started yet.",
            social_memory=self.social_memory,
        )

        tom_analysis = chatgpt_api(
            [{"role": "user", "content": tom_msg}],
            model=self.model,
            temperature=self.temperature,
            api_key=self.api_key,
            api_base=self.api_base,
        )

        if self.mode == "watch":
            print(f"{Fore.CYAN}[{self.name} ToM Analysis]: {tom_analysis[:200]}...{Style.RESET_ALL}")

        # Step 2: Strategy selection with domain constraints
        strat_msg = metamind_strategy_prompt.format(
            tom_analysis=tom_analysis,
            goal=self.goal,
            scenario=self.scenario,
        )

        strategy = chatgpt_api(
            [{"role": "user", "content": strat_msg}],
            model=self.model,
            temperature=self.temperature,
            api_key=self.api_key,
            api_base=self.api_base,
        )

        if self.mode == "watch":
            print(f"{Fore.CYAN}[{self.name} Strategy]: {strategy[:200]}...{Style.RESET_ALL}")

        # Step 3: Generate response
        sys_msg = system_prompt.format(
            name=self.name,
            character_profile=self.character_profile,
            scenario=self.scenario,
            goal=self.goal,
            relationship=self.relationship,
            strategy=self.strategy,
        )

        user_msg = response_prompt.format(
            name=self.name,
            phase="conversation",
            character_summary=self.character_summary,
            goal=self.goal,
            strategy=strategy,
            scenario=self.scenario,
            relationship=self.relationship,
            summary=summary if summary else "The conversation has not started yet.",
            action=f"Theory of Mind analysis:\n{tom_analysis}\n\nSelected strategy:\n{strategy}",
            question=question,
        )

        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ]

        response = chatgpt_api(
            messages,
            model=self.model,
            temperature=self.temperature,
            api_key=self.api_key,
            api_base=self.api_base,
        )

        clean_response = self._clean_response(response)

        # Update social memory after generating response
        self._update_social_memory(summary, clean_response)

        return clean_response

    def _update_social_memory(self, conv_summary: str, latest_response: str):
        """Update the social memory based on the latest exchange."""
        try:
            # Get the latest exchange (last few lines of conversation + our response)
            latest_exchange = ""
            if conv_summary:
                lines = conv_summary.strip().split("\n")
                latest_exchange = "\n".join(lines[-3:])  # Last 3 lines
            latest_exchange += f"\n{self.name}: {latest_response}"

            update_msg = metamind_memory_update_prompt.format(
                latest_exchange=latest_exchange,
                previous_memory=self.social_memory,
            )

            updated_memory = chatgpt_api(
                [{"role": "user", "content": update_msg}],
                model=self.model,
                temperature=0.1,
                api_key=self.api_key,
                api_base=self.api_base,
            )

            self.social_memory = updated_memory.strip()

        except Exception:
            # Silently fail - social memory update is not critical
            pass

    def identify_intent(
        self,
        speaker_name: str,
        speaker_utterance: str,
        follower_name: str,
        follower_goal: str,
        follower_profile: str,
        scenario: str,
        conv_summary: str,
        k: int = 3,
    ) -> Optional[Dict]:
        """
        Generate desired and undesired follower responses for training data.

        Returns:
            Dict with 'desired_responses' and 'undesired_responses' lists.
        """
        results = {}

        for response_type, description in [
            ("desired", "align well with the follower's goals and advance the conversation positively"),
            ("undesired", "are counterproductive, reveal secrets, damage relationships, or work against the follower's goals"),
        ]:
            prompt = intent_identification_prompt.format(
                scenario=scenario,
                dialog_history=conv_summary,
                speaker_name=speaker_name,
                speaker_role="participant",
                utterance=speaker_utterance,
                next_speaker_name=follower_name,
                next_speaker_role="participant",
                next_speaker_goal=follower_goal,
                next_speaker_profile=follower_profile,
                k=k,
                response_type=response_type,
                response_description=description,
            )

            try:
                response = chatgpt_api(
                    [{"role": "user", "content": prompt}],
                    model=self.model,
                    temperature=0.7,
                    api_key=self.api_key,
                    api_base=self.api_base,
                )

                # Parse numbered responses
                responses = []
                for line in response.strip().split("\n"):
                    line = line.strip()
                    match = re.match(r'^\d+[\.\)]\s*(.+)', line)
                    if match:
                        responses.append(match.group(1).strip())

                results[f"{response_type}_responses"] = responses[:k]

            except Exception as e:
                results[f"{response_type}_responses"] = []

        return results if any(results.values()) else None

    def _clean_response(self, response: str) -> str:
        """Clean up the agent's response."""
        response = response.strip()

        # Remove common prefixes like "Agent:" or the agent's name
        prefixes_to_remove = [
            f"{self.name}:",
            f"{self.name} says:",
            f"As {self.name},",
            f"As {self.name}:",
        ]
        for prefix in prefixes_to_remove:
            if response.lower().startswith(prefix.lower()):
                response = response[len(prefix):].strip()

        # Remove surrounding quotes if present
        if response.startswith('"') and response.endswith('"'):
            response = response[1:-1]

        return response


class RefinerWrapper:
    """
    Wraps any SotopiaAgent with a trained Refiner model for utterance refinement.

    Two-stage pipeline:
    1. The wrapped agent generates a base utterance
    2. The Refiner model refines it into a more persuasive version

    Usage: Set agent_type to "refiner+react", "refiner+direct", etc.
    """

    def __init__(
        self,
        wrapped_agent: SotopiaAgent,
        refiner_model_path: str,
        refiner_lora_path: Optional[str] = None,
        refiner_temperature: float = 0.7,
    ):
        self.wrapped_agent = wrapped_agent
        self.refiner_model_path = refiner_model_path
        self.refiner_lora_path = refiner_lora_path
        self.refiner_temperature = refiner_temperature
        self.name = wrapped_agent.name

        # Lazy-load model
        self._refiner_model = None
        self._refiner_tokenizer = None

    @classmethod
    def init_instance(cls, **kwargs) -> "RefinerWrapper":
        """Factory method - not used directly; created via run_sotopia_battle."""
        return cls(**kwargs)

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

        self._refiner_tokenizer = AutoTokenizer.from_pretrained(
            self.refiner_model_path,
            trust_remote_code=True,
            padding_side='left',
        )
        if self._refiner_tokenizer.pad_token is None:
            self._refiner_tokenizer.pad_token = self._refiner_tokenizer.eos_token

        self._refiner_model = AutoModelForCausalLM.from_pretrained(
            self.refiner_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        if self.refiner_lora_path:
            self._refiner_model = PeftModel.from_pretrained(
                self._refiner_model,
                self.refiner_lora_path,
            )
            self._refiner_model = self._refiner_model.merge_and_unload()

        self._refiner_model.eval()

    def _refine_utterance(self, base_utterance: str) -> str:
        """Refine a base utterance using the trained Refiner model."""
        self._load_refiner()
        import torch
        from prompt.sotopia_prompts import refine_prompt

        agent = self.wrapped_agent
        refine_input = refine_prompt.format(
            player_name=agent.name,
            player_role=agent.character_summary,
            game_state=f"Scenario: {agent.scenario}",
            dialog_history=f"Goal: {agent.goal}\nRelationship: {agent.relationship}",
            base_utterance=base_utterance,
        )

        messages = [
            {"role": "system", "content": "You are a communication expert specializing in persuasive dialogue refinement for social interactions."},
            {"role": "user", "content": refine_input},
        ]

        input_text = self._refiner_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
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

        generated_ids = outputs[0][input_ids.shape[1]:]
        refined = self._refiner_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        return refined if refined else base_utterance

    def set_scenario_info(self, **kwargs):
        """Delegate to wrapped agent."""
        self.wrapped_agent.set_scenario_info(**kwargs)

    def step(self, summary: str, question: str, turn: int = 0) -> str:
        """Two-stage: base agent generates, then Refiner refines."""
        base_utterance = self.wrapped_agent.step(summary, question, turn)
        refined = self._refine_utterance(base_utterance)

        if self.wrapped_agent.mode == "watch":
            print(f"{Fore.CYAN}[Refiner] Base: {base_utterance[:150]}...{Style.RESET_ALL}")
            print(f"{Fore.CYAN}[Refiner] Refined: {refined[:150]}...{Style.RESET_ALL}")

        return refined

    def identify_intent(self, **kwargs):
        """Delegate to wrapped agent."""
        return self.wrapped_agent.identify_intent(**kwargs)
