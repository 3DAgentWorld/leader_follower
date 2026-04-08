#!/usr/bin/env python
# encoding: utf-8
"""
Agent base class for Sotopia social simulation.
"""
from abc import abstractmethod


class AbsAgent:
    """Abstract base class for Sotopia agents."""

    def __init__(self, name: str, agent_type: str, model: str,
                 api_key: str, api_base: str = None,
                 temperature: float = 0.3, mode: str = "watch"):
        self.name = name
        self.agent_type = agent_type
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.temperature = temperature
        self.mode = mode

    @abstractmethod
    def step(self, summary: str, question: str, turn: int = 0) -> str:
        """
        Generate a response for the current turn.

        Args:
            summary: Conversation history summary.
            question: Current prompt/question.
            turn: Current turn number.

        Returns:
            The agent's response string.
        """
        pass

    @abstractmethod
    def set_scenario_info(self, **kwargs) -> None:
        """Set scenario context for the agent."""
        pass

    @classmethod
    def init_instance(cls, **kwargs):
        """Factory method to create an agent instance."""
        return cls(**kwargs)
