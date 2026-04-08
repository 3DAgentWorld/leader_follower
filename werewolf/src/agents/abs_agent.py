#!/usr/bin/env python
# encoding: utf-8
"""
Agent base class.
"""
from abc import abstractmethod


class Agent:
    """Abstract base class for all agents."""
    name = None
    role = None

    def __init__(self, **kwargs):
        self.name = kwargs.get('name')
        self.role = kwargs.get('role')

    @abstractmethod
    def step(self, message: str) -> str:
        """
        Interact with the agent.
        
        Args:
            message: Input message to the agent.
        
        Returns:
            The agent's response.
        """
        pass

    @abstractmethod
    def receive(self, name: str, message: str) -> None:
        """
        Receive a message from another agent.
        
        Args:
            name: Name of the sending agent.
            message: Message content.
        """
        pass

    def set_night_info(self, info: str) -> None:
        """
        Set night phase information, which will be merged into the system prompt.
        
        Args:
            info: Night phase information.
        """
        self.night_info = info

    def identify_intent(self, next_player: str) -> dict:
        """
        Intent identification: identify desired and undesired responses from the next player.
        
        Intent Identification:
        - Identify K desired responses (beneficial to the current player)
        - Identify K undesired responses (harmful to the current player)
        
        Args:
            next_player: Name of the next player to speak.
            
        Returns:
            dict: Contains desired_responses and undesired_responses,
                  or None if not implemented.
        """
        return None

    @classmethod
    def init_instance(cls, **kwargs):
        """Factory method to create an agent instance."""
        return cls(**kwargs)
