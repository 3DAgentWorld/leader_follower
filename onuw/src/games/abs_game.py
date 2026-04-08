#!/usr/bin/env python
# encoding: utf-8
"""
Game base class.
"""
from abc import abstractmethod


class Game:
    """Abstract base class for Games."""
    
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def start(self):
        """Start the game."""
        pass

    @abstractmethod
    def add_players(self, players: list):
        """Add players to the game."""
        pass
