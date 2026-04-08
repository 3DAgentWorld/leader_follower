#!/usr/bin/env python
# encoding: utf-8
"""
Extractor base class.
"""
from abc import abstractmethod


class Extractor:
    """Abstract base class for Extractors."""
    
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def step(self, input_text: str) -> str:
        """
        Extract information from input text.
        
        Args:
            input_text: The input text to extract from.
        
        Returns:
            Extracted information.
        """
        pass

    @classmethod
    def init_instance(cls, **kwargs):
        """Factory method to create an Extractor instance."""
        return cls(**kwargs)
