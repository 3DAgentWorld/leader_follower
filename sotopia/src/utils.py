#!/usr/bin/env python
# encoding: utf-8
"""
Utility functions.
"""
import time
import os
import json
from colorama import Fore


def print_text_animated(text, delay=0.01):
    """Animate text output character by character.
    
    Args:
        text: Text to print.
        delay: Delay between characters in seconds (default: 0.01).
    """
    for char in text:
        print(char, end="", flush=True)
        time.sleep(delay)


# Player color mapping
COLOR = {
    "player 1": Fore.BLUE,
    "player 2": Fore.GREEN,
    "player 3": Fore.YELLOW,
    "player 4": Fore.RED,
    "player 5": Fore.LIGHTGREEN_EX,
    "player 6": Fore.CYAN,
}


def create_dir(dir_path):
    """Create directory if it does not exist."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def write_data(data, path):
    """Write text data to file."""
    with open(path, mode='a+', encoding='utf-8') as f:
        f.write(data)
        f.write('\n')


def write_json(data, path):
    """Write JSON data to file."""
    with open(path, mode='w+', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def read_json(path):
    """Read JSON data from file."""
    with open(path, mode="r", encoding="utf-8") as f:
        json_data = json.load(f)
    return json_data
