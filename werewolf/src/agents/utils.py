#!/usr/bin/env python
# encoding: utf-8
import json


def write_json(data, path):
    """Write data to a JSON file."""
    with open(path, mode='w+', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
