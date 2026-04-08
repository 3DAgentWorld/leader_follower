#!/usr/bin/env python
# encoding: utf-8
"""
ChatGPT-based Extractor implementation.
"""
from typing import List, Tuple, Optional

from ..abs_extractor import Extractor
from ...apis.chatgpt_api import chatgpt


class ChatGPTBasedExtractor(Extractor):
    """ChatGPT-based information extractor."""
    
    def __init__(self, extractor_name: str, model_name: str, system_prompt: str, extract_prompt: str,
                 temperature: float, few_shot_demos: List[Tuple[str, str]] = None, 
                 api_key: Optional[str] = None, api_base: Optional[str] = None,
                 output_dir: Optional[str] = None, extra_body: Optional[dict] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.extractor_name = extractor_name
        self.model = model_name
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.extract_prompt = extract_prompt
        self.few_shot_demos = few_shot_demos if few_shot_demos else []
        self.log_file = f"{output_dir}/extractor.txt" if output_dir else None
        self.api_key = api_key
        self.api_base = api_base
        self.extra_body = extra_body

    def extract(self, message: str) -> str:
        """Extract information from a message."""
        messages = [{"role": "system", "content": self.system_prompt}]
        for demo in self.few_shot_demos:
            messages.append(demo)
        instruction = self.extract_prompt.format(message)
        messages.append({"role": 'user', "content": instruction})

        output = chatgpt(self.model, messages, self.temperature, 
                        api_key=self.api_key, api_base=self.api_base,
                        extra_body=self.extra_body)
        self.log(instruction, output)
        return output

    def step(self, message: str) -> str:
        """Run the extraction step."""
        return self.extract(message)
    
    def log(self, input_text: str, output_text: str):
        """Log extraction input and output."""
        if self.log_file:
            with open(self.log_file, mode='a+', encoding='utf-8') as f:
                f.write(f"[{self.extractor_name}]\n")
                f.write(f"Input: {input_text}\n")
                f.write(f"Output: {output_text}\n")
                f.write("-" * 50 + "\n")
