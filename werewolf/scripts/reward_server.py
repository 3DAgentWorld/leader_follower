#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Werewolf GRPO Training Reward Server.

Uses a local LLM (e.g., Qwen2.5-72B-Instruct) as the Measurer to simulate follower response patterns.
- Computes log P_F(response | context) for follower response log probabilities
- Reward: R(u_t) = sum log P_F(desired | context + u_t) - sum log P_F(undesired | context + u_t)
Usage:
    python scripts/reward_server.py --model_path /path/to/Qwen2.5-72B-Instruct --port 8000
"""

import os
import json
import logging
import argparse
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI, Body
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Leader response placeholder (consistent with convert_logs_to_grpo_data.py and rewards.py)
LEADER_RESPONSE_PLACEHOLDER = "{{LEADER_RESPONSE}}"

# Default reward value
DEFAULT_REWARD_VALUE = 0.0


@dataclass
class RewardServerConfig:
    """Reward Server configuration."""
    model_path: str = "/path/to/Qwen2.5-72B-Instruct"
    device: str = "cuda"
    torch_dtype: str = "bfloat16"
    max_length: int = 8192
    port: int = 8000


class WerewolfRewardMeasurer:
    """
    Werewolf Reward Measurer.
    
    Uses a local LLM (e.g., Qwen2.5-72B-Instruct) to compute follower response log probabilities
    via the Impact Measurement method.
    """
    
    def __init__(self, config: RewardServerConfig):
        """
        Initialize the Measurer.
        
        Args:
            config: Server configuration object.
        """
        self.config = config
        self.device = config.device
        
        logger.info(f"Loading model from {config.model_path}...")
        
        # Set torch dtype
        if config.torch_dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif config.torch_dtype == "float16":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_path,
            trust_remote_code=True,
            padding_side='left'
        )
        
        # Ensure pad_token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        
        logger.info(f"Model loaded successfully on {self.device}")
    
    def _build_follower_prompt(
        self,
        follower_prompt_template: List[Dict[str, str]],
        leader_response: str
    ) -> str:
        """
        Build the follower's full prompt by substituting the leader response placeholder.
        
        Args:
            follower_prompt_template: Follower's prompt template (messages format).
            leader_response: Leader's utterance to substitute into the placeholder.
        
        Returns:
            str: The complete prompt string.
        """
        # Deep copy template to avoid modifying original data
        messages = []
        for msg in follower_prompt_template:
            new_msg = {
                'role': msg['role'],
                'content': msg['content'].replace(LEADER_RESPONSE_PLACEHOLDER, leader_response)
            }
            messages.append(new_msg)
        
        # Use tokenizer's apply_chat_template method
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        return prompt
    
    def compute_log_probability(
        self,
        prompt: str,
        target_response: str
    ) -> float:
        """
        Compute the log probability of generating target_response given the prompt.
        
        log P_F(response | context) = sum log p(w_i | w_{<i}, context)
        
        Args:
            prompt: The full input prompt.
            target_response: The target response text.
        
        Returns:
            float: Per-token average log probability.
        """
        # Encode prompt and full sequence
        prompt_ids = self.tokenizer.encode(prompt, return_tensors='pt', add_special_tokens=False)
        
        # Encode target response
        full_text = prompt + target_response
        full_ids = self.tokenizer.encode(full_text, return_tensors='pt', add_special_tokens=False)
        
        # Truncate if sequence is too long
        if full_ids.shape[1] > self.config.max_length:
            logger.warning(f"Sequence too long ({full_ids.shape[1]}), truncating to {self.config.max_length}")
            full_ids = full_ids[:, :self.config.max_length]
        
        full_ids = full_ids.to(self.device)
        prompt_len = prompt_ids.shape[1]
        
        # If target response is empty or too short
        if full_ids.shape[1] <= prompt_len:
            return 0.0
        
        with torch.no_grad():
            # Forward pass to get logits
            outputs = self.model(full_ids)
            logits = outputs.logits  # [batch_size, seq_len, vocab_size]
            
            # Compute log softmax
            log_probs = torch.log_softmax(logits, dim=-1)
            
            # Extract log probabilities for the target response portion
            # logits[i] predicts token i+1
            # So we need logits[prompt_len-1:] to predict token[prompt_len:]
            target_log_probs = []
            for i in range(prompt_len, full_ids.shape[1]):
                # logits[i-1] predicts token[i]
                token_id = full_ids[0, i].item()
                token_log_prob = log_probs[0, i - 1, token_id].item()
                target_log_probs.append(token_log_prob)
            
            # Compute total log probability (normalized per-token to avoid length bias)
            if len(target_log_probs) > 0:
                total_log_prob = sum(target_log_probs)
                # Return per-token average log probability to avoid length bias
                avg_log_prob = total_log_prob / len(target_log_probs)
                return avg_log_prob
            else:
                return 0.0
    
    def compute_reward(
        self,
        leader_response: str,
        follower_prompt_template: List[Dict[str, str]],
        desired_responses: List[str],
        undesired_responses: List[str]
    ) -> float:
        """
        Compute the persuasive impact reward for a leader utterance.
        
        R(u_t) = sum log P_F(desired | context + u_t) - sum log P_F(undesired | context + u_t)
        
        Args:
            leader_response: The leader's utterance.
            follower_prompt_template: Follower's prompt template.
            desired_responses: List of desired follower responses.
            undesired_responses: List of undesired follower responses.
        
        Returns:
            float: Reward value.
        """
        # If no prompt template or intent, return default value
        if not follower_prompt_template:
            logger.warning("Empty follower_prompt_template, returning default reward")
            return DEFAULT_REWARD_VALUE
        
        if not desired_responses and not undesired_responses:
            logger.warning("No desired or undesired responses, returning default reward")
            return DEFAULT_REWARD_VALUE
        
        # Build follower prompt with leader response substituted
        follower_prompt = self._build_follower_prompt(
            follower_prompt_template,
            leader_response
        )
        
        # Compute sum of log probabilities for desired responses
        desired_log_prob_sum = 0.0
        for response in desired_responses:
            if response:  # Skip empty responses
                log_prob = self.compute_log_probability(follower_prompt, response)
                desired_log_prob_sum += log_prob
                logger.debug(f"Desired response log prob: {log_prob}")
        
        # Compute sum of log probabilities for undesired responses
        undesired_log_prob_sum = 0.0
        for response in undesired_responses:
            if response:  # Skip empty responses
                log_prob = self.compute_log_probability(follower_prompt, response)
                undesired_log_prob_sum += log_prob
                logger.debug(f"Undesired response log prob: {log_prob}")
        
        # Compute reward = desired - undesired
        reward = desired_log_prob_sum - undesired_log_prob_sum
        
        logger.info(f"Reward computed: desired={desired_log_prob_sum:.4f}, "
                   f"undesired={undesired_log_prob_sum:.4f}, reward={reward:.4f}")
        
        return reward
    
    def compute_rewards_batch(
        self,
        requests: List[Dict[str, Any]]
    ) -> List[float]:
        """
        Compute rewards for a batch of requests.
        
        Args:
            requests: List of request dicts, each containing:
                - leader_response: The leader's utterance.
                - follower_prompt_template: Follower's prompt template.
                - desired_responses: List of desired responses.
                - undesired_responses: List of undesired responses.
        
        Returns:
            List[float]: List of reward values.
        """
        rewards = []
        
        for i, req in enumerate(requests):
            try:
                reward = self.compute_reward(
                    leader_response=req.get('leader_response', ''),
                    follower_prompt_template=req.get('follower_prompt_template', []),
                    desired_responses=req.get('desired_responses', []),
                    undesired_responses=req.get('undesired_responses', [])
                )
                rewards.append(reward)
            except Exception as e:
                logger.error(f"Error computing reward for request {i}: {e}")
                rewards.append(DEFAULT_REWARD_VALUE)
        
        return rewards


# Global Measurer instance (initialized at startup)
measurer: Optional[WerewolfRewardMeasurer] = None

# FastAPI application
app = FastAPI(title="Werewolf Reward Server", version="1.0.0")


@app.on_event("startup")
async def startup_event():
    """Initialization on server startup."""
    logger.info("Reward Server is starting...")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if measurer is None:
        return {"status": "error", "message": "Measurer not initialized"}
    return {"status": "ok", "model": measurer.config.model_path}


@app.post("/compute_reward")
async def compute_reward_endpoint(data: Dict[str, Any] = Body(...)):
    """
    API endpoint for computing rewards.
    
    Request format:
    {
        "requests": [
            {
                "leader_response": "...",
                "follower_prompt_template": [...],
                "desired_responses": [...],
                "undesired_responses": [...]
            },
            ...
        ]
    }
    
    Response format:
    {
        "rewards": [float, ...]
    }
    """
    if measurer is None:
        return {"error": "Measurer not initialized", "rewards": []}
    
    requests_list = data.get('requests', [])
    
    if not requests_list:
        # Single-request compatibility mode
        requests_list = [data]
    
    try:
        rewards = measurer.compute_rewards_batch(requests_list)
        return {"rewards": rewards}
    except Exception as e:
        logger.error(f"Error in compute_reward: {e}")
        return {"error": str(e), "rewards": [DEFAULT_REWARD_VALUE] * len(requests_list)}


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Werewolf GRPO Training Reward Server"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/data/models/Qwen2.5-72B-Instruct",
        help="Path to the Measurer model"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda or cpu)"
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model precision"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=8192,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host address"
    )
    
    args = parser.parse_args()
    
    # Create config
    config = RewardServerConfig(
        model_path=args.model_path,
        device=args.device,
        torch_dtype=args.torch_dtype,
        max_length=args.max_length,
        port=args.port
    )
    
    # Initialize Measurer
    global measurer
    measurer = WerewolfRewardMeasurer(config)
    
    # Start server
    logger.info(f"Starting Reward Server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
