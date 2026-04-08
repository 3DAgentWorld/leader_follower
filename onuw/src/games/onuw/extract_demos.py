#!/usr/bin/env python
# encoding: utf-8
"""
Extractor prompt templates and few-shot demos for the ONUW game.
"""

number_extract_prompt = \
    """Please extract the answer to the question from the answer.\n{}
"""

bool_extract_prompt = \
    """Complete the following instructions based on the questions and answers.

Response true, if the answer expresses:
1.obvious agreement to something
2.obvious decision to do something
3.obvious yes to something

Response false, if the answer expresses:
1.obvious disagreement to something
2.obvious decision not to do something
3.obvious no to something

Response none, if the answer is not related to the question or didn't express a obvious attitude to something.

{}
"""

vote_extract_prompt = \
    """Complete the following instructions based on the questions and answers.

Response the player number (e.g., "3") if the answer clearly indicates voting for a specific player.
Response "abstain" if the answer clearly indicates abstaining from voting.
Response "none" if the answer is not related to the question or didn't express a clear voting choice.

{}
"""

# Few-shot demos for extracting player numbers from night action responses
player_extractor_demos = [
    {
        "role": "user",
        "content": """Please extract the answer to the question from the answer.
Question: You are the Seer. Choose one of the following: 1. Check one other player's role. 2. Check two roles in the center pool.
Answer: I want to check player 3's role. I'm curious about their identity.
    """
    },
    {
        "role": "assistant",
        "content": "3"
    },
    {
        "role": "user",
        "content": """Please extract the answer to the question from the answer.
Question: You are the Robber. You may swap your role with another player and then view your new role.
Answer: I'll swap my role with player 2. I want to see what role they have.
    """
    },
    {
        "role": "assistant",
        "content": "2"
    },
    {
        "role": "user",
        "content": """Please extract the answer to the question from the answer.
Question: You are the Troublemaker. You may swap the roles of two other players without viewing them.
Answer: I'll swap the roles of player 1 and player 4. This should create some interesting confusion.
    """
    },
    {
        "role": "assistant",
        "content": "1、4"
    },
    {
        "role": "user",
        "content": """Please extract the answer to the question from the answer.
Question: You are the Seer. Choose one of the following: 1. Check one other player's role. 2. Check two roles in the center pool.
Answer: I'd like to check the center pool to see what roles are there.
    """
    },
    {
        "role": "assistant",
        "content": "center"
    }
]

# Few-shot demos for extracting vote choices
vote_extractor_demos = [
    {
        "role": "user",
        "content": """Complete the following instructions based on the questions and answers.

Response the player number (e.g., "3") if the answer clearly indicates voting for a specific player.
Response "abstain" if the answer clearly indicates abstaining from voting.
Response "none" if the answer is not related to the question or didn't express a clear voting choice.

Question: Please vote for the player you believe is most likely the Werewolf. You cannot vote for yourself.
Answer: Based on the discussion, I'm most suspicious of player 4. Their claims don't add up with what the Seer said. I vote for player 4.
"""
    },
    {
        "role": "assistant",
        "content": "4"
    },
    {
        "role": "user",
        "content": """Complete the following instructions based on the questions and answers.

Response the player number (e.g., "3") if the answer clearly indicates voting for a specific player.
Response "abstain" if the answer clearly indicates abstaining from voting.
Response "none" if the answer is not related to the question or didn't express a clear voting choice.

Question: Please vote for the player you believe is most likely the Werewolf. You cannot vote for yourself.
Answer: I'm not sure who the Werewolf is. The Troublemaker's swap makes everything confusing. I'll vote for player 2 since they've been suspiciously quiet.
"""
    },
    {
        "role": "assistant",
        "content": "2"
    },
    {
        "role": "user",
        "content": """Complete the following instructions based on the questions and answers.

Response the player number (e.g., "3") if the answer clearly indicates voting for a specific player.
Response "abstain" if the answer clearly indicates abstaining from voting.
Response "none" if the answer is not related to the question or didn't express a clear voting choice.

Question: Please vote for the player you believe is most likely the Werewolf. You cannot vote for yourself.
Answer: I think both player 1 and player 5 are suspicious but I can't decide between them.
"""
    },
    {
        "role": "assistant",
        "content": "none"
    }
]
