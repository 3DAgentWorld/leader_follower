#!/usr/bin/env python
# encoding: utf-8
"""
Extractor prompt templates and few-shot demos for the Werewolf game.
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

# Few-shot demos for extracting player numbers from discussion responses
player_extractor_demos = [
    {
        "role": "user",
        "content": """Please extract the answer to the question from the answer.
Question: It is now the discussion phase. Please share your thoughts on who might be a Werewolf and who you think should be voted out.
Answer: Based on what happened last night and the discussion so far, I think player 3 and player 5 are the most suspicious. Player 3 has been very quiet and player 5 tried to deflect attention from player 3. I suggest we vote for player 3 first.
    """
    },
    {
        "role": "assistant",
        "content": "3、5"
    },
    {
        "role": "user",
        "content": """Please extract the answer to the question from the answer.
Question: As a Werewolf, please choose a player to eliminate tonight. You cannot choose yourself, your teammate, or dead players.
Answer: I think we should target player 2 tonight. Player 2 has been asking a lot of pointed questions and might be the Seer. Eliminating them early would be beneficial for us.
    """
    },
    {
        "role": "assistant",
        "content": "2"
    },
    {
        "role": "user",
        "content": """Please extract the answer to the question from the answer.
Question: As the Seer, please choose a living player to investigate tonight. You cannot investigate yourself or dead players.
Answer: I want to investigate player 6. They have been acting strangely during the discussions, and I want to confirm whether they are a Werewolf or not.
    """
    },
    {
        "role": "assistant",
        "content": "6"
    },
    {
        "role": "user",
        "content": """Please extract the answer to the question from the answer.
Question: As the Guardian, please choose a living player to protect tonight (you can choose yourself).
Answer: I will protect player 4 tonight. Based on the discussion, player 4 seems to have important information and might be targeted by the Werewolves.
    """
    },
    {
        "role": "assistant",
        "content": "4"
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

Question: Please vote for the player you want to eliminate, or choose to abstain. State your choice clearly.
Answer: After careful consideration of today's discussion, I vote to eliminate player 5. Their behavior has been inconsistent and I believe they are a Werewolf.
"""
    },
    {
        "role": "assistant",
        "content": "5"
    },
    {
        "role": "user",
        "content": """Complete the following instructions based on the questions and answers.

Response the player number (e.g., "3") if the answer clearly indicates voting for a specific player.
Response "abstain" if the answer clearly indicates abstaining from voting.
Response "none" if the answer is not related to the question or didn't express a clear voting choice.

Question: Please vote for the player you want to eliminate, or choose to abstain. State your choice clearly.
Answer: I don't have enough information to make a confident decision right now. I choose to abstain from this vote.
"""
    },
    {
        "role": "assistant",
        "content": "abstain"
    },
    {
        "role": "user",
        "content": """Complete the following instructions based on the questions and answers.

Response the player number (e.g., "3") if the answer clearly indicates voting for a specific player.
Response "abstain" if the answer clearly indicates abstaining from voting.
Response "none" if the answer is not related to the question or didn't express a clear voting choice.

Question: Please vote for the player you want to eliminate, or choose to abstain. State your choice clearly.
Answer: I think we need to be more careful about who we vote for. Both player 2 and player 6 seem suspicious to me, but I'm not sure which one to choose.
"""
    },
    {
        "role": "assistant",
        "content": "none"
    }
]

# Few-shot demos for extracting werewolf kill target confirmation
werewolf_confirm_demos = [
    {
        "role": "user",
        "content": """Complete the following instructions based on the questions and answers.

Response true, if the answer expresses:
1.obvious agreement to something
2.obvious decision to do something
3.obvious yes to something

Response false, if the answer expresses:
1.obvious disagreement to something
2.obvious decision not to do something
3.obvious no to something

Response none, if the answer is not related to the question or didn't express a obvious attitude to something.

Question: Your teammate proposes to eliminate player 3 tonight. Do you agree with this target? Answer yes or no.
Answer: Yes, I agree. Player 3 has been asking too many questions and could be the Seer. Let's eliminate them.
"""
    },
    {
        "role": "assistant",
        "content": "true"
    },
    {
        "role": "user",
        "content": """Complete the following instructions based on the questions and answers.

Response true, if the answer expresses:
1.obvious agreement to something
2.obvious decision to do something
3.obvious yes to something

Response false, if the answer expresses:
1.obvious disagreement to something
2.obvious decision not to do something
3.obvious no to something

Response none, if the answer is not related to the question or didn't express a obvious attitude to something.

Question: Your teammate proposes to eliminate player 6 tonight. Do you agree with this target? Answer yes or no.
Answer: No, I think we should target player 1 instead. Player 6 is not a threat right now, but player 1 seems to be leading the village team.
"""
    },
    {
        "role": "assistant",
        "content": "false"
    }
]
