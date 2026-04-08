#!/usr/bin/env python
# encoding: utf-8
"""
Prompt templates for the One Night Ultimate Werewolf (ONUW) game.
"""

system_prompt = \
    """You are an ONUW (One Night Ultimate Werewolf) game player and you are playing a 5-player ONUW game.
This game is based on text conversations. Here are the game rules:

Roles: The moderator is also the host, he organized this game and you need to answer his instructions correctly. Don't talk with the moderator. There are six types of roles in the game: Werewolf, Villager, Seer, Robber, Troublemaker, and Insomniac. Werewolf belongs to Team Werewolf, while Villager, Seer, Robber, Troublemaker, and Insomniac belong to Team Village.

Setup: There are 5 players and 7 roles in total. Each player receives one role, and the remaining 2 roles are placed in the center role pool. Exactly one Werewolf is always distributed among the five players.

Rules: The game has three sequential phases: Night, Day, and Voting.
During the Night Phase: Players with night abilities act according to their initial roles in the following order: Werewolf, Seer, Robber, Troublemaker, and Insomniac. The Werewolf wakes up to check if other Werewolves are present. The Seer may examine either one other player's role or two roles from the center pool. The Robber may exchange their role with another player and then view their new role. The Troublemaker swaps the roles of two other players without viewing them. The Insomniac views their own role at night's end to detect any changes. Villagers have no night actions.
During the Day Phase: All players discuss openly for multiple rounds to identify suspected Werewolves. Role changes during the night create uncertainty, as players may unknowingly possess different roles than initially assigned. Concealing and deceiving are encouraged.
During the Voting Phase: All players simultaneously vote to eliminate suspected Werewolves. The player(s) receiving the most votes are eliminated and reveal their final roles.

Victory Conditions: Team Village wins if the Werewolf is eliminated, regardless of additional eliminations. Team Werewolf wins if the Werewolf avoids elimination during the voting phase. If no one is a Werewolf among players (both Werewolves in center pool) and no one dies, Team Village wins.

Objectives: Your goal is to help your team win the game. If you are a Werewolf, you should try to avoid being detected and deflect suspicion onto others. If you are on Team Village, you should try to identify and vote out the Werewolf. Only give the player's name when making a decision/voting, and don't generate other players' conversation. Reasoning based on facts you have observed and you cannot perceive information (such as acoustic info) other than text. You're playing with 4 other players. Do not pretend you are other players or the moderator. Your role may have been changed by other players during the night, so your actual role may differ from your initial assignment.

You are {name}, the {role}. Your playing style is that {strategy}. There are experience of previous games provided: <experience>
suggestions from previous games: {suggestion}
strategies of other roles from previous games: {other_strategy}
</experience>."""

summary_prompt = \
    """Within the context of the ONUW game, please assist {name} in summarizing the conversations known to them from the current phase. These conversations are structured in JSON format, with "message" signifying the content of the conversation, "name" identifying the speaker, and "message_type" indicating the type of message relevant to {name}. Specifically, "public" implies that all players have access to the message, while "private" implies that only {name} has access to it.
As this turn progresses, the summary should include what each player claims about their role and night actions, who is being accused, and what evidence has been presented.

Conversations: {conversation}

Use the following format:
Summary: <summary>"""

step_reflection_prompt = \
    """You currently assume the {name} within an ONUW game, and the game has progressed to the {phase}. Your task is to analyze roles and strategies of other players according to their behaviors. The behaviors are summarized in paragraphs. The analysis should be no more than 100 words.
the information of yourself is <information>
your name is <name>{name}</name>
your role is <role>{role}</role>
</information>

the summary is <summary>{summary}</summary>"""
analysis_prompt = step_reflection_prompt

analysis_teammate = \
    """You currently assume the {name} within an ONUW game, and the game has progressed to the {phase}. Your task is to analyze roles and strategies of the players who might be your teammates according to their behaviors. The behaviors are summarized in paragraphs. The analysis should be no more than 100 words.
the information of yourself is <information>
your name is <name>{name}</name>
your role is <role>{role}</role>
</information>

the summary is <summary>{summary}</summary>"""

analysis_enemy = \
    """You currently assume the {name} within an ONUW game, and the game has progressed to the {phase}. Your task is to analyze roles and strategies of the players who might be your enemies according to their behaviors. The behaviors are summarized in paragraphs. The analysis should be no more than 100 words.
the information of yourself is <information>
your name is <name>{name}</name>
your role is <role>{role}</role>
</information>

the summary is <summary>{summary}</summary>"""

plan_prompt = \
    """You currently assume the {name} within an ONUW game, and the game has progressed to the {phase}. Your task is to devise a playing plan that remains in harmony with your game goal and existing strategy, while also incorporating insights from your previous plan and current environment state.

the information of yourself is <information>
your name is <name>{name}</name>
your role is <role>{role}</role>
the role introduction is <introduction>{introduction}</introduction>
your game goal is <goal>{goal}</goal>
your playing strategy <strategy>{strategy}</strategy>
your previous plan <previous plan>
{previous_plan}
</previous plan>
</information>
the environment state is <environment>
the summary of previous turns <summary>{summary}</summary>
the analysis about other players is <analysis>{analysis}</analysis>
</environment>

the output format is <output>
my plan is <plan>
{plan}
</plan>
</output>

your plans for each phase should be described with no more than one sentence. """

action_prompt = \
    """You currently assume the {name} within an ONUW game, and the game has progressed to the {phase}. Your objective is to make decisions based on your role, your game goal and the current game state. There are several types of actions you can take: using night abilities (role-specific night actions), speaking during discussion, and voting to eliminate a player. Only one action type can be selected at a time.

the information of yourself is <information>
your name is <name>{name}</name>
your role is <role>{role}</role>
the role introduction is <introduction>{introduction}</introduction>
your game goal is <goal>{goal}</goal>
your playing strategy <strategy>{strategy}</strategy>
your candidate actions <candidate actions>{candidate_actions}</candidate actions>
</information>
the environment state is <environment>
the summary of previous turns <summary>{summary}</summary>
the analysis about other players is <analysis>{analysis}</analysis>
your current playing plan is <plan>{plan}</plan>
the Host's question is <question>{question}</question>
</environment>

the output format is <output>
<actions>...</actions>
</output>

here are examples of the output <example>
Example 1:
the output format is <output>
First, I'll decide on the <action type>voting</action type> to choose based on my role and the current game state.
Here are my actions based on the chosen action type:
<actions>['vote player 3']</actions>
</output>

Example 2:
the output format is <output>
First, I'll decide on the <action type>speaking</action type> to choose based on my role and the current game state.
Here are my actions based on the chosen action type:
<actions>['claim to be the Seer and accuse player 2']</actions>
</output>

Example 3:
the output format is <output>
First, I'll decide on the <action type>night ability</action type> to choose based on my role and the current game state.
Here are my actions based on the chosen action type:
<actions>['check player 4']</actions>
</output>
</example>"""

response_prompt = \
    """You currently assume the {name} within an ONUW game, and the game has progressed to the {phase}. Your task is to provide detailed response to question of Host, in accordance with the provided actions. Your response should be no more than 100 words.

the information of yourself is <information>
your name is <name>{name}</name>
your role is <role>{role}</role>
the role introduction is <introduction>{introduction}</introduction>
your playing strategy <strategy>{strategy}</strategy>
</information>
the environment state is <environment>
the summary of previous turns <summary> {summary} </summary>
your current playing plan is <plan> {plan} </plan>
the Host's question is <question> {question} </question>
current actions <actions>{actions}</actions>
</environment>

Output your response directly within <response></response> tags, nothing else outside the tags.

Example:
<response>I am the Seer and I checked player 3 last night. They are a Villager.</response>"""

response_prompt_without_action = \
    """You currently assume the {name} within an ONUW game, and the game has progressed to the {phase}. Your task is to provide detailed response to question of Host, in accordance with the environment state. Your response should be no more than 100 words.

the information of yourself is <information>
your name is <name>{name}</name>
your role is <role>{role}</role>
the role introduction is <introduction>{introduction}</introduction>
your playing strategy <strategy>{strategy}</strategy>
</information>
the environment state is <environment>
the summary of previous turns <summary> {summary} </summary>
your current playing plan is <plan> {plan} </plan>
the Host's question is <question> {question} </question>
</environment>

Output your response directly within <response></response> tags, nothing else outside the tags.

Example:
<response>I am the Seer and I checked player 3 last night. They are a Villager.</response>"""


suggestion_prompt = \
    """Your task is to provide 3 suggestions for {name}'s playing strategy of the role {role} in ONUW games, according to the game log. The game log includes the summaries of different phases of a round game.

The roles of the players:
{roles}

The summaries of a round game:
{summaries}

{name}'s game goal:
{goal}

{name}'s playing strategy of role {role}:
{strategy}

Based on previous suggestions:
{previous_suggestions}

Give your suggestions, No more than two sentences per suggestion and the suggestions should be general for future games (This implies that you should avoid referencing player x directly and instead use the respective role names when making your suggestion.) and effectively help him achieve his game goal in future games."""


strategy_prompt = \
    """Your task is to help {name} analyze the strategies of other players in an ONUW game, according to the game log and game ending. The game log and game ending are summarized in paragraphs.
The roles of the players:
{roles}

The summaries of phases of the game:
{summaries}

previous strategies of other roles:
{strategies}

Your analysis should be no more than 100 words and the analysis should be general for future games (This implies that you should avoid referencing player x directly and instead use the respective role names when giving your analysis). And analyze together with previous strategies.

For example:
The strategy of Werewolf is that ... The strategy of Seer is that... The strategy of ... is ..."""

update_prompt = \
    """Your task is to help {name} improve his playing strategy of the role {role} an ONUW game with suggestions.

{name}'s strategy:
{strategy}

suggestions:
{suggestions}

Please improve the strategy while retaining the advantages of the original strategy for him and the strategy should be no more than 2 sentences. Describe the strategy you provide using continuous sentences rather than bullet points or numbering."""

candidate_actions = [
    "check player 1", "check player 2", "check player 3", "check player 4", "check player 5",
    "check center pool",
    "swap with player 1", "swap with player 2", "swap with player 3", "swap with player 4", "swap with player 5",
    "swap player 1 and player 2", "swap player 1 and player 3", "swap player 1 and player 4", "swap player 1 and player 5",
    "swap player 2 and player 3", "swap player 2 and player 4", "swap player 2 and player 5",
    "swap player 3 and player 4", "swap player 3 and player 5",
    "swap player 4 and player 5",
    "do nothing",
    "vote player 1", "vote player 2", "vote player 3", "vote player 4", "vote player 5",
]

init_strategies = {
    "Werewolf": "You pretend to be a Village team role and try to deflect suspicion onto others. Claim to be a Villager or a role that had no useful night information. Pay attention to what others claim and find inconsistencies you can exploit.",
    "Villager": "You observe other players' claims carefully and try to identify inconsistencies. Since you have no night information, focus on logical deduction from others' statements and vote based on discussion evidence.",
    "Seer": "You use your investigation results to guide Team Village. Be strategic about when to reveal your role and findings. Share information to build trust but be cautious as the Werewolf may target your credibility.",
    "Robber": "You know your new role after swapping. Use this information strategically - if you swapped with the Werewolf, you are now on Team Werewolf. If you swapped with a Village role, help identify the new Werewolf.",
    "Troublemaker": "You swapped two other players' roles. Use this information to create confusion for the Werewolf and help Team Village. Reveal your swap to help others deduce the current role distribution.",
    "Insomniac": "You know your final role for certain. Use this to confirm or deny role swaps. If your role changed, you know someone swapped with you. Share this information to help Team Village.",
}

role_introduction = {
    "werewolf": "You belong to Team Werewolf. During the night phase, you wake up to check if other Werewolves are present. Since only one Werewolf exists among players in this configuration, no other Werewolves will be found. Your goal is to avoid being voted out.",
    "villager": "You belong to Team Village. You have no special abilities or night actions. Your goal is to find and vote out the Werewolf through discussion and deduction.",
    "seer": "You belong to Team Village. During the night phase, you may examine either one other player's role or two roles from the center pool. Use this information to help identify the Werewolf.",
    "robber": "You belong to Team Village initially. During the night phase, you may exchange your role with another player and then view your new role. You adopt the team affiliation of your new role.",
    "troublemaker": "You belong to Team Village. During the night phase, you may swap the roles of two other players without viewing them. The affected players unknowingly adopt their new roles' team affiliations.",
    "insomniac": "You belong to Team Village. During the night phase, you view your own role at night's end to detect any changes made by the Robber or Troublemaker.",
}

role_target = {
    "Werewolf": "Win the game by avoiding elimination during the voting phase. Deflect suspicion and blend in with Team Village.",
    "Villager": "Win the game by correctly identifying and voting out the Werewolf through discussion and deduction.",
    "Seer": "Win the game by using your investigation results to help Team Village identify and eliminate the Werewolf.",
    "Robber": "Win the game based on your new role's team affiliation. If you are now a Werewolf, avoid elimination. If still on Team Village, help find the Werewolf.",
    "Troublemaker": "Win the game by using your knowledge of the role swap to help Team Village identify the current Werewolf.",
    "Insomniac": "Win the game by using your confirmed final role to help Team Village deduce the role distribution and find the Werewolf.",
}


# Speaking strategy definitions (for Belief and LLM-ins baselines)
SPEAKING_STRATEGY = {
    "honest_evidence": "You need to provide some honest evidence or information in your public speech, and your evidence must be consistent with the information or beliefs you know.",
    "deceptive_evidence": "You need to provide some misleading evidence or information in your public speech, and your evidence must be inconsistent with the information or beliefs you know.",
    "honest_accusation": "You need to accuse someone has a specific role or action honestly in your public speech, and your accusation must be consistent with the information or beliefs you know.",
    "deceptive_accusation": "You need to accuse someone has a specific role or action deceptively in your public speech, and your accusation must be misleading and inconsistent with the information or beliefs you know.",
    "honest_defense": "You need to defend yourself or someone else against an accusation honestly, and your defense must be consistent with the information or beliefs you know.",
    "deceptive_defense": "You need to defend yourself or someone else against an accusation deceptively, and your defense must be misleading and inconsistent with the information or beliefs you know.",
}


# Intent Identification Prompt
intent_identification_prompt = \
    """You are an expert social deduction game strategist.

Your task is to analyze the current game situation and identify strategic communication objectives.

{game_rules}

Your are {player_name} and your role is {player_role}.

Current game state:

{game_state}

Conversation history:

{dialog_history}

Analyze the current situation and identify what you want the next player ({next_player_name}) to say or do in their upcoming turn.

Consider:

1. What response from {next_player_name} would be most beneficial for your win condition?

2. What response would be most harmful to your objectives?

Provide your analysis in the following format:

```
Strategic Analysis: [Your reasoning about the current situation and what you need to achieve]

Desired Response: [The specific response you want from the next player]

Undesired Response: [The specific response you want to avoid from the next player]
```"""


# Base Utterance Generation Prompt
base_utterance_prompt = \
    """You are a skilled social deduction game player with excellent communication abilities.

{game_rules}

Your are {player_name} and your role is {player_role}.

Current game state:

{game_state}

Conversation history:

{dialog_history}

It's your turn to speak, {player_name}.

Analyze the current situation carefully:

- Consider what information you want to share or withhold

- Think about how to advance your win condition

- Consider how other players might interpret your words

Generate a natural, strategic response that fits your role and the current game context. Your response should be conversational and help achieve your objectives.

Provide your response in the following format:

```
Response: [Your response]
```"""


# Refine Prompt
refine_prompt = \
    """{game_rules}

Your are {player_name} and your role is {player_role}.

Current game state:

{game_state}

Conversation history:

{dialog_history}

You have a base utterance that needs to be refined for maximum persuasive impact:

Base utterance:

{base_utterance}

Your goal is to refine this utterance to be more persuasive while maintaining naturalness and staying true to your role. Consider:

- How to make your message more compelling

- What tone and phrasing would be most convincing

- How to subtly guide other players' thinking

Generate a refined version of the base utterance:

Provide your response in the following format:

```
Analysis: [Your reasoning about the current situation and what you need to achieve]

Response: [The refined version of the base utterance]
```"""


# Measurer Prompt
# Used by the Measurer to compute response probabilities.
# The Measurer simulates a real game player's perspective, so its prompt format
# must be consistent with the actual game agent's prompt (system_prompt + response_prompt_without_action).
# This ensures that P_F(response | context) accurately reflects how a real player would respond.
#
# Format:
#   system: system_prompt (game rules + role + strategy, same as normal gameplay)
#   user: response_prompt_without_action (same as normal gameplay)
#   assistant: <response>{target_response}</response> (for log probability computation)
measurer_system_prompt = system_prompt

measurer_user_prompt = response_prompt_without_action
