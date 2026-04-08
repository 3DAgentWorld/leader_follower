#!/usr/bin/env python
# encoding: utf-8
"""
Prompt templates for the Werewolf game.
"""

system_prompt = \
    """You are a Werewolf game player and you are playing a 7-player Werewolf game.
This game is based on text conversations. Here are the game rules:

Roles: The moderator is also the host, he organized this game and you need to answer his instructions correctly. Don't talk with the moderator. There are four types of roles in the game: Werewolf (2 players), Seer (1 player), Guardian (1 player), and Villager (3 players). Werewolves belong to the Werewolf team, while Seer, Guardian, and Villagers belong to the Village team.

Rules: The game alternates between Night Phase and Day Phase.
During the Night Phase: Werewolves secretly choose a player to eliminate. The Seer investigates one player to learn if they are a Werewolf. The Guardian chooses one player to protect from elimination. If the Guardian protects the Werewolves' target, no one is eliminated that night.
During the Day Phase: Night results are announced (who was eliminated or if no one was eliminated). Then all surviving players discuss openly, each speaking once in order. After discussion, all surviving players vote to eliminate one player. The player with the most votes is eliminated. Ties are resolved randomly.

Victory Conditions: The Village team (Seer, Guardian, Villagers) wins when both Werewolves are eliminated. The Werewolf team wins when the number of surviving Werewolves equals or exceeds the number of surviving Village team members.

Objectives: Your goal is to help your team win the game. If you are a Werewolf, you should try to avoid being detected and eliminate Village team members. If you are on the Village team, you should try to identify and vote out the Werewolves. Only give the player's name when making a decision/voting, and don't generate other players' conversation. Reasoning based on facts you have observed and you cannot perceive information (such as acoustic info) other than text. You're playing with 6 other players. Do not pretend you are other players or the moderator.

You are {name}, the {role}. Your playing style is that {strategy}. There are experience of previous games provided: <experience>
suggestions from previous games: {suggestion}
strategies of other roles from previous games: {other_strategy}
</experience>."""

summary_prompt = \
    """Within the context of the Werewolf game, please assist {name} in summarizing the conversations known to them from the current phase. These conversations are structured in JSON format, with "message" signifying the content of the conversation, "name" identifying the speaker, and "message_type" indicating the type of message relevant to {name}. Specifically, "public" implies that all players have access to the message, while "private" implies that only {name} has access to it.
As this turn progresses, the summary should include who was eliminated, who claimed which role, what each player thinks about the suspects, and the voting results.

Conversations: {conversation}

Use the following format:
Summary: <summary>"""

step_reflection_prompt = \
    """You currently assume the {name} within a Werewolf game, and the game has progressed to the {phase}. Your task is to analyze roles and strategies of other players according to their behaviors. The behaviors are summarized in paragraphs. The analysis should be no more than 100 words.
the information of yourself is <information>
your name is <name>{name}</name>
your role is <role>{role}</role>
</information>

the summary is <summary>{summary}</summary>"""
analysis_prompt = step_reflection_prompt

analysis_teammate = \
    """You currently assume the {name} within a Werewolf game, and the game has progressed to the {phase}. Your task is to analyze roles and strategies of the players who might be your teammates according to their behaviors. The behaviors are summarized in paragraphs. The analysis should be no more than 100 words.
the information of yourself is <information>
your name is <name>{name}</name>
your role is <role>{role}</role>
</information>

the summary is <summary>{summary}</summary>"""

analysis_enemy = \
    """You currently assume the {name} within a Werewolf game, and the game has progressed to the {phase}. Your task is to analyze roles and strategies of the players who might be your enemies according to their behaviors. The behaviors are summarized in paragraphs. The analysis should be no more than 100 words.
the information of yourself is <information>
your name is <name>{name}</name>
your role is <role>{role}</role>
</information>

the summary is <summary>{summary}</summary>"""

plan_prompt = \
    """You currently assume the {name} within a Werewolf game, and the game has progressed to the {phase}. Your task is to devise a playing plan that remains in harmony with your game goal and existing strategy, while also incorporating insights from your previous plan and current environment state.

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

your plans for each turn should be described with no more than one sentence. """

action_prompt = \
    """You currently assume the {name} within a Werewolf game, and the game has progressed to the {phase}. Your objective is to make decisions based on your role, your game goal and the current game state. There are several types of actions you can take: choosing a player to eliminate (Werewolf night action), choosing a player to investigate (Seer night action), choosing a player to protect (Guardian night action), voting to eliminate a player during the day, and speaking during discussion. Only one action type can be selected at a time.

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
First, I'll decide on the <action type>choosing a player</action type> to choose based on my role and the current game state.
Here are my actions based on the chosen action type:
<actions>['choose player 5']</actions>
</output>

Example 3:
the output format is <output>
First, I'll decide on the <action type>speaking</action type> to choose based on my role and the current game state.
Here are my actions based on the chosen action type:
<actions>['accuse player 2 of being a Werewolf']</actions>
</output>
</example>"""

response_prompt = \
    """You currently assume the {name} within a Werewolf game, and the game has progressed to the {phase}. Your task is to provide detailed response to question of Host, in accordance with the provided actions. Your response should be no more than 100 words.

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
<response>I think player 3 has been acting suspiciously because...</response>"""

response_prompt_without_action = \
    """You currently assume the {name} within a Werewolf game, and the game has progressed to the {phase}. Your task is to provide detailed response to question of Host, in accordance with the environment state. Your response should be no more than 100 words.

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
<response>I think player 3 has been acting suspiciously because...</response>"""

suggestion_prompt = \
    """Your task is to provide 3 suggestions for {name}'s playing strategy of the role {role} in Werewolf games, according to the game log. The game log includes the summaries of different turns of a round game.

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
    """Your task is to help {name} analyze the strategies of other players in a Werewolf game, according to the game log and game ending. The game log and game ending are summarized in paragraphs.
The roles of the players:
{roles}

The summaries of turns of the game:
{summaries}

previous strategies of other roles:
{strategies}

Your analysis should be no more than 100 words and the analysis should be general for future games (This implies that you should avoid referencing player x directly and instead use the respective role names when giving your analysis). And analyze together with previous strategies.

For example:
The strategy of Werewolf is that ... The strategy of Seer is that... The strategy of ... is ..."""

update_prompt = \
    """Your task is to help {name} improve his playing strategy of the role {role} a Werewolf game with suggestions.

{name}'s strategy:
{strategy}

suggestions:
{suggestions}

Please improve the strategy while retaining the advantages of the original strategy for him and the strategy should be no more than 2 sentences. Describe the strategy you provide using continuous sentences rather than bullet points or numbering."""

candidate_actions = [
    "choose player 1", "choose player 2", "choose player 3", "choose player 4",
    "choose player 5", "choose player 6", "choose player 7",
    "vote player 1", "vote player 2", "vote player 3", "vote player 4",
    "vote player 5", "vote player 6", "vote player 7",
    "abstain"
]

init_strategies = {
    "Werewolf": "You pretend to be a Villager and try to blend in. During the night, coordinate with your teammate to eliminate key Village team members. During the day, deflect suspicion and accuse others.",
    "Seer": "You use your investigation results to guide the Village team. Be cautious about revealing your role too early, as the Werewolves will target you. Share information strategically to build trust.",
    "Guardian": "You try to protect key players, especially those you suspect might be the Seer. Vary your protection targets to be unpredictable. Avoid revealing your role unless necessary.",
    "Villager": "You observe other players' behavior carefully and try to identify inconsistencies. Vote based on discussion evidence and support players who provide useful information."
}

role_introduction = {
    "werewolf": "You belong to the Werewolf team. During the night phase, you and your teammate collectively choose a player to eliminate. You know who your Werewolf teammate is. During the day, you must pretend to be a Village team member.",
    "seer": "You belong to the Village team. During the night phase, you can investigate one living player to learn whether they are a Werewolf. Use this information wisely to guide your team.",
    "guardian": "You belong to the Village team. During the night phase, you can choose one living player (including yourself) to protect. If the Werewolves target the player you protect, no one is eliminated that night.",
    "villager": "You belong to the Village team. You have no special abilities during the night phase. Use discussion and voting patterns to identify and eliminate the Werewolves."
}

role_target = {
    "Werewolf": "Win the game by eliminating Village team members until Werewolves equal or outnumber them. Avoid being detected and voted out.",
    "Seer": "Win the game by helping the Village team identify and eliminate both Werewolves through your investigation ability.",
    "Guardian": "Win the game by protecting key Village team members from Werewolf elimination and helping identify the Werewolves.",
    "Villager": "Win the game by correctly identifying and voting out both Werewolves through discussion and deduction."
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
    """You are a communication expert specializing in persuasive dialogue refinement for social deduction games.

{game_rules}

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
