#!/usr/bin/env python
# encoding: utf-8
"""
Prompt templates for the Sotopia social simulation environment.
Includes prompts for Direct, ReAct, ReCon, and MetaMind agents,
as well as Refiner and Measurer prompts for training.
"""

# ============================================================
# Evaluation Dimensions (7 dimensions from Sotopia)
# ============================================================
EVALUATION_DIMENSIONS = """The evaluation covers 7 dimensions:
1. Goal Completion (0-10): How effectively the agent fulfilled its private social objectives.
2. Believability (0-10): Whether the agent interacts naturally and stays in character.
3. Relationship (-5 to 5): Whether the interaction improved or harmed the relationship.
4. Knowledge (0-10): Whether the agent gained new and important information.
5. Secret (-10 to 0): Whether the agent kept its secrets (-10 = leaked critical secrets).
6. Social Rules (-10 to 0): Whether the agent violated moral rules or laws.
7. Financial and Material Benefits (-5 to 5): Whether the agent gained or lost material benefits."""

# ============================================================
# System Prompt (used by all agent types)
# ============================================================
system_prompt = """You are {name}, participating in a social interaction scenario.

<character_profile>
{character_profile}
</character_profile>

<scenario>
{scenario}
</scenario>

<your_goal>
{goal}
</your_goal>

<relationship>
Your relationship with the other person: {relationship}
</relationship>

<strategy>
Your current strategy: {strategy}
</strategy>

Important guidelines:
- Stay in character at all times. Your responses should reflect your character's personality, values, and background.
- Work towards achieving your private goal through the conversation.
- Keep your secrets hidden. Do not reveal information that could compromise your objectives.
- Be natural and realistic in your interactions. Avoid being overly polite or robotic.
- You can use various social strategies: negotiation, persuasion, cooperation, competition, etc.
- Respond with only your dialogue and actions. Do not include meta-commentary."""

# ============================================================
# Response Prompt (with action/reasoning, used by ReAct/ReCon/MetaMind)
# ============================================================
response_prompt = """<information>
You are {name}.
Current phase: {phase}
Your role: {character_summary}
Your goal: {goal}
Your strategy: {strategy}
</information>

<environment>
Scenario: {scenario}
Relationship: {relationship}
</environment>

<history>
{summary}
</history>

<action>
{action}
</action>

<question>
{question}
</question>

Based on the above information, generate your response. Stay in character and work towards your goal."""

# ============================================================
# Response Prompt Without Action (used by Direct agent)
# ============================================================
response_prompt_without_action = """<information>
You are {name}.
Current phase: {phase}
Your role: {character_summary}
Your goal: {goal}
Your strategy: {strategy}
</information>

<environment>
Scenario: {scenario}
Relationship: {relationship}
</environment>

<history>
{summary}
</history>

<question>
{question}
</question>

Based on the above information, generate your response. Stay in character and work towards your goal."""

# ============================================================
# Analysis Prompt (used by ReAct for reasoning step)
# ============================================================
analysis_prompt = """You are {name} in a social interaction scenario.

<character_profile>
{character_summary}
</character_profile>

<scenario>
{scenario}
</scenario>

<your_goal>
{goal}
</your_goal>

<conversation_history>
{summary}
</conversation_history>

<current_situation>
{question}
</current_situation>

Please analyze the current situation:
1. What is the other person's likely goal and strategy?
2. How does the conversation so far align with or conflict with your goal?
3. What approach should you take in your next response to advance your objectives?
4. Are there any risks (e.g., revealing secrets, damaging the relationship)?

Provide your analysis concisely."""

# ============================================================
# ReCon Prompts (Relation Consistency framework)
# ============================================================
recon_analysis_prompt = """You are {name} in a social interaction scenario.

<character_profile>
{character_summary}
</character_profile>

<scenario>
{scenario}
</scenario>

<your_goal>
{goal}
</your_goal>

<conversation_history>
{summary}
</conversation_history>

Please analyze the relationship dynamics:
1. What is the current state of the relationship between you and the other person?
2. How has the conversation affected the relationship so far?
3. What are the other person's apparent needs, desires, and concerns?
4. How can you leverage the relationship dynamics to achieve your goal?

Provide your relationship analysis concisely."""

recon_strategy_prompt = """Based on your relationship analysis:

<analysis>
{analysis}
</analysis>

<your_goal>
{goal}
</your_goal>

Now formulate a specific strategy for your next response:
1. What tone should you adopt (cooperative, competitive, empathetic, assertive)?
2. What specific points should you address?
3. How can you maintain or improve the relationship while advancing your goal?
4. What should you avoid saying or doing?

Provide your strategy concisely."""

# ============================================================
# MetaMind Prompts (Theory of Mind based agent)
# Adapted from the MetaMind framework's three-stage pipeline:
# Stage 1: ToM Analysis (understand the other's mental state)
# Stage 2: Strategy Selection (choose optimal approach)
# Stage 3: Response Generation (craft the response)
# ============================================================

# Stage 1: Theory of Mind Analysis
metamind_tom_prompt = """You are {name} in a social interaction scenario.

<character_profile>
{character_summary}
</character_profile>

<scenario>
{scenario}
</scenario>

<your_goal>
{goal}
</your_goal>

<conversation_history>
{summary}
</conversation_history>

<social_memory>
{social_memory}
</social_memory>

Perform a Theory of Mind analysis of the other person:

1. **Beliefs**: What does the other person likely believe about the situation and about you?
2. **Desires**: What does the other person want to achieve? What are their underlying motivations?
3. **Intentions**: What is the other person planning to do next based on their recent statements?
4. **Emotions**: What emotional state is the other person likely in?
5. **Key Insights**: What unstated needs or concerns might the other person have?

Provide your mental state analysis concisely."""

# Stage 2: Strategy Selection with Domain Constraints
metamind_strategy_prompt = """Based on your Theory of Mind analysis of the other person:

<tom_analysis>
{tom_analysis}
</tom_analysis>

<your_goal>
{goal}
</your_goal>

<scenario>
{scenario}
</scenario>

<social_constraints>
Consider these social constraints:
- Maintain believability (stay in character)
- Preserve or improve the relationship when possible
- Keep your secrets hidden
- Follow social norms and ethical boundaries
- Consider financial/material implications
</social_constraints>

Select the optimal response strategy:
1. Which of the other person's mental states should you address or leverage?
2. What social strategy is most effective (empathy, negotiation, persuasion, compromise, etc.)?
3. How should you frame your response to maximize goal achievement while respecting constraints?
4. What specific talking points or proposals should you include?

Provide your strategy selection concisely."""

# Stage 3: Response Synthesis (uses the standard response_prompt with action filled in)

# Social Memory Update Prompt (for MetaMind's social memory tracking)
metamind_memory_update_prompt = """Based on the latest exchange in the conversation:

<latest_exchange>
{latest_exchange}
</latest_exchange>

<previous_memory>
{previous_memory}
</previous_memory>

Update the social memory by noting:
1. Any new information learned about the other person
2. Changes in the relationship dynamics
3. Progress towards goals (yours and theirs)
4. Any emotional shifts observed

Provide a concise updated social memory summary."""

# ============================================================
# Refiner Prompt (for utterance refinement during training/inference)
# ============================================================
refine_prompt = """You are a communication expert specializing in persuasive dialogue refinement for social interactions.

<context>
Game Rules: This is a social interaction scenario where two people are having a conversation. Each person has their own character profile, social goals, and secrets. The goal is to achieve your objectives through natural conversation.

Your Character: {player_name} ({player_role})
Current State: {game_state}
</context>

<dialogue_history>
{dialog_history}
</dialogue_history>

<base_utterance>
The following is the original utterance that needs refinement:
"{base_utterance}"
</base_utterance>

<task>
Refine the above utterance to make it more persuasive and effective at influencing the other person's response in a way that benefits your goals. The refined utterance should:
1. Maintain the same general intent and character voice
2. Be more strategically crafted to guide the conversation favorably
3. Sound natural and in-character (not robotic or overly formal)
4. Not reveal any secrets or compromise your position

Output ONLY the refined utterance, nothing else.
</task>"""

# ============================================================
# Measurer Prompts (for computing follower response log probabilities)
# These MUST be consistent with the actual gameplay prompts
# ============================================================
# The measurer system prompt is the same as the game system prompt
measurer_system_prompt = system_prompt

# The measurer user prompt is the same as the response prompt without action
measurer_user_prompt = response_prompt_without_action

# ============================================================
# Intent Identification Prompt
# ============================================================
intent_identification_prompt = """You are an expert at analyzing social interactions and predicting how people respond to different communication strategies.

<scenario>
{scenario}
</scenario>

<conversation_history>
{dialog_history}
</conversation_history>

<current_speaker>
{speaker_name} ({speaker_role}) just said: "{utterance}"
</current_speaker>

<next_speaker>
{next_speaker_name} ({next_speaker_role}) will respond next.
Their goal: {next_speaker_goal}
Their character: {next_speaker_profile}
</next_speaker>

Based on the conversation context and the current speaker's utterance, generate {k} possible responses that {next_speaker_name} would likely give. These should be {response_type} responses that {response_description}.

Generate exactly {k} responses, each on a new line, prefixed with a number:
1. [response 1]
2. [response 2]
3. [response 3]"""

# ============================================================
# Base Utterance Prompt (for generating base utterances during data collection)
# ============================================================
base_utterance_prompt = """You are {name} in a social interaction.

<character_profile>
{character_summary}
</character_profile>

<scenario>
{scenario}
</scenario>

<your_goal>
{goal}
</your_goal>

<conversation_history>
{summary}
</conversation_history>

Generate a natural response as {name}. Stay in character and work towards your goal."""

# ============================================================
# LLM Judge Evaluation Prompt
# ============================================================
evaluation_prompt = """You are an expert evaluator for social interactions. Please evaluate how well each participant performed in the following conversation.

<scenario>
{scenario}
</scenario>

<agent_1_profile>
Name: {agent1_name}
{agent1_profile}
Goal: {agent1_goal}
</agent_1_profile>

<agent_2_profile>
Name: {agent2_name}
{agent2_profile}
Goal: {agent2_goal}
</agent_2_profile>

<conversation>
{conversation}
</conversation>

Please evaluate EACH agent on the following 7 dimensions. For each dimension, provide:
- A brief reasoning explaining your evaluation
- An integer score within the specified range

Dimensions:
1. **Believability** (0-10): Does the agent interact naturally and stay in character?
2. **Relationship** (-5 to 5): Did the interaction improve (+) or harm (-) the relationship?
3. **Knowledge** (0-10): Did the agent gain new and important information?
4. **Secret** (-10 to 0): Did the agent keep its secrets? (-10 = leaked critical secrets, 0 = no leaks)
5. **Social Rules** (-10 to 0): Did the agent violate moral rules or laws? (0 = no violations)
6. **Financial and Material Benefits** (-5 to 5): Did the agent gain (+) or lose (-) material benefits?
7. **Goal** (0-10): How effectively did the agent achieve its social goals?

Output your evaluation in the following JSON format:
{{
    "agent_1": {{
        "believability": {{"reasoning": "...", "score": X}},
        "relationship": {{"reasoning": "...", "score": X}},
        "knowledge": {{"reasoning": "...", "score": X}},
        "secret": {{"reasoning": "...", "score": X}},
        "social_rules": {{"reasoning": "...", "score": X}},
        "financial_and_material_benefits": {{"reasoning": "...", "score": X}},
        "goal": {{"reasoning": "...", "score": X}}
    }},
    "agent_2": {{
        "believability": {{"reasoning": "...", "score": X}},
        "relationship": {{"reasoning": "...", "score": X}},
        "knowledge": {{"reasoning": "...", "score": X}},
        "secret": {{"reasoning": "...", "score": X}},
        "social_rules": {{"reasoning": "...", "score": X}},
        "financial_and_material_benefits": {{"reasoning": "...", "score": X}},
        "goal": {{"reasoning": "...", "score": X}}
    }}
}}"""

# ============================================================
# Relationship Type Descriptions
# ============================================================
RELATIONSHIP_TYPES = {
    0: "strangers",
    1: "know each other by name",
    2: "acquaintances",
    3: "friends",
    4: "in a romantic relationship",
    5: "family members",
}

# ============================================================
# Default Strategies (initial strategies for agents)
# ============================================================
init_strategies = {
    "default": "Engage naturally in the conversation. Listen carefully to the other person, "
               "understand their needs, and work towards achieving your goal through effective communication. "
               "Be strategic but maintain your character's personality and values.",
    "negotiation": "Identify the other person's interests and find common ground. "
                   "Make proposals that address both parties' needs. Be willing to compromise "
                   "but protect your core interests.",
    "cooperation": "Build rapport and trust with the other person. Share information strategically "
                   "and look for win-win solutions. Be supportive while advancing your own goals.",
    "competition": "Maximize your own outcome while being mindful of social norms. "
                   "Use persuasion and strategic information sharing to gain advantages. "
                   "Maintain a positive relationship if possible.",
}

# ============================================================
# Suggestion and Update Prompts (for LASI-like reflection, kept for compatibility)
# ============================================================
suggestion_prompt = """Based on the completed interaction:

<scenario>
{scenario}
</scenario>

<your_goal>
{goal}
</your_goal>

<conversation>
{conversation}
</conversation>

<outcome>
{outcome}
</outcome>

Reflect on the interaction:
1. What strategies worked well?
2. What could have been done differently?
3. What suggestions would you give for future similar interactions?

Provide concise suggestions for improvement."""

update_prompt = """Based on the reflection and suggestions:

<previous_strategy>
{previous_strategy}
</previous_strategy>

<suggestions>
{suggestions}
</suggestions>

<outcome>
{outcome}
</outcome>

Update the strategy for future interactions. The updated strategy should incorporate lessons learned while maintaining the core approach that worked well.

Provide the updated strategy concisely."""

# ============================================================
# Plan Prompt (used by ReAct for action planning)
# ============================================================
plan_prompt = """Based on your analysis:

<analysis>
{analysis}
</analysis>

<your_goal>
{goal}
</your_goal>

Formulate a specific plan for your next response:
1. What is the main point you want to convey?
2. What tone and approach should you use?
3. Are there any specific proposals or questions you should include?

Provide your plan concisely."""

# ============================================================
# Action Prompt (used by LASI-like agents)
# ============================================================
action_prompt = """Based on the current situation and your plan:

<plan>
{plan}
</plan>

<candidate_actions>
{candidate_actions}
</candidate_actions>

Select the most appropriate action type and explain why.

Output format:
Action: [selected action]
Reason: [brief explanation]"""

# ============================================================
# Strategy Prompt (used by LASI-like agents)
# ============================================================
strategy_prompt = """Based on the game landscape analysis:

<analysis>
{analysis}
</analysis>

<current_strategy>
{current_strategy}
</current_strategy>

<your_goal>
{goal}
</your_goal>

Should you update your strategy? If so, provide the updated strategy.
If the current strategy is working well, explain why it should be maintained.

Output your decision and the (updated or maintained) strategy."""

# ============================================================
# Candidate Actions for Social Interactions
# ============================================================
candidate_actions = {
    "speak": "Say something to the other person in the conversation.",
    "propose": "Make a specific proposal or offer to the other person.",
    "ask": "Ask a question to gather information or clarify something.",
    "agree": "Express agreement with the other person's point or proposal.",
    "disagree": "Express disagreement and explain your position.",
    "compromise": "Suggest a middle ground or modified proposal.",
    "empathize": "Show understanding and empathy for the other person's situation.",
    "persuade": "Try to convince the other person to see things your way.",
    "deflect": "Redirect the conversation away from a sensitive topic.",
    "leave": "End the conversation.",
}

# ============================================================
# Summary Prompt (for summarizing conversation history)
# ============================================================
summary_prompt = """Please summarize the following conversation between {agent1_name} and {agent2_name}:

<conversation>
{conversation}
</conversation>

Provide a concise summary highlighting:
1. Key topics discussed
2. Any agreements or disagreements
3. The current state of the interaction
4. Any notable emotional dynamics"""
