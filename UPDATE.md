# Finalists Update for the Additional Prompts

## Throttling Problem Description

Like most other participants I have faced the "GenAI Throttling Problem" which made my application unusable. In my case, the system would go into the retry loop raising the warnings like the below even on the very first call of a model:

```python
WARNING:llama_index.llms.bedrock.utils:Retrying llama_index.llms.bedrock.utils.completion_with_retry.<locals>._completion_with_retry in 4.0 seconds as it raised ThrottlingException: An error occurred (ThrottlingException) when calling the InvokeModel operation (reached max retries: 10): Too many requests, please wait before trying again. You have sent too many requests.  Wait before trying again..
```

I tried switching between the US regions and other models with no success. My next step would be to test the local model but I didn't have enough time to do a local setup. Fortunately, the organizers allowed participants to modify the write-up describing the potential modifications. This document is to address these potential modifications.

## What would I do if I didn't have throttling issues

There are some additional prompts that the system is expected to tackle as part of the final selection stage. I will address them one by one since each may require specific tweaks. But I will first describe the problem with my original design that prevented handling these additional prompts in the first place. The agent I initially implemented extends the user prompt with a system prompt to add additional instructions. For example:

```
Use the provided tools to get the list of up to 10 players in the specified league,
and optionally check their region if requested to do so.
If the league is not specified, then combine the list from all available leagues.
...
```

This technique greatly improved the output of the model for the team composition task. Unfortunately, this is also a limitation, because if the user requested a task different from composing a team (as in some of the prompts below), the agent would still extend it with the same system prompt and this would confuse the model. And it did, based on my early test results. One way to fix this would probably be re-writing the original prompt while infusing only the instructions that are relevant to the user prompt. For example, I could break down my system instructions into a list e.g.:

```python
system_instructions = [
    "If the user requested to make a new team, start with the list of 10 players",
    "Filter by region when calling the tools if the user has specified it in their prompt",
    "Include player statistics in the final response if the user requested their metrics"
    ...
]
```

I could then call an LLM asking it to re-write the original user prompt while injecting the relevant instructions from the list. This pre-processed prompt would then be provided to the team manager agent (my reasoning agent) to generate the actual user response. I think this would be a great way to generalize the agent to various tasks while keeping it guided by the set of built-in instructions. This would require quite a bit of testing though. So my alternative to that (which is what I would most likely implement in reality) would be to simply add more system prompt templates to account for the additional use cases. I would then analyze the original user prompt and extend it with the appropriate system prompt. I would incorporate those templates into the corresponding agent methods:

```python
def run_task(user_prompt: str, history: str):
    if "build a team" in user_prompt:
        manager.make_team(user_prompt)  # this will use the system prompt to provide instructions on building a team
    elif "suitable replacement" in user_prompt:
        manager.find_replacement(user_prompt, history) # this method will use a different system prompt providing instructions to find a good replacement
    ...
```

While it obviously is an error-prone and less flexible approach, I think it may work fine for a set of standardized prompts. It would be interesting to implement and compare these two approaches together!

Ok, so in addition to handling the user prompt and system instructions, I would likely need to implement some additional changes as well. Let's now go over each prompt:

> "What recent performances or statistics justify the inclusion of this player in the team?"

This one will work out of the box. The reasoning agent is already collecting each user's stats to reason about. One thing that comes to my mind is filtering the stats by year, which would require an additional filter to be implemented in the retrieval tool `get_player_stats`. I would consider this a **minor** update.

> "If this player were unavailable, who would be a suitable replacement and why?"

This will likely require an additional parameter to be added to an existing retrieval tool `get_random_players`. The agent would extract the filtering parameters for the player that needs to be replaced (these are already supported) along with their assigned role (this will need to be added). The agent will then use the shortlisted players returned by the tool to select the best player for the role that needs to be filled. In theory, this is no different from assigning the roles to the players in the original team composition task. I would also consider this a **minor** update.

> “Which maps would this team composition excel in? Why?”

This one will unfortunately require rebuilding my dataset. The problem is that I haven't extracted/included any map information into my prepared game stats dataset. I built and aggregated the performance stats by the game, player, and team role. So to enable this prompt I would first need to extract the map information (probably from the config events), add it to the game stats, and then also modify my retrieval tools to accept the additional map filter. I would say it is a **medium** update.

> “Which player would take a leadership (IGL) role in this team? Why?”

This prompt will likely require extracting extra stats from the game files and a different aggregation. The problem is to understand what data is relevant. And this is a rather complex analysis for me since it requires a deep understanding of the game dynamics. For example, one thing that could be relevant is the pistol rounds (#1 and #13) - winning these requires a lot of coordination and may be an indication of strong leadership skills, but again without knowing a game very well it is really hard to justify. Ultimately, it is the data analysis task to find the correlation between the data patterns and leadership skills. It is definitely the **hardest** update I would consider making.

> Role-specific metrics and justification (Duelists, Sentinels, Controllers, Initiators)

This prompt will work out of the box with no modifications. The system is built by relying on these metrics in the first place, so I often requested the agent to display them while testing.
