"""
Prompt templates for genetic algorithm operations.
Inspired by PromptBreeder paper and promptimizer.
"""

INIT_POPULATION_PROMPT = """You are an expert AI prompt engineer. You will be given a prompt and your job is to come up with {population_size} better prompts. The user wants to improve the following about the given prompt:

<improvement_request>
{improvement_request}
</improvement_request>

Each prompt you generate should employ a different strategy to improve the initial prompt.

Use prompting strategies that are appropriate for the task. For logic and math, encourage more chain-of-thought reasoning, or include reasoning trajectories to induce better performance. For creative tasks, consider adding style guidelines. Or consider including examples.

Your improved prompts must:
- Keep all original input variables.
- Maintain any special formatting or delimiters.
- Be clear and concise. 
- Do NOT use complicated language or jargon.
- Avoid repeating mistakes.

You MUST generate {population_size} prompts that are better than the provided prompt."""

CROSSOVER_PROMPT = """You are an expert AI prompt engineer tasked with improving the following prompt:

<initial_prompt>
{initial_prompt}
</initial_prompt>

The user would like to improve the following about the prompt:

<improvement_request>
{improvement_request}
</improvement_request>

Using this information above, your job is to generate a better prompt by combining elements from two prompts that are known to be successful.

The goal is to create a prompt that is better than either of the original prompts, while still staying true to the intent of the initial prompt and the improvement request.

In your head, plan the optimization step-by-step:
1. Analyze the two prompts and where they fall short.
2. Identify patterns in the prompts that are more or less likely to be successful.
3. Propose specific improvements to address the shortcomings.
4. Generate an improved prompt that maintains all required formatting.

The improved prompt must:
- Keep all original input variables.
- Maintain any special formatting or delimiters.
- Be clear and concise. 
- Do NOT use complicated language or jargon.
- Avoid repeating mistakes.
- Combine the best elements of both prompts.

Output both your step-by-step analysis and the improved prompt."""

EVAL_PROMPT = """You are an expert AI prompt engineer tasked with improving the following prompt:

<initial_prompt>
{initial_prompt}
</initial_prompt>

The user would like to improve the following about the prompt:

<improvement_request>
{improvement_request}
</improvement_request>

Using this information above, your job is to evaluate a potentially improved prompt given to you.

You should grade the prompt in the following categories:
- **Clarity:** Precisely defines the task with unambiguous language.
- **Context:** Provides essential background and purpose of the request.
- **Specificity:** Outlines exact requirements, expected format, and constraints.
- **Guidance:** Breaks complex tasks into clear, sequential steps.
- **Examples:** Includes concrete samples of desired input/output.
- **Role Definition:** Specifies the persona or perspective to adopt.
- **Boundaries:** Sets clear limitations and ethical guidelines.
- **Reasoning:** Requests explanation of logic and self-validation.
- **Flexibility:** Allows space for creative interpretation.
- **Structure:** Defines preferred output format and presentation.

Your final evaluation should be a score between 1 and 10, with 10 being the highest."""
