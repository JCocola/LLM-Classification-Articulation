import re
import numpy as np
from typing import Callable, List, Dict
import json
from pathlib import Path
from textwrap import dedent
from typing import Optional, List, Tuple, Callable, Dict
import numpy as np
import re
import asyncio
from inspect_ai.model import get_model, GenerateConfig


def build_classification_prompt(
    json_path: str,
    intro: str,
    question: str,
    instructions: str,
    outro: Optional[str] = None
) -> str:
    """
    Build a prompt from:
      • an intro line
      • an instructions line
      • a training block
      • a question header
      • a test block
      • [optional] an outro line

    Args:
        json_path: Path to JSON with "train" and "test" lists.
        intro:     Leading context for the assistant.
        instructions: Directive on output format.
        question:  Header for the test block.
        outro:     Optional trailing text (e.g., “Thank you.”).
    """
    data = json.loads(Path(json_path).read_text())
    train = data["train"]
    test  = data["test"]

    lines = []
    # 1. Intro & instructions
    lines.append(intro.strip())
    lines.append(instructions.strip())
    lines.append("")  # blank line

    # 2. Training examples
    lines.append("### Training examples")
    for item in train:
        lines.append(f"\"{item['input']}\" -> {item['label']}")
    lines.append("")  # blank line

    # 3. Question header + test examples
    lines.append(f"### {question.strip()}")
    for item in test:
        lines.append(f"\"{item['input']}\" ->")

    # 4. Optional outro
    if outro and outro.strip():
        lines.append("")               # blank line before outro
        lines.append(outro.strip())

    return "\n".join(lines)

def build_articulation_prompt(
    json_path: str,
    intro: str,
    question: str,
    instructions: str,
    outro: Optional[str] = None
) -> str:
    """
    Build a prompt asking the LLM to articulate the rule used to classify inputs,
    using only the labelled training examples and no test examples.

    Args:
        json_path: Path to JSON file with "train" (and optionally "test", ignored here).
        intro: Introductory system or context message.
        question: A question like "What rule did you follow?"
        instructions: Additional directions, e.g., "Use natural language and be concise."
        outro: Optional final message (e.g., "Explain your reasoning if unsure.")
    """
    data = json.loads(Path(json_path).read_text())
    train = data["train"]

    lines = []
    # Intro and instructions
    lines.append(intro.strip())
    lines.append(instructions.strip())
    lines.append("")  # blank line

    # Training examples
    lines.append("### Training examples")
    for item in train:
        lines.append(f"\"{item['input']}\" -> {item['label']}")
    lines.append("")  # blank line

    # Rule articulation question
    lines.append(f"### {question.strip()}")

    # Optional outro
    if outro and outro.strip():
        lines.append("")
        lines.append(outro.strip())

    return "\n".join(lines)

def make_inference_fn(model_name: str = "openai/gpt-4o",
                      temperature: float = 0.0,
                      num_choices: int = 1):
    """
    Returns a callable `inference_fn(prompt:str)->str` that
    1. sends `prompt` to the chosen model with the given temperature
    2. returns the content of the first choice as a plain string.
    """

    async def _chat_once(prompt: str) -> str:
        model = get_model(model_name,
                          config=GenerateConfig(temperature=temperature,
                                                num_choices=num_choices))
        reply = await model.generate(prompt)
        return reply.choices[0].message.content

    # wrap the async coroutine in a sync function expected by the evaluator
    def inference_fn(prompt: str) -> str:
        return asyncio.run(_chat_once(prompt))

    return inference_fn


async def chat(model_name: str, prompt: str, *,
               n: int = 1, temperature: float = 0.0, print_choices: bool = True):
    model = get_model(
        model_name,
        config=GenerateConfig(temperature=temperature, num_choices=n)
    )
    reply = await model.generate(prompt)
    if print_choices:
        for i, choice in enumerate(reply.choices, 1):
            print(f"\n— Choice {i} ({model_name}) —\n{choice.message.content}")
    
    return reply
