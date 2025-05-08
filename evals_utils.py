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
from inspect_ai.model import get_model, GenerateConfig, ChatMessageUser, ChatMessageAssistant


def load_expected_labels(json_path: str) -> List[str]:
    """
    Load expected labels from the 'test' section of the dataset JSON.

    Args:
        json_path: Path to JSON file with "train" and "test" keys.

    Returns:
        List of string labels (e.g., ["TRUE", "FALSE", ...]).
    """
    data = json.loads(Path(json_path).read_text())
    test_data = data.get("test", [])

    # Convert boolean labels to uppercase strings if needed
    expected_labels = [
        str(item["label"]).upper()
        for item in test_data
        if "label" in item
    ]

    return expected_labels

def parse_predictions(output: str, n: int, label_set: List[str]) -> List[str]:
    lines = output.strip().splitlines()
    preds = []
    for line in lines:
        match = re.search(r'->\s*(\w+)', line)
        if match and match.group(1).upper() in label_set:
            preds.append(match.group(1).upper())
        if len(preds) == n:
            break
    return preds

def flexible_parse_predictions(output: str, n: int, label_set: List[str]) -> List[str]:
    lines = output.strip().splitlines()
    preds = []
    
    for line in lines:
        # Try the original arrow pattern first
        match = re.search(r'->\s*(\w+)', line)
        if match and match.group(1).upper() in label_set:
            preds.append(match.group(1).upper())
        # If no arrow pattern, check if the entire line is a label
        elif line.strip().upper() in label_set:
            preds.append(line.strip().upper())
        
        if len(preds) == n:
            break
    
    return preds

def evaluate_classification_accuracy(
    prompt: str,
    inference_fn: Callable[[str], str],
    expected_labels: List[str],
    label_set: List[str] = ["TRUE", "FALSE"],
    num_runs: int = 5,
) -> Dict[str, float]:
    """
    Args:
      prompt: the full classification prompt
      inference_fn: fn(prompt)->model_output
      expected_labels: list of gold labels (e.g. ["TRUE","FALSE",...])
      label_set: allowed labels (e.g. ["TRUE","FALSE"] or ["A","B"])
      num_runs: how many times to run

    Returns:
      {
        "run_accuracies": [...],
        "mean_accuracy": ...,
        "std_accuracy": ...,
        "overall_accuracy": ...,
        "all_outs": [...],
      }
    """
    n = len(expected_labels)
    run_accuracies = []
    all_preds = []
    all_outs = []

    for j in range(num_runs):
        out = inference_fn(prompt)
        all_outs.append(out)
        preds = flexible_parse_predictions(out, n, label_set)
        if len(preds) != n:
            print(f"Warning: expected {n} preds, got {len(preds)}")
            print(f"Iteration {j}")
            print(f"Output: {out}")
            print(f"Preds: {preds}")
            print(f"Expected: {expected_labels}")
            raise ValueError(f"Expected {n} preds, got {len(preds)}")
        acc = np.mean([p == g for p, g in zip(preds, expected_labels)])
        run_accuracies.append(acc)
        all_preds.extend(preds)

    overall_acc = np.mean([
        pred == gold
        for pred, gold in zip(all_preds, expected_labels * num_runs)
    ])

    return {
        "run_accuracies": run_accuracies,
        "mean_accuracy": float(np.mean(run_accuracies)),
        "std_accuracy": float(np.std(run_accuracies)),
        "overall_accuracy": float(overall_acc),
        "all_outs": all_outs,
    }

# ---- your helpers (unchanged) ---------------------------------------------
# load_expected_labels, parse_predictions
# ---------------------------------------------------------------------------

def classify_then_explain(
    first_prompt: str,
    second_prompt: str,
    model_name: str,
    dataset_json: str,
    *,
    label_set: List[str] = ["TRUE", "FALSE"],
    temperature: float = 0.0,
    num_runs: int = 3,
) -> Dict[str, any]:
    """Run a two‑turn conversation num_runs times and return metrics + texts."""
    gold = load_expected_labels(dataset_json)
    n    = len(gold)

    async def _one_run() -> tuple[float, str, str]:
        model = get_model(
            model_name,
            config=GenerateConfig(temperature=temperature, num_choices=1)
        )

        # ---- Turn 1: classification ---------------------------------------
        reply1 = await model.generate(first_prompt)
        cls_text = reply1.choices[0].message.content

        preds = flexible_parse_predictions(cls_text, n=n, label_set=label_set)
        if len(preds) != n:
            raise ValueError(f"Expected {n} predictions, got {len(preds)}")
        acc = float(np.mean([p == g for p, g in zip(preds, gold)]))

        # ---- Turn 2: explanation (with history) ---------------------------
        history = [
            ChatMessageUser(content=first_prompt),
            ChatMessageAssistant(content=cls_text),
            ChatMessageUser(content=second_prompt),
        ]
        reply2 = await model.generate(history)
        expl_text = reply2.choices[0].message.content
        return acc, cls_text, expl_text

    # run repeatedly ---------------------------------------------------------
    run_accs, cls_outs, expls = [], [], []
    for _ in range(num_runs):
        acc, o1, o2 = asyncio.run(_one_run())
        print(f"Run {_+1}: acc {acc:.3f}")
        run_accs.append(acc)
        cls_outs.append(o1)
        expls.append(o2)

    print(f"✅ Classification finished: mean acc {np.mean(run_accs):.3f} "
          f"(±{np.std(run_accs):.3f}) across {num_runs} runs\n")
    print("📝 Explanation from last run:\n")
    print(expls[-1])

    return {
        "run_accuracies": run_accs,
        "classification_outputs": cls_outs,
        "explanations": expls,
        "mean_accuracy": float(np.mean(run_accs)),
        "std_accuracy":  float(np.std(run_accs)),
    }