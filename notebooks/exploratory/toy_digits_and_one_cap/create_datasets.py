import random
import string
import json
import os


import os
import random
import string
import json
from typing import Optional

def generate_toy_dataset(
    num_samples: int,
    output_folder: str,
    file_name: str = "dataset.json",
    test_samples: Optional[int] = None,
):
    """
    Generates a dataset of examples for train and test splits,
    each a 6-character string.
    - Label = True : exactly one random uppercase letter and 5 digits.
    - Label = False: all 6 characters are digits.
    Saves a JSON file with fields "train" and "test": lists of {"input": str, "label": bool}.

    Args:
        num_samples: number of examples for the train split.
        output_folder: directory where the JSON will be written.
        file_name: name of the JSON file (default "dataset.json").
        test_samples: number of examples for the test split (defaults to same as num_samples).
    """
    os.makedirs(output_folder, exist_ok=True)

    def make_samples(n: int):
        samples = []
        for _ in range(n):
            # start with 6 random digits
            chars = [random.choice(string.digits) for _ in range(6)]
            # decide label
            label = random.random() < 0.5
            if label:
                # pick a random position to turn into an uppercase letter
                i = random.randrange(6)
                chars[i] = random.choice(string.ascii_uppercase)
            s = "".join(chars)
            samples.append({"input": s, "label": label})
        return samples

    # Determine test set size
    test_n = test_samples if test_samples is not None else num_samples

    train_samples = make_samples(num_samples)
    test_samples_list = make_samples(test_n)

    out_path = os.path.join(output_folder, file_name)
    with open(out_path, "w") as f:
        json.dump({"train": train_samples, "test": test_samples_list}, f, indent=2)

    print(
        f"Saved {len(train_samples)} train and {len(test_samples_list)} test samples to {out_path}"
    )


# conterfactual dataset 

import os
import json
import random
import string
from typing import Optional

import os
import json
import random
import string
from typing import Optional

def generate_toy_dataset_counterfactual(
    input_json: str,
    output_folder: str,
    file_name: str = "dataset.json",
    n1: int = 0,  # Type 1 false: 2 uppercase + 4 digits
    n2: int = 0,  # Type 2 false: 1 lowercase + 1 uppercase + 4 digits
    n3: int = 0,  # Type 3 false: 1 uppercase + 5 lowercase (always at pos 0)
    n4: int = 0,  # Type 4 false: 1 uppercase + 5 lowercase (random pos)
    n5: int = 50,  # Type 5 false: 6 digits
    n6: int = 0,  # Type 6 false: 1 lowercase + 5 digits
):
    """
    Args:
        input_json:    path to an existing JSON file with a "train" field
        output_folder: directory where the new JSON will be written
        file_name:     name of the output JSON file
        n1-n6:         counts for each of six *false* test–string types
    Test-set size will be (n1+n2+n3+n4+n5+n6) false + (n1+n2+n3+n4+n5+n6) true examples.
    """

    os.makedirs(output_folder, exist_ok=True)

    # --- load train split from user file ---
    with open(input_json, "r") as f:
        data = json.load(f)
    train_samples = data.get("train", [])

    # --- helper generators for each false–type ---
    def type1(n):  # 2 uppercase, 4 digits
        out = []
        for _ in range(n):
            pos = random.sample(range(6), 2)
            chars = [random.choice(string.digits) for _ in range(6)]
            for i in pos:
                chars[i] = random.choice(string.ascii_uppercase)
            out.append({"input": "".join(chars), "label": False})
        return out

    def type2(n):  # 1 lowercase + 1 uppercase, rest digits
        out = []
        for _ in range(n):
            pos_upper, pos_lower = random.sample(range(6), 2)
            chars = [random.choice(string.digits) for _ in range(6)]
            chars[pos_upper] = random.choice(string.ascii_uppercase)
            chars[pos_lower] = random.choice(string.ascii_lowercase)
            out.append({"input": "".join(chars), "label": False})
        return out

    def type3(n):  # no digits, uppercase at pos 0 + 5 lowercase
        out = []
        for _ in range(n):
            chars = [random.choice(string.ascii_lowercase) for _ in range(6)]
            chars[0] = random.choice(string.ascii_uppercase)
            out.append({"input": "".join(chars), "label": False})
        return out

    def type4(n):  # no digits, exactly 1 uppercase at random pos + 5 lowercase
        out = []
        for _ in range(n):
            chars = [random.choice(string.ascii_lowercase) for _ in range(6)]
            i = random.randrange(6)
            chars[i] = random.choice(string.ascii_uppercase)
            out.append({"input": "".join(chars), "label": False})
        return out

    def type5(n):  # all digits
        out = []
        for _ in range(n):
            s = "".join(random.choice(string.digits) for _ in range(6))
            out.append({"input": s, "label": False})
        return out

    def type6(n):  # 1 lowercase + 5 digits
        out = []
        for _ in range(n):
            pos_lower = random.randrange(6)
            chars = [random.choice(string.digits) for _ in range(6)]
            chars[pos_lower] = random.choice(string.ascii_lowercase)
            out.append({"input": "".join(chars), "label": False})
        return out

    # --- true examples: exactly 1 uppercase + 5 digits ---
    def make_true(n):
        out = []
        for _ in range(n):
            chars = [random.choice(string.digits) for _ in range(6)]
            i = random.randrange(6)
            chars[i] = random.choice(string.ascii_uppercase)
            out.append({"input": "".join(chars), "label": True})
        return out

    # build false + true sets
    counts = [n1, n2, n3, n4, n5, n6]
    false_parts = (
        type1(n1)
        + type2(n2)
        + type3(n3)
        + type4(n4)
        + type5(n5)
        + type6(n6)
    )
    true_parts = make_true(sum(counts))

    test_samples = false_parts + true_parts
    random.shuffle(test_samples)

    # --- write out ---
    out_path = os.path.join(output_folder, file_name)
    with open(out_path, "w") as f:
        json.dump({"train": train_samples, "test": test_samples}, f, indent=2)

    print(
        f"Loaded {len(train_samples)} train samples from {input_json} "
        f"and saved {len(test_samples)} test samples "
        f"({sum(counts)} false + {sum(counts)} true) to {out_path}"
    )
