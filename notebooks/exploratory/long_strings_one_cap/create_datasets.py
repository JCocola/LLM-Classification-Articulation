import random
import string
import json
import os


import json
from typing import Optional

import json, os, random, string
from typing import Optional

import json, os, random, string
from typing import Optional

# def generate_long_string_dataset(
#     num_samples: int,
#     output_folder: str,
#     file_name: str = "dataset.json",
#     test_samples: Optional[int] = None,
#     N: int = 100,
#     p_digit: float = 0.5,
#     case_mode: int = 1,     # 1 = “left‑only uppercase”, 2 = “left or none”
# ):
#     """
#     Generates train/test splits where each example is an N‑char string.

#     Label = True  : exactly one uppercase letter placed uniformly in the right half.
#     Label = False :
#         case_mode 1 – exactly one uppercase in the left half
#         case_mode 2 – (a) one uppercase in the left half  OR  (b) no uppercase
#     All other characters: digit w.p. p_digit, lowercase letter otherwise.

#     JSON output: {"train": [...], "test": [...]}, each item {"input": str, "label": bool}.
#     """
#     assert 0 <= p_digit <= 1, "p_digit must be in [0,1]"
#     assert case_mode in (1, 2), "case_mode must be 1 or 2"

#     os.makedirs(output_folder, exist_ok=True)
#     left_size   = N // 2
#     right_start = left_size             # works for even/odd N
#     right_size  = N - right_start

#     def random_char() -> str:
#         return random.choice(string.digits) if random.random() < p_digit else random.choice(string.ascii_lowercase)

#     def make_samples(n: int):
#         samples = []
#         for _ in range(n):
#             chars = [random_char() for _ in range(N)]
#             label = random.random() < 0.5  # True vs False, 50/50

#             if label:  # TRUE: one uppercase in right half
#                 idx = right_start + random.randrange(right_size)
#                 chars[idx] = random.choice(string.ascii_uppercase)

#             else:      # FALSE
#                 if case_mode == 2 and random.random() < 0.5:
#                     # 50% of false examples have NO uppercase at all
#                     pass
#                 else:
#                     idx = random.randrange(left_size)
#                     chars[idx] = random.choice(string.ascii_uppercase)

#             samples.append({"input": "".join(chars), "label": label})
            
#             random.shuffle(samples)
            
#         return samples

#     test_n = test_samples if test_samples is not None else num_samples
#     data = {
#         "train": make_samples(num_samples),
#         "test":  make_samples(test_n),
#     }

#     out_path = os.path.join(output_folder, file_name)
#     with open(out_path, "w") as f:
#         json.dump(data, f, indent=2)

#     print(f"Saved {len(data['train'])} train and {len(data['test'])} test samples to {out_path}")


def generate_marker_dataset(
    num_samples: int,
    output_folder: str,
    file_name: str = "marker_dataset.json",
    test_samples: Optional[int] = None,
    N: int = 100,
    marker_character: str = "*",
    filler_character: str = "_",
    case_mode: int = 1,     # 1 = "left‑only marker", 2 = "left or none"
):
    """
    Generates train/test splits where each example is an N‑char string.

    Label = True  : exactly one marker_character placed uniformly in the right half.
    Label = False :
        case_mode 1 – exactly one marker_character in the left half
        case_mode 2 – (a) one marker_character in the left half  OR  (b) no marker_character
    All other characters are filled with filler_character.

    JSON output: {"train": [...], "test": [...]}, each item {"input": str, "label": bool}.
    """
    assert len(marker_character) == 1, "marker_character must be a single character"
    assert len(filler_character) == 1, "filler_character must be a single character"
    assert case_mode in (1, 2), "case_mode must be 1 or 2"

    os.makedirs(output_folder, exist_ok=True)
    left_size   = N // 2
    right_start = left_size             # works for even/odd N
    right_size  = N - right_start

    def make_samples(n: int):
        samples = []
        for _ in range(n):
            # Fill the string with filler characters
            chars = [filler_character for _ in range(N)]
            label = random.random() < 0.5  # True vs False, 50/50

            if label:  # TRUE: one marker in right half
                idx = right_start + random.randrange(right_size)
                chars[idx] = marker_character

            else:      # FALSE
                if case_mode == 2 and random.random() < 0.5:
                    # 50% of false examples have NO marker at all
                    pass
                else:
                    idx = random.randrange(left_size)
                    chars[idx] = marker_character

            samples.append({"input": "".join(chars), "label": label})
        
        random.shuffle(samples)
        return samples

    test_n = test_samples if test_samples is not None else num_samples
    data = {
        "train": make_samples(num_samples),
        "test":  make_samples(test_n),
    }

    out_path = os.path.join(output_folder, file_name)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved {len(data['train'])} train and {len(data['test'])} test samples to {out_path}")


def generate_adversarial_marker_dataset(
    num_samples: int,
    output_folder: str,
    file_name: str = "adversarial_marker_dataset.json",
    test_samples: Optional[int] = None,
    N: int = 100,
    marker_character: str = "*",
    filler_character: str = "_",
    window_size: int = 10,
    case_mode: int = 1,     # 1 = "left‑only marker", 2 = "left or none"
):
    """
    Generates train/test splits where each example is an N‑char string with adversarial positioning.
    
    Instead of random positioning across the whole string, the marker is placed in a small window
    around the center (defined by window_size). This creates more challenging edge cases.

    Label = True  : exactly one marker_character placed in the small window, but on the right side of center.
    Label = False :
        case_mode 1 – exactly one marker_character in the small window, but on the left side of center
        case_mode 2 – (a) one marker_character in the small window, on the left side of center  OR  (b) no marker_character
    All other characters are filled with filler_character.

    JSON output: {"train": [...], "test": [...]}, each item {"input": str, "label": bool}.
    """
    assert len(marker_character) == 1, "marker_character must be a single character"
    assert len(filler_character) == 1, "filler_character must be a single character"
    assert case_mode in (1, 2), "case_mode must be 1 or 2"
    assert window_size > 0, "window_size must be positive"
    assert window_size <= N, "window_size cannot exceed string length N"

    os.makedirs(output_folder, exist_ok=True)
    
    # Calculate the center position
    center = N // 2
    
    # Calculate left and right window boundaries
    half_window = window_size // 2
    left_window_start = max(0, center - half_window)
    left_window_end = center
    right_window_start = center
    right_window_end = min(N, center + half_window)
    
    # Ensure there's at least one position on each side
    assert left_window_end > left_window_start, "Window too small for left side"
    assert right_window_end > right_window_start, "Window too small for right side"

    def make_samples(n: int):
        samples = []
        for _ in range(n):
            # Fill the string with filler characters
            chars = [filler_character for _ in range(N)]
            label = random.random() < 0.5  # True vs False, 50/50

            if label:  # TRUE: one marker in right half of window
                idx = random.randint(right_window_start, right_window_end - 1)
                chars[idx] = marker_character

            else:      # FALSE
                if case_mode == 2 and random.random() < 0.5:
                    # 50% of false examples have NO marker at all
                    pass
                else:
                    idx = random.randint(left_window_start, left_window_end - 1)
                    chars[idx] = marker_character

            samples.append({"input": "".join(chars), "label": label})
        
        random.shuffle(samples)
        return samples

    test_n = test_samples if test_samples is not None else num_samples
    data = {
        "train": make_samples(num_samples),
        "test":  make_samples(test_n),
    }

    out_path = os.path.join(output_folder, file_name)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved {len(data['train'])} train and {len(data['test'])} test samples to {out_path}")
