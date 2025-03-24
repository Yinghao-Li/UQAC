"""
# Author: Yinghao Li
# Modified: February 28th, 2025
# ---------------------------------------
Description: Check whether two math expressions are equivalent. This is used to check the correctness of the model's output.
Reference: https://github.com/hendrycks/math/blob/main/modeling/math_equivalence.py
"""

import regex
from logging import getLogger
from seqlbtoolkit.data import txt_to_token_span

__all__ = ["is_math_express_equiv", "is_answer_correct", "get_boxed_content", "get_boxed_token_span"]

logger = getLogger(__name__)


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string


def _remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def _strip_string(string):
    # linebreaks
    string = string.replace("\n", "")
    # print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    # print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    # print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    # print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    # print(string)

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string


def is_math_express_equiv(ans, ref, verbose=False):
    assert ans is not None, ValueError("Failed to get the answer.")
    assert ref is not None, ValueError("Failed to get the reference.")

    try:
        ss1 = _strip_string(ans)
        ss2 = _strip_string(ref)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except:
        return ans == ref


def get_boxed_content(text):
    """
    Reference: https://stackoverflow.com/questions/16258143/get-all-nested-curly-braces
    """
    # Use regex to find all \boxed{} blocks, allowing for nested braces
    boxed_blocks = regex.findall(r"\\boxed(?=\{((?:[^{}]++|\{(?1)\})++)\})", text)

    if boxed_blocks:
        # Get the content of the right-most \boxed{} block
        rightmost_boxed_content = boxed_blocks[-1]
        return rightmost_boxed_content
    else:
        return None


def get_boxed_token_span(tokenizer, generated):
    # Extract the answer text and its start/end indices in the generated text
    ans = get_boxed_content(generated)
    answer_start_idx = generated.rindex(ans)
    answer_end_idx = answer_start_idx + len(ans)

    # Tokenize the generated text and find the token span of the answer
    tokenized_result = tokenizer(generated, add_special_tokens=False)
    generated_token_ids = tokenized_result["input_ids"]
    generated_tokens = [tokenizer.decode(t_id) for t_id in generated_token_ids]
    answer_token_span = txt_to_token_span(generated_tokens, generated, [(answer_start_idx, answer_end_idx)])

    return answer_token_span


def is_answer_correct(ans: str, ref: str):
    """
    Check whether the answer is correct by comparing the answer with the reference.
    """
    ans = get_boxed_content(ans)
    ref = get_boxed_content(ref)
    is_correct = is_math_express_equiv(ans, ref)
    return is_correct


def evaluate_math_responses(df):
    """
    Evaluate responses from a GSM8k dataset by checking correctness.

    Parameters
    ----------
    merged_df : pandas.DataFrame
        A DataFrame with columns:
        - "idx": Unique identifiers for each sample.
        - "generated": The generated model answer.
        - "response": The reference or expected answer.

    Returns
    -------
    tuple of lists
        (correct_ids, incorrect_ids, failed_ids)

        correct_ids   : List of indices where the generated answer is correct.
        incorrect_ids : List of indices where the generated answer is incorrect.
        failed_ids    : List of indices where the correctness check failed
                        (e.g., due to a ValueError).
    """
    correct_ids = []
    incorrect_ids = []
    failed_ids = []

    # Iterate over each row's data simultaneously
    for sample_id, generated, reference in zip(df["idx"], df["generated"], df["response"]):
        try:
            # Check if the generated answer is correct according to a given criterion
            if is_answer_correct(generated, reference):
                correct_ids.append(sample_id)
            else:
                incorrect_ids.append(sample_id)
        except (ValueError, AssertionError):
            # In case the correctness function fails (e.g., parsing error),
            # we record that attempt as a failure.
            failed_ids.append(sample_id)

    return correct_ids, incorrect_ids, failed_ids
