import re
import os.path as osp
import numpy as np

from seqlbtoolkit.data import txt_to_token_span
from .math import get_boxed_content, get_boxed_token_span


def extract_last_number(text):
    # Split the text into lines
    lines = text.strip().split("\n")
    line = lines[-1]

    # Define the regex pattern
    pattern = r"""
    # Match either a number inside \boxed{} or a standalone number
    (?:\\boxed{\s*
    ([\$\£\€\¥]?           # Optional currency symbol
    \d{1,3}(?:,\d{3})*     # Integer part with optional commas
    (?:\.\d+)?             # Optional decimal part
    %?)\s*}                # Optional percentage sign and closing brace
    |
    ([\$\£\€\¥]?           # Optional currency symbol
    \d{1,3}(?:,\d{3})*     # Integer part with optional commas
    (?:\.\d+)?             # Optional decimal part
    %?))                   # Optional percentage sign
    """

    # Compile the regex with the VERBOSE flag for better readability
    regex = re.compile(pattern, re.VERBOSE)

    # Process the lines in reverse order to find the last line with a number
    line = line.strip()

    # Find all matches in the line
    matches = regex.findall(line)

    # Extract the numbers from the matches
    numbers = []
    for match in matches:
        # Each match is a tuple; pick the non-empty group
        number = match[0] if match[0] else match[1]
        numbers.append(number)

    if numbers:
        # Return the last number found in this line
        return numbers[-1]

    # Return None if no numbers are found in any line
    return None


def convert_to_float(number_str):
    """
    Converts a number string to a float, handling currency symbols, percentages,
    commas, and optional \boxed{} wrapping.

    Args:
        number_str (str): The number string to convert.

    Returns:
        float or None: The numeric value as a float, or None if conversion fails.
    """
    # Remove \boxed{} wrapping if present
    number_str = re.sub(r"^\\boxed{\s*(.*?)\s*}$", r"\1", number_str)

    # Remove any currency symbols
    number_str = re.sub(r"^[\$\£\€\¥]", "", number_str)

    # Check for percentage and adjust accordingly
    is_percentage = False
    if number_str.endswith("%"):
        is_percentage = True
        number_str = number_str[:-1]

    # Remove commas from the number
    number_str = number_str.replace(",", "")

    # Strip any leading/trailing whitespace
    number_str = number_str.strip()

    try:
        value = float(number_str)
        if is_percentage:
            value /= 100.0  # Convert percentage to decimal
        return value
    except ValueError:
        return None  # Return None if the string cannot be converted to float


def removing_preceeding_tailing_chars(number_str):
    """
    Converts a number string to a float, handling currency symbols, percentages,
    commas, and optional \boxed{} wrapping.

    Args:
        number_str (str): The number string to convert.

    Returns:
        float or None: The numeric value as a float, or None if conversion fails.
    """
    # Remove \boxed{} wrapping if present
    number_str = re.sub(r"^\\boxed{\s*(.*?)\s*}$", r"\1", number_str)

    # Remove any currency symbols
    number_str = re.sub(r"^[\$\£\€\¥]", "", number_str)

    # Check for percentage and adjust accordingly
    is_percentage = False
    if number_str.endswith("%"):
        is_percentage = True
        number_str = number_str[:-1]

    # Strip any leading/trailing whitespace
    number_str = number_str.strip()

    return number_str


def extract_gsm8k_target(response):
    return re.search(r"####\s*(.+)", response).group(1).strip()


def check_gsm8k_correctness(generated, response):
    pred = extract_last_number(generated)
    if not pred:
        pred = get_boxed_content(generated)
    tgt = extract_gsm8k_target(response)

    if not pred:
        raise ValueError("No number found in the generated text")

    pred_num = convert_to_float(pred)
    tgt_num = convert_to_float(tgt)

    return pred_num == tgt_num


def locate_gsm8k_answer_in_text(generated):
    answer_text = extract_last_number(generated)
    if not answer_text:
        return None, None
    answer_text = removing_preceeding_tailing_chars(answer_text)

    # Locate the answer text in the generated output
    generated = generated.rstrip()
    last_line = generated.split("\n")[-1]
    answer_start_idx = generated.rindex(last_line) + last_line.rindex(answer_text)
    answer_end_idx = answer_start_idx + len(answer_text)

    return answer_start_idx, answer_end_idx


def get_gsm8k_answer_token_span(tokenizer, txtresp):
    # Extract the answer text and its start/end indices in the generated text
    answer_start_idx, answer_end_idx = locate_gsm8k_answer_in_text(txtresp)
    if answer_start_idx is not None:
        # Tokenize the generated text and find the token span of the answer
        tokenized_result = tokenizer(txtresp, add_special_tokens=False)
        generated_token_ids = tokenized_result["input_ids"]
        generated_tokens = [tokenizer.decode(t_id) for t_id in generated_token_ids]
        answer_token_span = txt_to_token_span(generated_tokens, txtresp, [(answer_start_idx, answer_end_idx)])
        return answer_token_span

    return get_boxed_token_span(tokenizer, txtresp)


def get_gsm8k_answer_prob(tokenizer, preds, idx, generated):
    answer_token_span = get_gsm8k_answer_token_span(tokenizer, generated)

    # Adjust the token indices by the number of instruction tokens
    # Note: -1 offset because each predicted token is generated from the previous step
    adjusted_answer_token_span = [(start_idx - 1, end_idx - 1) for start_idx, end_idx in answer_token_span]

    # Return the probability of the first token in the predicted answer
    return preds[idx]["top5_pred_probs"][adjusted_answer_token_span[0][0], 0]


def get_gsm8k_generated_avg_prob(preds, instance_idx):

    top5_answer_probs = preds[instance_idx]["top5_pred_probs"]
    avg_generated_token_probs = top5_answer_probs[:-1, 0].mean()

    return avg_generated_token_probs


def evaluate_gsm8k_responses(df):
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
    for sample_id, generated_answer, reference_answer in zip(df["idx"], df["generated"], df["response"]):
        try:
            # Check if the generated answer is correct according to a given criterion
            if check_gsm8k_correctness(generated_answer, reference_answer):
                correct_ids.append(sample_id)
            else:
                incorrect_ids.append(sample_id)
        except Exception as e:
            # In case the correctness function fails (e.g., parsing error),
            # we record that attempt as a failure.
            failed_ids.append(sample_id)

    return correct_ids, incorrect_ids, failed_ids
