import re
from .math import get_boxed_content


def remove_parentheses(s):
    """
    Removes the outer parentheses from a string if the entire string
    is enclosed in them. Otherwise, returns the string unchanged.
    """
    return re.sub(r"^\((.*)\)$", r"\1", s)


def is_answer_correct(generated, response):
    """
    Check if the generated answer is correct.
    """
    g = get_boxed_content(generated)
    if remove_parentheses(g.lower()) == remove_parentheses(response.lower()):
        return True
    return False


def evaluate_bbh_responses(df):
    """
    Evaluate the responses in the dataframe and return the accuracy.
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
        except Exception:
            # In case the correctness function fails (e.g., parsing error),
            # we record that attempt as a failure.
            failed_ids.append(sample_id)

    return correct_ids, incorrect_ids, failed_ids
