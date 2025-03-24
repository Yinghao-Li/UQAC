"""
Modified: March 18th, 2025
---------------------------------------
Description: Collection of dataset processing and formatting functions.
"""

import torch
from datasets import Dataset

from src.utils.macro import VU_PROMPT
from src.utils.support import thresholded_topk_1d, thresholded_topk_2d, find_column_indices, filter_indices


def process_default(
    sample,
    tokenizer=None,
    dataset_name=None,
    task=None,
    model_name=None,
    dataset_text_field: str = "text",
    **kwargs,
):
    """
    Entry point for default tokenization/formatting of a single dataset row.
    Calls specialized sub-functions based on (partition, task).
    """
    sample["keep_instance"] = True

    # Always apply the Gemma "math" tweak if relevant
    instruction = sample.get("instruction", "")
    if "gemma" in type(tokenizer).__name__.lower() and dataset_name == "math":
        instruction = _math_instruction_update_for_gemma(instruction)
        sample["instruction"] = instruction  # update in sample for consistency

    # Dispatch logic
    if task == "infer-uq":
        text = _format_infer_uq_sample(sample, tokenizer, model_name)
    elif task == "infer-vu":
        text = _format_infer_vu_sample(sample, tokenizer)
    else:
        text = _format_test_sample(sample, tokenizer)

    # Count the number of tokens
    token_ids = tokenizer(text, truncation=False, add_special_tokens=False)["input_ids"]
    sample["n_tks"] = len(token_ids)

    # Assign the final text into the designated dataset text field
    sample[dataset_text_field] = text

    return sample


# ----------------------------------------------------------------
# Below are the sub-functions called by process_default
# ----------------------------------------------------------------


def _format_infer_uq_sample(sample, tokenizer, model_name):
    """
    For 'infer-uq': user instruction + user-generated answer
    (we treat it as the assistant's generation).
    """
    instruction = sample.get("instruction", "")
    generated = sample.get("generated", "")

    text = apply_chat_template(instruction, generated, tokenizer, model_name)

    # Optionally, store how many tokens are in the instruction alone
    instr_msg = [{"role": "user", "content": instruction}]
    instr_tks = tokenizer.apply_chat_template(instr_msg, tokenize=True, add_generation_prompt=True)
    sample["n_instr_tks"] = len(instr_tks)

    return text


def _format_infer_vu_sample(sample, tokenizer):
    """
    For 'infer-vu': after the user's (real) instruction and the
    model's generated answer, we add a second user query (VU_PROMPT).
    """
    instruction = sample.get("instruction", "")
    generated = sample.get("generated", "")

    # Remove any EOS token to avoid repeated ends
    generated = generated.replace(tokenizer.eos_token, "")
    text = apply_chat_template_for_vu(instruction, generated, tokenizer)
    return text


def _format_test_sample(sample, tokenizer):
    """
    For 'test' partition: user instruction only (assistant will generate).
    """
    instruction = sample.get("instruction", "")

    msgs = [{"role": "user", "content": instruction}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return text


def expand_samples_for_ens_inference(
    sample,
    tokenizer=None,
    dataset_name=None,
    model_name=None,
    attn_kwargs=None,
    **kwargs,
):
    """
    Specialized expansion logic for 'infer-ens' tasks, returning multiple
    expansions (replacements) of each sample. The expansions are stored in
    sample["expanded"], which will then be flattened after mapping.
    """
    sample["keep_instance"] = True
    try:
        txtinstr = sample.get("instruction", "")
        txtresp = sample.get("generated", "")

        # Handle Gemma-specific instructions for "math" dataset
        if "gemma" in type(tokenizer).__name__.lower() and dataset_name == "math":
            txtinstr = _math_instruction_update_for_gemma(txtinstr)

        eos_token = tokenizer.eos_token or ""
        txtresp = txtresp.replace(eos_token, "")

        # Format the text as a user+assistant conversation
        txtconv = apply_chat_template(txtinstr, txtresp, tokenizer, model_name)

        # Determine how many tokens in the instruction
        instr_msg = [{"role": "user", "content": txtinstr}]
        n_instr_tks = len(tokenizer.apply_chat_template(instr_msg, tokenize=True, add_generation_prompt=True))

        # Convert text to token IDs
        tks_int = tokenizer(txtconv, add_special_tokens=False, truncation=False, return_tensors="pt")["input_ids"][0]
        mask = torch.ones_like(tks_int)
        sample["n_tks"] = len(tks_int)

        tkans_srcidx = torch.as_tensor(sample["tkans_srcidx"]) + n_instr_tks
        tkans_int = torch.as_tensor(sample["tkresp_int"])[tkans_srcidx + 1 - n_instr_tks]

        tkattn_sims = torch.as_tensor(sample["tkattn_to_tkans_sims"])[: -len(tkans_srcidx)]
        sims_top_idx = thresholded_topk_1d(
            tkattn_sims, k=attn_kwargs.get("sims-k", 10), threshold=attn_kwargs.get("sims-threshold", 0.0)
        )
        tksim_srcids = torch.sort(torch.as_tensor(sample["tkattn_srcidx"])[sims_top_idx]).values
        tksim_ids = tksim_srcids + n_instr_tks + 1
        tksim_int = tks_int[tksim_ids]

        # Candidate tokens (top5 predictions)
        tksim_candint = torch.as_tensor(sample["top5_tkresp_int"]).view(-1, 5)[tksim_srcids]
        tksim_candprobs = torch.as_tensor(sample["top5_tkresp_prob"]).view(-1, 5)[tksim_srcids]

        # Threshold in 2D
        _, topk_prob_row_ids, topk_prob_col_ids = thresholded_topk_2d(
            tksim_candprobs,
            k=attn_kwargs.get("ens-k", 10) + len(tksim_int),
            threshold=attn_kwargs.get("ens-threshold", 1e-2),
        )

        # Identify columns that correspond exactly to the original tokens
        attn_cols = find_column_indices(tksim_candint, tksim_int)
        # Filter out the original tokens from the replacement set
        topk_prob_row_ids, topk_prob_col_ids = filter_indices(
            topk_prob_row_ids, topk_prob_col_ids, torch.arange(len(attn_cols)), attn_cols
        )

        # Create expansions
        n_total = len(topk_prob_row_ids) + 1

        tks_int = tks_int.unsqueeze(0).expand(n_total, -1).clone()
        mask = mask.unsqueeze(0).expand(n_total, -1)
        tksim_int = tksim_int.unsqueeze(0).expand(n_total, -1).clone()

        # Insert the replacements
        if n_total > 1:
            replace_row_ids = torch.arange(1, n_total)
            replace_tk_idx = tksim_ids[topk_prob_row_ids]
            replace_tk_int = tksim_candint[topk_prob_row_ids, topk_prob_col_ids]
            tks_int[replace_row_ids, replace_tk_idx] = replace_tk_int
            tksim_int[replace_row_ids, topk_prob_row_ids] = replace_tk_int

        # Gather expansions
        expanded_samples = []
        for row_idx in range(n_total):
            expanded_samples.append(
                {
                    "idx": sample["idx"],
                    "input_tks": tks_int[row_idx],
                    "attention_mask": mask[row_idx],
                    "attn_tks": tksim_int[row_idx],
                    "attn_src_ids": (tksim_ids - 1),
                    "ans_tks": tkans_int,
                    "ans_src_ids": tkans_srcidx,
                }
            )

        sample["expanded"] = expanded_samples

    except Exception:
        sample["keep_instance"] = False
        sample["expanded"] = []

    return sample


# ----------------------------------------------------------------
# Shared or generic helper functions
# ----------------------------------------------------------------


def apply_chat_template(
    instruction: str,
    generated: str,
    tokenizer,
    model_name: str,
) -> str:
    """
    Format the conversation for DeepSeek models or fallback format for others.
    """
    if "deepseek" in model_name.lower():
        msgs = [{"role": "user", "content": instruction}]
        txt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        txt += generated
        if not txt.endswith(tokenizer.eos_token):
            txt += tokenizer.eos_token
    else:
        msgs = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": generated},
        ]
        txt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    return txt


def apply_chat_template_for_vu(
    instruction: str,
    generated: str,
    tokenizer,
) -> str:
    """
    For "VU" tasks, we ask a new user query (VU_PROMPT) after the assistant's
    initial response.
    """
    msgs = [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": generated},
        {"role": "user", "content": VU_PROMPT},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def _math_instruction_update_for_gemma(instruction: str) -> str:
    """
    Append an instruction for final answer format when using Gemma on math tasks.
    """
    return f"{instruction.strip()}\n\nPut your final answer in `\\boxed{{ }}`."


def flatten_ds(ds: Dataset, field_name: str) -> Dataset:
    """
    Flatten a dataset column (field_name) that contains a list of dictionaries.
    Each dictionary becomes a separate row in the new dataset.
    """
    original_keys = [k for k in ds.column_names if k != field_name]
    # assume at least one non-empty list to discover sub-keys
    sub_keys = list(ds[0][field_name][0].keys())

    inst_dict = {k: [] for k in original_keys + sub_keys}

    for row in ds:
        expansions = row[field_name]
        for sub_dict in expansions:
            # Replicate original columns
            for k in original_keys:
                inst_dict[k].append(row[k])
            # Add expanded sub-keys
            for sk in sub_keys:
                inst_dict[sk].append(sub_dict[sk])

    return Dataset.from_dict(inst_dict)
