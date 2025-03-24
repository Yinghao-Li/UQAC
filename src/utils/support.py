"""
# Author: Yinghao Li
# Modified: March 19th, 2025
# ---------------------------------------
# Description: Support functions.
"""

import re
import numpy as np
import random
import torch


def thresholded_topk_2d(tensor: torch.Tensor, k: int = 10, threshold: float = 1e-2):
    R, C = tensor.shape
    flattened = tensor.view(-1)  # (R*C,)
    mask = flattened > threshold  # boolean mask
    masked_values = flattened[mask]  # only keep above threshold

    if masked_values.numel() == 0:
        return torch.empty(0), torch.empty(0), torch.empty(0)

    # If we have fewer than k valid values, just sort descending
    if masked_values.numel() <= k:
        sorted_vals, sorted_idx = masked_values.sort(descending=True)
    else:
        sorted_vals, sorted_idx = masked_values.topk(k, largest=True)

    # Map sorted_idx back to original flattened indices
    # We need the actual indices in 'flattened' that correspond to the masked values
    masked_indices = torch.nonzero(mask, as_tuple=False).squeeze(1)
    topk_indices = masked_indices[sorted_idx]

    row_indices = topk_indices // C
    col_indices = topk_indices % C

    return sorted_vals, row_indices, col_indices


def thresholded_topk_1d_numpy(arr: np.ndarray, k: int, threshold: float):
    """
    Returns the indices of the top-k largest values in 'arr'
    that are >= 'threshold'. If fewer than k values qualify,
    all qualifying indices are returned (in descending order of their values).
    """
    # Step 1: Apply the threshold
    mask = arr >= threshold
    valid_indices = np.where(mask)[0]

    # If no elements pass the threshold, return an empty array
    if len(valid_indices) == 0:
        return np.array([], dtype=int)

    valid_values = arr[valid_indices]

    # Step 2: If fewer than k elements qualify, just sort them by descending value
    if len(valid_values) <= k:
        sorted_local_indices = np.argsort(valid_values)[::-1]  # sort descending
        return valid_indices[sorted_local_indices]

    # Step 3: Otherwise, find the top-k among the qualifying values
    # argpartition is more efficient for top-k than a full sort
    part = np.argpartition(valid_values, -k)[-k:]  # indices of top-k in valid_values
    # Now sort those k indices by actual value, descending
    top_k_local_indices = part[np.argsort(valid_values[part])][::-1]

    # Convert local indices to global indices
    return valid_indices[top_k_local_indices]


def thresholded_topk_1d(tensor: torch.Tensor, k: int, threshold: float):
    tensor = tensor.float()

    # 1) Apply the threshold
    mask = tensor >= threshold
    valid_indices = mask.nonzero(as_tuple=True)[0]  # shape: (num_valid,)
    if valid_indices.numel() == 0:
        # If nothing passes the threshold, return an empty LongTensor
        return torch.empty(0, dtype=torch.long)

    valid_values = tensor[valid_indices]  # shape: (num_valid,)

    # 2) If fewer than k elements qualify, just sort them in descending order
    if valid_values.numel() <= k:
        sorted_local_indices = torch.argsort(valid_values, descending=True)
        return valid_indices[sorted_local_indices]

    # 3) Otherwise, find the top-k among the qualifying values
    _, topk_local_indices = torch.topk(valid_values, k, largest=True, sorted=True)
    # torch.topk(..., sorted=True) ensures they're returned in descending order
    return valid_indices[topk_local_indices]


def find_column_indices(matrix: torch.Tensor | np.ndarray, values: torch.Tensor | np.ndarray) -> torch.Tensor:
    """
    Given an (N x D) matrix and a 1D tensor of length N, return a 1D tensor of column
    indices where matrix[i, col_idx[i]] == values[i].
    """
    # Shape of (matrix == values.unsqueeze(1)) is [N, D], broadcasting values[i] over row i.
    # argmax(dim=1) returns the index of the first True/1 for each row.
    assert type(matrix) == type(values), "Matrix and values must have the same type."

    if isinstance(matrix, np.ndarray):
        return (matrix == values[:, None]).argmax(axis=1)

    return (matrix == values.unsqueeze(1)).to(torch.int64).argmax(dim=1)


def filter_indices(
    row_indices: torch.Tensor,
    col_indices: torch.Tensor,
    rows_to_remove: torch.Tensor,
    cols_to_remove: torch.Tensor,
):
    ids_pairs = torch.stack((row_indices, col_indices), dim=1)
    ids_to_remove_pairs = torch.stack((rows_to_remove, cols_to_remove), dim=1)
    eq = ids_pairs.unsqueeze(1).eq(ids_to_remove_pairs.unsqueeze(0))  # (N, M, 2)

    rowwise_match = eq.all(dim=2)
    mask_remove = rowwise_match.any(dim=1)

    mask_keep = ~mask_remove
    return row_indices[mask_keep], col_indices[mask_keep]


def ids_to_tks(tokenizer, ids: tuple[int] | list[int] | np.ndarray | torch.Tensor, normalize: bool = True):
    """
    Convert a list of token IDs to a list of token strings.

    Parameters
    ----------
    tokenizer : transformers.PreTrainedTokenizer
        The tokenizer to use for decoding.
    ids : list of int
        The list of token IDs to convert.
    normalized : bool, default=True
        If True, decode the token IDs using the tokenizer's `decode` method.
        If False, convert the token IDs to token strings using `ids_to_tokens`.

    Returns
    -------
    list of str
        The list of token strings corresponding to the input IDs.
    """
    if normalize:
        tks = [tokenizer.decode(t_id) for t_id in ids]
    else:
        tks = tokenizer.convert_ids_to_tokens(ids)
    if isinstance(ids, (torch.Tensor, list, tuple)):
        return tks
    elif isinstance(ids, np.ndarray):
        return np.asarray(tks)
    else:
        raise ValueError("Unsupported input type for 'ids' parameter.")


def tokenize(tokenizer, text_sequence, normalize=True):
    if not normalize:
        return tokenizer.tokenize(text_sequence, add_special_tokens=False)

    token_ids = tokenizer.encode(text_sequence, add_special_tokens=False)
    return ids_to_tks(tokenizer, token_ids, normalize=normalize)


def remove_whitespace(text: str) -> str:
    """
    Remove all whitespace characters from a string.
    """
    return re.sub(r"\s+", "", text)


def ids_selection(ids: list[str | int], n: int) -> list[str | int]:
    """
    Randomly select n elements from a list of IDs.
    """
    return random.sample(ids, n)
