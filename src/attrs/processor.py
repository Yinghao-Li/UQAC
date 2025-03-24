import logging
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from seqlbtoolkit.io import ProgressBar
from seqlbtoolkit.timer import Timer

from src.utils.macro import STOP_WORDS, STOP_CHARACTERS, INSTR_TOKEN, CALIBRATION_RATIO
from src.utils.support import ids_to_tks, remove_whitespace
from .data import TokenAttributes


logger = logging.getLogger(__name__)


class TokenAttributesProcessor:
    """
    This class contains the methods for processing data stored in a TokenAttributesData instance.
    It includes methods for reducing attention matrices, building attention
    chains, and calculating similarity scores.
    """

    def __init__(
        self,
        data: TokenAttributes = None,
        tokenizer=None,
        extract_answer_span_func=None,
        keep_hidden_states=False,
    ):
        self.data = data
        self._tokenizer = tokenizer
        self._extract_answer_span_function = extract_answer_span_func
        self._keep_hidden_states = keep_hidden_states

        # Timers
        self._attention_reduction_timer = Timer()
        self._attention_chain_timer = Timer()
        self._similarity_timer = Timer()

        self._calibration_factor = torch.as_tensor(list(reversed(CALIBRATION_RATIO)))
        self._stop_words = STOP_WORDS.union(STOP_CHARACTERS).union({INSTR_TOKEN})

    def reset_timers(self):
        self._attention_reduction_timer.init()
        self._attention_chain_timer.init()
        self._similarity_timer.init()

    # ---------------
    # Core processing
    # ---------------
    def reduce_attention_manual(
        self,
        attn: np.ndarray,
        layer_selection: list[int] = None,
        layer_reduction: str = "mean",
        head_selection: list[int] = None,
        head_reduction: str = "mean",
    ) -> np.ndarray:
        """
        Manually reduce attention across selected layers/heads (e.g. mean or max).
        """
        attn_reduced = attn
        if layer_selection is not None:
            attn_reduced = attn_reduced[layer_selection, ...]

        if layer_reduction == "mean":
            attn_reduced = attn_reduced.mean(axis=0)
        elif layer_reduction == "max":
            attn_reduced = attn_reduced.max(axis=0)
        else:
            raise ValueError(f"Unknown layer_reduction: {layer_reduction}")

        if head_selection is not None:
            attn_reduced = attn_reduced[head_selection, ...]

        if head_reduction == "mean":
            attn_reduced = attn_reduced.mean(axis=0)
        elif head_reduction == "max":
            attn_reduced = attn_reduced.max(axis=0)
        else:
            raise ValueError(f"Unknown head_reduction: {head_reduction}")

        return attn_reduced

    def attn_mat_preprocessing(self, attn_mats: torch.Tensor) -> torch.Tensor:
        """
        Preprocess the attention matrix by normalizing it and removing the diagonal.
        Applies a calibration factor as well.
        """
        device = attn_mats.device
        n_tks = attn_mats.shape[-1]
        n_ans_tks = attn_mats.shape[-2]
        n_instr_tks = n_tks - n_ans_tks

        ans_range = torch.arange(n_ans_tks, device=device)
        instr_range = torch.arange(n_instr_tks, n_tks, device=device)

        # Zero out cross-attention from the answer side to the instruction side
        attn_mats[..., ans_range, instr_range] = 0
        attn_mats[..., 0] = 0

        attn_mats = self.attention_calibration(attn_mats)
        return attn_mats

    def attention_calibration(self, attention_matrices: torch.Tensor) -> torch.Tensor:
        """
        Multiply specific indices by a calibration factor.
        """
        device = attention_matrices.device
        n_tks = attention_matrices.shape[-1]
        n_ans_tks = attention_matrices.shape[-2]
        n_instr_tks = n_tks - n_ans_tks

        calibration_tensor = self._calibration_factor.to(dtype=attention_matrices.dtype, device=device)
        n_cal_elements = calibration_tensor.shape[0]

        rows = torch.arange(n_ans_tks, device=device).unsqueeze(-1)
        cols = torch.arange(n_cal_elements, device=device)

        # Apply calibration to "diagonal" region near the answer tokens
        attention_matrices[:, :, rows, rows + cols + (n_instr_tks - n_cal_elements)] *= calibration_tensor

        return attention_matrices

    @staticmethod
    def select_by_ids(matrices: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        """
        Select specific heads from attention_matrices based on head_ids.

        e.g. output shape: [n_heads_to_keep, n_ans_tks, n_tks].
        """
        mats = matrices.transpose(0, 1)
        ids = ids.transpose(0, 1)

        row_idx = torch.arange(mats.shape[0], device=mats.device).unsqueeze(-1)
        mats_selected = mats[row_idx, ids, :]
        return mats_selected.transpose(0, 1)

    def attn_mat_reduction(
        self,
        attn_mats: torch.tensor,
        attn_entropy: torch.tensor = None,
        n_heads_to_keep: int = 16,
        reduction: str = "max",
    ) -> np.ndarray:
        """
        Reduce attention matrices (e.g. using top-k heads with lowest entropy, then mean or max).
        """
        if attn_entropy is None:
            # Shape: [num_layers, num_heads, n_ans_tks, n_tks]
            attn_proc = self.attn_mat_preprocessing(attn_mats)
            n_ans_tks, n_tks = attn_proc.shape[-2], attn_proc.shape[-1]

            # In-place flatten: -> [num_layers*num_heads, n_ans_tks, n_tks]
            attn_proc = attn_proc.view(-1, n_ans_tks, n_tks)
            # In-place normalization to avoid allocating a second large tensor
            attn_proc.div_(attn_proc.sum(dim=-1, keepdim=True) + 1e-12)

            # Compute entropy = -sum(p * log(p)) in a memory-friendly way
            attn_log = (attn_proc + 1e-12).log()  # temporary log
            attn_entropy = -(attn_proc * attn_log).sum(dim=-1)  # shape: [num_layers*num_heads, n_ans_tks]
            del attn_log  # free log tensor if needed

            # Select heads with smallest entropy
            _, selected_head_ids = torch.topk(attn_entropy, k=n_heads_to_keep, dim=0, largest=False)
            attn_mat_selected = self.select_by_ids(attn_proc, selected_head_ids)

        else:
            _, selected_head_ids = torch.topk(attn_entropy, k=n_heads_to_keep, dim=0, largest=False)
            attn_mat_selected = self.select_by_ids(attn_mats, selected_head_ids)

        # Final reduction
        if reduction == "mean":
            attn_mat_reduced = attn_mat_selected.mean(dim=0)
        elif reduction == "max":
            attn_mat_reduced, _ = attn_mat_selected.max(dim=0)
        else:
            raise ValueError(f"Unsupported reduction method: {reduction}")

        return attn_mat_reduced

    def backtrack(
        self,
        attn_mat: torch.Tensor,
        tkans_srcids: torch.Tensor,
        tkresp_str: list[str],
        backtrack_buffer_size: int = 3,
        backtrack_score_threshold: float = 0.5,
        min_attn_chain_len: int = 5,
        max_steps: int = 1000,
    ):
        """
        Recursively build a chain of attended tokens starting from the given target token indices.
        Optimized to use top-k and avoid repeated sorting over the entire row.
        """
        attn_mat = attn_mat.cpu()
        device = attn_mat.device
        dtype = attn_mat.dtype

        n_instr_tks = attn_mat.shape[1] - attn_mat.shape[0]
        tk_str = [INSTR_TOKEN] * n_instr_tks + tkresp_str

        # Prepend rows of zeros in attn_mat (shape: [n_instr_tks, attn_mat.shape[-1]])
        padding = torch.zeros((n_instr_tks, attn_mat.shape[-1]), device=device, dtype=dtype)
        attn_mat = torch.cat([padding, attn_mat], dim=0)

        # Shift the "answer source indices" forward by n_instr_tks
        tkans_srcids = tkans_srcids + n_instr_tks

        # Adjust min chain length to account for the initial answer tokens
        min_attn_chain_len += tkans_srcids.shape[0]

        cleaned_tokens = [remove_whitespace(t).lower() for t in tk_str]
        stop_word_mask = [cleaned_tokens[i] in self._stop_words for i in range(len(tk_str))]

        tkattn_srcids_chain = tkans_srcids.tolist()  # store chain of token indices
        tkattn_score_chain = [0.0] * tkans_srcids.shape[0]  # parallel list for attention scores

        processed_tks_srcids = set()  # track tokens we've already used
        tksrc_srcids = tkattn_srcids_chain

        num_tks = attn_mat.shape[0]  # total padded tokens (instr + response)

        for step_i in range(max_steps):
            # Once we run out of tokens to backtrack from, break
            if not tksrc_srcids:
                break
            processed_tks_srcids.update(tksrc_srcids)
            tkattn_scores = torch.zeros(num_tks, device=device, dtype=dtype)

            # For each source token in this iteration, gather top-k attentions
            for src_tk_idx in tksrc_srcids:
                src_tk_attn_vec = attn_mat[src_tk_idx]
                top_tk_ids = torch.argsort(src_tk_attn_vec, descending=True)

                # Merge scores into attended_scores
                for i in range(top_tk_ids.size(0)):
                    if i >= 5 and ((tkattn_scores > 0).any() or len(tkattn_srcids_chain) >= min_attn_chain_len):
                        break
                    tk_idx = top_tk_ids[i].item()
                    if tk_idx == src_tk_idx or (tk_idx - 1) in processed_tks_srcids or stop_word_mask[tk_idx]:
                        continue
                    tkattn_scores[tk_idx] += src_tk_attn_vec[tk_idx]

            nonzero_mask = tkattn_scores > 0
            if not torch.any(nonzero_mask):
                break

            # Indices of attended tokens and their scores. Sort by descending attention score
            nonzero_indices = nonzero_mask.nonzero(as_tuple=True)[0]
            nonzero_scores = tkattn_scores[nonzero_indices]
            sorted_scores, sorted_idx_order = torch.sort(nonzero_scores, descending=True)
            sorted_indices = nonzero_indices[sorted_idx_order]  # these are the token IDs

            # pick up to `backtrack_buffer_size` new tokens in descending score order
            tktgt_ids = []
            for i in range(sorted_indices.size(0)):
                if len(tktgt_ids) >= backtrack_buffer_size:
                    break

                tk_idx = sorted_indices[i].item()
                score = sorted_scores[i].item()

                # If the score is below threshold, but the chain is already above min length, we can stop
                if (
                    score < backtrack_score_threshold
                    and (len(tkattn_srcids_chain) + len(tktgt_ids)) >= min_attn_chain_len
                ):
                    break

                tktgt_ids.append(tk_idx)
                tkattn_srcids_chain.append(tk_idx - 1)  # shift back by 1 to keep the "raw" index
                tkattn_score_chain.append(score)

            # Prepare for next iteration
            tksrc_srcids = [t - 1 for t in tktgt_ids]

            # If we didn't pick any new tokens, break
            if not tktgt_ids:
                break

        pairs = sorted(zip(tkattn_srcids_chain, tkattn_score_chain), key=lambda x: x[0])
        sorted_ids, sorted_scores = zip(*pairs)

        # Convert to torch Tensors
        tkattn_srcids = torch.tensor(sorted_ids, dtype=torch.long)
        tkattn_scores = torch.tensor(sorted_scores, dtype=dtype)
        tkattn_srcids = tkattn_srcids - n_instr_tks

        return tkattn_srcids, tkattn_scores

    def constr_attn_chain(
        self,
        tkresp_int: torch.tensor,
        attn_mats: torch.tensor,
        attn_entropy: torch.tensor = None,
        attention_reduction: str = "mean",
        attention_reduction_n_heads: int = 8,
        attention_backgracking_buffer_size: int = 3,
        attention_backgracking_threshold: float = 0.1,
        minimum_attention_chain_length: int = 5,
    ):
        """
        Builds an attention chain by first reducing attention across heads,
        then backtracking from the answer token(s).
        """

        txtresp = self._tokenizer.decode(tkresp_int)
        try:
            tkans_span_in_tkresp = self._extract_answer_span_function(self._tokenizer, txtresp)
        except Exception:
            return None, None, None

        tkans_srcids = torch.arange(tkans_span_in_tkresp[0][0], tkans_span_in_tkresp[-1][-1]) - 1

        with self._attention_reduction_timer:
            attn_mat_reduced = self.attn_mat_reduction(
                attn_mats=attn_mats,
                attn_entropy=attn_entropy,
                reduction=attention_reduction,
                n_heads_to_keep=attention_reduction_n_heads,
            )

        tkresp_str = ids_to_tks(self._tokenizer, tkresp_int)
        with self._attention_chain_timer:
            tkattn_srcids, tkattn_scores = self.backtrack(
                attn_mat=attn_mat_reduced,
                tkans_srcids=tkans_srcids,
                tkresp_str=tkresp_str,
                backtrack_buffer_size=attention_backgracking_buffer_size,
                backtrack_score_threshold=attention_backgracking_threshold,
                min_attn_chain_len=minimum_attention_chain_length,
            )

        return tkattn_srcids, tkattn_scores, tkans_srcids

    def get_tkattn_to_tkans_sims(
        self,
        tkans_ids: torch.tensor,
        tkattn_ids: torch.tensor,
        output_logits: torch.tensor,
        similarity_reduction: str = "sim-mean",
    ) -> torch.tensor:
        """
        Compute similarity between attended tokens and answer tokens.
        """
        if similarity_reduction == "starting-token":
            embs_ans = output_logits[tkans_ids[0]].reshape(1, -1)
        elif similarity_reduction == "emb-mean":
            embs_ans = output_logits[tkans_ids].mean(axis=0, keepdims=True)
        elif similarity_reduction == "sim-mean":
            embs_ans = output_logits[tkans_ids]
        else:
            raise ValueError(f"Unknown similarity aggregation: {similarity_reduction}")

        embs_attn = output_logits[tkattn_ids]

        with self._similarity_timer:
            sims = torch.cosine_similarity(embs_ans.unsqueeze(1), embs_attn, dim=-1)
            if similarity_reduction == "sim-mean":
                sims = sims.mean(dim=0)
            else:
                sims = sims.squeeze()

        return sims

    def process(
        self,
        attention_reduction="mean",
        n_attention_heads=8,
        backtracking_buffer_size=3,
        backtracking_threshold=0.1,
        minimum_attention_chain_length=5,
        similarity_aggregation="sim-mean",
        disable_progress_bar=False,
        sims_only=False,
        **kwargs,
    ):
        """
        Orchestrates building attention chains and computing similarities
        for each instance in self.data.
        """
        assert self.data is not None, "No data to process."

        # Prepare to store processed results
        self.data.tkattn_srcidx_list = []
        self.data.tkattn_scores_list = []
        self.data.tkattn_to_tkans_sims_list = []
        self.data.tkans_srcidx_list = []

        # Timers
        self.reset_timers()

        # Quick length checks
        self.data.check_equal_lengths(keep_hidden_states=self._keep_hidden_states)
        if not self.data.attn_entropy_list:
            self.data.attn_entropy_list = [None] * len(self.data)

        failed_ids = list()
        pbar = ProgressBar(
            total=len(self.data), transient=True, desc="Build Attention Chains", disable=disable_progress_bar
        )
        with pbar:
            for idx, tkresp_int, attn_mats, attn_entropy, output_logits in zip(
                self.data.instance_ids,
                self.data.tkresp_int_list,
                self.data.attn_mats_list,
                self.data.attn_entropy_list,
                self.data.output_logits_list,
            ):
                # Build chain
                tkattn_srcids, tkattn_scores, tkans_srcids = self.constr_attn_chain(
                    tkresp_int=tkresp_int,
                    attn_mats=attn_mats,
                    attn_entropy=attn_entropy,
                    attention_reduction=attention_reduction,
                    attention_reduction_n_heads=n_attention_heads,
                    attention_backgracking_buffer_size=backtracking_buffer_size,
                    attention_backgracking_threshold=backtracking_threshold,
                    minimum_attention_chain_length=minimum_attention_chain_length,
                )
                if tkattn_srcids is None:
                    failed_ids.append(idx)
                    pbar.update()
                    continue

                if sims_only:
                    tkattn_srcids = torch.arange(tkans_srcids.max() + 1)
                    tkattn_scores = torch.zeros_like(tkattn_srcids)

                # Save chain
                self.data.tkattn_srcidx_list.append(tkattn_srcids.cpu())
                self.data.tkattn_scores_list.append(tkattn_scores.cpu())
                self.data.tkans_srcidx_list.append(tkans_srcids.cpu())

                # Similarities
                tkattn_to_tkans_sims = self.get_tkattn_to_tkans_sims(
                    tkans_ids=tkans_srcids + 1,
                    tkattn_ids=tkattn_srcids + 1,
                    output_logits=output_logits,
                    similarity_reduction=similarity_aggregation,
                )
                self.data.tkattn_to_tkans_sims_list.append(tkattn_to_tkans_sims.cpu())

                pbar.update()

        return failed_ids


def probability_aggregation(probabilities: list | np.ndarray, aggregation="mean"):
    """
    Simple helper function for aggregating token probabilities in various ways.
    """
    assert aggregation in ("mean", "log-mean", "entropy", "prod"), f"Unsupported aggregation method: {aggregation}"

    probabilities = np.array(probabilities)

    if aggregation == "mean":
        return np.mean(probabilities)
    elif aggregation == "log-mean":
        return np.exp(np.mean(np.log(probabilities + 1e-12)))
    elif aggregation == "prod":
        return np.exp(np.sum(np.log(probabilities + 1e-12)))
    elif aggregation == "entropy":
        return -np.sum(probabilities * np.log(probabilities + 1e-12))

    return None
