"""
# Author: Yinghao Li
# Modified: March 18th, 2025
# ---------------------------------------
# Description: Collate function for ChemBERTa.
"""

import logging
from seqlbtoolkit.training.dataset import Batch, unpack_instances

logger = logging.getLogger(__name__)


class Collator:
    def __init__(self, tokenizer, dataset_text_field: str = "text"):
        self.tokenizer = tokenizer
        self.dataset_text_field = dataset_text_field

    def __call__(self, instances: list[dict], *args, **kwargs) -> Batch:
        txt_seqs, instance_ids, n_instr_tks = unpack_instances(instances, ["text", "idx", "n_instr_tks"])
        tks = self.tokenizer(txt_seqs, add_special_tokens=False, truncation=False, padding=True, return_tensors="pt")

        return Batch(
            input_ids=tks["input_ids"],
            attention_mask=tks["attention_mask"],
            idx=instance_ids,
            n_instr_tks=n_instr_tks,
        )


class EnsCollator:
    def __init__(self, tokenizer, dataset_text_field: str = "text"):
        self.tokenizer = tokenizer
        self.txt_kw = dataset_text_field

    def __call__(self, instances: list[dict]):
        # Unpack needed fields
        (instance_ids, input_tks, attention_mask, attn_tks, attn_src_ids, ans_tks, ans_src_ids) = unpack_instances(
            instances,
            ("idx", "input_tks", "attention_mask", "attn_tks", "attn_src_ids", "ans_tks", "ans_src_ids"),
        )

        batch = self.tokenizer.pad(
            {"input_ids": input_tks, "attention_mask": attention_mask},
            padding=True,
            return_tensors="pt",
        )
        batch_input_tks = batch["input_ids"]
        batch_attention_mask = batch["attention_mask"]

        return Batch(
            input_ids=batch_input_tks,
            attention_mask=batch_attention_mask,
            idx=instance_ids,
            attn_src_ids=attn_src_ids,
            attn_tks=attn_tks,
            ans_src_ids=ans_src_ids,
            ans_tks=ans_tks,
        )
