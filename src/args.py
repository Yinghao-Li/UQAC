"""
# Modified: March 24th, 2025
# ---------------------------------------
# Description: Self-defined arguments
"""

import os.path as osp
from transformers import TrainingArguments  # Import for easy references
from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator
from accelerate.logging import get_logger

from src.eval import (
    evaluate_gsm8k_responses,
    get_gsm8k_answer_token_span,
    evaluate_math_responses,
    get_boxed_token_span,
    evaluate_bbh_responses,
)

logger = get_logger(__name__)
accelerator = Accelerator()

__all__ = ["TrainingArguments", "PipelineArguments", "ModelArguments", "DataArguments", "AttentionProcessingArguments"]

FUNC_MAP = {
    "gsm8k": {
        "eval_answer_correctness_function": evaluate_gsm8k_responses,
        "extract_answer_span_function": get_gsm8k_answer_token_span,
    },
    "math": {
        "eval_answer_correctness_function": evaluate_math_responses,
        "extract_answer_span_function": get_boxed_token_span,
    },
    "bbh": {
        "eval_answer_correctness_function": evaluate_bbh_responses,
        "extract_answer_span_function": get_boxed_token_span,
    },
}


@dataclass
class PipelineArguments:
    task: Optional[str] = field(
        default="train",
        metadata={
            "choices": ("infer", "infer-vllm", "infer-uq", "infer-ens", "infer-vu"),
            "help": "The task to run the pipeline.",
        },
    )
    project_input_root: Optional[str] = field(
        default=".",
        metadata={"help": "The root input directory of the project."},
    )
    project_output_root: Optional[str] = field(
        default=".",
        metadata={"help": "The root output directory of the project."},
    )


# Define and parse arguments.
@dataclass
class ModelArguments:

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    use_nested_quant: Optional[bool] = field(
        default=False,
        metadata={"help": "Activate nested quantization for 4bit base models"},
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default="float16",
        metadata={"help": "Compute dtype for 4bit base models"},
    )
    bnb_4bit_quant_storage_dtype: Optional[str] = field(
        default="uint8",
        metadata={"help": "Quantization storage dtype for 4bit base models"},
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4",
        metadata={"help": "Quantization type fp4 or nf4"},
    )
    use_flash_attn: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables Flash attention for training."},
    )
    use_peft_lora: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables PEFT LoRA for training."},
    )
    use_8bit_quantization: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables loading model in 8bit."},
    )
    use_4bit_quantization: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables loading model in 4bit."},
    )
    use_reentrant: Optional[bool] = field(
        default=False,
        metadata={"help": "Gradient Checkpointing param. Refer the related docs"},
    )
    inference_temperature: Optional[float] = field(
        default=0,
        metadata={"help": "Temperature for inference."},
    )
    logprobs: Optional[int] = field(
        default=None,
        metadata={"help": "Number of logprobs to return."},
    )


@dataclass
class DataArguments:
    dataset_dir: Optional[str] = field(default=None, metadata={"help": "the directory to local dataset"})
    dataset_name: Optional[str] = field(
        default="gsm8k",
        metadata={"help": "The preference dataset to use."},
    )
    disable_dataset_cache: Optional[bool] = field(
        default=False,
        metadata={"help": "Disable dataset caching when `mapping` is called."},
    )
    disable_seq_length_filter: Optional[bool] = field(
        default=False,
        metadata={"help": "Disable filtering samples based on sequence length."},
    )
    dataset_num_mapping_proc: Optional[int] = field(default=8, metadata={"help": "the number of mapping processes"})
    dataset_num_filtering_proc: Optional[int] = field(default=1, metadata={"help": "the number of filtering processes"})
    dataset_text_field: str = field(default="text", metadata={"help": "Dataset field to use as input text."})
    dataset_sorted_by_length: Optional[bool] = field(
        default=False,
        metadata={"help": "Sort dataset by length for efficient batching."},
    )
    max_seq_length: Optional[int] = field(default=4096)
    max_new_tokens: Optional[int] = field(default=128)
    subsample: float = field(
        default=0,
        metadata={"help": "The fraction or number of the training dataset to use for training."},
    )
    resp_dir: str = field(
        default="none",
        metadata={"help": "The folder to save the inference results."},
    )
    resp_file: Optional[str] = field(
        default="results.json",
        metadata={"help": "The file name to save the inference output."},
    )
    attn_kwargs: Optional[dict] = field(default=None)

    def __post_init__(self):
        if self.subsample < 0:
            raise ValueError("Subsample values must be greater than or equal to 0.")
        if self.subsample > 1:
            self.subsample = int(self.subsample)


@dataclass
class AttentionProcessingArguments:
    attrs_folder: Optional[str] = field(
        default="attrs",
        metadata={"help": "The folder to save the simplified token inference details."},
    )
    attrs_file_name: Optional[str] = field(
        default="attrs.h5",
        metadata={"help": "The file name to save the token inference details."},
    )
    attrs_save_frequency: Optional[int] = field(
        default=50,
        metadata={"help": "The frequency to save the token inference details."},
    )
    ensembles_pred_folder: Optional[str] = field(
        default="ens-results",
        metadata={"help": "The file name to save the ensemble predictions."},
    )
    attention_reduction: str = field(
        default="max",
        metadata={"choices": ("mean", "max"), "help": "How to aggregate the attention weights from different heads."},
    )
    n_attention_heads: int = field(
        default=16,
        metadata={"help": "Number of attention heads to use for the model."},
    )
    backtracking_buffer_size: int = field(
        default=3,
        metadata={"help": "Size of the backtracking buffer for attention backtracking."},
    )
    backtracking_threshold: float = field(
        default=0.5,
        metadata={"help": "Threshold for backtracking attention weights."},
    )
    minimum_attention_chain_length: int = field(
        default=5,
        metadata={"help": "Minimum length of the attention chain to consider."},
    )
    similarity_aggregation: str = field(
        default="sim-mean",
        metadata={
            "choices": ("sim-mean", "emb-mean", "starting-token"),
            "help": "How to aggregate the attention token to answer token similarities.",
        },
    )
    top_k_similarity: Optional[int] = field(
        default=10,
        metadata={"help": "The top-k most similar tokens to process attention."},
    )
    similarity_threshold: Optional[float] = field(
        default=0,
        metadata={"help": "The threshold to filter attention similarities."},
    )
    sims_only: Optional[bool] = field(
        default=False,
        metadata={"help": "Ablation Study. Only use local similarities for attention processing."},
    )


def argument_processing(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArguments,
    attn_args: AttentionProcessingArguments,
    *args,
    **kwargs,
) -> None:

    model_name = osp.basename(model_args.model_name_or_path)
    if data_args.resp_dir == "none":
        data_args.resp_dir = osp.join(training_args.output_dir, data_args.dataset_name, model_name)
    if attn_args.sims_only:
        attn_args.attrs_folder += "-sims-only"

    data_args.attn_kwargs = {
        "eval_answer_correctness_function": FUNC_MAP[data_args.dataset_name]["eval_answer_correctness_function"],
        "extract_answer_span_function": FUNC_MAP[data_args.dataset_name]["extract_answer_span_function"],
        "sims-k": attn_args.top_k_similarity,
        "sims-threshold": attn_args.similarity_threshold,
        "attrs_folder": attn_args.attrs_folder,
        "attrs_file_name": attn_args.attrs_file_name,
    }

    return None
