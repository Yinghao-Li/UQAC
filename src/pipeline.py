"""
# Modified: March 24th, 2025
# ---------------------------------------
# Description:

Causal LM Trainer including fine-tuning and inference
"""

import os
import os.path as osp
import torch
import logging
from accelerate import Accelerator
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoTokenizer,
)
from torch.utils.data import DataLoader
from vllm import LLM, SamplingParams

from trl import is_xpu_available
from seqlbtoolkit.io import save_json, ProgressBar

from src.dataset import CausalLMDataset, Collator, EnsCollator
from src.utils.io import save_attrs
from src.attrs import TokenAttributes, TokenAttributesProcessor
from src.args import ModelArguments, DataArguments, TrainingArguments, AttentionProcessingArguments

logger = logging.getLogger(__name__)
accelerator = Accelerator()

__all__ = ["Pipeline"]


class Pipeline:

    def __init__(
        self,
        task: str,
        model_args: ModelArguments,
        data_args: DataArguments,
        training_args: TrainingArguments,
        attn_args: AttentionProcessingArguments,
    ):
        """Initializes the Trainer object."""

        self.model_args: ModelArguments = model_args
        self.data_args: DataArguments = data_args
        self.training_args: TrainingArguments = training_args
        self.attn_args: AttentionProcessingArguments = attn_args

        self.model = None
        self.tokenizer = None
        self.datasets = None
        self.trainer = None
        self.optimizer_state = None

        self.tokenizer_size_changed = False
        self.task = task
        self.model_name = osp.basename(self.model_args.model_name_or_path)

        if task in ("infer-vllm", "infer-vu"):
            self.prepare_for_inference_vllm()
        elif task in ("infer", "infer-uq", "infer-ens"):
            self.prepare_for_inference()
        else:
            raise ValueError(f"Invalid task: {task}.")

    def prepare_for_inference(self):
        logger.info("Loading tokenizer and base model")
        self.initialize_tokenizer()

        self.initialize_model()
        self.tokenizer.padding_side = "left"

        self.initialize_datasets()
        return None

    def prepare_for_inference_vllm(self):
        self.model = LLM(
            model=self.model_args.model_name_or_path,
            enable_lora=False,
            max_model_len=self.data_args.max_seq_length,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_args.model_name_or_path)
        self.setup_padding()
        self.tokenizer.padding_side = "left"

        self.initialize_datasets()
        return None

    def initialize_model(self):

        # Step 1: Load the model

        bnb_config = None
        quant_storage_dtype = None
        device_map = "auto"

        if self.model_args.use_4bit_quantization:
            compute_dtype = getattr(torch, self.model_args.bnb_4bit_compute_dtype)
            quant_storage_dtype = getattr(torch, self.model_args.bnb_4bit_quant_storage_dtype)

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=self.model_args.use_4bit_quantization,
                bnb_4bit_quant_type=self.model_args.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=self.model_args.use_nested_quant,
                bnb_4bit_quant_storage=quant_storage_dtype,
            )

            if compute_dtype == torch.float16 and self.model_args.use_4bit_quantization:
                major, _ = torch.cuda.get_device_capability()
                if major >= 8 and accelerator.is_local_main_process:
                    logger.warning("=" * 80)
                    logger.warning("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
                    logger.warning("=" * 80)

        elif self.model_args.use_8bit_quantization:
            bnb_config = BitsAndBytesConfig(load_in_8bit=self.model_args.use_8bit_quantization)

        torch_dtype = (
            quant_storage_dtype if quant_storage_dtype and quant_storage_dtype.is_floating_point else torch.float32
        )

        if bnb_config:
            device_map = (
                {"": f"xpu:{Accelerator().process_index}"} if is_xpu_available() else {"": Accelerator().process_index}
            )

        # Setup model arguments
        model_args = {
            "pretrained_model_name_or_path": self.model_args.model_name_or_path,
            "quantization_config": bnb_config,
            "device_map": device_map,
            "trust_remote_code": True,
            "torch_dtype": torch_dtype,
        }

        if self.task == "infer-uq":
            model_args["use_cache"] = False
            model_args["return_dict_in_generate"] = True
            model_args["output_hidden_states"] = True
            model_args["output_attentions"] = True
            model_args["attn_implementation"] = "eager"

            if "gemma" in self.model_args.model_name_or_path.lower():
                model_args["cache_implementation"] = None

        if self.training_args.gradient_checkpointing:
            model_args["use_cache"] = False

        # Load the model
        self.model = AutoModelForCausalLM.from_pretrained(**model_args)

        if self.tokenizer_size_changed:
            # Resize the embeddings
            self.model.resize_token_embeddings(len(self.tokenizer))
            # Configure the pad token in the model
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        return None

    def initialize_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.model_name_or_path,
            trust_remote_code=True,
            add_bos_token=False,
            add_eos_token=False,
            legacy=False,
        )
        self.setup_padding()
        return None

    def setup_padding(self):
        if "llama" in self.model_args.model_name_or_path.lower():
            self.tokenizer.pad_token = "<|finetune_right_pad_id|>"
            self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)

        elif not self.tokenizer.pad_token:
            self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
            self.tokenizer_size_changed = True

        self.tokenizer.padding_side = "right"
        return None

    def initialize_datasets(self):
        self.datasets = (
            CausalLMDataset(
                task=self.task,
                model_name=self.model_name,
                seed=self.training_args.seed,
                partition="test" if "infer" in self.task else "train",
                tokenizer=self.tokenizer,
                **asdict(self.data_args),
            )
            .load()
            .process(
                tokenizer=self.tokenizer,
                dataset_text_field=self.data_args.dataset_text_field,
                num_mapping_proc=self.data_args.dataset_num_mapping_proc,
                num_filtering_proc=self.data_args.dataset_num_filtering_proc,
            )
        )

        return None

    @torch.no_grad()
    def infer_vllm(self):
        logger.info(
            f"Initializing pipeline for text generation with temperature {self.model_args.inference_temperature}"
        )
        test_ds = self.datasets.ds

        logger.info("Generating responses...")

        outputs = self.model.generate(
            test_ds["text"],
            sampling_params=SamplingParams(
                temperature=self.model_args.inference_temperature,
                logprobs=self.model_args.logprobs,
                max_tokens=self.data_args.max_new_tokens,
                seed=self.training_args.seed,
            ),
            lora_request=None,
        )

        logger.info("Saving results")
        indexed_results = list()
        indexed_probs = list()
        for idx, output in zip(test_ds["idx"], outputs):
            generated_text = output.outputs[0].text
            if self.model_args.logprobs is not None:
                logprobs = [{k: v.logprob for k, v in p.items()} for p in output.outputs[0].logprobs]
                indexed_probs.append({"idx": idx, "logprobs": logprobs})
            indexed_results.append({"idx": idx, "generated": generated_text})

        output_dir = self.data_args.resp_dir
        output_path = osp.join(output_dir, self.data_args.resp_file)
        if self.task == "infer-vu":
            output_path = output_path.replace(".json", "_vu.json")
        save_json(indexed_results, output_path)

        if self.model_args.logprobs is not None:
            probs_output_path = osp.join(output_dir, self.data_args.resp_file.replace(".json", "_logprobs.pt"))
            torch.save(indexed_probs, probs_output_path)

        logger.info(f"Results saved to {output_path}.")
        logger.info("Done.")

        return None

    @torch.inference_mode()
    def infer_uq(self):
        """
        Perform inference and periodically save results in parallel using multiprocessing.
        """

        # Set up DataLoader
        test_loader = DataLoader(
            self.datasets.ds,
            batch_size=self.training_args.per_device_eval_batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=Collator(self.tokenizer),
        )
        attrs_processor = TokenAttributesProcessor(
            tokenizer=self.tokenizer,
            extract_answer_span_func=self.data_args.attn_kwargs["extract_answer_span_function"],
            keep_hidden_states=False,
        )
        attrs_data = TokenAttributes()

        self.model.eval()
        pbar = ProgressBar(total=len(test_loader), desc="Inference")
        with pbar:
            for batch in test_loader:
                # Move batch to device
                batch.to(self.model.device)

                # Forward pass
                outputs = self.model(**batch.tensors)
                logits = outputs.logits  # (batch_size, seq_len, vocab_size)
                attns = outputs.attentions  # tuple of (batch_size, n_heads, seq_len, seq_len)
                last_states = outputs.hidden_states[-1]

                # Compute top-5 probabilities and ids
                probs = torch.softmax(logits, dim=-1)
                top5_probs, top5_tks = torch.topk(probs, 5, dim=-1)

                # shape: (batch_size, num_layers, num_heads, seq_len, seq_len)
                attns_stacked = torch.stack(attns, dim=0).transpose(0, 1)

                batch.to("cpu")
                inputs = (batch.idx, batch.input_ids, batch.n_instr_tks)
                results = (top5_tks, top5_probs, attns_stacked, last_states)

                for idx, input_tks, n_instr_tks, top_tks, top_probs, attns, last_state in zip(*inputs, *results):
                    # Number of pad tokens + instruction tokens to remove
                    n_pad_tks = torch.sum(input_tks == self.tokenizer.pad_token_id)
                    n_rm_tks = n_pad_tks + n_instr_tks

                    # Remove leading padding/instruction tokens
                    input_tks, top_tks, top_probs, last_state = (
                        x[n_rm_tks:] for x in (input_tks, top_tks, top_probs, last_state)
                    )
                    # Slice attention: [num_layers, num_heads, seq_len_ans, seq_len]
                    attns = attns[..., n_rm_tks:, n_pad_tks:].contiguous()

                    # TokenAttributes processing
                    attrs_processor.data = TokenAttributes(
                        instance_ids=[idx],
                        tkresp_int_list=[input_tks],
                        top5_tkresp_int_list=[top_tks],
                        top5_tkresp_prob_list=[top_probs],
                        attn_mats_list=[attns],
                        output_logits_list=[last_state],
                    )
                    failed_ids = attrs_processor.process(disable_progress_bar=True, **asdict(self.attn_args))
                    if failed_ids:
                        logger.error(f"Failed to process idx {idx}")
                        continue
                    attrs_data += attrs_processor.data.to("cpu")

                # Update progress
                pbar.update()

                # Save periodically to avoid large memory usage
                if len(attrs_data) >= self.attn_args.attrs_save_frequency:
                    pbar.update(advance=0, description="[red]Saving Predictions")
                    attrs_data.save(
                        file_dir=osp.join(self.data_args.resp_dir, self.attn_args.attrs_folder),
                        file_name=self.attn_args.attrs_file_name,
                        disable_progress_bar=True,
                    )
                    attrs_data.clear()
                    pbar.update(advance=0, description="Inference")

            # Final save of any remaining items
            if len(attrs_data) > 0:
                pbar.update(advance=0, description="[red]Save Predictions")
                attrs_data.save(
                    file_dir=osp.join(self.data_args.resp_dir, self.attn_args.attrs_folder),
                    file_name=self.attn_args.attrs_file_name,
                    disable_progress_bar=True,
                )

        logger.info("Done.")
        return None

    @torch.inference_mode()
    def infer_ens(self):

        test_loader = DataLoader(
            self.datasets.ds,
            batch_size=self.training_args.per_device_eval_batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=EnsCollator(self.tokenizer),
            num_workers=self.data_args.dataset_num_mapping_proc,
        )

        self.model.eval()
        result_dict = dict()

        pbar = ProgressBar(total=len(test_loader), desc="Inference")
        with pbar:
            for batch in test_loader:
                # Move batch to device once
                batch.to(self.model.device)

                # Forward pass
                outputs = self.model(**batch.tensors)
                batch_probs = torch.softmax(outputs.logits, dim=-1)

                batch_n_padding = (batch.attention_mask.shape[1] - batch.attention_mask.sum(dim=1)).cpu()
                for idx, attn_src_ids, attn_tks, ans_src_ids, ans_tks, probs, n_padding in zip(
                    batch.idx,
                    batch.attn_src_ids,
                    batch.attn_tks,
                    batch.ans_src_ids,
                    batch.ans_tks,
                    batch_probs,
                    batch_n_padding,
                ):
                    if not attn_src_ids:
                        logger.warning(f"Empty `attn_src_ids` for idx {idx}")

                    if idx not in result_dict:
                        result_dict[idx] = {"ans_prob": list(), "seq_prob": list()}

                    try:
                        ans_src_ids = torch.as_tensor(ans_src_ids) + n_padding
                        attn_src_ids = torch.as_tensor(attn_src_ids) + n_padding

                        ans_prob = (probs[ans_src_ids, ans_tks].clamp_min(1e-12).log().sum()).exp()
                        seq_prob = (probs[attn_src_ids, attn_tks].clamp_min(1e-12).log().sum()).exp()

                        result_dict[idx]["ans_prob"].append(ans_prob.item())
                        result_dict[idx]["seq_prob"].append(seq_prob.item())

                    except Exception as e:
                        logger.error(f"Error processing idx {idx}: {e}")

                pbar.update()

        # Save the results
        logger.info("Saving results")
        sims_params = f"{self.attn_args.top_k_similarity}-{self.attn_args.similarity_threshold}"
        save_json(
            result_dict,
            osp.join(self.data_args.resp_dir, self.attn_args.ensembles_pred_folder, f"{sims_params}.json"),
        )

        logger.info("Done.")
        return None
