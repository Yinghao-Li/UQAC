#!/usr/bin/env bash
set -e

export TOKENIZERS_PARALLELISM="false"
export CUDA_VISIBLE_DEVICES="1"
export PYTHONPATH="."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for dataset_name in "gsm8k" "math" "bbh"; do
for model_name in Llama-3.2-1B-Instruct gemma-2-2b-it Qwen2.5-1.5B-Instruct;  do

home_path="/localscratch/yli3100/LLM-Uncertainty" 

model_path="$home_path/../models/$model_name"
output_dir="./output"
resp_file="results.json"

per_device_eval_batch_size=4

dataset_dir="./datasets/"
disable_dataset_cache=true
dataset_num_mapping_proc=16
dataset_sorted_by_length=true
token_attrs_save_frequency=500

attention_reduction="max"
n_attention_heads=16
backtracking_buffer_size=3
backtracking_threshold=0.5

max_seq_length=1024
max_new_tokens=1024
if [ $dataset_name == "bbh" ]; then
  max_seq_length=1536
  max_new_tokens=1536
fi
disable_seq_length_filter=false

fp16=false
bf16=true
use_4bit_quantization=false
bnb_4bit_compute_dtype="bfloat16"

sims_only=true

report_to="none"

python ./run/main.py \
  --model_name_or_path $model_path \
  --output_dir $output_dir \
  --resp_file $resp_file \
  --dataset_dir $dataset_dir \
  --dataset_name $dataset_name \
  --disable_dataset_cache $disable_dataset_cache \
  --dataset_num_mapping_proc $dataset_num_mapping_proc \
  --dataset_sorted_by_length $dataset_sorted_by_length \
  --attrs_save_frequency $token_attrs_save_frequency \
  --disable_seq_length_filter $disable_seq_length_filter \
  --attention_reduction $attention_reduction \
  --n_attention_heads $n_attention_heads \
  --backtracking_buffer_size $backtracking_buffer_size \
  --backtracking_threshold $backtracking_threshold \
  --max_seq_length $max_seq_length \
  --max_new_tokens $max_new_tokens \
  --per_device_eval_batch_size $per_device_eval_batch_size \
  --fp16 $fp16 \
  --bf16 $bf16 \
  --use_4bit_quantization $use_4bit_quantization \
  --bnb_4bit_compute_dtype $bnb_4bit_compute_dtype \
  --sims_only $sims_only \
  --report_to $report_to \
  --task infer-uq

done
done
