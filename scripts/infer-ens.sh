set -e

export TOKENIZERS_PARALLELISM="false"
export CUDA_VISIBLE_DEVICES="3"
export PYTHONPATH="."

dataset_name="math"

for model_name in DeepSeek-R1-Distill-Llama-8B; do

top_k_similarity=10
similarity_threshold=0

home_path="." 
model_path="$home_path/../models/$model_name"
output_dir="./output"

adapter_name=$dataset_name
disable_adapters=true

per_device_eval_batch_size=6

dataset_dir="./datasets/"
disable_dataset_cache=true
dataset_num_mapping_proc=8
dataset_sorted_by_length=true

max_seq_length=1536
max_new_tokens=1536
disable_seq_length_filter=false

fp16=false
bf16=true
use_4bit_quantization=false
bnb_4bit_compute_dtype="bfloat16"

report_to="none"

python ./run/main.py \
  --model_name_or_path $model_path \
  --output_dir $output_dir \
  --dataset_dir $dataset_dir \
  --dataset_name $dataset_name \
  --disable_dataset_cache $disable_dataset_cache \
  --dataset_num_mapping_proc $dataset_num_mapping_proc \
  --dataset_sorted_by_length $dataset_sorted_by_length \
  --max_seq_length $max_seq_length \
  --max_new_tokens $max_new_tokens \
  --disable_seq_length_filter $disable_seq_length_filter \
  --per_device_eval_batch_size $per_device_eval_batch_size \
  --top_k_similarity $top_k_similarity \
  --similarity_threshold $similarity_threshold \
  --fp16 $fp16 \
  --bf16 $bf16 \
  --use_4bit_quantization $use_4bit_quantization \
  --bnb_4bit_compute_dtype $bnb_4bit_compute_dtype \
  --report_to $report_to \
  --task infer-ens

done