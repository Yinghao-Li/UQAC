set -e

export TOKENIZERS_PARALLELISM="false"
export CUDA_VISIBLE_DEVICES="0"
export PYTHONPATH="."

dataset_name=$1

for model_name in Llama-3.2-1B-Instruct Llama-3.2-3B-Instruct Meta-Llama-3.1-8B-Instruct gemma-2-2b-it gemma-2-9b-it Qwen2.5-1.5B-Instruct Qwen2.5-3B-Instruct Qwen2.5-7B-Instruct DeepSeek-R1-Distill-Llama-8B; do
# for model_name in DeepSeek-R1-Distill-Llama-8B; do

model_path="../models/$model_name"
output_dir="./output-sc"

adapter_name=$dataset_name
disable_adapters=true

per_device_eval_batch_size=8

dataset_dir="./datasets/"
disable_dataset_cache=true
dataset_num_mapping_proc=8
dataset_sorted_by_length=true

report_to="none"

python ./run/eval-sc.py \
  --model_name_or_path $model_path \
  --disable_adapters $disable_adapters \
  --output_dir $output_dir \
  --dataset_dir $dataset_dir \
  --dataset_name $dataset_name \
  --report_to $report_to \

done