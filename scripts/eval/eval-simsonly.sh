set -e

export TOKENIZERS_PARALLELISM="false"
export CUDA_VISIBLE_DEVICES="0"
export PYTHONPATH="."

for dataset_name in gsm8k math bbh; do
for model_name in Llama-3.2-1B-Instruct gemma-2-2b-it Qwen2.5-1.5B-Instruct; do

for top_k_similarity in 10; do
for similarity_threshold in 0; do

home_path="." 
model_path="$home_path/../models/$model_name"

output_dir="./output"

per_device_eval_batch_size=8

dataset_dir="./datasets/"
disable_dataset_cache=true
dataset_num_mapping_proc=8
dataset_sorted_by_length=true

sims_only=true

report_to="none"

python ./run/eval/eval-simsonly.py \
  --model_name_or_path $model_path \
  --output_dir $output_dir \
  --dataset_dir $dataset_dir \
  --dataset_name $dataset_name \
  --top_k_similarity $top_k_similarity \
  --similarity_threshold $similarity_threshold \
  --sims_only $sims_only \
  --report_to $report_to \

done
done
done
done