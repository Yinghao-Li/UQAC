set -e

export TOKENIZERS_PARALLELISM="false"
export CUDA_VISIBLE_DEVICES="0"
export PYTHONPATH="."


dataset_name="gsm8k"
model_name="Llama-3.2-1B-Instruct"

for top_k_similarity in 10; do
for similarity_threshold in 0; do

model_path="../models/$model_name"
output_dir="./output"

adapter_name=$dataset_name
disable_adapters=true
token_attrs_folder="key-attrs"

per_device_eval_batch_size=8

dataset_dir="./datasets/"
disable_dataset_cache=true
dataset_num_mapping_proc=8
dataset_sorted_by_length=true

report_to="none"

python ./assist/plot-correlation-acc.py \
  --model_name_or_path $model_path \
  --disable_adapters $disable_adapters \
  --output_dir $output_dir \
  --dataset_dir $dataset_dir \
  --dataset_name $dataset_name \
  --token_attrs_folder $token_attrs_folder \
  --top_k_similarity $top_k_similarity \
  --similarity_threshold $similarity_threshold \
  --report_to $report_to \

done
done
