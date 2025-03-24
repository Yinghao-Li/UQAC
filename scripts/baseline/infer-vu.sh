set -e

export CUDA_VISIBLE_DEVICES="1"
export PYTHONPATH="."

for dataset_name in bbh; do
for model_name in DeepSeek-R1-Distill-Llama-8B; do

home_path="/localscratch/yli3100/LLM-Uncertainty" 

model_path="$home_path/../models/$model_name"
output_dir="./output"
resp_file="results.json"

adapter_name=$dataset_name

disable_adapters=true

dataset_dir="./datasets/"
disable_dataset_cache=true
dataset_num_mapping_proc=8

max_seq_length=3000
max_new_tokens=1024
disable_seq_length_filter=false

fp16=false
bf16=true
use_4bit_quantization=false
bnb_4bit_compute_dtype="bfloat16"

report_to="none"

python ./run/main.py \
  --model_name_or_path $model_path \
  --output_dir $output_dir \
  --resp_file $resp_file \
  --dataset_dir $dataset_dir \
  --dataset_name $dataset_name \
  --disable_dataset_cache $disable_dataset_cache \
  --dataset_num_mapping_proc $dataset_num_mapping_proc \
  --disable_seq_length_filter $disable_seq_length_filter \
  --max_seq_length $max_seq_length \
  --max_new_tokens $max_new_tokens \
  --fp16 $fp16 \
  --bf16 $bf16 \
  --use_4bit_quantization $use_4bit_quantization \
  --bnb_4bit_compute_dtype $bnb_4bit_compute_dtype \
  --report_to $report_to \
  --task infer-vu

done
done
