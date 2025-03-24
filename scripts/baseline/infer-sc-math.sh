set -e

export CUDA_VISIBLE_DEVICES="2"
export PYTHONPATH="."

dataset_name="math"

for model_name in Llama-3.2-1B-Instruct Llama-3.2-3B-Instruct Meta-Llama-3.1-8B-Instruct gemma-2-2b-it gemma-2-9b-it Qwen2.5-1.5B-Instruct Qwen2.5-3B-Instruct Qwen2.5-7B-Instruct DeepSeek-R1-Distill-Llama-8B; do
for seed in 0 1 2 3 4; do

model_path="../models/$model_name"
output_dir="./output-sc"
pred_file="results-$seed.json"

adapter_name=$dataset_name

disable_adapters=true

dataset_dir="./datasets/"
disable_dataset_cache=true
dataset_num_mapping_proc=8

max_seq_length=2048
max_new_tokens=1024
if [ "$model_name" = "DeepSeek-R1-Distill-Llama-8B" ]; then
  max_new_tokens=1536
fi

disable_seq_length_filter=false
inference_temperature=0.3

fp16=false
bf16=true
use_4bit_quantization=false
bnb_4bit_compute_dtype="bfloat16"

report_to="none"

python ./run/main.py \
  --model_name_or_path $model_path \
  --adapter_name $adapter_name \
  --disable_adapters $disable_adapters \
  --output_dir $output_dir \
  --pred_file $pred_file \
  --dataset_dir $dataset_dir \
  --dataset_name $dataset_name \
  --disable_dataset_cache $disable_dataset_cache \
  --dataset_num_mapping_proc $dataset_num_mapping_proc \
  --disable_seq_length_filter $disable_seq_length_filter \
  --max_seq_length $max_seq_length \
  --max_new_tokens $max_new_tokens \
  --inference_temperature $inference_temperature \
  --fp16 $fp16 \
  --bf16 $bf16 \
  --use_4bit_quantization $use_4bit_quantization \
  --bnb_4bit_compute_dtype $bnb_4bit_compute_dtype \
  --report_to $report_to \
  --seed $seed \
  --logprobs 0 \
  --task infer-vllm

done
done
