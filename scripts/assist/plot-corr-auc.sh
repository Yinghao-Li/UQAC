set -e

export TOKENIZERS_PARALLELISM="false"
export CUDA_VISIBLE_DEVICES="0"
export PYTHONPATH="."

python ./assist/plot-correlation-auc.py \