#!/bin/bash
source /fast/wangk/virtual_env/multibitsq_env/bin/activate

WORK_DIR="/home/wangk/"
SAVE_DIR="/fast/wangk/"
HF_TOKEN=""
python $WORK_DIR/MultiBitsQ/scripts/download_data_from_hf.py --output_dir $SAVE_DIR/MultiBitsQ/train_data/ --target_count -1 --dataset_name "HuggingFaceFW/fineweb-edu" --subset_name "sample-100BT" --output_filename "finewebedu_train_samples.jsonl" --token $HF_TOKEN
