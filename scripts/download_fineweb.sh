#!/bin/bash
source /fast/wangk/virtual_env/multibitsq_env/bin/activate

WORK_DIR="/home/wangk/"
SAVE_DIR="/fast/wangk/"

python $WORK_DIR/MultiBitsQ/scripts/download_data.py --output_dir $SAVE_DIR/MultiBitsQ/train_data/ --target_count -1
