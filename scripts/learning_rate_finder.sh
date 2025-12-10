torchrun --nnodes=1 --nproc_per_node=1 ParetoQ/train.py \
  --do_train True --lr_find True \
  --lr_find_min_lr 1e-5 --lr_find_max_lr 1e-1 --lr_find_num_iter 200 \
  --train_data_local_path /path/to/train.jsonl \
  --input_model_filename /path/to/model --local_dir /tmp/run --output_dir /tmp/run/out
