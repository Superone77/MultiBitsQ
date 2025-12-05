HF_TOKEN=<>
WORK_DIR="/home/wangk/"
SAVE_DIR="/fast/wangk/"

mkdir -p $WORK_DIR/tmp_model/
python $WORK_DIR/MultiBitsQ/scripts/download_hfmodel.py --output_dir $WORK_DIR/tmp_model/ --models facebook/MobileLLM-ParetoQ-125M-BF16 --token $HF_TOKEN
python $WORK_DIR/MultiBitsQ/scripts/download_model.py --output_dir $WORK_DIR/tmp_model/ --models LLM-Research/Llama-3.2-1B-BF16 --token $HF_TOKEN


mv $WORK_DIR/tmp_model/ $SAVE_DIR/MultiBitsQ/model/