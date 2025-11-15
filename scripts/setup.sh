source ~/miniforge3/etc/profile.d/conda.sh

conda create -n multibitsq_env python=3.10 -y
conda activate multibitsq_env

pip install -r ParetoQ/requirements.txt