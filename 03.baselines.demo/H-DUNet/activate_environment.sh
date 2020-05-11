conda deactivate
module purge
module load machine_learning/2019.01-cudnn7.6-cuda10.0-py3.6
unset PYTHONPATH
conda activate H-DenseUNet
