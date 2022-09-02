#! /bin/sh
#SBATCH --job-name=plays_pretrain
#SBATCH --output=./plays_pretrain.out
#SBATCH --error=./plays_pretrain.err
#SBATCH --partition=studentbatch
#SBATCH --time=1-0:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=s-006
#SBATCH --gpus=4
python run_mlm.py \
    --model_name_or_path ./pretrained_again \
    --train_file ../../Data/corpus_pretrain.txt \
    --validation_file ../../Data/corpus_validation.txt \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --overwrite_output_dir \
    --output_dir ./pretrained