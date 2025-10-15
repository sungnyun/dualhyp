#!/usr/bin/env bash

# source activate <your_conda_env>

# "data" specifies the dataset name
# "train_path" specifies the training data path
# "val_path" specifies the valid data path

exp_name='relprompt_lrs2'
train_path=(
    "/path/to/dataset/LipHyp-AVSR/LRS2_train_whisper-large_babble_braven-large_coco.json"
    "/path/to/dataset/LipHyp-AVSR/LRS2_train_whisper-large_music_braven-large_hands.json"
    "/path/to/dataset/LipHyp-AVSR/LRS2_train_whisper-large_noise_braven-large_pixelate.json"
    "/path/to/dataset/LipHyp-AVSR/LRS2_train_whisper-large_speech_braven-large_blur.json"
)
val_path='/path/to/dataset/LipHyp-AVSR/LRS2_val_whisper-large_babble_braven-large_coco.json'

python -m finetune.relprompt --exp_name ${exp_name} \
       --train_path ${train_path[@]} \
       --val_path ${val_path} \
       --dual_hypotheses \
       --prompts_format RelPrompt \
       --micro_batch_size 1 \
       --lr 2e-4 \
       --classifier_lr 1e-4 \
       --num_epochs 5 \
       --llm_checkpoint ./checkpoints/TinyLlama/TinyLlama-1.1B-Chat-v1.0
