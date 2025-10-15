#!/usr/bin/env bash

# "test_data" specifies the test set, e.g., test_chime4/test_real, test_chime4/test_simu

exp_name='relprompt_lrs2'
test_path=(
    '/path/to/dataset/LipHyp-AVSR/LRS2_test_whisper-large_babble-whole_braven-large_coco-chunk50.json'
    '/path/to/dataset/LipHyp-AVSR/LRS2_test_whisper-large_music-whole_braven-large_coco-chunk50.json'
    '/path/to/dataset/LipHyp-AVSR/LRS2_test_whisper-large_noise-whole_braven-large_coco-chunk50.json'
    '/path/to/dataset/LipHyp-AVSR/LRS2_test_whisper-large_speech-whole_braven-large_coco-chunk50.json'

    # '/path/to/dataset/LipHyp-AVSR/LRS2_test_whisper-large_speech-whole_braven-large_coco_snr0.json'
    # '/path/to/dataset/LipHyp-AVSR/LRS2_test_whisper-large_speech-whole_braven-large_hands_snr0.json'
    # '/path/to/dataset/LipHyp-AVSR/LRS2_test_whisper-large_speech-whole_braven-large_pixelate_snr0.json'
    # '/path/to/dataset/LipHyp-AVSR/LRS2_test_whisper-large_speech-whole_braven-large_blur_snr0.json'
)

for path in "${test_path[@]}"; do
    python -m inference.relprompt --test_path "$path" \
        --model_path "./runs/${exp_name}/best_model.pth" \
        --dual_hypotheses \
        --prompts_format RelPrompt \
        --llm_checkpoint "./checkpoints/TinyLlama/TinyLlama-1.1B-Chat-v1.0"
done