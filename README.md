# Two Heads Are Better Than One: Audio-Visual Speech Error Correction with Dual Hypotheses

> This paper introduces a new paradigm for generative error correction (GER) framework in audio-visual speech recognition (AVSR) that reasons over modality-specific evidences directly in the language space. Our framework, **DualHyp**, empowers a large language model (LLM) to compose independent N-best hypotheses from separate automatic speech recognition (ASR) and visual speech recognition (VSR) models. To maximize the effectiveness of DualHyp, we further introduce **RelPrompt**, a noise-aware guidance mechanism that provides modality-grounded prompts to the LLM. RelPrompt offers the temporal reliability of each stream, guiding the model to dynamically switch its focus between ASR and VSR hypotheses for an accurate correction.

This repository contains the implementation of three approaches:

1. **GER (Generative Error Correction)**: Baseline ASR-only error correction model
2. **DualHyp**: Novel dual-stream hypothesis framework that employs separate ASR and VSR models to generate independent hypotheses, allowing the LLM to intelligently compose these dual-stream hypotheses in the language space
3. **RelPrompt**: Noise-aware guidance mechanism with reliability predictors that assess the quality of audio and visual streams, enabling more informed corrections by dynamically weighing ASR and VSR hypotheses

## Dataset Release

We are releasing the **DualHyp dataset**, which comprises ASR and VSR hypotheses generated from **Whisper-large-v3** and **BRAVEn-large** models, respectively. This dataset provides:

- Pre-generated N-best hypotheses from state-of-the-art ASR and VSR models
- Faster training of LLMs by eliminating the need to run inference on audio-visual models during training
- A valuable resource for future research within the DualHyp framework

**Dataset Access**: The DualHyp dataset is available through this [Google Drive link](https://drive.google.com/drive/folders/1lfnsOmek6I_F05tQLSfbyPag-zdNxJec?usp=sharing). Change `/path/to/dataset/` to your data root directory.

## Environment Setup

For environment setup and basic data preprocessing, please refer to the original [LipGER repository](https://github.com/Sreyan88/LipGER) as we follow their setup procedures.

### Requirements
- Python 3.10
- CUDA-compatible GPU

```bash
pip install -r requirements.txt
git submodule update --init --recursive
```

## Quick Start

### 1. Prepare LLM Checkpoint

Download and convert the base language model:

```bash
pip install huggingface_hub
export HF_TOKEN=[your_hf_token]
python scripts/download.py --repo_id TinyLlama/TinyLlama-1.1B-Chat-v1.0 --from_safetensors True
python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

### 2. Prepare LRS2 dataset

Follow [this repo](https://github.com/sungnyun/cav2vec) for data processing (mouth ROI cropping, HDF5 conversion) and detailed corruption instructions for LRS2.


### 3. Create Hypotheses Dataset

Before running any experiments, you need to generate JSON files containing ASR/VSR hypotheses:

#### Generate ASR Hypotheses
```bash
cd data
python make_json_asr.py --config=conf/asr_config.yaml
```

#### Generate VSR Hypotheses  
```bash
cd data
python make_json_vsr.py +vsr_config=conf/vsr_config.yaml
```

**Configuration**: Edit `data/conf/asr_config.yaml` and `data/conf/vsr_config.yaml` to specify:
- Dataset paths (`original_dataset_path`, `cropped_hdf5_path`, etc.)
- Output paths (`output_file_path`)
- Model settings (`model_name`)
- Corruption settings (`noise_type`, `occ_type`, etc.)

The JSON files will contain hypotheses data in this format:
```json
{
    "Dataset": "LRS2",
    "Uid": "unique_id", 
    "Caption": "ground truth transcription",
    "Clean_Wav": "path_to_audio.wav",
    "Mouthroi": "path_to_mouth_roi.hdf5",
    "Video": "path_to_video.mp4",
    "nhyps": ["hypothesis 1", "hypothesis 2", ...],
}
```

**Sample Data**: See the attached data file to find the sample of our generated trian/val/test hypotheses data.

## Running Experiments

All training and inference scripts are located in the `scripts/` directory. Update the data paths in the scripts before running.

### Training
```bash
bash scripts/finetune_ger.sh          # GER (ASR-only Error Correction)
bash scripts/finetune_ger_dual.sh     # DualHyp (Dual-Hypothesis Approach)
bash scripts/finetune_relprompt.sh    # RelPrompt (Reliability-aware Prompting)
```

### Inference
```bash
bash scripts/infer_ger.sh             # GER (ASR-only Error Correction)
bash scripts/infer_ger_dual.sh        # DualHyp (Dual-Hypothesis Approach)
bash scripts/infer_relprompt.sh       # RelPrompt (Reliability-aware Prompting)
```

## Script Configuration

Before running the scripts, update the following paths in each script:

- `train_path`: Path(s) to training JSON files
- `val_path`: Path to validation JSON file  
- `test_path`: Path(s) to test JSON files
- `exp_name`: Experiment name for saving checkpoints and results

Example paths in scripts:
```bash
train_path=(
    "/path/to/dataset/LipHyp-AVSR/LRS2_train_whisper-large_babble_braven-large_coco.json"
    "/path/to/dataset/LipHyp-AVSR/LRS2_train_whisper-large_music_braven-large_hands.json"
)
val_path='/path/to/dataset/LipHyp-AVSR/LRS2_val_whisper-large_babble_braven-large_coco.json'
```

## Model Differences

| Method | Module | Key Features |
|--------|--------|--------------|
| **GER** | `finetune.ger` | Baseline ASR-only error correction using LLM |
| **DualHyp** | `finetune.ger` | Dual-stream framework with separate ASR and VSR hypotheses composition using `--dual_hypotheses` flag |
| **RelPrompt** | `finetune.relprompt` | Noise-aware guidance with reliability predictors for robustness |

## Output Structure

Results are saved under `./runs/{exp_name}/`:
- `best_model.pth`: Best model checkpoint
- `predictions/`: Inference results in JSON format
- Training logs and metrics

## Data Usage and Compliance

All datasets used in this work (including LRS2 and LRS3) and the hypotheses generated from them are used strictly in accordance with their original intended use for research purposes only. Users must:

- Obtain proper licenses for LRS2 and LRS3 datasets from their respective providers
- Comply with all terms and conditions of the original dataset licenses
- Use the data exclusively for academic research and non-commercial purposes
- Respect any restrictions on data redistribution or sharing as specified in the original licenses

Please ensure you have appropriate permissions before using this codebase with LRS2/LRS3 data.

## Acknowledgments

This work builds upon:
- [LipGER](https://github.com/Sreyan88/LipGER) for the foundational framework
- [CAV2vec](https://github.com/sungnyun/cav2vec) for data preprocessing and corruption protocols 
