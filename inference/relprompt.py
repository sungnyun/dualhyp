import os
import sys
import time
import warnings
import json
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Literal, Optional
from evaluate import load

import torch
import lightning as L
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from lightning.fabric.strategies import FSDPStrategy

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

# support running without installing as a package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/raven")))

from ger import Tokenizer
from ger.relprompt import GPT, Block, Config
from ger.utils import check_valid_checkpoint_dir, get_default_supported_precision, lazy_load, quantization
from generate.relprompt import generate
from data.av_dataset import AVDataset, DualHypothesesMaskAVDataset

from finetune.relprompt import init_whisper, init_raven

def result(adapter_path, model, audio_encoder, visual_encoder, tokenizer, args):
    # LOADING CORRESPOINDG ADAPTER MODEL
    with lazy_load(adapter_path) as checkpoint:
        x,y = model.load_state_dict(checkpoint['model'], strict=False)
        print('Missing:', x)
        print('Unexpected:', y)

    # Using final checkpoint for loading noise classifiers
    final_adapter_path = adapter_path.replace('best_model', 'lit_model_adapter_finetuned')

    with lazy_load(final_adapter_path) as checkpoint:
        load_dict = {}
        for k, v in checkpoint['model'].items():
            if '_noise_classifier' in k:
                load_dict[k] = v
        x,y = model.load_state_dict(load_dict, strict=False)
        print('Loading noise classifiers...')
        # print('Missing:', x)
        print('Unexpected:', y)

    # Dataset loading
    if args.dual_hypotheses:
        dataset_class = DualHypothesesMaskAVDataset
    else:
        raise NotImplementedError
    test_dataset = dataset_class(split='test', 
                              json_path=args.data_path,
                              max_nhyps=args.max_nhyps,
                              tokenizer=tokenizer,
                              audio_mel=True,
                              audio_pad=False,
                              audio_corruption_enabled=(not args.audio_corruption_disabled),
                              visual_corruption_enabled=(not args.visual_corruption_disabled),
                              prompts_format=args.prompts_format,
                              leave_masks=True,  # Keep masks in the prompt for inference
                              mask_threshold=args.mask_threshold,
                              time_window=args.time_window,
                              )
    test_dataloader = DataLoader(test_dataset, batch_size=1, collate_fn=test_dataset.collate_fn, shuffle=False, num_workers=4, pin_memory=True)

    c = 0
    return_dict = {}
    pr = []
    gt = []
    to_json = []

    # For mask classification metrics
    all_audio_predictions = []
    all_audio_targets = []
    all_visual_predictions = []
    all_visual_targets = []

    audio_encoder.eval()
    visual_encoder.eval()
    encoder_device = next(audio_encoder.parameters()).device

    # Add the token map for converting from indices to special tokens
    idx_to_mask_token = {0: "<<C>>", 1: "<<M>>", 2: "<<N>>"}

    for datapoint in tqdm(test_dataloader):
        encoded = datapoint['input_ids_no_response'][0].to(model.device)
        ground_truth = datapoint['ground_truth'][0]
        audio_features = datapoint['audio'].to(encoder_device)
        visual_features = datapoint['video'].to(encoder_device)
        audio_bin_labels = datapoint['audio_bin_labels'][0]
        visual_bin_labels = datapoint['video_bin_labels'][0]

        with torch.no_grad():
            # Get encoder features
            audio_enc_features = audio_encoder(audio_features) # (B, n_mel, 4*T) -> (B, 2*T, whisper_dim) (100fps -> 50fps)
            visual_enc_features, _ = visual_encoder(visual_features.squeeze(1), None) # (B, C, T, H, W) -> (B, T, raven_dim) (25fps)

            # Convert encoder features to match model precision
            model_dtype = next(model.parameters()).dtype
            audio_enc_features = audio_enc_features.to(dtype=model_dtype, device=model.device)
            visual_enc_features = visual_enc_features.to(dtype=model_dtype, device=model.device)

            # Get predicted noise masks - these are already at chunk level due to pooling in classifiers
            audio_mask_logits = model.audio_noise_classifier(audio_enc_features)  # (B, T_audio//20, 3)
            visual_mask_logits = model.visual_noise_classifier(visual_enc_features)  # (B, T_visual//10, 3)
            
            # Get mask class predictions
            audio_mask_pred = torch.argmax(audio_mask_logits, dim=-1)[0]  # (T_audio//20,)
            visual_mask_pred = torch.argmax(visual_mask_logits, dim=-1)[0]  # (T_visual//10,)
            
            # Convert ground truth labels to indices for metrics calculation
            def labels_to_indices(labels_list, prefix=""):
                # Map special tokens to indices: <<C>> -> 0, <<M>> -> 1, <<N>> -> 2
                idx = [0 if l == f"<<{prefix}C>>" else (1 if l == f"<<{prefix}M>>" else 2) for l in labels_list]
                return torch.tensor(idx)
            
            audio_mask_targets = labels_to_indices(audio_bin_labels)  # (T_chunk)
            visual_mask_targets = labels_to_indices(visual_bin_labels)  # (T_chunk)
            
            # Ensure prediction and target lengths match
            min_audio_len = min(len(audio_mask_pred), len(audio_mask_targets))
            min_visual_len = min(len(visual_mask_pred), len(visual_mask_targets))
            
            audio_mask_pred_trimmed = audio_mask_pred[:min_audio_len]
            audio_mask_targets_trimmed = audio_mask_targets[:min_audio_len]
            visual_mask_pred_trimmed = visual_mask_pred[:min_visual_len]
            visual_mask_targets_trimmed = visual_mask_targets[:min_visual_len]
            
            # Collect predictions and targets for overall metrics
            all_audio_predictions.append(audio_mask_pred_trimmed.cpu())
            all_audio_targets.append(audio_mask_targets_trimmed.cpu())
            all_visual_predictions.append(visual_mask_pred_trimmed.cpu())
            all_visual_targets.append(visual_mask_targets_trimmed.cpu())
            
            # Convert to token strings
            audio_mask_tokens = [idx_to_mask_token[idx.item()] for idx in audio_mask_pred]
            visual_mask_tokens = [idx_to_mask_token[idx.item()] for idx in visual_mask_pred]
            
            # Replace mask placeholders with predicted masks
            audio_mask_str = ''.join(audio_mask_tokens)
            visual_mask_str = ''.join(visual_mask_tokens)
            
            # Replace placeholders in the encoded prompt
            prompt_with_masks = datapoint['input_no_response'][0].replace('<<<ASR_MASKS>>>', audio_mask_str).replace('<<<VSR_MASKS>>>', visual_mask_str)
            
            # Re-encode with the predicted masks
            encoded = tokenizer.encode(prompt_with_masks)
            encoded = torch.tensor(encoded, device=model.device)

        max_returned_tokens = encoded.size(0) + 150

        y = generate(
            model=model,
            audio_features=audio_features,
            visual_features=visual_features,
            idx=encoded,
            audio_encoder=audio_encoder,
            visual_encoder=visual_encoder,
            max_returned_tokens=max_returned_tokens,
            max_seq_length=max_returned_tokens,
            temperature=0.2,
            top_k=1,
            eos_id=tokenizer.eos_token_id
        )

        model.reset_cache()
        output = tokenizer.decode(y)

        inf = output[len(tokenizer.decode(encoded)):].split('\n')[0].strip()
        ref = ground_truth.strip()
        if inf == ref:
            c = c + 1
        pr.append(inf)
        gt.append(ref)
        # print(f'REF: {ref}')
        # print(f'INF: {inf}\n')
        to_json.append({'inference': inf, 'ground_truth': ref})

    print(f'\nFor {adapter_path}')
    noise_types = ['babble-whole', 'babble-chunk', 'noise-whole', 'noise-chunk', 'music-whole', 'music-chunk', 'speech-whole', 'speech-chunk']
    noise_type = next((nt for nt in noise_types if nt in args.data_path), "unknown")
    print(f"Noise type: {noise_type}")
    
    wer_ = wer.compute(predictions=pr, references=gt)
    print(f'WER is {wer_}')
    return_dict['WER'] = wer_
    print(f'Ground truth matches is {c}/{len(test_dataloader)}')
    to_json.append({'wer': wer_, 'gtms': f'{c}/{len(test_dataloader)}'})
    return_dict['gtms'] = c / len(test_dataloader)

    print('the post string normalization wer is')
    x = 0
    for i in range(len(pr)):
        pr[i] = pr[i].lower().replace('.', '').replace(',', '').replace('-', '').replace('?', '').replace("'", '')
        gt[i] = gt[i].lower().replace('.', '').replace(',', '').replace('-', '').replace('?', '').replace("'", '')
        if pr[i] == gt[i]:
            x = x + 1
    post_wer = wer.compute(predictions=pr, references=gt)
    print('WER', post_wer)
    return_dict['post_ST_wer'] = post_wer
    print(x, '/', len(pr))
    return_dict['post_gtms'] = x / len(pr)
    to_json.append({'post_wer': post_wer, 'post_gtms': x / len(pr)})
    print('*********************')

    # Calculate mask classification metrics
    if all_audio_predictions and all_visual_predictions:
        # Concatenate all predictions and targets
        all_preds = torch.cat(all_audio_predictions + all_visual_predictions)
        all_targs = torch.cat(all_audio_targets + all_visual_targets)
        
        # Overall accuracy
        overall_acc = (all_preds == all_targs).float().mean().item()
        
        # Precision and Recall for noise detection (classes 1,2 vs 0)
        # Convert to binary: 0 = clean, 1 = noise (M or N)
        pred_binary = (all_preds > 0).long()  # 1 if pred is M or N, 0 if C
        targ_binary = (all_targs > 0).long()  # 1 if target is M or N, 0 if C
        
        # Calculate precision and recall
        tp = ((pred_binary == 1) & (targ_binary == 1)).sum().item()  # True Positive: predicted noise, actually noise
        fp = ((pred_binary == 1) & (targ_binary == 0)).sum().item()  # False Positive: predicted noise, actually clean
        fn = ((pred_binary == 0) & (targ_binary == 1)).sum().item()  # False Negative: predicted clean, actually noise
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        print(f'Mask Classification Metrics:')
        print(f'  Accuracy: {overall_acc:.4f}')
        print(f'  Precision: {precision:.4f}')
        print(f'  Recall: {recall:.4f}')
        print(f'  F1: {f1:.4f}')
        
        # Add to return dict and JSON
        return_dict['mask_acc'] = overall_acc
        return_dict['mask_precision'] = precision
        return_dict['mask_recall'] = recall
        return_dict['mask_f1'] = f1
        to_json.append({
            'mask_acc': overall_acc,
            'mask_precision': precision,
            'mask_recall': recall,
            'mask_f1': f1
        })
    else:
        print('No mask predictions available for metrics calculation')
    print('*********************')

    os.system(f'mkdir -p {args.predict_dir}')
    with open(os.path.join(args.predict_dir, adapter_path.split('/')[-1].replace('.pth', '.json')), 'w') as f:
        f.write(json.dumps(to_json, indent=4, ensure_ascii=False))
    print('Results in ', os.path.join(args.predict_dir, adapter_path.split('/')[-2] + '.json'), '\n')
    return return_dict


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--llm_checkpoint', type=str, default='checkpoints/TinyLlama/TinyLlama-1.1B-Chat-v1.0')
    parser.add_argument('--dual_hypotheses', action='store_true', help='Whether to use dual hypotheses (default: False)')
    parser.add_argument('--max_nhyps', type=int, default=None, help='max number of hypotheses to use for training')
    parser.add_argument('--d', type=int, default=1, help='lNo of GPUs (default: 1)')
    parser.add_argument('--audio_corruption_disabled', action='store_true', help='Whether to use audio corruption (default: False)')
    parser.add_argument('--visual_corruption_disabled', action='store_true', help='Whether to use visual corruption (default: False)')
    parser.add_argument('--seed', type=int, default=1337)

    parser.add_argument('--prompts_format', type=str, default='DualMask')
    parser.add_argument('--mask_threshold', type=int, default=None, help='SNR threshold for mask reliability')
    parser.add_argument('--time_window', type=float, default=0.4, help='Time segment duration in seconds')
    parser.add_argument('--pool_size', type=int, default=10, help='Pooling size for noise classifiers')

    # TODO: align with finetuning options - how?
    parser.add_argument('--lora_r', type=int, default=16)  # for Llama-7b, 16?
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument('--lora_query', type=bool, default=True)
    parser.add_argument('--lora_key', type=bool, default=True)
    parser.add_argument('--lora_value', type=bool, default=True)
    parser.add_argument('--lora_projection', type=bool, default=True)
    parser.add_argument('--lora_mlp', type=bool, default=False)
    parser.add_argument('--lora_head', type=bool, default=False)
    
    args = parser.parse_args()

    precision = None
    quantize = None
    strategy: str = "auto"
    torch.set_float32_matmul_precision("high")

    precision = precision or get_default_supported_precision(training=False)
    fabric = L.Fabric(devices=args.d, precision=precision, strategy=strategy)
    fabric.launch()
    args.dtype = torch.bfloat16 if fabric.device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32

    fabric.seed_everything(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Load model and tokenizer
    checkpoint_dir = Path(args.llm_checkpoint) # LLM checkpoint direcotry (before finetuning)
    check_valid_checkpoint_dir(checkpoint_dir)
    
    # Initialize encoders to get dimensions
    audio_encoder, (_, whisper_dim) = init_whisper('large')
    visual_encoder, raven_dim = init_raven()
    
    config = Config.from_name(
        name=checkpoint_dir.name,
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        to_query=args.lora_query,
        to_key=args.lora_key,
        to_value=args.lora_value,
        to_projection=args.lora_projection,
        to_mlp=args.lora_mlp,
        to_head=args.lora_head,
        whisper_dim=whisper_dim,
        raven_dim=raven_dim,
        pool_size=args.pool_size,
        )
    if 'llama-3' in config.name.lower():
        config.block_size = 4096
    with fabric.init_module(empty_init=False):
        model = GPT(config)

    hf_tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True)
    hf_tokenizer.add_special_tokens({"additional_special_tokens": ['<<C>>', '<<M>>', '<<N>>']}) # add special tokens for AVDataset
    model.resize_token_embeddings(new_vocab_size=3)

    if 'phi-' in config.name.lower():
        hf_tokenizer.eos_token = "<|endoftext|>"
        
    if hf_tokenizer.pad_token is None:
        hf_tokenizer.pad_token = hf_tokenizer.eos_token

    model.eval()
    model = fabric.setup(model)

    # Load dataset
    args.data_path = args.test_path
    args.predict_dir = Path(args.model_path).parent / 'predictions'

    # Load WER metric
    wer = load("wer")

    # Inference
    result_dict = result(args.model_path, model, audio_encoder, visual_encoder, hf_tokenizer, args)
    wer_percent = result_dict['WER'] * 100
    wer_percent_post = result_dict['post_ST_wer'] * 100

    gt_percent = result_dict['gtms'] * 100
    gt_percent_post = result_dict['post_gtms'] * 100

    print('Model: ', args.model_path, 'WER: ', wer_percent, "WER_post: ", wer_percent_post, "GTM: ", gt_percent, "GTM_post: ", gt_percent_post)