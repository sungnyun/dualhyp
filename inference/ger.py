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

from ger import Tokenizer
from ger.lora import GPT, Block, Config
from ger.utils import check_valid_checkpoint_dir, get_default_supported_precision, lazy_load, quantization
from generate.base import generate
from data.av_dataset import AVDataset, DualHypothesesAVDataset, DualHypothesesAlignAVDataset, DualHypothesesAlign2AVDataset


def result(adapter_path, model, tokenizer, args):
    # LOADING CORRESPOINDG ADAPTER MODEL
    with lazy_load(adapter_path) as checkpoint:
        x,y = model.load_state_dict(checkpoint['model'], strict=False)
        print('Missing:', x)
        print('Unexpected:', y)

    # Dataset loading
    if args.dual_hypotheses:
        dataset_class = DualHypothesesAVDataset
        if "Dual" not in args.prompts_format:
            print('Warning: dual hypotheses is enabled, but prompts format is not Dual.')
        elif "DualAlign" in args.prompts_format:
            if args.prompts_format == "DualAlign2":
                dataset_class = DualHypothesesAlign2AVDataset
            else:
                dataset_class = DualHypothesesAlignAVDataset
    else:
        dataset_class = AVDataset
    test_dataset = dataset_class(split='test', 
                             json_path=args.data_path, 
                             nhyps_key=args.nhyps_key,
                             max_nhyps=args.max_nhyps, 
                             tokenizer=tokenizer,
                             audio_corruption_enabled=(not args.audio_corruption_disabled),
                             visual_corruption_enabled=(not args.visual_corruption_disabled),
                             prompts_format=args.prompts_format,
                             apply_chat_template=args.apply_chat_template,
                             language=args.language,
                             )
    test_dataloader = DataLoader(test_dataset, batch_size=1, collate_fn=test_dataset.collate_fn, shuffle=False, num_workers=4, pin_memory=True)

    c = 0
    return_dict = {}
    pr = []
    gt = []
    to_json = []
    for datapoint in tqdm(test_dataloader):
        encoded = datapoint['input_ids_no_response'][0].to(model.device)
        ground_truth = datapoint['ground_truth'][0]

        max_returned_tokens = encoded.size(0) + 150
        # max_returned_tokens = datapoint['input_ids'][0].size(0) + 10

        y = generate(
            model=model,
            idx=encoded,
            max_returned_tokens=max_returned_tokens,
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
    return_dict['adapter_path'] = adapter_path
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
    parser.add_argument('--nhyps_key', type=str, default='nhyps_asr')
    parser.add_argument('--dual_hypotheses', action='store_true', help='Whether to use dual hypotheses (default: False)')
    parser.add_argument('--max_nhyps', type=int, default=None, help='max number of hypotheses to use for training')
    parser.add_argument('--d', type=int, default=1, help='lNo of GPUs (default: 1)')
    parser.add_argument('--audio_corruption_disabled', action='store_true', help='Whether to use audio corruption (default: False)')
    parser.add_argument('--visual_corruption_disabled', action='store_true', help='Whether to use visual corruption (default: False)')
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--prompts_format', type=str, default='GER')
    parser.add_argument('--apply_chat_template', action='store_true', help='Whether to use apply_chat_template for tokenization (default: False, but always True for phi-3.5)')
    parser.add_argument('--language', type=str, default=None, help='Language to specify in the prompt, e.g., "German" (default: None)')

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
    # with open(checkpoint_dir / "lit_config.json") as fp:
    #     config = Config(**json.load(fp))
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
        )
    if 'llama-3' in config.name.lower():
        config.block_size = 4096
    with fabric.init_module(empty_init=False):
        model = GPT(config)
    # tokenizer = Tokenizer(checkpoint_dir)
    hf_tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True)

    if 'phi-' in config.name.lower():
        # tokenizer.eos_token = "<|endoftext|>"
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
    result_dict = result(args.model_path, model, hf_tokenizer, args)
    wer_percent = result_dict['WER'] * 100
    wer_percent_post = result_dict['post_ST_wer'] * 100

    gt_percent = result_dict['gtms'] * 100
    gt_percent_post = result_dict['post_gtms'] * 100

    print('Model: ', args.model_path, 'WER: ', wer_percent, "WER_post: ", wer_percent_post, "GTM: ", gt_percent, "GTM_post: ", gt_percent_post)