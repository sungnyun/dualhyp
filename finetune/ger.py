import os
import sys
import time
import math
import numpy as np
import logging
import json
import argparse
import itertools
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from tqdm import tqdm
from collections import OrderedDict

import torch
import lightning as L
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from lightning.fabric.strategies import FSDPStrategy, XLAStrategy, DDPStrategy

# support running without installing as a package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# from generate.base import generate
from ger.lora import GPT, Block, Config, lora_filter, mark_only_lora_as_trainable
from ger.speed_monitor import SpeedMonitorFabric as SpeedMonitor
from ger.utils import (
    check_valid_checkpoint_dir,
    chunked_cross_entropy,
    get_default_supported_precision,
    lazy_load,
    num_parameters,
    step_csv_logger,
)
from data.av_dataset import AVDataset, DualHypothesesAVDataset, DualHypothesesAlignAVDataset, DualHypothesesAlign2AVDataset
# from scripts.prepare_alpaca import generate_prompt


def setup_logger(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ]
    )

def setup(
    args,
    checkpoint_dir: Path = None,
    out_dir: Path = None,
    precision: Optional[str] = None,
    tpu: bool = False,
):
    precision = precision or get_default_supported_precision(training=True, tpu=tpu)
    print('precision: ', precision)

    fabric_devices = args.d
    if fabric_devices > 1:
        if tpu:
            # For multi-host TPU training, the device count for Fabric is limited to the count on a single host.
            fabric_devices = "auto"
            strategy = XLAStrategy(sync_module_states=False)
        else:
            strategy = FSDPStrategy(
                auto_wrap_policy={Block},
                activation_checkpointing_policy={Block},
                state_dict_type="full",
                limit_all_gathers=True,
                cpu_offload=False
            )
    else:
        strategy = DDPStrategy(find_unused_parameters=True)

    logger = step_csv_logger(out_dir.parent, out_dir.name, flush_logs_every_n_steps=args.log_interval)
    fabric = L.Fabric(devices=fabric_devices, accelerator='cuda', strategy=strategy, precision=precision, loggers=logger)
    # fabric.print(hparams)
    log_path = out_dir/"train.log"
    setup_logger(log_path)
    logging.info(f"CLI arguments: {args.__dict__}")
    fabric.launch(main, args, checkpoint_dir, out_dir)


def main(fabric: L.Fabric, args, checkpoint_dir: Path, out_dir: Path):
    check_valid_checkpoint_dir(checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    speed_monitor = SpeedMonitor(fabric, window_size=50, time_unit="seconds")

    fabric.seed_everything(1337)  # same seed for every process to init model (FSDP)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    if not any((args.lora_query, args.lora_key, args.lora_value, args.lora_projection, args.lora_mlp, args.lora_head)):
        fabric.print("Warning: all LoRA layers are disabled!")
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
    checkpoint_path = checkpoint_dir / "lit_model.pth"
    logging.info(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}")
    with fabric.init_module(empty_init=False):
        model = GPT(config)
        print(model)

    if 'phi-' in config.name.lower():
        tokenizer.eos_token = "<|endoftext|>"
        
    with lazy_load(checkpoint_path) as checkpoint:
        # strict=False because missing keys due to adapter weights not contained in state dict
        x, y = model.load_state_dict(checkpoint, strict=False)

    mark_only_lora_as_trainable(model)

    logging.info(f"Number of trainable parameters: {num_parameters(model, requires_grad=True):,}")
    logging.info(f"Number of non trainable parameters: {num_parameters(model, requires_grad=False):,}")
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    model, optimizer = fabric.setup(model, optimizer)

    fabric.seed_everything(1337 + fabric.global_rank)

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
    train_dataset = dataset_class(split='train', 
                              json_path=args.train_path,
                              nhyps_key=args.nhyps_key,
                              max_input_length=args.max_input_length,
                              max_nhyps=args.max_nhyps, 
                              tokenizer=tokenizer,
                              audio_corruption_enabled=(not args.audio_corruption_disabled),
                              visual_corruption_enabled=(not args.visual_corruption_disabled),
                              prompts_format=args.prompts_format,
                              apply_chat_template=args.apply_chat_template,
                              language=args.language,
                              )
    val_dataset = dataset_class(split='val', 
                            json_path=args.val_path,
                            nhyps_key=args.nhyps_key, 
                            max_input_length=args.max_input_length, 
                            max_nhyps=args.max_nhyps,
                            tokenizer=tokenizer,
                            audio_corruption_enabled=(not args.audio_corruption_disabled),
                            visual_corruption_enabled=(not args.visual_corruption_disabled),
                            prompts_format=args.prompts_format,
                            apply_chat_template=args.apply_chat_template,
                            language=args.language,
                            )
    train_dataloader = DataLoader(train_dataset, batch_size=args.micro_batch_size, collate_fn=train_dataset.collate_fn, shuffle=True, num_workers=4, pin_memory=False)
    val_dataloader = DataLoader(val_dataset, batch_size=args.micro_batch_size, collate_fn=val_dataset.collate_fn, shuffle=False, num_workers=4, pin_memory=False)

    train_data_len = len(train_dataset)
    epoch_size = train_data_len // args.micro_batch_size  # train dataset size
    max_iters = args.num_epochs * epoch_size // args.d
    val_data_len = len(val_dataset)
    eval_iters = val_data_len // args.micro_batch_size // args.d
    warmup_steps = int(epoch_size * args.wp) // args.d
    warmup_steps_m = int(epoch_size * args.wp) // args.d
    train_cfg = {
        'train_data_len': train_data_len,
        'num_epochs': args.num_epochs,
        'epoch_size': epoch_size, 
        'max_iters': max_iters, 
        'eval_iters': eval_iters, 
        'warmup_steps': warmup_steps, 
        'warmup_steps_m': warmup_steps_m,
        'gradient_accumulation_iters': args.gradient_accumulation_iters,
        'learning_rate': args.learning_rate,
        'use_cosine_scheduler': args.use_cosine_scheduler,
        'min_lr_ratio': args.min_lr_ratio,
        'log_interval': args.log_interval,
        'save_interval': args.save_interval,
    }
    if fabric.global_rank == 0: logging.info(f"Training configs: {train_cfg}")

    train_time = time.perf_counter()
    train(fabric, model, optimizer, train_dataloader, val_dataloader, train_cfg, checkpoint_dir, out_dir, speed_monitor)
    if fabric.global_rank == 0:
        logging.info(f"Total training time: {(time.perf_counter()-train_time):.2f}s")
        if fabric.device.type == "cuda":
            logging.info(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")

    # Save the final checkpoint at the end of training
    save_path = out_dir / "lit_model_lora_finetuned.pth"
    save_lora_checkpoint(fabric, model, save_path)


def train(
    fabric: L.Fabric,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    train_cfg: Dict,
    checkpoint_dir: Path,
    out_dir: Path,
    speed_monitor: SpeedMonitor,
) -> None:
    # Create an infinite iterator using cycle.

    # deprecated: max_seq_length set to max_input_length as default
    # max_seq_length, longest_seq_length, longest_seq_ix = train_dataloader.dataset.get_max_seq_length()
    # max_seq_length = min(max_seq_length, max_input_length)
    # longest_seq_length = min(longest_seq_length, max_input_length)

    # sanity check
    # validate(fabric, model, val_data, tokenizer, longest_seq_length)

    gradient_accumulation_iters = train_cfg["gradient_accumulation_iters"]
    log_interval = train_cfg["log_interval"]
    save_interval = train_cfg["save_interval"]

    cumulative_loss = 0.0
    cumulative_loss_interval = 0.0
    best_val_loss = 99999
    log_start_time = time.time()

    if fabric.device.type == "xla":
        import torch_xla.core.xla_model as xm
        xm.mark_step()

    real_iter = 0
    micro_step = 0
    for epoch in range(train_cfg["num_epochs"]):
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}:", dynamic_ncols=True):

            input_ids, targets = batch["input_ids"], batch["labels"]
            input_ids, targets = fabric.to_device((input_ids, targets))

            # Learning rate scheduling
            if real_iter <= train_cfg["warmup_steps"]:
                # Warmup phase: linear increase from 0 to max_lr
                lr = train_cfg["learning_rate"] * real_iter / train_cfg["warmup_steps"]
            elif train_cfg["use_cosine_scheduler"]:
                # Cosine annealing after warmup
                progress = (real_iter - train_cfg["warmup_steps"]) / (train_cfg["max_iters"] - train_cfg["warmup_steps"])
                progress = min(progress, 1.0)  # Clamp progress to [0, 1]
                min_lr = train_cfg["learning_rate"] * train_cfg["min_lr_ratio"]
                lr = min_lr + (train_cfg["learning_rate"] - min_lr) * (1 + math.cos(math.pi * progress)) / 2
            else:
                # Keep constant learning rate after warmup
                lr = train_cfg["learning_rate"]
            
            # Apply learning rate to optimizer
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            if micro_step == 0:
                mark_only_lora_as_trainable(model)
                optimizer.zero_grad()

            t0 = time.time()
            with fabric.no_backward_sync(model, enabled=(micro_step + 1) % gradient_accumulation_iters != 0):
                logits = model(input_ids, lm_head_chunk_size=128)
                # shift the targets such that output n predicts token n+1
                logits[-1] = logits[-1][..., :-1, :]
                loss = chunked_cross_entropy(logits, targets[..., 1:])
                cumulative_loss += loss.item()
                cumulative_loss_interval += loss.item()

                fabric.backward(loss / gradient_accumulation_iters)
                micro_step += 1

            if (micro_step + 1) % gradient_accumulation_iters == 0:
                mark_only_lora_as_trainable(model)
                optimizer.step()
                optimizer.zero_grad()
                micro_step = 0

            if (real_iter + 1) % log_interval == 0 and fabric.global_rank == 0:
                avg_loss = cumulative_loss / log_interval
                elapsed = time.time() - log_start_time
                current_lr = optimizer.param_groups[0]['lr']
                logging.info(f"iter {(real_iter + 1)}: train loss = {avg_loss:.4f}, lr = {current_lr:.2e}, time / {log_interval} iters = {elapsed:.2f}s")
                cumulative_loss = 0.0
                log_start_time = time.time()

            if (real_iter + 1) % save_interval == 0:
                # checkpoint_path = out_dir / f"iter-{real_iter:06d}.pth"
                # save_adapter_checkpoint(fabric, model, checkpoint_path)
                avg_loss = cumulative_loss_interval / save_interval
                logging.info(f"Average train loss = {avg_loss:.4f}")
                cumulative_loss_interval = 0.0

                val_loss = validate(fabric, model, val_dataloader)
                if fabric.global_rank == 0:
                    logging.info(f"iter {(real_iter + 1)}: val loss {val_loss:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint_path = out_dir / "best_model.pth"
                    save_lora_checkpoint(fabric, model, checkpoint_path)
                    logging.info(f"iter {(real_iter + 1)} -> best model saved")
                fabric.barrier()

            real_iter += 1
    
    # after training, validate the model
    val_loss = validate(fabric, model, val_dataloader)
    if fabric.global_rank == 0:
        logging.info(f"iter {(real_iter + 1)}: val loss {val_loss:.4f}")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        checkpoint_path = out_dir / "best_model.pth"
        save_lora_checkpoint(fabric, model, checkpoint_path)
        logging.info(f"iter {(real_iter + 1)} -> best model saved")

@torch.no_grad()
def validate(
    fabric: L.Fabric, model: GPT, val_dataloader: DataLoader,
) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()
    losses = []
    for batch in val_dataloader:
        input_ids, targets = batch["input_ids"], batch["labels"]
        input_ids, targets = fabric.to_device((input_ids, targets))
        valid_tokens = (targets[..., 1:] != -1).sum().item()
        if valid_tokens == 0:
            # logging.warning(f"[Validation] All targets are masked (-1) at step {k}, skipping batch.")
            continue
        logits = model(input_ids)
        loss = chunked_cross_entropy(logits[..., :-1, :], targets[..., 1:], chunk_size=0)
        losses.append(loss.item())
    val_loss = sum(losses) / len(losses)

    torch.cuda.empty_cache()
    model.reset_cache()
    model.train()
    return val_loss


def save_lora_checkpoint(fabric, model, file_path: Path):
    fabric.print(f"Saving LoRA weights to {str(file_path)!r}")
    fabric.save(file_path, {"model": model})


def mark_as_trainable(model):
    for name, param in model.named_parameters():
        param.requires_grad = True


def mark_as_untrainable(model):
    for name, param in model.named_parameters():
        param.requires_grad = False


if __name__ == "__main__":
    # cli setup
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, nargs='+', help='Path to the training data')
    parser.add_argument('--val_path', type=str)
    parser.add_argument('--exp_name', type=str, default='finetune')
    parser.add_argument('--llm_checkpoint', type=str, default='checkpoints/TinyLlama/TinyLlama-1.1B-Chat-v1.0')
    parser.add_argument('--nhyps_key', type=str, default='nhyps_asr')
    parser.add_argument('--dual_hypotheses', action='store_true', help='Whether to use dual hypotheses (default: False)')
    parser.add_argument('--max_nhyps', type=int, default=None, help='max number of hypotheses to use for training')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--micro_batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--weight_decay', type=float, default=0.02)
    parser.add_argument('--d', type=int, default=1, help='lNo of GPUs (default: 1)')
    parser.add_argument('--lamda', type=float, default=0.5)
    parser.add_argument('--wp', type=float, default=0.2, help='warmup proportion of total training steps')
    parser.add_argument('--use_cosine_scheduler', action='store_true', help='Use cosine annealing scheduler after warmup (default: False)')
    parser.add_argument('--min_lr_ratio', type=float, default=0.01, help='Minimum learning rate ratio for cosine scheduler (default: 0.01)')
    parser.add_argument('--log_interval', type=int, default=100, help='log interval')
    parser.add_argument('--save_interval', type=int, default=10000, help='checkpoint save interval')
    parser.add_argument('--audio_corruption_disabled', action='store_true', help='Whether to use audio corruption (default: False)')
    parser.add_argument('--visual_corruption_disabled', action='store_true', help='Whether to use visual corruption (default: False)')
    parser.add_argument('--prompts_format', type=str, default='GER')
    parser.add_argument('--apply_chat_template', action='store_true', help='Whether to use apply_chat_template for tokenization (default: False, but always True for phi-3.5)')
    parser.add_argument('--language', type=str, default=None, help='Language to specify in the prompt, e.g., "German" (default: None)')

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

    # Batch and device stuff
    devices = args.d
    args.batch_size = args.batch_size // devices  # trained atis with 32BS 1 gpu == 64BS with 2 GPUs
    args.gradient_accumulation_iters = args.batch_size // args.micro_batch_size
    args.learning_rate = args.lr
    args.learning_rate_m = args.lr * 0.1
    args.save_interval = args.save_interval // devices
    # change this value to force a maximum sequence length
    args.override_max_seq_length = None
    
    args.max_input_length = 1024  # 800 for v100 wo k,v ; 700 works for v100 w k,v
    if os.path.exists(Path(args.llm_checkpoint) / "tokenizer_config.json"):
        with open(Path(args.llm_checkpoint) / "tokenizer_config.json") as fp:
            tokenizer_config = json.load(fp)
        args.max_input_length = tokenizer_config.get("model_max_length")

    # args.hparams = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}

    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")
    setup(
        args,
        checkpoint_dir=Path(args.llm_checkpoint),
        out_dir=Path(f"./runs/{args.exp_name}"),
        )