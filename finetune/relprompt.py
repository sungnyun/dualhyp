import os
import sys
import time
import numpy as np
import logging
import json
import argparse
import itertools
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn.functional as F
import lightning as L
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from lightning.fabric.strategies import FSDPStrategy, XLAStrategy, DDPStrategy

# support running without installing as a package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/raven")))

from ger.relprompt import GPT, Block, Config, mark_only_lora_as_trainable
from ger.speed_monitor import SpeedMonitorFabric as SpeedMonitor
from ger.speed_monitor import estimate_flops, measure_flops
from ger.utils import (
    check_valid_checkpoint_dir,
    chunked_cross_entropy,
    get_default_supported_precision,
    lazy_load,
    num_parameters,
    step_csv_logger,
)
import data.whisper as whisper
from data.raven.finetune_learner import Learner
from data.av_dataset import AVDataset, DualHypothesesMaskAVDataset
from hydra import initialize, compose
# from scripts.prepare_alpaca import generate_prompt

os.environ["TOKENIZERS_PARALLELISM"] = "false"
ASR_VSR_PRJ_DEVICE = "cuda"  # TODO: make this configurable

def setup_logger(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ]
    )

def init_whisper(size):
    if size == "large":
        model = whisper.load_model(size, device="cpu").encoder.to(ASR_VSR_PRJ_DEVICE)
        n_mel, whisper_dim = 128, 1280
    print(f"=> Whisper model loaded")
    return model, (n_mel, whisper_dim)

def init_raven():
    with initialize(config_path="../data/raven/conf/", version_base=None):
        cfg = compose(config_name="config_test")
        if cfg["model"]["pretrained_model_path"].startswith("raven/"):
            cfg["model"]["pretrained_model_path"] = cfg["model"]["pretrained_model_path"].replace("raven/", "data/raven/")
        model = Learner(cfg).model.encoder.to(ASR_VSR_PRJ_DEVICE)
        raven_dim = cfg["model"]["visual_backbone"]["adim"]
    print(f"=> BRAVEn model loaded")
    return model, raven_dim

def labels_to_indices(labels_list, device, prefix=""):
    indices = []
    for labels in labels_list:
        # Map special tokens to indices: <<C>> -> 0, <<M>> -> 1, <<N>> -> 2
        idx = [0 if l == f"<<{prefix}C>>" else (1 if l == f"<<{prefix}M>>" else 2) for l in labels]
        indices.append(torch.tensor(idx, device=device))
    return torch.stack(indices)

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
    tokenizer.add_special_tokens({"additional_special_tokens": ['<<C>>', '<<M>>', '<<N>>']})  # add special tokens for AVDataset
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    speed_monitor = SpeedMonitor(fabric, window_size=50, time_unit="seconds")

    fabric.seed_everything(1337)  # same seed for every process to init model (FSDP)

    audio_encoder, (_, whisper_dim) = init_whisper('large')
    visual_encoder, raven_dim = init_raven()
    args.whisper_dim = whisper_dim
    args.raven_dim = raven_dim
    audio_encoder.eval()
    visual_encoder.eval()

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
        whisper_dim=args.whisper_dim,
        raven_dim=args.raven_dim,
        pool_size=args.pool_size,
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

    model.resize_token_embeddings(new_vocab_size=3)
    mark_only_lora_as_trainable(model)

    logging.info(f"Number of trainable parameters: {num_parameters(model, requires_grad=True):,}")
    logging.info(f"Number of non trainable parameters: {num_parameters(model, requires_grad=False):,}")
    
    # Separate parameters for different learning rates
    noise_classifier_params = []
    llm_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'noise_classifier' in name:
                noise_classifier_params.append(param)
            else:
                llm_params.append(param)
    
    logging.info(f"LLM trainable parameters: {sum(p.numel() for p in llm_params):,}")
    logging.info(f"Noise classifier parameters: {sum(p.numel() for p in noise_classifier_params):,}")
    logging.info(f"LLM learning rate: {args.learning_rate:.2e}")
    logging.info(f"Noise classifier learning rate: {args.classifier_learning_rate:.2e}")
    
    # Create optimizer with different learning rates
    param_groups = [
        {'params': llm_params, 'lr': args.learning_rate, 'weight_decay': args.weight_decay},
        {'params': noise_classifier_params, 'lr': args.classifier_learning_rate, 'weight_decay': args.weight_decay}
    ]
    optimizer = torch.optim.AdamW(param_groups)
    model, optimizer = fabric.setup(model, optimizer)

    fabric.seed_everything(1337 + fabric.global_rank)

    # Dataset loading
    if args.dual_hypotheses:
        dataset_class = DualHypothesesMaskAVDataset
    else:
        raise NotImplementedError
    train_dataset = dataset_class(split='train', 
                            json_path=args.train_path,
                            max_input_length=args.max_input_length,
                            max_nhyps=args.max_nhyps, 
                            tokenizer=tokenizer,
                            audio_mel=True,
                            audio_pad=False,
                            audio_corruption_enabled=(not args.audio_corruption_disabled),
                            visual_corruption_enabled=(not args.visual_corruption_disabled),
                            prompts_format=args.prompts_format,
                            leave_masks=False,
                            mask_threshold=args.mask_threshold,
                            time_window=args.time_window,
                            )
    val_dataset = dataset_class(split='val', 
                            json_path=args.val_path,
                            max_input_length=args.max_input_length, 
                            max_nhyps=args.max_nhyps,
                            tokenizer=tokenizer,
                            audio_mel=True,
                            audio_pad=False,
                            audio_corruption_enabled=(not args.audio_corruption_disabled),
                            visual_corruption_enabled=(not args.visual_corruption_disabled),
                            prompts_format=args.prompts_format,
                            leave_masks=False,
                            mask_threshold=args.mask_threshold,
                            time_window=args.time_window,
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
        'max_input_length': args.max_input_length,
        'gradient_accumulation_iters': args.gradient_accumulation_iters,
        'learning_rate': args.learning_rate,
        'classifier_learning_rate': args.classifier_learning_rate,
        'use_cosine_scheduler': args.use_cosine_scheduler,
        'min_lr_ratio': args.min_lr_ratio,
        'mask_loss_weight': args.mask_loss_weight,
        'log_interval': args.log_interval,
        'save_interval': args.save_interval,
    }
    if fabric.global_rank == 0: logging.info(f"Training configs: {train_cfg}")

    train_time = time.perf_counter()
    train(fabric, model, optimizer, audio_encoder, visual_encoder, train_dataloader, val_dataloader, train_cfg, checkpoint_dir, out_dir, speed_monitor)
    if fabric.global_rank == 0:
        logging.info(f"Total training time: {(time.perf_counter()-train_time):.2f}s")
        if fabric.device.type == "cuda":
            logging.info(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")

    # Save the final checkpoint at the end of training
    save_path = out_dir / "lit_model_adapter_finetuned.pth"
    save_adapter_checkpoint(fabric, model, save_path)


def train(
    fabric: L.Fabric,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    audio_encoder,
    visual_encoder,
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
    cumulative_mask_loss = 0.0
    cumulative_llm_loss = 0.0
    best_val_loss = 99999
    log_start_time = time.time()

    if fabric.device.type == "xla":
        import torch_xla.core.xla_model as xm
        xm.mark_step()

    real_iter = 0
    micro_step = 0
    for epoch in range(train_cfg["num_epochs"]):
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}:", dynamic_ncols=True):

            input_ids, targets, audio_features, visual_features = batch["input_ids"], batch["labels"], batch["audio"], batch["video"]
            input_ids, targets = fabric.to_device((input_ids, targets))

            # Get ground truth masks if available (during training)
            audio_bin_labels = batch.get("audio_bin_labels", None)
            video_bin_labels = batch.get("video_bin_labels", None)

            if real_iter <= train_cfg["warmup_steps"]:
                llm_lr = train_cfg["learning_rate"] * real_iter / train_cfg["warmup_steps"]
                classifier_lr = train_cfg["classifier_learning_rate"] * real_iter / train_cfg["warmup_steps"]
            elif train_cfg["use_cosine_scheduler"]:
                progress = (real_iter - train_cfg["warmup_steps"]) / (train_cfg["max_iters"] - train_cfg["warmup_steps"])
                progress = min(progress, 1.0)  # Clamp progress to [0, 1]
                min_llm_lr = train_cfg["learning_rate"] * train_cfg["min_lr_ratio"]
                min_classifier_lr = train_cfg["classifier_learning_rate"] * train_cfg["min_lr_ratio"]
                llm_lr = min_llm_lr + (train_cfg["learning_rate"] - min_llm_lr) * (1 + math.cos(math.pi * progress)) / 2
                classifier_lr = min_classifier_lr + (train_cfg["classifier_learning_rate"] - min_classifier_lr) * (1 + math.cos(math.pi * progress)) / 2
            else:
                llm_lr = train_cfg["learning_rate"]
                classifier_lr = train_cfg["classifier_learning_rate"]

            # Update learning rates for different parameter groups
            optimizer.param_groups[0]['lr'] = llm_lr  # LLM parameters
            optimizer.param_groups[1]['lr'] = classifier_lr  # Noise classifier parameters

            if micro_step == 0:
                # mark_only_lora_as_trainable(model)
                optimizer.zero_grad()

            with torch.no_grad():
                # Get encoder features
                audio_features, visual_features = audio_features.to(ASR_VSR_PRJ_DEVICE), visual_features.to(ASR_VSR_PRJ_DEVICE)
                audio_enc_features = audio_encoder(audio_features) # (B, n_mel, 4*T) -> (B, 2*T, whisper_dim) (100fps -> 50fps)
                visual_enc_features, _ = visual_encoder(visual_features.squeeze(1), None) # (B, C, T, H, W) -> (B, T, raven_dim) (25fps)

            t0 = time.time()
            with fabric.no_backward_sync(model, enabled=(micro_step + 1) % gradient_accumulation_iters != 0):
                # Get mask predictions from classifiers
                audio_mask_logits = model.audio_noise_classifier(audio_enc_features.to(fabric.device))
                visual_mask_logits = model.visual_noise_classifier(visual_enc_features.to(fabric.device))

                # Convert string labels to indices if ground truth masks are available
                mask_loss = 0.0
                if audio_bin_labels is not None and video_bin_labels is not None:
                    # Convert ground truth labels to target indices
                    audio_mask_targets = labels_to_indices(audio_bin_labels, device=fabric.device, prefix="")  # (B, T_chunk)
                    visual_mask_targets = labels_to_indices(video_bin_labels, device=fabric.device, prefix="")  # (B, T_chunk)
                    
                    # Get predicted logits dimensions - these should now match the target dimensions
                    B, T_audio_pred, _ = audio_mask_logits.shape  # (B, T_audio//20, 3)
                    B, T_visual_pred, _ = visual_mask_logits.shape  # (B, T_visual//10, 3)
                    
                    # Ensure target and prediction lengths match by trimming if necessary
                    audio_target_len = audio_mask_targets.shape[1]
                    visual_target_len = visual_mask_targets.shape[1]
                    
                    if T_audio_pred > audio_target_len:
                        audio_mask_logits = audio_mask_logits[:, :audio_target_len, :]
                    elif T_audio_pred < audio_target_len:
                        audio_mask_targets = audio_mask_targets[:, :T_audio_pred]
                    
                    if T_visual_pred > visual_target_len:
                        visual_mask_logits = visual_mask_logits[:, :visual_target_len, :]
                    elif T_visual_pred < visual_target_len:
                        visual_mask_targets = visual_mask_targets[:, :T_visual_pred]
                        
                    # Calculate mask classification loss
                    audio_mask_loss = F.cross_entropy(audio_mask_logits.view(-1, 3), audio_mask_targets.view(-1))
                    visual_mask_loss = F.cross_entropy(visual_mask_logits.view(-1, 3), visual_mask_targets.view(-1))
                    mask_loss = audio_mask_loss + visual_mask_loss
                    
                # Forward pass through LLM
                logits = model(input_ids,
                               audio_query=None,
                               lip_query=None,
                               max_seq_length=train_cfg["max_input_length"],
                               lm_head_chunk_size=128)

                # shift the targets such that output n predicts token n+1
                logits[-1] = logits[-1][..., :-1, :]
                llm_loss = chunked_cross_entropy(logits, targets[..., 1:])

                # Combine losses
                mask_loss_weight = train_cfg["mask_loss_weight"]
                mask_loss *= mask_loss_weight
                loss = llm_loss + mask_loss

                cumulative_loss += loss.item()
                cumulative_loss_interval += loss.item()
                cumulative_llm_loss += llm_loss.item()
                cumulative_mask_loss += mask_loss.item()

                fabric.backward(loss / gradient_accumulation_iters)
                micro_step += 1

            if (micro_step + 1) % gradient_accumulation_iters == 0:
                # mark_only_lora_as_trainable(model)
                optimizer.step()
                optimizer.zero_grad()
                micro_step = 0

            dt = time.time() - t0
            if (real_iter + 1) % log_interval == 0 and fabric.global_rank == 0:
                avg_loss = cumulative_loss / log_interval
                avg_mask_loss = cumulative_mask_loss / log_interval
                avg_llm_loss = cumulative_llm_loss / log_interval
                elapsed = time.time() - log_start_time
                current_llm_lr = optimizer.param_groups[0]['lr']
                current_classifier_lr = optimizer.param_groups[1]['lr']
                logging.info(f"iter {(real_iter + 1)}: train loss = {avg_loss:.4f}, llm loss = {avg_llm_loss:.4f}, mask loss = {avg_mask_loss:.4f}, llm_lr = {current_llm_lr:.2e}, cls_lr = {current_classifier_lr:.2e}, time / {log_interval} iters = {elapsed:.2f}s")
                cumulative_loss = 0.0
                cumulative_mask_loss = 0.0
                cumulative_llm_loss = 0.0
                log_start_time = time.time()

            if (real_iter + 1) % save_interval == 0:
                # checkpoint_path = out_dir / f"iter-{real_iter:06d}.pth"
                # save_adapter_checkpoint(fabric, model, checkpoint_path)
                avg_loss = cumulative_loss_interval / save_interval
                logging.info(f"Average train loss = {avg_loss:.4f}")
                cumulative_loss_interval = 0.0

                val_loss = validate(fabric, model, audio_encoder, visual_encoder, val_dataloader, train_cfg)
                if fabric.global_rank == 0:
                    logging.info(f"iter {(real_iter + 1)}: val loss {val_loss:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint_path = out_dir / "best_model.pth"
                    save_adapter_checkpoint(fabric, model, checkpoint_path)
                    logging.info(f"iter {(real_iter + 1)} -> best model saved")
                fabric.barrier()

            real_iter += 1            
    # after training, validate the model
    val_loss = validate(fabric, model, audio_encoder, visual_encoder, val_dataloader, train_cfg)
    if fabric.global_rank == 0:
        logging.info(f"iter {(real_iter + 1)}: val loss {val_loss:.4f}")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        checkpoint_path = out_dir / "best_model.pth"
        save_adapter_checkpoint(fabric, model, checkpoint_path)
        logging.info(f"iter {(real_iter + 1)} -> best model saved")

@torch.no_grad()
def validate(
    fabric: L.Fabric, 
    model: GPT,
    audio_encoder,
    visual_encoder,
    val_dataloader: DataLoader,
    train_cfg: Dict,
) -> torch.Tensor:
    
    fabric.print("Validating ...")
    model.eval()
    losses = []
    llm_losses = []
    
    # Aggregate all predictions and targets for overall metrics
    all_predictions = []
    all_targets = []
    
    for batch in tqdm(val_dataloader):
        input_ids, targets, audio_features, visual_features = batch["input_ids"], batch["labels"], batch["audio"], batch["video"]
        input_ids, targets = fabric.to_device((input_ids, targets))
        
        # Get ground truth masks if available (during validation)
        audio_bin_labels = batch.get("audio_bin_labels", None)
        video_bin_labels = batch.get("video_bin_labels", None)
        
        valid_tokens = (targets[..., 1:] != -1).sum().item()
        if valid_tokens == 0:
            continue
            
        # Get encoder features
        audio_features, visual_features = audio_features.to(ASR_VSR_PRJ_DEVICE), visual_features.to(ASR_VSR_PRJ_DEVICE)
        audio_enc_features = audio_encoder(audio_features)
        visual_enc_features, _ = visual_encoder(visual_features.squeeze(1), None)
        
        # Get mask predictions from classifiers
        audio_mask_logits = model.audio_noise_classifier(audio_enc_features.to(fabric.device))
        visual_mask_logits = model.visual_noise_classifier(visual_enc_features.to(fabric.device))
        
        # Calculate mask loss if labels are available
        mask_loss = 0.0
        if audio_bin_labels is not None and video_bin_labels is not None:           
            # Convert ground truth labels to target indices
            audio_mask_targets = labels_to_indices(audio_bin_labels, device=fabric.device, prefix="")  # (B, T_chunk)
            visual_mask_targets = labels_to_indices(video_bin_labels, device=fabric.device, prefix="")  # (B, T_chunk)
            
            # Get predicted logits dimensions - these should now match the target dimensions
            B, T_audio_pred, _ = audio_mask_logits.shape  # (B, T_audio//20, 3)
            B, T_visual_pred, _ = visual_mask_logits.shape  # (B, T_visual//10, 3)
            
            # Ensure target and prediction lengths match by trimming if necessary
            audio_target_len = audio_mask_targets.shape[1]
            visual_target_len = visual_mask_targets.shape[1]
            
            if T_audio_pred > audio_target_len:
                audio_mask_logits = audio_mask_logits[:, :audio_target_len, :]
            elif T_audio_pred < audio_target_len:
                audio_mask_targets = audio_mask_targets[:, :T_audio_pred]
            
            if T_visual_pred > visual_target_len:
                visual_mask_logits = visual_mask_logits[:, :visual_target_len, :]
            elif T_visual_pred < visual_target_len:
                visual_mask_targets = visual_mask_targets[:, :T_visual_pred]
                
            # Calculate mask classification loss and accuracy
            audio_mask_loss = F.cross_entropy(audio_mask_logits.view(-1, 3), audio_mask_targets.view(-1))
            visual_mask_loss = F.cross_entropy(visual_mask_logits.view(-1, 3), visual_mask_targets.view(-1))
            mask_loss = audio_mask_loss + visual_mask_loss
            
            # Calculate accuracy
            audio_predictions = torch.argmax(audio_mask_logits, dim=-1)  # (B, T_audio)
            visual_predictions = torch.argmax(visual_mask_logits, dim=-1)  # (B, T_visual)
            
            # Collect all predictions and targets for overall metrics
            all_predictions.append(audio_predictions.cpu().flatten())
            all_predictions.append(visual_predictions.cpu().flatten())
            all_targets.append(audio_mask_targets.cpu().flatten())
            all_targets.append(visual_mask_targets.cpu().flatten())
        
        # Forward pass through LLM
        logits = model(input_ids,
                       audio_query=None,
                       lip_query=None,
                       lm_head_chunk_size=0)
        llm_loss = chunked_cross_entropy(logits[..., :-1, :], targets[..., 1:], chunk_size=0)
        
        # Combine losses
        mask_loss_weight = train_cfg["mask_loss_weight"]
        mask_loss *= mask_loss_weight
        total_loss = llm_loss + mask_loss
        
        losses.append(total_loss.item())
        llm_losses.append(llm_loss.item())
    
    val_loss = sum(losses) / len(losses)
    val_llm_loss = sum(llm_losses) / len(llm_losses)
    
    # Calculate overall mask classification metrics
    if all_predictions:
        # Concatenate all predictions and targets
        all_preds = torch.cat(all_predictions)
        all_targs = torch.cat(all_targets)
        
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
    else:
        overall_acc = 0.0
        precision = 0.0
        recall = 0.0
        f1 = 0.0
    
    if fabric.global_rank == 0:
        logging.info(f"Validation - total loss: {val_loss:.4f}, llm loss: {val_llm_loss:.4f}")
        logging.info(f"Mask metrics - acc: {overall_acc:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}")

    torch.cuda.empty_cache()
    model.reset_cache()
    model.train()
    # Only LLM loss is used for best model selection
    return val_llm_loss


def save_adapter_checkpoint(fabric, model, file_path: Path):
    fabric.print(f"Saving adapter weights to {str(file_path)!r}")
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
    parser.add_argument('--dual_hypotheses', action='store_true', help='Whether to use dual hypotheses (default: False)')
    parser.add_argument('--max_nhyps', type=int, default=None, help='max number of hypotheses to use for training')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--micro_batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--classifier_lr', type=float, default=1e-4, help='Learning rate for noise classifiers')

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

    parser.add_argument('--prompts_format', type=str, default='RelPrompt')
    parser.add_argument('--mask_loss_weight', type=float, default=0.02, help='Weight for mask classification loss')
    parser.add_argument('--mask_threshold', type=int, default=None, help='SNR threshold for mask reliability')
    parser.add_argument('--time_window', type=float, default=0.4, help='Time segment duration in seconds')
    parser.add_argument('--pool_size', type=int, default=10, help='Pooling size for noise classifiers')

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
    args.classifier_learning_rate = args.classifier_lr
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