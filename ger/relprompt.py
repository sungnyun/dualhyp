# Derived from https://github.com/microsoft/LoRA
#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

r"""
    Low Ranking Adaptation for LLMs scheme.

             ┌───────────────────┐
             ┆         h         ┆
             └───────────────────┘
                       ▲
                       |
                       +
                    /     \
    ┌─────────────────┐    ╭───────────────╮     Matrix initialization:
    ┆                 ┆     \      B      /      B = 0
    ┆   pretrained    ┆      \    r*d    /       A = N(0, sigma^2)
    ┆    weights      ┆       ╰─────────╯
    ┆                 ┆       |    r    |        r - rank
    ┆   W e R^(d*d)   ┆       | ◀─────▶ |
    ┆                 ┆       ╭─────────╮
    └─────────────────┘      /     A     \
              ▲             /     d*r     \
               \           ╰───────────────╯
                \                ▲
                 \              /
                  \            /
             ┌───────────────────┐
             ┆         x         ┆
             └───────────────────┘

With LoRA (Low Ranking Adaptation: https://arxiv.org/abs/2106.09685) instead of learning weights of size d*d,
we can freeze the pretrained weights and instead learn two matrices of size d*r and r*d (they will store weight updates
for the pretrained weights): the number of parameters in this case will be reduced drastically (depending on the rank of
course) yet after multiplication of matrices d*r and r*d we will get a matrix d*d which we can sum with frozen
pretrained weights and thus fine-tune the model.

The goal of this approach is to move weight updates into a separate matrix which is decomposed with
two matrices of a lower rank.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing_extensions import Self

import ger
from ger.config import Config as BaseConfig
from ger.model import GPT as BaseModel
from ger.model import Block as BaseBlock
from ger.model import CausalSelfAttention as BaseCausalSelfAttention
from ger.model import LLaMAMLP as BaseLLaMAMLP
from ger.model import KVCache, RoPECache
from ger.lora import LoRALayer, LoRALinear, LoRAQKVLinear
from ger.utils import map_old_state_dict_weights


class AdapterV2Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, **kwargs) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, **kwargs)
        self.adapter_bias = torch.nn.Parameter(torch.zeros(out_features), requires_grad=False)
        self.adapter_scale = torch.nn.Parameter(torch.ones(out_features), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.adapter_scale * (self.linear(x) + self.adapter_bias)

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.adapter_bias)
        nn.init.ones_(self.adapter_scale)


def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    """Freeze all modules except LoRA's and depending on 'bias' value unfreezes bias weights.

    Args:
        model: model with LoRA layers
        bias:
            ``"none"``: all bias weights will be frozen,
            ``"lora_only"``: only bias weight for LoRA layers will be unfrozen,
            ``"all"``: all bias weights will be unfrozen.

    Raises:
        NotImplementedError: if `bias` not in ["none", "lora_only", "all"]
    """
    # freeze all layers except LoRA's
    for n, p in model.named_parameters():
        if "audio_proj" in n or "visual_proj" in n:
            p.requires_grad = True
            continue
        if "noise_classifier" in n:
            p.requires_grad = True
            continue
        if "lora_" not in n:
            # freeze for now
            # if "adapter_scale" in n or "adapter_bias" in n:
            #     p.requires_grad = True
            #     continue
            p.requires_grad = False

    # depending on the `bias` value unfreeze bias weights
    if bias == "none":
        return
    if bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoRALayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


def lora_filter(key: str, value: Any) -> bool:
    return "lora_" in key


class NoiseMaskClassifier(nn.Module):
    """Classifier to predict noise masks for audio or visual features with configurable temporal reduction"""
    def __init__(self, input_dim, hidden_dim=256, dropout=0.1, pool_size=10):
        super().__init__()
        self.pool_size = pool_size
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.pool = nn.AvgPool1d(kernel_size=pool_size, stride=pool_size, ceil_mode=True)
        self.classifier = nn.Linear(hidden_dim, 3)  # 3 classes: clean, mixed, noisy
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (B, T, C)
        x = x.transpose(1, 2)  # (B, C, T)
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        # Apply pooling to reduce temporal dimension by pool_size
        x = self.pool(x)  # (B, C, T//pool_size)
        x = x.transpose(1, 2)  # (B, T//pool_size, C)
        logits = self.classifier(x)  # (B, T//pool_size, 3)
        return logits


@dataclass
class Config(BaseConfig):
    """
    Args:
        r: rank of the weight update matrices. To make sense of using LoRA the rank should be smaller than the rank of
            the weights of the model. The rank can be as low as 1: https://arxiv.org/pdf/2106.09685.pdf (section 7.2)
        alpha: alpha is needed for scaling updates as alpha/r
            "This scaling helps to reduce the need to retune hyperparameters when we vary r"
            https://arxiv.org/pdf/2106.09685.pdf (section 4.1)
        dropout: dropout that is applied on the input in the LoRA branch (before multiplying by matrix A)
        to_*: either apply LoRA to the specified weights or not
    """

    r: int = 16
    alpha: int = 1
    dropout: float = 0.0
    to_query: bool = True
    to_key: bool = True
    to_value: bool = True
    to_projection: bool = True
    to_mlp: bool = False
    to_head: bool = False
    lora_start_layer: int = 0  # start from 0 to train projectors
    whisper_dim: int = 1280
    raven_dim: int = 1024
    pool_size: int = 10

    @property
    def mlp_class(self) -> Type:
        return getattr(ger.relprompt, self._mlp_class)


class GPT(BaseModel):
    def __init__(self, config: Config) -> None:
        nn.Module.__init__(self)
        assert config.padded_vocab_size is not None
        self.config = config

        if config.to_head:
            self.lm_head = LoRALinear(
                config.n_embd,
                config.padded_vocab_size,
                bias=config.lm_head_bias,
                r=(config.r if config.to_head else 0),
                lora_alpha=config.alpha,
                lora_dropout=(config.dropout if config.to_head else 0),
            )
        else:
            self.lm_head = AdapterV2Linear(config.n_embd, config.padded_vocab_size, bias=config.lm_head_bias)
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
                h=nn.ModuleList(Block(config, i) for i in range(config.n_layer)),
                ln_f=config.norm_class(config.n_embd, eps=config.norm_eps),
            )
        )
        self.rope_cache: Optional[RoPECache] = None
        self.max_seq_length = self.config.block_size
        self.mask_cache: Optional[torch.Tensor] = None
        self.kv_caches: List[KVCache] = []

        # Add noise mask classifiers
        self.audio_noise_classifier = NoiseMaskClassifier(config.whisper_dim, pool_size=2*config.pool_size)
        self.visual_noise_classifier = NoiseMaskClassifier(config.raven_dim, pool_size=config.pool_size)

    def resize_token_embeddings(self, new_vocab_size: int):
        old_embed: nn.Embedding = self.transformer.wte
        old_num, dim = old_embed.weight.shape

        if new_vocab_size == old_num:
            return

        new_embed = nn.Embedding(old_num + new_vocab_size, dim)
        new_embed.weight.data[:old_num].copy_(old_embed.weight.data)
        nn.init.normal_(
            new_embed.weight.data[old_num:],
            mean=0.0,
            std=old_embed.weight.data.std().item(),
        )
        # replace
        self.transformer.wte = new_embed

    def forward(
        self,
        idx: torch.Tensor,
        audio_query: torch.Tensor = None,
        lip_query: torch.Tensor = None,
        max_seq_length: Optional[int] = None,
        input_pos: Optional[torch.Tensor] = None,
        lm_head_chunk_size: int = 0,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        B, T = idx.size()
        use_kv_cache = input_pos is not None #if input_pos is not None then True, if None then False
        block_size = self.config.block_size

        if use_kv_cache:  # not relevant otherwise
            assert (
                self.max_seq_length >= T
            ), f"Cannot forward sequence of length {T}, max seq length is only {self.max_seq_length}"
        assert self.max_seq_length <= block_size, f"Cannot attend to {self.max_seq_length}, block size is only {block_size}"
        assert block_size >= T, f"Cannot forward sequence of length {T}, block size is only {block_size}"

        if self.rope_cache is None:
            self.rope_cache = self.build_rope_cache(idx)
        # passing `attn_mask` to SDPA downgrades it to use the inefficient implementation. since we only need the mask
        # for the kv-cache support (only during inference), we only create it in that situation
        # this will be resolved by https://github.com/pytorch/pytorch/issues/96099
        if use_kv_cache and self.mask_cache is None:
            self.mask_cache = self.build_mask_cache(idx)

        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)

        if (not use_kv_cache) or (use_kv_cache and not self.kv_caches):
            input_pos = torch.arange(0, x.shape[1], device=x.device)
            if use_kv_cache:
                self.av_prompt_length = x.shape[1] - T
        else:
            # when decoding with kv-caching, need to shift the input positions
            input_pos = input_pos + self.av_prompt_length

        cos, sin = self.rope_cache
        if use_kv_cache:
            cos = cos.index_select(0, input_pos)
            sin = sin.index_select(0, input_pos)
            mask = self.mask_cache.index_select(2, input_pos)
            mask = mask[:, :, :, :self.max_seq_length]
        else:
            cos = cos[:x.shape[1]]
            sin = sin[:x.shape[1]]
            mask = None


        if not use_kv_cache:
            for block in self.transformer.h:
                x, *_ = block(x, (cos, sin), self.max_seq_length)
        else:
            self.kv_caches = self.kv_caches or self.build_kv_caches(x, self.max_seq_length, cos.size(-1))
            for i, block in enumerate(self.transformer.h):
                x, self.kv_caches[i] = block(x, (cos, sin), self.max_seq_length, mask, input_pos, self.kv_caches[i])

        x = self.transformer.ln_f(x)[:,-idx.shape[-1]:,:]
        if lm_head_chunk_size > 0:
            # chunk the lm head logits to reduce the peak memory used by autograd
            return [self.lm_head(x_i) for x_i in x.split(lm_head_chunk_size, dim=1)]
        return self.lm_head(x)  # (B, T, vocab_size)

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(Config.from_name(name, **kwargs))

    def _init_weights(self, module: nn.Module) -> None:
        """Meant to be used with `gpt.apply(gpt._init_weights)`. Unused method left for completeness."""
        super()._init_weights(module)
        if isinstance(module, LoRALinear):
            module.reset_parameters()

    def _load_from_state_dict(self, state_dict: Dict, prefix: str, *args: Any, **kwargs: Any) -> None:
        """For compatibility with base checkpoints."""
        mapping = {"lm_head.weight": "lm_head.linear.weight", "lm_head.bias": "lm_head.linear.bias"}
        state_dict = map_old_state_dict_weights(state_dict, mapping, prefix)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class Block(BaseBlock):
    def __init__(self, config: Config, block_idx: int) -> None:
        nn.Module.__init__(self)
        self.norm_1 = config.norm_class(config.n_embd, eps=config.norm_eps)
        if block_idx >= config.lora_start_layer:
            self.attn = CausalSelfAttention(config)
        else:
            self.attn = BaseCausalSelfAttention(config)
        if not config.shared_attention_norm:
            self.norm_2 = config.norm_class(config.n_embd, eps=config.norm_eps)
        if block_idx >= config.lora_start_layer:
            self.mlp = config.mlp_class(config)
        else:
            self.mlp = BaseLLaMAMLP(config)

        self.config = config


class CausalSelfAttention(BaseCausalSelfAttention):
    def __init__(self, config: Config) -> None:
        # Skip the parent class __init__ altogether and replace it to avoid
        # useless allocations
        nn.Module.__init__(self)
        shape = (config.n_head + 2 * config.n_query_groups) * config.head_size
        # key, query, value projections for all heads, but in a batch
        self.attn = LoRAQKVLinear(
            in_features=config.n_embd,
            out_features=shape,
            r=config.r,
            lora_alpha=config.alpha,
            lora_dropout=config.dropout,
            enable_lora=(config.to_query, config.to_key, config.to_value),
            bias=config.bias,
            # for MQA/GQA support
            n_head=config.n_head,
            n_query_groups=config.n_query_groups,
        )
        # output projection
        self.proj = LoRALinear(
            config.n_embd,
            config.n_embd,
            bias=config.bias,
            r=(config.r if config.to_projection else 0),
            lora_alpha=config.alpha,
            lora_dropout=config.dropout,
        )
        # disabled by default
        self.kv_cache: Optional[KVCache] = None

        self.config = config

    def _load_from_state_dict(self, state_dict: Dict, prefix: str, *args: Any, **kwargs: Any) -> None:
        """For compatibility with base checkpoints."""
        mapping = {
            "attn.weight": "attn.linear.weight",
            "attn.bias": "attn.linear.bias",
            "proj.weight": "proj.linear.weight",
            "proj.bias": "proj.linear.bias",
        }
        state_dict = map_old_state_dict_weights(state_dict, mapping, prefix)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class GptNeoxMLP(ger.model.GptNeoxMLP):
    def __init__(self, config: Config) -> None:
        nn.Module.__init__(self)
        self.fc = LoRALinear(
            config.n_embd,
            config.intermediate_size,
            bias=config.bias,
            r=(config.r if config.to_mlp else 0),
            lora_alpha=config.alpha,
            lora_dropout=(config.dropout if config.to_mlp else 0),
        )
        self.proj = LoRALinear(
            config.intermediate_size,
            config.n_embd,
            bias=config.bias,
            r=(config.r if config.to_mlp else 0),
            lora_alpha=config.alpha,
            lora_dropout=(config.dropout if config.to_mlp else 0),
        )

        self.config = config

    def _load_from_state_dict(self, state_dict: Dict, prefix: str, *args: Any, **kwargs: Any) -> None:
        """For compatibility with base checkpoints."""
        mapping = {
            "fc.weight": "fc.linear.weight",
            "fc.bias": "fc.linear.bias",
            "proj.weight": "proj.linear.weight",
            "proj.bias": "proj.linear.bias",
        }
        state_dict = map_old_state_dict_weights(state_dict, mapping, prefix)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class LLaMAMLP(BaseLLaMAMLP):
    def __init__(self, config: Config) -> None:
        nn.Module.__init__(self)
        self.fc_1 = LoRALinear(
            config.n_embd,
            config.intermediate_size,
            bias=config.bias,
            r=(config.r if config.to_mlp else 0),
            lora_alpha=config.alpha,
            lora_dropout=(config.dropout if config.to_mlp else 0),
        )
        self.fc_2 = LoRALinear(
            config.n_embd,
            config.intermediate_size,
            bias=config.bias,
            r=(config.r if config.to_mlp else 0),
            lora_alpha=config.alpha,
            lora_dropout=(config.dropout if config.to_mlp else 0),
        )
        self.proj = LoRALinear(
            config.intermediate_size,
            config.n_embd,
            bias=config.bias,
            r=(config.r if config.to_mlp else 0),
            lora_alpha=config.alpha,
            lora_dropout=(config.dropout if config.to_mlp else 0),
        )

    def _load_from_state_dict(self, state_dict: Dict, prefix: str, *args: Any, **kwargs: Any) -> None:
        """For compatibility with base checkpoints."""
        mapping = {
            "fc_1.weight": "fc_1.linear.weight",
            "fc_1.bias": "fc_1.linear.bias",
            "fc_2.weight": "fc_2.linear.weight",
            "fc_2.bias": "fc_2.linear.bias",
            "proj.weight": "proj.linear.weight",
            "proj.bias": "proj.linear.bias",
        }
        state_dict = map_old_state_dict_weights(state_dict, mapping, prefix)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


def merge_lora_weights(model: GPT) -> None:
    """Merge LoRA weights into the full-rank weights to speed up inference."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge()
