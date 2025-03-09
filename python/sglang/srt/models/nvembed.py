from typing import Iterable, Optional, Tuple, TypedDict

import torch
import torch.nn as nn
from einops import rearrange, repeat

from sglang.srt.configs.nv_embed_config import LatentAttentionConfig, NVEmbedConfig
from sglang.srt.layers.pooler import EmbeddingPoolerOutput, Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.model_runner import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.llama_embedding import MistralModel


def exists(val):
    return val is not None


class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = torch.nn.LayerNorm(dim)
        self.norm_context = (
            torch.nn.LayerNorm(context_dim) if exists(context_dim) else None
        )

    def forward(self, x, **kwargs):
        x = self.norm(x)
        if exists(self.norm_context):
            context = kwargs["context"]
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)
        return self.fn(x, **kwargs)


class GEGLU(torch.nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * torch.nn.functional.gelu(gates)


class FeedForward(torch.nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            torch.nn.Linear(dim * mult, dim),
        )

    def forward(self, x):
        return self.net(x)


def default(val, d):
    return val if exists(val) else d


class Attention(torch.nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = torch.nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = torch.nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = torch.nn.Linear(inner_dim, query_dim, bias=False)

    def forward(self, x, context=None, mask=None):
        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_mem_efficient=True
        ):
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


class LatentAttentionModel(nn.Module):
    config_class = LatentAttentionConfig

    def __init__(self, config: LatentAttentionConfig):
        super().__init__()
        ## cross-attention block
        num_latents, latent_dim, cross_heads, cross_dim_head = (
            config.num_latents_value,
            config.latent_dim,
            config.num_cross_heads,
            config.cross_dim_head,
        )
        dim = config.hidden_dim
        # init latent_attention and latents
        self.cross_attend_blocks = torch.nn.ModuleList(
            [
                PreNorm(
                    latent_dim,
                    Attention(
                        latent_dim, dim, heads=cross_heads, dim_head=cross_dim_head
                    ),
                    context_dim=dim,
                ),
                PreNorm(latent_dim, FeedForward(latent_dim)),
            ]
        )
        self.output_normalize = config.output_normalize
        self.register_parameter(
            "latents", torch.nn.Parameter(torch.randn(num_latents, latent_dim))
        )

    def forward(self, hiddens, attention_mask: torch.Tensor = None):
        ## cross-attention block
        cross_attn, cross_ff = self.cross_attend_blocks
        x = repeat(self.latents, "n d -> b n d", b=1)
        hiddens = cross_attn(hiddens, context=x, mask=None) + hiddens
        hiddens = cross_ff(hiddens) + hiddens
        if attention_mask != None:
            s = torch.sum(hiddens * attention_mask.unsqueeze(-1).float(), dim=1)
            d = attention_mask.sum(dim=1, keepdim=True).float()
            hiddens = s / d
            if self.output_normalize:
                hiddens = torch.nn.functional.normalize(hiddens, p=2, dim=-1)
        return hiddens


class NVEmbedModel(nn.Module):

    _no_split_modules = ["MistralDecoderLayer", "LatentAttentionModel"]

    def __init__(
        self,
        config: NVEmbedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):

        super().__init__()
        self.config = config
        self.embedding_model = MistralModel(config=config.text_config)
        self.latent_attention_model = LatentAttentionModel(
            config=config.latent_attention_config
        )
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = True,
    ) -> EmbeddingPoolerOutput:

        assert get_embedding

        hidden_states = self.embedding_model(
            input_ids, positions, forward_batch, input_embeds, return_embeds=True
        )
        hidden_states = hidden_states.unsqueeze(0)
        embeds = self.latent_attention_model(hidden_states)
        embeds = embeds.squeeze(0)

        return self.pooler(embeds, forward_batch)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "latent_attention_model" in name:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            else:
                name = name.replace("embedding_model.", "")
                self.embedding_model.load_weights([(name, loaded_weight)])


EntryClass = NVEmbedModel
