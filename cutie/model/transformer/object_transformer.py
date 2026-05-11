from typing import Dict, Optional
from omegaconf import DictConfig

import torch
import torch.nn as nn
from cutie.model.group_modules import GConv2d
from cutie.utils.tensor_utils import aggregate
from cutie.model.transformer.positional_encoding import PositionalEncoding
from cutie.model.transformer.transformer_layers import *


class QueryTransformerBlock(nn.Module):
    def __init__(self, model_cfg: DictConfig):
        super().__init__()

        this_cfg = model_cfg.object_transformer
        self.embed_dim = this_cfg.embed_dim
        self.num_heads = this_cfg.num_heads
        self.num_queries = this_cfg.num_queries
        self.ff_dim = this_cfg.ff_dim

        self.read_from_pixel = CrossAttention(self.embed_dim,
                                              self.num_heads,
                                              add_pe_to_qkv=this_cfg.read_from_pixel.add_pe_to_qkv)
        self.self_attn = SelfAttention(self.embed_dim,
                                       self.num_heads,
                                       add_pe_to_qkv=this_cfg.query_self_attention.add_pe_to_qkv)
        self.ffn = FFN(self.embed_dim, self.ff_dim)
        self.read_from_query = CrossAttention(self.embed_dim,
                                              self.num_heads,
                                              add_pe_to_qkv=this_cfg.read_from_query.add_pe_to_qkv,
                                              norm=this_cfg.read_from_query.output_norm)
        self.pixel_ffn = PixelFFN(self.embed_dim)

    def forward(
            self,
            x: torch.Tensor,
            pixel: torch.Tensor,
            query_pe: torch.Tensor,
            pixel_pe: torch.Tensor,
            attn_mask: torch.Tensor,
            need_weights: bool = False) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        # x: (bs*num_objects)*num_queries*embed_dim
        # pixel: bs*num_objects*C*H*W
        # query_pe: (bs*num_objects)*num_queries*embed_dim
        # pixel_pe: (bs*num_objects)*(H*W)*C
        # attn_mask: (bs*num_objects*num_heads)*num_queries*(H*W)

        # bs*num_objects*C*H*W -> (bs*num_objects)*(H*W)*C
        pixel_flat = pixel.flatten(3, 4).flatten(0, 1).transpose(1, 2).contiguous()
        x, q_weights = self.read_from_pixel(x,
                                            pixel_flat,
                                            query_pe,
                                            pixel_pe,
                                            attn_mask=attn_mask,
                                            need_weights=need_weights)
        x = self.self_attn(x, query_pe)
        x = self.ffn(x)

        pixel_flat, p_weights = self.read_from_query(pixel_flat,
                                                     x,
                                                     pixel_pe,
                                                     query_pe,
                                                     need_weights=need_weights)
        pixel = self.pixel_ffn(pixel, pixel_flat)

        if need_weights:
            bs, num_objects, _, h, w = pixel.shape
            q_weights = q_weights.view(bs, num_objects, self.num_heads, self.num_queries, h, w)
            p_weights = p_weights.transpose(2, 3).view(bs, num_objects, self.num_heads,
                                                       self.num_queries, h, w)

        return x, pixel, q_weights, p_weights


class QueryTransformer(nn.Module):
    def __init__(self, model_cfg: DictConfig):
        super().__init__()

        this_cfg = model_cfg.object_transformer
        self.value_dim = model_cfg.value_dim
        self.embed_dim = this_cfg.embed_dim
        self.num_heads = this_cfg.num_heads
        self.num_queries = this_cfg.num_queries

        # query initialization and embedding
        self.query_init = nn.Embedding(self.num_queries, self.embed_dim)
        self.query_emb = nn.Embedding(self.num_queries, self.embed_dim)

        # projection from object summaries to query initialization and embedding
        self.summary_to_query_init = nn.Linear(self.embed_dim, self.embed_dim)
        self.summary_to_query_emb = nn.Linear(self.embed_dim, self.embed_dim)

        self.pixel_pe_scale = model_cfg.pixel_pe_scale
        self.pixel_pe_temperature = model_cfg.pixel_pe_temperature
        self.pixel_init_proj = GConv2d(self.embed_dim, self.embed_dim, kernel_size=1)
        self.pixel_emb_proj = GConv2d(self.embed_dim, self.embed_dim, kernel_size=1)
        self.spatial_pe = PositionalEncoding(self.embed_dim,
                                             scale=self.pixel_pe_scale,
                                             temperature=self.pixel_pe_temperature,
                                             channel_last=False,
                                             transpose_output=True)

        # transformer blocks
        self.num_blocks = this_cfg.num_blocks
        self.blocks = nn.ModuleList(
            QueryTransformerBlock(model_cfg) for _ in range(self.num_blocks))
        self.mask_pred = nn.ModuleList(
            nn.Sequential(nn.ReLU(), GConv2d(self.embed_dim, 1, kernel_size=1))
            for _ in range(self.num_blocks + 1))

        self.act = nn.ReLU(inplace=True)

    def forward(self,
                pixel: torch.Tensor,
                obj_summaries: torch.Tensor,
                selector: Optional[torch.Tensor] = None,
                need_weights: bool = False,
                use_explicit_shift: bool = False, 
                pixel_shift_uv: Optional[torch.Tensor] = None) -> tuple:
        
        bs, num_objects = obj_summaries.shape[:2]
        obj_values = obj_summaries.flatten(0, 1)

        obj_init = self.summary_to_query_init(obj_values)
        obj_emb = self.summary_to_query_emb(obj_values)

        query = self.query_init.weight.unsqueeze(0).expand(bs * num_objects, -1, -1) + obj_init
        query_emb = self.query_emb.weight.unsqueeze(0).expand(bs * num_objects, -1, -1) + obj_emb

        pixel_init = self.pixel_init_proj(pixel)
        pixel_emb = self.pixel_emb_proj(pixel)
        
        # 取得空間位置編碼
        pixel_pe_spatial = self.spatial_pe(pixel.flatten(0, 1))

        # =================================================================
        # Trajectory-Aware Explicit 2D Shift
        # 利用 Affine_grid 對空間位置編碼進行無幽靈疊影 (Ghost-free) 的平移
        # =================================================================
        if use_explicit_shift and (pixel_shift_uv is not None):
            du = pixel_shift_uv[0, 0].item()
            dv = pixel_shift_uv[0, 1].item()
            B_obj, C_pe, H_pe, W_pe = pixel_pe_spatial.shape
            
            # 計算正規化平移量 [-1, 1]。注意反向平移 (-)
            tx = -du / (W_pe / 2.0)
            ty = -dv / (H_pe / 2.0)
            
            # 建立 Affine 平移矩陣
            translation_matrix = torch.tensor([
                [1.0, 0.0, tx],
                [0.0, 1.0, ty]
            ], device=pixel_pe_spatial.device, dtype=torch.float32).unsqueeze(0).repeat(B_obj, 1, 1)
            
            # 產生採樣網格並套用 (padding_mode='zeros' 填零補齊)
            grid = F.affine_grid(translation_matrix, pixel_pe_spatial.size(), align_corners=True)
            pixel_pe_spatial = F.grid_sample(pixel_pe_spatial, grid, align_corners=True, padding_mode='zeros')
        # =================================================================

        pixel_emb = pixel_emb.flatten(3, 4).flatten(0, 1).transpose(1, 2).contiguous()
        pixel_pe = pixel_pe_spatial.flatten(1, 2) + pixel_emb
        pixel_init = pixel_init.flatten(3, 4).flatten(0, 1).transpose(1, 2).contiguous()

        aux_features = {}
        for block in self.blocks:
            query, pixel_init, aux_features = block(query, query_emb, pixel_init, pixel_pe,
                                                    selector, need_weights)

        # read from query
        pixel = self.read_from_query(pixel_init, pixel_pe, query, query_emb, selector)
        pixel = self.ffn(pixel) + pixel

        pixel = pixel.transpose(1, 2).contiguous().view(bs, num_objects, self.embed_dim,
                                                        pixel_pe_spatial.shape[-2],
                                                        pixel_pe_spatial.shape[-1])
        return pixel, aux_features

    def _get_aux_mask(self, logits: torch.Tensor, selector: torch.Tensor) -> torch.Tensor:
        # logits: batch_size*num_objects*H*W
        # selector: batch_size*num_objects*1*1
        # returns a mask of shape (batch_size*num_objects*num_heads)*num_queries*(H*W)
        # where True means the attention is blocked

        if selector is None:
            prob = logits.sigmoid()
        else:
            prob = logits.sigmoid() * selector
        logits = aggregate(prob, dim=1)

        is_foreground = (logits[:, 1:] >= logits.max(dim=1, keepdim=True)[0])
        foreground_mask = is_foreground.bool().flatten(start_dim=2)
        inv_foreground_mask = ~foreground_mask
        inv_background_mask = foreground_mask

        aux_foreground_mask = inv_foreground_mask.unsqueeze(2).unsqueeze(2).repeat(
            1, 1, self.num_heads, self.num_queries // 2, 1).flatten(start_dim=0, end_dim=2)
        aux_background_mask = inv_background_mask.unsqueeze(2).unsqueeze(2).repeat(
            1, 1, self.num_heads, self.num_queries // 2, 1).flatten(start_dim=0, end_dim=2)

        aux_mask = torch.cat([aux_foreground_mask, aux_background_mask], dim=1)

        aux_mask[torch.where(aux_mask.sum(-1) == aux_mask.shape[-1])] = False

        return aux_mask
