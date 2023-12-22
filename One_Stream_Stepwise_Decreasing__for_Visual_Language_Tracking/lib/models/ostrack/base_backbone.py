from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import resize_pos_embed
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from lib.models.layers.patch_embed import PatchEmbed
from lib.models.ostrack.utils import combine_tokens, recover_tokens
from  torchvision.utils import save_image

class BaseBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        # for original ViT
        self.pos_embed = None
        self.img_size = [224, 224]
        self.patch_size = 16
        self.embed_dim = 384

        self.cat_mode = 'direct'

        self.pos_embed_z = None
        self.pos_embed_x = None

        self.template_segment_pos_embed = None
        self.search_segment_pos_embed = None

        self.return_inter = False
        self.return_stage = [2, 5, 8, 11]

        self.add_cls_token = False
        self.add_sep_seg = False

        self.text_channela = SpatialAttentiona(dim=768,spatial_dim=30)
        self.text_channelb = SpatialAttentiona(dim=768,spatial_dim=15)
        self.text_channelc = SpatialAttentiona(dim=768,spatial_dim=5)

    def finetune_track(self, cfg, patch_start_index=1):

        search_size = to_2tuple(cfg.DATA.SEARCH.SIZE)
        template_size = to_2tuple(cfg.DATA.TEMPLATE.SIZE)
        new_patch_size = cfg.MODEL.BACKBONE.STRIDE

        self.cat_mode = cfg.MODEL.BACKBONE.CAT_MODE
        self.return_inter = cfg.MODEL.RETURN_INTER
        self.return_stage = cfg.MODEL.RETURN_STAGES
        self.add_sep_seg = cfg.MODEL.BACKBONE.SEP_SEG

        # resize patch embedding
        if new_patch_size != self.patch_size:
            print('Inconsistent Patch Size With The Pretrained Weights, Interpolate The Weight!')
            old_patch_embed = {}
            for name, param in self.patch_embed.named_parameters():
                if 'weight' in name:
                    param = nn.functional.interpolate(param, size=(new_patch_size, new_patch_size),
                                                      mode='bicubic', align_corners=False)
                    param = nn.Parameter(param)
                old_patch_embed[name] = param
            self.patch_embed = PatchEmbed(img_size=self.img_size, patch_size=new_patch_size, in_chans=3,
                                          embed_dim=self.embed_dim)
            self.patch_embed.proj.bias = old_patch_embed['proj.bias']
            self.patch_embed.proj.weight = old_patch_embed['proj.weight']

        # for patch embedding
        patch_pos_embed = self.pos_embed[:, patch_start_index:, :]
        patch_pos_embed = patch_pos_embed.transpose(1, 2)
        B, E, Q = patch_pos_embed.shape
        P_H, P_W = self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size
        patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)

        # for search region
        H, W = search_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        search_patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic',
                                                           align_corners=False)
        search_patch_pos_embed = search_patch_pos_embed.flatten(2).transpose(1, 2)

        # for template region
        H, W = template_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        template_patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic',
                                                             align_corners=False)
        template_patch_pos_embed = template_patch_pos_embed.flatten(2).transpose(1, 2)

        self.pos_embed_z = nn.Parameter(template_patch_pos_embed)
        self.pos_embed_x = nn.Parameter(search_patch_pos_embed)

        # for cls token (keep it but not used)
        if self.add_cls_token and patch_start_index > 0:
            cls_pos_embed = self.pos_embed[:, 0:1, :]
            self.cls_pos_embed = nn.Parameter(cls_pos_embed)

        # separate token and segment token
        if self.add_sep_seg:
            self.template_segment_pos_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.template_segment_pos_embed = trunc_normal_(self.template_segment_pos_embed, std=.02)
            self.search_segment_pos_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.search_segment_pos_embed = trunc_normal_(self.search_segment_pos_embed, std=.02)

        # self.cls_token = None
        # self.pos_embed = None

        if self.return_inter:
            for i_layer in self.return_stage:
                if i_layer != 11:
                    norm_layer = partial(nn.LayerNorm, eps=1e-6)
                    layer = norm_layer(self.embed_dim)
                    layer_name = f'norm{i_layer}'
                    self.add_module(layer_name, layer)

    def forward_features(self, z, x,text):
        B, H, W = x.shape[0], x.shape[2], x.shape[3]

        x = self.patch_embed(x)
        z = self.patch_embed(z)

        if self.add_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            cls_tokens = cls_tokens + self.cls_pos_embed

        z += self.pos_embed_z
        x += self.pos_embed_x
        if self.add_sep_seg:
            x += self.search_segment_pos_embed
            z += self.template_segment_pos_embed

        x = combine_tokens(z, x, mode=self.cat_mode)
        if self.add_cls_token:
            x = torch.cat([cls_tokens, x], dim=1)

        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            if i == 0 :
                x = combine_tokens(text,x,mode=self.cat_mode)
            elif i ==  1 :
                text_prompt = self.text_channela(text)
                x = combine_tokens(text_prompt, x, mode=self.cat_mode)
            elif i ==  4 :
                text = x[:,:30,:]
                text_prompt = self.text_channelb(text)
                x = x[:,30:,:]
                x = combine_tokens(text_prompt, x, mode=self.cat_mode)
            elif i ==  7 :
                text = x[:, :15, :]
                text_prompt = self.text_channelc(text)
                x = x[:,15:,:]
                x = combine_tokens(text_prompt, x, mode=self.cat_mode)
            elif i == 10:
                x = x[:,5:,:]
            x = blk(x)
            if i < 1:
                text = x[:,:40,:]
                x = x[:,40:,:]
            elif i < 4:
                x = x[:,30:,:]
                x = combine_tokens(text_prompt, x, mode=self.cat_mode)
            elif i < 7:
                x = x[:, 15:, :]
                x = combine_tokens(text_prompt, x, mode=self.cat_mode)
            elif i < 10:
                x = x[:,5:,:]
                x = combine_tokens(text_prompt, x, mode=self.cat_mode)


        lens_z = self.pos_embed_z.shape[1]
        lens_x = self.pos_embed_x.shape[1]
        x = recover_tokens(x, lens_z, lens_x, mode=self.cat_mode)

        aux_dict = {"attn": None}
        return self.norm(x), aux_dict

    def forward(self, z, x,text, **kwargs):
        """
        Joint feature extraction and relation modeling for the basic ViT backbone.
        Args:
            z (torch.Tensor): template feature, [B, C, H_z, W_z]
            x (torch.Tensor): search region feature, [B, C, H_x, W_x]

        Returns:
            x (torch.Tensor): merged template and search region feature, [B, L_z+L_x, C]
            attn : None
        """
        x, aux_dict = self.forward_features(z, x,text,)

        return x, aux_dict

class SpatialAttentiona(nn.Module):
    def __init__(self, dim, num_heads=8, spatial_dim=20, qkv_bias=False, attn_drop=0., proj_drop=0., kernel_size=3):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.spatial_dim = spatial_dim
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        # self.summation = nn.Conv2d(dim, spatial_dim, kernel_size=kernel_size, padding=1)
        # if kernel_size is 3:
        #     self.summation = nn.Conv2d(dim, spatial_dim, kernel_size=kernel_size, padding=1)
        # elif kernel_size is 1:
        #     self.summation = nn.Conv2d(dim, spatial_dim, kernel_size=kernel_size)
        self.summation = nn.Linear(dim,spatial_dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.norm = nn.BatchNorm1d(40)
        self.cross_heads = nn.Linear(self.spatial_dim, self.spatial_dim, bias=qkv_bias)

    def forward(self, x, prompts=None, return_token=False):
        # x: Bx, Cx, W, H
        # mask: [B, N, ] torch.bool
        Bx, N ,Cx = x.shape
        space_attn = self.summation(x)
        x_expanded = x.view(Bx, Cx, -1).transpose(1, 2).unsqueeze(-2)
        space_attn = space_attn.view(Bx, self.spatial_dim, -1).transpose(1, 2)
        space_attn_expanded = space_attn.softmax(dim=1).unsqueeze(-1).expand(Bx, -1, self.spatial_dim, Cx)
        spatial_x = space_attn_expanded * x_expanded
        # attn_topk
        tokens = torch.topk(spatial_x, 1, dim=1, largest=True)[0].squeeze(1)
        qkv = self.qkv(tokens).reshape(Bx, self.spatial_dim, 3, self.num_heads, Cx // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn_spatial_x = (attn @ v).transpose(1, 2).reshape(Bx, -1, Cx)
        attn_spatial_x = self.proj_drop(attn_spatial_x).transpose(1, 2)
        attn_spatial_x = self.cross_heads(attn_spatial_x).transpose(1, 2)
        # if prompts is not None:
        #     prompts_weight = torch.cosine_similarity(prompts, attn_spatial_x, dim=-1).unsqueeze(-1)
        #     attn_spatial_x = attn_spatial_x + prompts_weight * attn_spatial_x
        # out = (space_attn @ attn_spatial_x).reshape(Bx, N, Cx).permute(0,1, 2).contiguous()
        # out = self.norm(out) + x
        # if return_token:
        #     return out, attn_spatial_x
        return attn_spatial_x
