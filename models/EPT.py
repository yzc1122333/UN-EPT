import math
import numpy as np
import warnings
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from mmseg.ops import resize
from mmcv.runner import force_fp32
from mmseg.models.builder import build_loss, build_backbone
from mmseg.models.losses import accuracy

from .vision_transformer import deit_base_distilled_patch16_384
from .spatial_branch import spatial_branch
from .context_branch import context_branch
from .base import base_segmentor

import sys
sys.path.append("..")
from builder import SEGMENTORS
   
   
@SEGMENTORS.register_module()
class EPT(base_segmentor):
    def __init__(self, 
                 heads, 
                 feat_dim, 
                 k, 
                 L,
                 dropout,
                 depth,
                 hidden_dim,
                 backbone_cfg,
                 loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 ignore_index=255,
                 activation="relu",
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 auxiliary_head=None):

        """
        params:
        heads: head number of the transformer in the context branch;
        feat_dim: input feature dimension of the context branch;
        k: #points for each scale;
        L: #scale;
        depth: transformer encoder/decoder number in the context branch;
        hidden_dim: transforme hidden dimension in the context branch.

        """

        super(EPT, self).__init__()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.test_cfg.stride = test_cfg.stride
        self.test_cfg.crop_size = test_cfg.crop_size
        self.num_classes = self.test_cfg.num_classes
        self.ignore_index = ignore_index
        self.align_corners = False
        self.feat_dim = feat_dim

        self.loss_decode = build_loss(loss_decode)

        if pretrained is not None:
            logger = logging.getLogger()
            logger.info(f'load model from: {pretrained}')

        if backbone_cfg.type == 'DeiT':
            self.backbone = deit_base_distilled_patch16_384(
                            img_size=backbone_cfg.img_size,
                            patch_size=backbone_cfg.patch_size,
                            embed_dim=backbone_cfg.embed_dim,
                            depth=backbone_cfg.bb_depth,
                            num_heads=backbone_cfg.num_heads,
                            mlp_ratio=backbone_cfg.mlp_ratio,
                            pretrained=pretrained)
        elif backbone_cfg.type == 'ResNetV1c':
            self.backbone = build_backbone(backbone_cfg)
            self.backbone.init_weights(pretrained=pretrained)
        
       
        self.cls = nn.Conv2d(feat_dim, self.num_classes, kernel_size=1)

        # get pyramid features
        self.layers = nn.ModuleList([])
        self.backbone_type = backbone_cfg.type
        if self.backbone_type == 'DeiT':
            self.layers.append(nn.Conv2d(backbone_cfg.embed_dim, feat_dim, kernel_size=1, stride=1))
            self.layers.append(nn.Conv2d(backbone_cfg.embed_dim, feat_dim, kernel_size=1, stride=1))
            self.layers.append(nn.Conv2d(backbone_cfg.embed_dim, feat_dim, kernel_size=1, stride=1))
        elif self.backbone_type == 'ResNetV1c':
            self.layers.append(nn.Conv2d(512, feat_dim, kernel_size=1, stride=1))
            self.layers.append(nn.Conv2d(1024, feat_dim, kernel_size=1, stride=1))
            self.layers.append(nn.Conv2d(2048, feat_dim, kernel_size=1, stride=1))

        self.context_branch = context_branch(d_model=feat_dim, nhead=heads,
                    num_encoder_layers=depth, num_decoder_layers=depth, dim_feedforward=hidden_dim, dropout=dropout,
                    activation=activation, num_feature_levels=L, dec_n_points=k,  enc_n_points=k)
        
        self.num_queries = self.test_cfg.num_queries
        self.query_embed = nn.Embedding(self.num_queries, feat_dim)
        self.spatial_branch = spatial_branch()


    def encode_decode(self, x):

        bsize, c, h, w = x.shape
        backbone_feats = self.backbone(x)

        if self.backbone_type == 'ResNetV1c':
            backbone_feats = backbone_feats[1:]
       
        context = self.spatial_branch(x)
        context = context.flatten(2).permute(2, 0, 1)

        pyramid_feats = [] 
        for i, conv_layer in enumerate(self.layers):
            feature = conv_layer(backbone_feats[i]) 
            pyramid_feats.append(feature)
    
        out = self.context_branch(pyramid_feats, context, self.query_embed.weight) 
        
        out = out.unsqueeze(0).reshape([h//8, w//8, bsize, self.feat_dim]).permute(2, 3, 0, 1)

        out = self.cls(out)

        seg_logits = resize(
            input=out,
            size=x.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        return seg_logits

    def forward_train(self, img, img_metas, gt_semantic_seg):
        
        seg_logits = self.encode_decode(img)
        losses = dict()

        loss_decode = self.losses(seg_logits, gt_semantic_seg)
        losses.update(loss_decode)

        return losses
    
    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, seg_label):
        """Compute segmentation loss."""
        loss = dict()
        
        seg_weight = None

        seg_label = seg_label.squeeze(1)

        loss['loss_seg'] = self.loss_decode(
            seg_logit,
            seg_label,
            weight=seg_weight,
            ignore_index=self.ignore_index)
        
        loss['acc_seg'] = accuracy(seg_logit, seg_label)

        return loss

