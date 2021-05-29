# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/openseg-group/openseg.pytorch
# and https://github.com/fundamentalvision/Deformable-DETR
# ------------------------------------------------------------------------------------------------

from collections import OrderedDict
import math
import numpy as np
import copy
import warnings
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.init import normal_

from mmseg.ops import resize
from mmcv.runner import force_fp32
from mmseg.models.builder import build_loss
from mmseg.models.losses import accuracy


from .vision_transformer import VisionTransformer, _cfg

from .ms_deform_attn import MSDeformAttn

import sys
sys.path.append("..")
from builder import SEGMENTORS

## deit
# modified from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
class DistilledVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        
        trunc_normal_(self.pos_embed, std=.02)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
       
        x = x + self.pos_embed
        x = self.pos_drop(x)

        feats = []
        cnt = 1
        for blk in self.blocks:
            
            x = blk(x)
            if cnt%4 == 0:
                feat = x[:,2:,]
                feat = feat.permute(0,2,1).unsqueeze(2).reshape([x[:,2:,].shape[0], self.embed_dim, 30, 30])
                feats.append(feat)
            cnt += 1 

        return feats

    def forward(self, x):
        
        feats = self.forward_features(x)
        return feats

# modified from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
def deit_base_distilled_patch16_384(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        img_size=480, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load('/home/ubuntu/work/deit/models/deit_base_distilled_patch16_384-d0272ac0.pth', map_location='cpu')
        
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
       
        # interpolate position embedding
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model['pos_embed'] = new_pos_embed
        
        model.load_state_dict(checkpoint["model"], strict=False)
    return model


## context path
class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, bsize, h, w):
        #import pdb; pdb.set_trace()
        mask = torch.ones(bsize, h, w).bool().cuda()
        assert mask is not None
        y_embed = mask.cumsum(1, dtype=torch.float32)
        x_embed = mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32).cuda()
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=True, kernel_size=3, stride=2,padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self. bn = nn.SyncBatchNorm(out_channels)
        # self.bn = nn.BatchNorm2d(out_channels)
        self.activation = activation
        if activation:
            self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        if self.activation:
            return self.relu(self.bn(x))
        else:
            return self.bn(x)

class context_path(nn.Module):
    def __init__(self):
        super().__init__()
        self.convblock1 = ConvBlock(in_channels=3, out_channels=64)
        self.convblock2 = ConvBlock(in_channels=64, out_channels=128)
        self.convblock3 = ConvBlock(in_channels=128, out_channels=256, activation=False)

    def forward(self, input):
        x = self.convblock1(input)
        x = self.convblock2(x)
        x = self.convblock3(x)
        return x

## UN-EPT
class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ffn, dropout, activation,
                 n_levels, n_heads, n_points):

        super().__init__()
        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src

class Encoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, device):
        
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / H_
            ref_x = ref_x.reshape(-1)[None] / W_
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        
        reference_points = torch.cat(reference_points_list, 1)
        
        reference_points = reference_points[:, :, None].repeat(1, 1, len(spatial_shapes), 1)

        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, pos):
        output = src

        reference_points = self.get_reference_points(spatial_shapes, device=src.device)

        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index)
        
        return output

class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ffn, dropout, activation, n_levels, n_heads, n_points):
        
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
    
    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    @staticmethod
    def get_reference_points(spatial_shapes, device):
        
        # H, W = spatial_shapes[0]
        H, W = 60, 60

        ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device),
                                        torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device))
        ref_y = ref_y.reshape(-1)[None] / H
        ref_x = ref_x.reshape(-1)[None] / W
        ref = torch.stack((ref_x, ref_y), -1)
        
        reference_points = ref[:, :, None].repeat(1, 1, len(spatial_shapes), 1)

        return reference_points


    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, src, src_spatial_shapes, level_start_index):
        
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        reference_points = self.get_reference_points(src_spatial_shapes, device=src.device)
        # cross attention
        
        tgt = tgt.permute(1, 0, 2)
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos), reference_points, src, src_spatial_shapes, level_start_index)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt.permute(1,0,2)

class Decoder(nn.Module):

    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, tgt, src, src_spatial_shapes, src_level_start_index, query_pos=None):
        output = tgt

        for lid, layer in enumerate(self.layers):
            output = layer(output, query_pos, src, src_spatial_shapes, src_level_start_index)
  
        return output

class EPT(nn.Module):
    def __init__(self, d_model, nhead,
                 num_encoder_layers, num_decoder_layers, dim_feedforward, dropout,
                 activation, num_feature_levels, dec_n_points,  enc_n_points):

        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_feature_levels = num_feature_levels

        encoder_layer = EncoderLayer(d_model, dim_feedforward,
                                        dropout, activation,
                                        num_feature_levels, nhead, enc_n_points)
        self.encoder = Encoder(encoder_layer, num_encoder_layers)

        decoder_layer = DecoderLayer(d_model, dim_feedforward, dropout, activation, num_feature_levels, nhead, dec_n_points)

        self.decoder = Decoder(decoder_layer, num_decoder_layers)

        self.pos_embed = PositionEmbeddingSine(d_model//2, normalize=True)
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        
        normal_(self.level_embed)

    def forward(self, ms_feats, context, query_embed):

        src_flatten = []
        spatial_shapes = []
        lvl_pos_embed_flatten = []
        
        for lvl, src in enumerate(ms_feats):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            pos_embed = self.pos_embed(bs, h, w).flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
        
        src_flatten = torch.cat(src_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))

        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, lvl_pos_embed_flatten)
        # import ipdb; ipdb.set_trace()
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        context = context + query_embed
        out = self.decoder(context, memory, spatial_shapes, level_start_index)
        
        return out

 
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
   
   
@SEGMENTORS.register_module()
class UNEPT(nn.Module):
    def __init__(self, 
                 heads, 
                 feat_dim, 
                 k, 
                 L,
                 dropout,
                 depth,
                 hidden_dim,
                 loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 ignore_index=255,
                 activation="relu",
                 train_cfg=None,
                 test_cfg=None):

        super(UNEPT, self).__init__()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.test_cfg.stride = test_cfg.stride
        self.test_cfg.crop_size = test_cfg.crop_size
        self.num_classes = self.test_cfg.num_classes
        self.ignore_index = ignore_index
        self.align_corners = False
        self.feat_dim = feat_dim

        self.loss_decode = build_loss(loss_decode)
        self.backbone = deit_base_distilled_patch16_384(pretrained=True)
       
        self.cls = nn.Conv2d(feat_dim, self.num_classes, kernel_size=1)

        # get multi-scale feature maps
        self.layers = nn.ModuleList([])
        self.layers.append(nn.Conv2d(768, feat_dim, kernel_size=1, stride=1))
        self.layers.append(nn.Conv2d(768, feat_dim, kernel_size=1, stride=1))
        self.layers.append(nn.Conv2d(768, feat_dim, kernel_size=1, stride=1))
        # self.layers.append(nn.Conv2d(2048, feat_dim, kernel_size=3, stride=2, padding=1))

        self.transformer = EPT(d_model=feat_dim, nhead=heads,
                    num_encoder_layers=depth, num_decoder_layers=depth, dim_feedforward=hidden_dim, dropout=dropout,
                    activation=activation, num_feature_levels=L, dec_n_points=k,  enc_n_points=k)
        self.num_queries = self.test_cfg.num_queries
        self.query_embed = nn.Embedding(self.num_queries, feat_dim)
        self.context_path = context_path()

        self.dir_head = nn.Sequential(
            nn.Conv2d(256,
                      256,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.SyncBatchNorm(256),
            nn.ReLU(),
            nn.Conv2d(256,
                      8,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False))
        
        self.mask_head = nn.Sequential(
            nn.Conv2d(256,
                      256,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.SyncBatchNorm(256),
            nn.ReLU(),
            nn.Conv2d(256,
                      2,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False))


    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.
        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.
        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.
        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        # import ipdb; ipdb.set_trace()
        losses = self(**data_batch)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(data_batch['img'].data))
        return outputs

    @staticmethod
    def _parse_losses(losses):
        """Parse the raw outputs (losses) of the network.
        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.
        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor
                which may be a weighted sum of all losses, log_vars contains
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars


    def encode_decode(self, x):

        
        bsize, c, h, w = x.shape
        backbone_feats = self.backbone(x)
        #import ipdb; ipdb.set_trace()
        context = self.context_path(x)

        mask_map = self.mask_head(context)
        dir_map = self.dir_head(context)
        context = context.flatten(2).permute(2, 0, 1)


        ms_feats = [] 
        for i, conv_layer in enumerate(self.layers):
            feature = conv_layer(backbone_feats[i]) 
            ms_feats.append(feature)
       
        out = self.transformer(ms_feats, context, self.query_embed.weight) # 3600 x b x 256
        # import ipdb; ipdb.set_trace()
        out = out.unsqueeze(0).reshape([60, 60, bsize, self.feat_dim]).permute(2, 3, 0, 1)

        out = self.cls(out)

        seg_logits = resize(
            input=out,
            size=x.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        return seg_logits, mask_map, dir_map


    def distance_to_mask_label(self, distance_map, 
                               seg_label_map, 
                               return_tensor=False):

        if return_tensor:
            assert isinstance(distance_map, torch.Tensor)
            assert isinstance(seg_label_map, torch.Tensor)
        else:
            assert isinstance(distance_map, np.ndarray)
            assert isinstance(seg_label_map, np.ndarray)

        if return_tensor:
            mask_label_map = torch.zeros_like(seg_label_map).long().to(distance_map.device)
        else:
            mask_label_map = np.zeros(seg_label_map.shape, dtype=np.int)

        keep_mask = (distance_map <= 5) & (distance_map >= 0)
        mask_label_map[keep_mask] = 1
        mask_label_map[seg_label_map == -1] = -1

        return mask_label_map

    def calc_weights(self, label_map, num_classes):

        weights = []
        for i in range(num_classes):
            weights.append((label_map == i).sum().data)
        weights = torch.FloatTensor(weights)
        weights_sum = weights.sum()
        return (1 - weights / weights_sum).cuda()  

    def align_angle(self, angle_map, 
                    num_classes=8, 
                    return_tensor=False):

        # if num_classes == 4 and not DTOffsetConfig.c4_align_axis:
        #     return DTOffsetHelper.align_angle_c4(angle_map, return_tensor=return_tensor)

        if return_tensor:
            assert isinstance(angle_map, torch.Tensor)
        else:
            assert isinstance(angle_map, np.ndarray)

        step = 360 / num_classes
        if return_tensor:
            new_angle_map = torch.zeros(angle_map.shape).float().to(angle_map.device)
            angle_index_map = torch.zeros(angle_map.shape).long().to(angle_map.device)
        else:
            new_angle_map = np.zeros(angle_map.shape, dtype=np.float)
            angle_index_map = np.zeros(angle_map.shape, dtype=np.int)
        mask = (angle_map <= (-180 + step/2)) | (angle_map > (180 - step/2))
        new_angle_map[mask] = -180
        angle_index_map[mask] = 0

        for i in range(1, num_classes):
            middle = -180 + step * i
            mask = (angle_map > (middle - step / 2)) & (angle_map <= (middle + step / 2))
            new_angle_map[mask] = middle
            angle_index_map[mask] = i

        return new_angle_map, angle_index_map

    def angle_to_direction_label(self, angle_map, 
                                 seg_label_map=None, 
                                 distance_map=None, 
                                 num_classes=8, 
                                 extra_ignore_mask=None, 
                                 return_tensor=False):

        if return_tensor:
            assert isinstance(angle_map, torch.Tensor)
            assert isinstance(seg_label_map, torch.Tensor) or seg_label_map is None
        else:
            assert isinstance(angle_map, np.ndarray)
            assert isinstance(seg_label_map, np.ndarray) or seg_label_map is None

        _, label_map = self.align_angle(angle_map, 
                                                  num_classes=num_classes, 
                                                  return_tensor=return_tensor)
        if distance_map is not None:
            label_map[distance_map > 5] = num_classes
        if seg_label_map is None:
            if return_tensor:
                ignore_mask = torch.zeros(angle_map.shape, dtype=torch.uint8).to(angle_map.device)
            else:
                ignore_mask = np.zeros(angle_map.shape, dtype=np.bool)
        else:
            ignore_mask = seg_label_map == -1
        # import ipdb; ipdb.set_trace()

        if extra_ignore_mask is not None:
            extra_ignore_mask = extra_ignore_mask.unsqueeze(1)
            ignore_mask = ignore_mask | extra_ignore_mask
        label_map[ignore_mask] = -1

        return label_map

    def forward_train(self, img, img_metas, gt_semantic_seg, distance_map, angle_map):
        # import ipdb; ipdb.set_trace()
        seg_logits, pred_mask, pred_direction = self.encode_decode(img)
        
        losses = dict()
        
        loss_decode = self.losses(seg_logits, pred_mask, pred_direction, gt_semantic_seg, distance_map, angle_map)
        losses.update(loss_decode)

        return losses
    
    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, pred_mask, pred_direction, seg_label, distance_map, angle_map):
        """Compute segmentation loss."""
        loss = dict()
        
        seg_weight = None

        gt_mask = self.distance_to_mask_label(distance_map, seg_label, return_tensor=True)
        gt_size = gt_mask.shape[2:]
        mask_weights = self.calc_weights(gt_mask, 2)
        
        pred_direction = F.interpolate(pred_direction, size=gt_size, mode="bilinear", align_corners=True)
        pred_mask = F.interpolate(pred_mask, size=gt_size, mode="bilinear", align_corners=True)
        mask_loss = F.cross_entropy(pred_mask, gt_mask[:,0], weight=mask_weights, ignore_index=-1)
        
        mask_threshold = 0.5
        binary_pred_mask = torch.softmax(pred_mask, dim=1)[:, 1, :, :] > mask_threshold
        
        gt_direction = self.angle_to_direction_label(
            angle_map,
            seg_label_map=seg_label,
            extra_ignore_mask=(binary_pred_mask == 0),
            return_tensor=True
        )
        
        direction_loss_mask = gt_direction != -1
        direction_weights = self.calc_weights(gt_direction[direction_loss_mask], pred_direction.size(1))
        direction_loss = F.cross_entropy(pred_direction, gt_direction[:,0], weight=direction_weights, ignore_index=-1)
        # import ipdb; ipdb.set_trace()

        offset = self._get_offset(pred_mask, pred_direction)
        
        # seg_map = seg_logit.argmax(dim=1)

        refine_map = self.shift(seg_logit, offset.permute(0,3,1,2))

        # out_0 = torch.from_numpy(out_0).unsqueeze(0).cuda()
        # out_1 = torch.from_numpy(out_1).unsqueeze(0).cuda()
        # refine_map = torch.cat((out_0,out_1), dim=0)

        # import ipdb; ipdb.set_trace()

        seg_label = seg_label.squeeze(1)
        loss['loss_seg'] = 0.8*self.loss_decode(
            seg_logit,
            seg_label,
            weight=seg_weight,
            ignore_index=self.ignore_index) + 5*mask_loss + 0.6*direction_loss + \
            self.loss_decode(
            refine_map,
            seg_label,
            weight=seg_weight,
            ignore_index=self.ignore_index)

        
        loss['acc_seg'] = accuracy(seg_logit, seg_label)

        return loss

    def shift(self, x, offset):
        """
        x: b x c x h x w
        offset: b x 2 x h x w
        """
        def gen_coord_map(H, W):
            coord_vecs = [torch.arange(length, dtype=torch.float) for length in (H, W)]
            coord_h, coord_w = torch.meshgrid(coord_vecs)
            coord_h = coord_h.cuda()
            coord_w = coord_w.cuda()
            return coord_h, coord_w
        
        b, c, h, w = x.shape
        

        coord_map = gen_coord_map(h, w)
        norm_factor = torch.FloatTensor([(w-1)/2, (h-1)/2]).cuda()
        grid_h = offset[:, 0]+coord_map[0]
        grid_w = offset[:, 1]+coord_map[1]
        grid = torch.stack([grid_w, grid_h], dim=-1) / norm_factor - 1

        x = F.grid_sample(x.float(), grid, padding_mode='border', mode='bilinear', align_corners=True)
       
        return x

    def _get_offset(self, mask_logits, dir_logits):
       
        edge_mask = mask_logits[:, 1] > 0.5
        dir_logits = torch.softmax(dir_logits, dim=1)
        n, _, h, w = dir_logits.shape

        keep_mask = edge_mask

        dir_label = torch.argmax(dir_logits, dim=1).float()
        offset = self.label_to_vector(dir_label)
        offset = offset.permute(0, 2, 3, 1)
        offset[~keep_mask, :] = 0
        
        return offset
    
    def label_to_vector(self, labelmap, 
                        num_classes=8):

        assert isinstance(labelmap, torch.Tensor)

        label_to_vector_mapping = {
            # 4: [
            #     [-1, -1], [-1, 1], [1, 1], [1, -1]
            # ] if not DTOffsetConfig.c4_align_axis else [
            #     [0, -1], [-1, 0], [0, 1], [1, 0]
            # ],    
            8: [
                [0, -1], [-1, -1], [-1, 0], [-1, 1],
                [0, 1], [1, 1], [1, 0], [1, -1]
            ],
            16: [
                [0, -2], [-1, -2], [-2, -2], [-2, -1], 
                [-2, 0], [-2, 1], [-2, 2], [-1, 2],
                [0, 2], [1, 2], [2, 2], [2, 1],
                [2, 0], [2, -1], [2, -2], [1, -2]
            ],
            32: [
                [0, -4], [-1, -4], [-2, -4], [-3, -4], [-4, -4], [-4, -3], [-4, -2], [-4, -1],
                [-4, 0], [-4, 1], [-4, 2], [-4, 3], [-4, 4], [-3, 4], [-2, 4], [-1, 4],
                [0, 4], [1, 4], [2, 4], [3, 4], [4, 4], [4, 3], [4, 2], [4, 1],
                [4, 0], [4, -1], [4, -2], [4, -3], [4, -4], [3, -4], [2, -4], [1, -4],
            ]
        }

        mapping = label_to_vector_mapping[num_classes]
        offset_h = torch.zeros_like(labelmap).long()
        offset_w = torch.zeros_like(labelmap).long()

        for idx, (hdir, wdir) in enumerate(mapping):
            mask = labelmap == idx
            offset_h[mask] = hdir
            offset_w[mask] = wdir

        return torch.stack([offset_h, offset_w], dim=-1).permute(0, 3, 1, 2).to(labelmap.device)



    def forward(self, img, img_metas, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)

    def forward_test(self, imgs, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got '
                                f'{type(var)}')

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) != '
                             f'num of image meta ({len(img_metas)})')
        # all images in the same aug batch all of the same ori_shape and pad
        # shape
        for img_meta in img_metas:
            ori_shapes = [_['ori_shape'] for _ in img_meta]
            assert all(shape == ori_shapes[0] for shape in ori_shapes)
            img_shapes = [_['img_shape'] for _ in img_meta]
            assert all(shape == img_shapes[0] for shape in img_shapes)
            pad_shapes = [_['pad_shape'] for _ in img_meta]
            assert all(shape == pad_shapes[0] for shape in pad_shapes)

        if num_augs == 1:
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            return self.aug_test(imgs, img_metas, **kwargs)

    def slide_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap."""

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                pad_img = crop_img.new_zeros(
                    (crop_img.size(0), crop_img.size(1), h_crop, w_crop))
                pad_img[:, :, :y2 - y1, :x2 - x1] = crop_img
                pad_seg_logit, _, _  = self.encode_decode(pad_img)
                preds[:, :, y1:y2,
                      x1:x2] += pad_seg_logit[:, :, :y2 - y1, :x2 - x1]
                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        preds = preds / count_mat
        if rescale:
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)

        return preds

    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image."""

        seg_logit = self.encode_decode(img)
        if rescale:
            seg_logit = resize(
                seg_logit,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)

        return seg_logit

    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.
        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.
        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale)
        output = F.softmax(seg_logit, dim=1)
        flip = img_meta[0]['flip']
        flip_direction = img_meta[0]['flip_direction']
        if flip:
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2, ))

        return output

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, rescale)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.
        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)
        
    

