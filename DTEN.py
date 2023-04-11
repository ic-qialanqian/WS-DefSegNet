import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from mmcv.cnn import xavier_init

#from mmdet.core import auto_fp16
#from ..registry import NECKS
#from ..utils import ConvModule
import cv2
from Deformable_DETR.models.deformable_transformer import DeformableTransformerEncoderLayer
from Deformable_DETR.models.position_encoding import build_position_encoding
from Deformable_DETR.models.ops.modules import MSDeformAttn
from Deformable_DETR.models.util.misc import NestedTensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_


#@NECKS.register_module
class DTEN(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 hidden_dim,
                 position_embedding,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None):
        super(DTEN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        assert self.num_ins == 3
        self.num_outs = num_outs
        self.activation = activation
        self.fp16_enabled = False

        #self.p3 = nn.ConvTranspose2d(384, 256, kernel_size=4, stride=2, padding=1)  # 1/4
        self.p3 = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0) # 1/8
        #self.p4 = nn.ConvTranspose2d(384, 256, kernel_size=4, stride=2, padding=1)  # 1/8
        self.p4 = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0) #1/16
        self.p5 = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # 1/16
        # self.p4 = nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=4, padding=0)
        # self.p5 = nn.ConvTranspose2d(2048, 256, kernel_size=4, stride=4, padding=0)

        # self.smooth_p3 = nn.Conv2d(256, 256, 3, 1, 1)
        # self.smooth_p4 = nn.Conv2d(256, 256, 3, 1, 1)

        # self.p3_l2 = L2Norm(256, 10)
        # self.p4_l2 = L2Norm(256, 10)
        # self.p5_l2 = L2Norm(256, 10)

        self.group_norm3 = nn.GroupNorm(32, 256)
        self.group_norm4 = nn.GroupNorm(32, 256)
        self.group_norm5 = nn.GroupNorm(32, 256)

        self.level_embed = nn.Parameter(torch.Tensor(3, 256))
        posargs = PosEncodingArgs(hidden_dim=hidden_dim, position_embedding=position_embedding)
        self.pos_emb = build_position_encoding(posargs)
        self.deformale_encoder = DeformableTransformerEncoderLayer(d_model=256, d_ffn=1024, n_levels=3, n_heads=8,
                                                                   n_points=4)
        #self.deconv = nn.ConvTranspose2d(256, 256, 4, 4)  # nn.ConvTranspose2d(256, 256, 4, 2, 1)
        self.d3 = nn.Sequential(nn.Conv2d(2048, 256, kernel_size=3, padding=1),  nn.PReLU())
        self.d4 = nn.Sequential(nn.Conv2d(2048, 256, kernel_size=3, padding=1),  nn.PReLU())
        self.d5 = nn.Sequential(nn.Conv2d(2048, 256, kernel_size=3, padding=1),  nn.PReLU())
        # self.smooth_feature_map = nn.Conv2d(256, 256, 3, 1, 1)

        self._reset_parameters()
        self.init_weights()

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        normal_(self.level_embed)

    def normalize(self, x):
        x = x - x.min()
        x = x / x.max()
        return x

    def feature_map_visualization(self, x, y):
        x = x[0].detach().cpu().numpy()
        y = y[0].detach().cpu().numpy()
        first = self.normalize(x[0])
        second = self.normalize(y[0])
        cv2.imshow('1', first)
        cv2.imshow('2', second)
        cv2.waitKey(0)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    #@staticmethod
    def get_reference_points(self,spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]  # 奇怪,为啥要对归一化位置做乘法
        return reference_points

    #@auto_fp16()
    def forward(self, inputs, padding_masks):
        assert len(inputs) == len(self.in_channels)
        # self.feature_map_visualization(inputs[0], p3)

        p3 = self.p3(inputs[0])
        p4 = self.p4(inputs[1])
        p5 = self.p5(inputs[2])
        
        

        p3 = self.group_norm3(p3)
        p4 = self.group_norm4(p4)
        p5 = self.group_norm5(p5)



        mask_p3 = F.interpolate(padding_masks[None].float(), size=p3.shape[-2:]).to(torch.bool)[0]
        mask_p4 = F.interpolate(padding_masks[None].float(), size=p4.shape[-2:]).to(torch.bool)[0]
        mask_p5 = F.interpolate(padding_masks[None].float(), size=p5.shape[-2:]).to(torch.bool)[0]
        masks = [mask_p3, mask_p4, mask_p5]

        nest_p3 = NestedTensor(p3, mask_p3)
        nest_p4 = NestedTensor(p4, mask_p4)
        nest_p5 = NestedTensor(p5, mask_p5)

        pos_embeds = []
        pos_embeds.append(self.pos_emb(nest_p3).to(nest_p3.tensors.dtype))
        pos_embeds.append(self.pos_emb(nest_p4).to(nest_p4.tensors.dtype))
        pos_embeds.append(self.pos_emb(nest_p5).to(nest_p5.tensors.dtype))

        srcs = [p3, p4, p5]

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)

        if isinstance(self.deformale_encoder, DeformableTransformerEncoderLayer):
            out = self.deformale_encoder(src_flatten, lvl_pos_embed_flatten, reference_points, spatial_shapes,
                                         level_start_index, mask_flatten)

        level_out = 2
        b, h, w, c, npoints = src_flatten.shape[0], spatial_shapes[level_out][0].item(), spatial_shapes[level_out][1].item(), \
                              src_flatten.shape[-1], src_flatten.shape[1]
        level_start_index = torch.cat((level_start_index, torch.tensor([npoints], device=level_start_index.device)))
        ind = [i for i in range(level_start_index[level_out], level_start_index[level_out+1])]
        output = out[:, ind, :].view(b, h, w, c).permute(0, 3, 1, 2)

        output = F.interpolate(output, inputs[0].size()[2:], mode="bilinear")+ self.d3(inputs[0])
        output = F.interpolate(output, inputs[1].size()[2:], mode="bilinear")+ self.d4(inputs[1])
        output = F.interpolate(output, inputs[2].size()[2:], mode="bilinear")+ self.d5(inputs[2])
        #output = self.deconv(output)
        # output = self.smooth_feature_map(output)
        if isinstance(self.deformale_encoder, DeformableTransformerEncoderLayer):
            #return tuple([output])
            return output


class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out


class PosEncodingArgs():
    def __init__(self, hidden_dim=256, position_embedding='sine'):
        self.hidden_dim = hidden_dim
        self.position_embedding = position_embedding


def generate_mask(x):
    pass

