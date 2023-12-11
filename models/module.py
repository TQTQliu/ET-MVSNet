import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib
import math
import numpy as np
from .transformer import Transformer


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
            import math
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        '''
        Input:
        x: (b, h, w)
        not_mask: position to embed 
        '''
        not_mask = (x >= 0)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2) # b  2*num_pos_feats  h  w
        return pos 



def homo_warping(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, Ndepth] o [B, Ndepth, H, W]
    # out: [B, C, Ndepth, H, W]
    C = src_fea.shape[1]
    Hs,Ws = src_fea.shape[-2:]
    B,num_depth,Hr,Wr = depth_values.shape

    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([torch.arange(0, Hr, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, Wr, dtype=torch.float32, device=src_fea.device)])
        y = y.reshape(Hr*Wr)
        x = x.reshape(Hr*Wr)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(B, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.reshape(B, 1, num_depth, -1)  # [B, 3, Ndepth, H*W]
        proj_xyz = rot_depth_xyz + trans.reshape(B, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        # FIXME divide 0
        temp = proj_xyz[:, 2:3, :, :]
        temp[temp==0] = 1e-9
        proj_xy = proj_xyz[:, :2, :, :] / temp  # [B, 2, Ndepth, H*W]
        # proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]

        proj_x_normalized = proj_xy[:, 0, :, :] / ((Ws - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :, :] / ((Hs - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy
    if len(src_fea.shape)==4:
        warped_src_fea = F.grid_sample(src_fea, grid.reshape(B, num_depth * Hr, Wr, 2), mode='bilinear', padding_mode='zeros', align_corners=True)
        warped_src_fea = warped_src_fea.reshape(B, C, num_depth, Hr, Wr)
    elif len(src_fea.shape)==5:
        warped_src_fea = []
        for d in range(src_fea.shape[2]):
            warped_src_fea.append(F.grid_sample(src_fea[:,:,d], grid.reshape(B, num_depth, Hr, Wr, 2)[:,d], mode='bilinear', padding_mode='zeros', align_corners=True))
        warped_src_fea = torch.stack(warped_src_fea, dim=2)

    return warped_src_fea

def init_range(cur_depth, ndepths, device, dtype, H, W):
    cur_depth_min = cur_depth[:, 0]  # (B,)
    cur_depth_max = cur_depth[:, -1]
    new_interval = (cur_depth_max - cur_depth_min) / (ndepths - 1)  # (B, )
    new_interval = new_interval[:, None, None]  # B H W
    depth_range_samples = cur_depth_min.unsqueeze(1) + (torch.arange(0, ndepths, device=device, dtype=dtype,
                                                                requires_grad=False).reshape(1, -1) * new_interval.squeeze(1)) #(B, D)
    depth_range_samples = depth_range_samples.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W) #(B, D, H, W)
    return depth_range_samples

def init_inverse_range(cur_depth, ndepths, device, dtype, H, W):
    inverse_depth_min = 1. / cur_depth[:, 0]  # (B,)
    inverse_depth_max = 1. / cur_depth[:, -1]
    itv = torch.arange(0, ndepths, device=device, dtype=dtype, requires_grad=False).reshape(1, -1,1,1).repeat(1, 1, H, W)  / (ndepths - 1)  # 1 D H W
    inverse_depth_hypo = inverse_depth_max[:,None, None, None] + (inverse_depth_min - inverse_depth_max)[:,None, None, None] * itv

    return 1./inverse_depth_hypo

def schedule_inverse_range(inverse_min_depth, inverse_max_depth, ndepths, H, W):
    #cur_depth_min, (B, H, W)
    #cur_depth_max: (B, H, W)
    itv = torch.arange(0, ndepths, device=inverse_min_depth.device, dtype=inverse_min_depth.dtype, requires_grad=False).reshape(1, -1,1,1).repeat(1, 1, H//2, W//2)  / (ndepths - 1)  # 1 D H W

    inverse_depth_hypo = inverse_max_depth[:,None, :, :] + (inverse_min_depth - inverse_max_depth)[:,None, :, :] * itv  # B D H W
    inverse_depth_hypo = F.interpolate(inverse_depth_hypo.unsqueeze(1), [ndepths, H, W], mode='trilinear', align_corners=True).squeeze(1)
    return 1./inverse_depth_hypo

def schedule_range(cur_depth, ndepth, depth_inteval_pixel, H, W):
    #shape, (B, H, W)
    #cur_depth: (B, H, W)
    #return depth_range_values: (B, D, H, W)
    cur_depth_min = (cur_depth - ndepth / 2 * depth_inteval_pixel[:,None,None])  # (B, H, W)
    cur_depth_max = (cur_depth + ndepth / 2 * depth_inteval_pixel[:,None,None])
    new_interval = (cur_depth_max - cur_depth_min) / (ndepth - 1)  # (B, H, W)

    depth_range_samples = cur_depth_min.unsqueeze(1) + (torch.arange(0, ndepth, device=cur_depth.device, dtype=cur_depth.dtype,
                                                                  requires_grad=False).reshape(1, -1, 1, 1) * new_interval.unsqueeze(1))
    depth_range_samples = F.interpolate(depth_range_samples.unsqueeze(1), [ndepth, H, W], mode='trilinear', align_corners=True).squeeze(1)
    return depth_range_samples

def init_bn(module):
    if module.weight is not None:
        nn.init.ones_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return

def init_uniform(module, init_method):
    if module.weight is not None:
        if init_method == "kaiming":
            nn.init.kaiming_uniform_(module.weight)
        elif init_method == "xavier":
            nn.init.xavier_uniform_(module.weight)
    return

class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)

class ConvBnReLU3D_CAM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU3D_CAM, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.linear_agg = nn.Sequential(
            nn.Linear(out_channels, out_channels//2),
            nn.ReLU(),
            nn.Linear(out_channels//2, out_channels)
        )

    def forward(self, input):
        x = self.conv(input)
        B,C,D,H,W = x.shape
        avg_attn = self.linear_agg(x.reshape(B,C,D*H*W).mean(2))
        max_attn = self.linear_agg(x.reshape(B,C,D*H*W).max(2)[0])  # B C
        attn = F.sigmoid(max_attn+avg_attn)[:,:,None,None,None]  # B C,1,1,1
        x = x * attn
        return F.relu(self.bn(x+input), inplace=True)

class ConvBnReLU3D_DCAM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU3D_DCAM, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.linear_agg = nn.Sequential(
            nn.Linear(out_channels, out_channels//2),
            nn.ReLU(),
            nn.Linear(out_channels//2, out_channels)
        )

    def forward(self, input):
        x = self.conv(input)
        B,C,D,H,W = x.shape
        avg_attn = self.linear_agg(x.reshape(B,C,D,H*W).mean(3).permute(0,2,1).reshape(B*D,C)).reshape(B,D,C).permute(0,2,1)
        max_attn = self.linear_agg(x.reshape(B,C,D,H*W).max(3)[0].permute(0,2,1).reshape(B*D,C)).reshape(B,D,C).permute(0,2,1)  # B C D
        attn = F.sigmoid(max_attn+avg_attn)[:,:,:,None,None]  # B C,D,1,1
        x = x * attn
        return F.relu(self.bn(x+input), inplace=True)

class ConvBnReLU3D_PAM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU3D_PAM, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.pixel_conv = nn.Conv2d(2,1,7,stride=1,padding='same')

    def forward(self, input):
        x = self.conv(input)
        B,C,D,H,W = x.shape
        max_attn = x.reshape(B,C*D,H,W).max(1, keepdim=True)[0]
        avg_attn = x.reshape(B,C*D,H,W).mean(1, keepdim=True)  # B 1 H W
        attn = F.sigmoid(self.pixel_conv(torch.cat([max_attn, avg_attn], dim=1)))[:,:,None,:,:]  # B 1,1,H,W
        x = x * attn
        return F.relu(self.bn(x+input), inplace=True)

class ConvBnReLU3D_PDAM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU3D_PDAM, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.spatial_conv = nn.Conv3d(2,1,7,stride=1,padding='same')

    def forward(self, input):
        x = self.conv(input)
        B,C,D,H,W = x.shape
        max_attn = x.max(1, keepdim=True)[0]
        avg_attn = x.mean(1, keepdim=True)  # B 1 D H W
        attn = F.sigmoid(self.spatial_conv(torch.cat([max_attn, avg_attn], dim=1)))  # B 1,D,H,W
        x = x * attn
        return F.relu(self.bn(x+input), inplace=True)

class Deconv3d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Deconv3d, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

    def forward(self, x):
        y = self.conv(x)
        if self.bn is not None:
            x = self.bn(y)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)

class Conv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn_momentum=0.1, init_method="xavier", gn=False, group_channel=8, **kwargs):
        super(Conv2d, self).__init__()
        bn = not gn
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.gn = nn.GroupNorm(int(max(1, out_channels / group_channel)), out_channels) if gn else None
        self.relu = relu

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        else:
            x = self.gn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)

class Deconv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Deconv2d, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu


class reg2d(nn.Module):
    def __init__(self, input_channel=128, base_channel=32, conv_name='ConvBnReLU3D'):
        super(reg2d, self).__init__()
        module = importlib.import_module("models.module")
        stride_conv_name = 'ConvBnReLU3D'
        self.conv0 = getattr(module, stride_conv_name)(input_channel, base_channel, kernel_size=(1,3,3), pad=(0,1,1))
        self.conv1 = getattr(module, stride_conv_name)(base_channel, base_channel*2, kernel_size=(1,3,3), stride=(1,2,2), pad=(0,1,1))
        self.conv2 = getattr(module, conv_name)(base_channel*2, base_channel*2)

        self.conv3 = getattr(module, stride_conv_name)(base_channel*2, base_channel*4, kernel_size=(1,3,3), stride=(1,2,2), pad=(0,1,1))
        self.conv4 = getattr(module, conv_name)(base_channel*4, base_channel*4)

        self.conv5 = getattr(module, stride_conv_name)(base_channel*4, base_channel*8, kernel_size=(1,3,3), stride=(1,2,2), pad=(0,1,1))
        self.conv6 = getattr(module, conv_name)(base_channel*8, base_channel*8)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(base_channel*8, base_channel*4, kernel_size=(1,3,3), padding=(0,1,1), output_padding=(0,1,1), stride=(1,2,2), bias=False),
            nn.BatchNorm3d(base_channel*4),
            nn.ReLU(inplace=True))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(base_channel*4, base_channel*2, kernel_size=(1,3,3), padding=(0,1,1), output_padding=(0,1,1), stride=(1,2,2), bias=False),
            nn.BatchNorm3d(base_channel*2),
            nn.ReLU(inplace=True))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(base_channel*2, base_channel, kernel_size=(1,3,3), padding=(0,1,1), output_padding=(0,1,1), stride=(1,2,2), bias=False),
            nn.BatchNorm3d(base_channel),
            nn.ReLU(inplace=True))

        self.prob = nn.Conv3d(8, 1, 1, stride=1, padding=0)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)
        x = conv0 + self.conv11(x)
        x = self.prob(x)

        return x.squeeze(1)

class reg3d(nn.Module):
    def __init__(self, in_channels, base_channels, down_size=3):
        super(reg3d, self).__init__()
        self.down_size = down_size
        self.conv0 = ConvBnReLU3D(in_channels, base_channels, kernel_size=3, pad=1)
        self.conv1 = ConvBnReLU3D(base_channels, base_channels*2, kernel_size=3, stride=2, pad=1)
        self.conv2 = ConvBnReLU3D(base_channels*2, base_channels*2)
        if down_size >= 2:
            self.conv3 = ConvBnReLU3D(base_channels*2, base_channels*4, kernel_size=3, stride=2, pad=1)
            self.conv4 = ConvBnReLU3D(base_channels*4, base_channels*4)
        if down_size >= 3:
            self.conv5 = ConvBnReLU3D(base_channels*4, base_channels*8, kernel_size=3, stride=2, pad=1)
            self.conv6 = ConvBnReLU3D(base_channels*8, base_channels*8)
            self.conv7 = nn.Sequential(
                nn.ConvTranspose3d(base_channels*8, base_channels*4, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
                nn.BatchNorm3d(base_channels*4),
                nn.ReLU(inplace=True))
        if down_size >= 2:
            self.conv9 = nn.Sequential(
                nn.ConvTranspose3d(base_channels*4, base_channels*2, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
                nn.BatchNorm3d(base_channels*2),
                nn.ReLU(inplace=True))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(base_channels*2, base_channels, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True))
        self.prob = nn.Conv3d(base_channels, 1, 3, stride=1, padding=1, bias=False)

    def forward(self, x):
        if self.down_size==3:
            conv0 = self.conv0(x)
            conv2 = self.conv2(self.conv1(conv0))
            conv4 = self.conv4(self.conv3(conv2))
            x = self.conv6(self.conv5(conv4))
            x = conv4 + self.conv7(x)
            x = conv2 + self.conv9(x)
            x = conv0 + self.conv11(x)
            x = self.prob(x)
        elif self.down_size==2:
            conv0 = self.conv0(x)
            conv2 = self.conv2(self.conv1(conv0))
            x = self.conv4(self.conv3(conv2))
            x = conv2 + self.conv9(x)
            x = conv0 + self.conv11(x)
            x = self.prob(x)
        else:
            conv0 = self.conv0(x)
            x = self.conv2(self.conv1(conv0))
            x = conv0 + self.conv11(x)
            x = self.prob(x)
        return x.squeeze(1)  # B D H W


# feature map --> epipolar sequence
def map2seq(ref_flag,src_flag,ref,src,pos):
    ref = ref.data.cpu().numpy()
    src = src.data.cpu().numpy()
    pos = pos.data.cpu().numpy()
    ref_flag = ref_flag.data.cpu().numpy().astype(np.int32)
    src_flag = src_flag.data.cpu().numpy().astype(np.int32)
    value = np.intersect1d(ref_flag[ref_flag!=0],src_flag[src_flag!=0])
    if len(value) <= 1:
        return None,None,None,None,None,None,None
    ref_ls, src_ls = [], []
    ref_pos_ls, src_pos_ls = [], []
    ref_m, src_m = 0, 0
    ref_idx_ori = np.empty([0,2])
    ref_idx_epi = np.empty([0,2])
    src_idx_ori = np.empty([0,2])
    src_idx_epi = np.empty([0,2])
        
    for i,v in enumerate(value):
        idx_ref_y,idx_ref_x = np.where(ref_flag==v)
        ref_ls.append(ref[:,idx_ref_y,idx_ref_x])
        ref_pos_ls.append(pos[:,idx_ref_y,idx_ref_x])
        ref_m = max(ref_m,len(idx_ref_y))
        yx = np.stack((idx_ref_y,idx_ref_x),axis=1)
        ref_idx_ori = np.concatenate((ref_idx_ori,yx))
        y = i*np.ones(len(idx_ref_y))
        x = np.arange(len(idx_ref_y))
        yx = np.stack((y,x),axis=1)
        ref_idx_epi = np.concatenate((ref_idx_epi,yx))

        idx_src_y,idx_src_x = np.where(src_flag==v)
        src_ls.append(src[:,idx_src_y,idx_src_x])
        src_pos_ls.append(pos[:,idx_src_y,idx_src_x])
        src_m = max(src_m,len(idx_src_y))
        yx = np.stack((idx_src_y,idx_src_x),axis=1)
        src_idx_ori = np.concatenate((src_idx_ori,yx))
        y = i*np.ones(len(idx_src_y))
        x = np.arange(len(idx_src_y))
        yx = np.stack((y,x),axis=1)
        src_idx_epi = np.concatenate((src_idx_epi,yx))

    C = ref.shape[0]
    ref_epipolar = np.zeros((C,len(ref_ls),ref_m))
    ref_pos = np.zeros((C,len(ref_ls),ref_m))
    ref_mask = np.ones((len(ref_ls),ref_m))
    src_epipolar = np.zeros((C,len(src_ls),src_m))
    src_pos = np.zeros((C,len(src_ls),src_m))
    src_mask = np.ones((len(src_ls),src_m))
    
    for i,v in enumerate(value):
        ref_epipolar[:,i,:ref_ls[i].shape[-1]] = ref_ls[i]
        ref_pos[:,i,:ref_pos_ls[i].shape[-1]] = ref_pos_ls[i]
        ref_mask[i,:ref_ls[i].shape[-1]] = 0

        src_epipolar[:,i,:src_ls[i].shape[-1]] = src_ls[i]
        src_pos[:,i,:src_pos_ls[i].shape[-1]] = src_pos_ls[i]
        src_mask[i,:src_ls[i].shape[-1]] = 0
    
    return ref_epipolar,ref_pos,ref_mask,src_epipolar,src_pos,src_mask,(ref_idx_ori,ref_idx_epi,src_idx_ori,src_idx_epi)


def seq2map(epipolar_seq, idx_back, size):
    device = epipolar_seq.device
    feature_map = torch.zeros(size).to(device)
    idx_map = torch.from_numpy(idx_back[0]).type(torch.long)
    idx_seq = torch.from_numpy(idx_back[1]).type(torch.long)
    feature_map[:,idx_map[:,0],idx_map[:,1]] = epipolar_seq[:,idx_seq[:,0],idx_seq[:,1]]
    
    return feature_map


class FPN4(nn.Module):
    """
    FPN aligncorners downsample 4x"""
    def __init__(self, base_channels, gn=False):
        super(FPN4, self).__init__()
        self.base_channels = base_channels

        self.conv0 = nn.Sequential(
            Conv2d(3, base_channels, 3, 1, padding=1, gn=gn),
            Conv2d(base_channels, base_channels, 3, 1, padding=1, gn=gn),
        )

        self.conv1 = nn.Sequential(
            Conv2d(base_channels, base_channels * 2, 5, stride=2, padding=2, gn=gn),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1, gn=gn),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1, gn=gn),
        )

        self.conv2 = nn.Sequential(
            Conv2d(base_channels * 2, base_channels * 4, 5, stride=2, padding=2, gn=gn),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1, gn=gn),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1, gn=gn),
        )

        self.conv3 = nn.Sequential(
            Conv2d(base_channels * 4, base_channels * 8, 5, stride=2, padding=2, gn=gn),
            Conv2d(base_channels * 8, base_channels * 8, 3, 1, padding=1, gn=gn),
            Conv2d(base_channels * 8, base_channels * 8, 3, 1, padding=1, gn=gn),
        )

        self.out_channels = [8 * base_channels]
        final_chs = base_channels * 8
        
        self.pos_enc = PositionEmbeddingSine(num_pos_feats=final_chs//2)
        self.epipolar_encoder = Transformer(d_model=final_chs, nhead=4, num_encoder_layers=1, dim_feedforward=final_chs*4, dropout=0.1,
                    activation="relu")

        self.inner1 = nn.Conv2d(base_channels * 4, final_chs, 1, bias=True)
        self.inner2 = nn.Conv2d(base_channels * 2, final_chs, 1, bias=True)
        self.inner3 = nn.Conv2d(base_channels * 1, final_chs, 1, bias=True)

        self.out1 = nn.Conv2d(final_chs, base_channels * 8, 1, bias=False)
        self.out2 = nn.Conv2d(final_chs, base_channels * 4, 3, padding=1, bias=False)
        self.out3 = nn.Conv2d(final_chs, base_channels * 2, 3, padding=1, bias=False)
        self.out4 = nn.Conv2d(final_chs, base_channels, 3, padding=1, bias=False)

        self.la = nn.Conv2d(final_chs, final_chs, 3, bias=False,padding=1)
        self.res = nn.Conv2d(final_chs, final_chs, 1, bias=False)

        self.out_channels.append(base_channels * 4)
        self.out_channels.append(base_channels * 2)
        self.out_channels.append(base_channels)


    def forward(self, imgs,flag):
        ref_img, src_imgs = imgs[0], imgs[1:]
        device = ref_img.device

        ref_outputs, src_outputs = [],[]

        ref_conv0 = self.conv0(ref_img)
        ref_conv1 = self.conv1(ref_conv0)
        ref_conv2 = self.conv2(ref_conv1)
        ref_conv3 = self.conv3(ref_conv2)
        B,C,H,W = ref_conv3.shape

        ref_intra = ref_conv3.clone()
        intra = ref_conv3
        ref_out1 = self.out1(intra)
        intra = F.interpolate(intra, scale_factor=2, mode="bilinear", align_corners=True) + self.inner1(ref_conv2)
        ref_out2 = self.out2(intra)
        intra = F.interpolate(intra, scale_factor=2, mode="bilinear", align_corners=True) + self.inner2(ref_conv1)
        ref_out3 = self.out3(intra)
        intra = F.interpolate(intra, scale_factor=2, mode="bilinear", align_corners=True) + self.inner3(ref_conv0)
        ref_out4 = self.out4(intra)

        ref_outputs = {}
        ref_outputs["stage1"] = ref_out1
        ref_outputs["stage2"] = ref_out2
        ref_outputs["stage3"] = ref_out3
        ref_outputs["stage4"] = ref_out4

        pos = torch.ones((1,H,W))
        pos = self.pos_enc(pos).squeeze(0).to(device) # CHW

        for src_idx, src_img in enumerate(src_imgs):
            src_out ={}
            src_conv0 = self.conv0(src_img)
            src_conv1 = self.conv1(src_conv0)
            src_conv2 = self.conv2(src_conv1)
            src_conv3 = self.conv3(src_conv2)
            src_intra = src_conv3.clone()

            for b in range(B):
                ref_flag, src_flag = flag[b,src_idx,0,:,:], flag[b,src_idx,1,:,:]

                ref_epipolar, ref_pos, ref_mask, src_epipolar, src_pos, src_mask, idx_back = map2seq(ref_flag, src_flag, ref_intra[b], src_intra[b], pos)
                if idx_back is None:
                    continue
                ref_epipolar = torch.from_numpy(ref_epipolar).to(device).permute(1,2,0).type(torch.float32)
                ref_pos = torch.from_numpy(ref_pos).to(device).permute(1,2,0).type(torch.float32)
                ref_mask = torch.from_numpy(ref_mask).to(device).type(torch.bool)
                src_epipolar = torch.from_numpy(src_epipolar).to(device).permute(1,2,0).type(torch.float32)
                src_pos = torch.from_numpy(src_pos).to(device).permute(1,2,0).type(torch.float32)
                src_mask = torch.from_numpy(src_mask).to(device).type(torch.bool)


                ref_epipolar, src_epipolar = self.epipolar_encoder(ref=ref_epipolar, src=src_epipolar, mask_ref=ref_mask, mask_src=src_mask, pos_ref=ref_pos, pos_src=src_pos)
                ref_epipolar = ref_epipolar.permute(2,1,0)
                src_epipolar = src_epipolar.permute(2,1,0)

                src_enhanced_map = seq2map(src_epipolar,idx_back[-2:],size=(C,H,W))
                src_enhanced_map_la = self.la(src_enhanced_map)
                src_enhanced_map = torch.where((torch.sum(abs(src_enhanced_map),dim=0)==0).squeeze(0).repeat(C,1,1),src_enhanced_map_la,src_enhanced_map)
                src_intra[b] = src_intra[b] + self.res(src_enhanced_map)

 
            src_out1 = self.out1(src_intra)
            src_intra = F.interpolate(src_intra, scale_factor=2, mode="bilinear", align_corners=True) + self.inner1(src_conv2)
            src_out2 = self.out2(src_intra)
            src_intra = F.interpolate(src_intra, scale_factor=2, mode="bilinear", align_corners=True) + self.inner2(src_conv1)
            src_out3 = self.out3(src_intra)
            src_intra = F.interpolate(src_intra, scale_factor=2, mode="bilinear", align_corners=True) + self.inner3(src_conv0)
            src_out4 = self.out4(src_intra)
            
            src_out["stage1"] = src_out1
            src_out["stage2"] = src_out2
            src_out["stage3"] = src_out3
            src_out["stage4"] = src_out4
            src_outputs.append(src_out)

        return ref_outputs, src_outputs


class stagenet(nn.Module):
    def __init__(self, inverse_depth=False, attn_fuse_d=True, attn_temp=2):
        super(stagenet, self).__init__()
        self.inverse_depth = inverse_depth
        self.attn_fuse_d = attn_fuse_d
        self.attn_temp = attn_temp

    def forward(self, ref_feature, src_features, proj_matrices, depth_hypo, regnet, stage_idx, group_cor=False, group_cor_dim=8, split_itv=1):

        # step 1. feature extraction
        proj_matrices = torch.unbind(proj_matrices, 1)
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]
        B,D,H,W = depth_hypo.shape
        C = ref_feature.shape[1]

        cor_weight_sum = 1e-8
        cor_feats = 0
        ref_volume =  ref_feature.unsqueeze(2).repeat(1, 1, D, 1, 1)
        ref_proj_new = ref_proj[:, 0].clone()
        ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])

        # step 2. Epipolar Transformer Aggregation
        for src_idx, (src_fea, src_proj) in enumerate(zip(src_features, src_projs)):
            src_proj_new = src_proj[:, 0].clone()
            src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])
            warped_src = homo_warping(src_fea, src_proj_new, ref_proj_new, depth_hypo)  # B C D H W
            if group_cor:
                warped_src = warped_src.reshape(B, group_cor_dim, C//group_cor_dim, D, H, W)
                ref_volume = ref_volume.reshape(B, group_cor_dim, C//group_cor_dim, D, H, W)
                cor_feat = (warped_src * ref_volume).mean(2)  # B G D H W
            else:
                cor_feat = (ref_volume - warped_src)**2 # B C D H W 
            del warped_src, src_proj, src_fea

            if not self.attn_fuse_d:
                cor_weight = torch.softmax(cor_feat.sum(1), 1).max(1)[0]  # B H W
                cor_weight_sum += cor_weight  # B H W
                cor_feats += cor_weight.unsqueeze(1).unsqueeze(1) * cor_feat  # B C D H W
            else:
                cor_weight = torch.softmax(cor_feat.sum(1) / self.attn_temp, 1) / math.sqrt(C)  # B D H W
                cor_weight_sum += cor_weight  # B D H W
                cor_feats += cor_weight.unsqueeze(1) * cor_feat  # B C D H W
            del cor_weight, cor_feat
        if not self.attn_fuse_d:
            cor_feats = cor_feats / cor_weight_sum.unsqueeze(1).unsqueeze(1)  # B C D H W
        else:
            cor_feats = cor_feats / cor_weight_sum.unsqueeze(1)  # B C D H W

        del cor_weight_sum, src_features
        
    
        # step 3. regularization
        attn_weight = regnet(cor_feats)  # B D H W
        del cor_feats
        attn_weight = F.softmax(attn_weight, dim=1)  # B D H W

        # step 4. depth argmax
        attn_max_indices = attn_weight.max(1, keepdim=True)[1]  # B 1 H W
        depth = torch.gather(depth_hypo, 1, attn_max_indices).squeeze(1)  # B H W

        if not self.training:
            with torch.no_grad():
                photometric_confidence = attn_weight.max(1)[0]  # B H W
                photometric_confidence = F.interpolate(photometric_confidence.unsqueeze(1), scale_factor=2**(3-stage_idx), mode='bilinear', align_corners=True).squeeze(1)
        else:
            photometric_confidence = torch.tensor(0.0, dtype=torch.float32, device=ref_feature.device, requires_grad=False)
        
        ret_dict = {"depth": depth,  "photometric_confidence": photometric_confidence, "hypo_depth": depth_hypo, "attn_weight": attn_weight}
        
        if self.inverse_depth:
            last_depth_itv = 1./depth_hypo[:,2,:,:] - 1./depth_hypo[:,1,:,:]
            inverse_min_depth = 1/depth + split_itv * last_depth_itv  # B H W
            inverse_max_depth = 1/depth - split_itv * last_depth_itv  # B H W
            ret_dict['inverse_min_depth'] = inverse_min_depth
            ret_dict['inverse_max_depth'] = inverse_max_depth
            
        return ret_dict