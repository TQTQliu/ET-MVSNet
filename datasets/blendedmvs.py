from torch.utils.data import Dataset
from datasets.data_io import *
import os
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms as T
import random
import copy
from copy import deepcopy

def check_invalid_input(imgs, depths, masks, depth_mins, depth_maxs):
    for img in imgs:
        assert np.isnan(img).sum() == 0
        assert np.isinf(img).sum() == 0
    for depth in depths.values():
        assert np.isnan(depth).sum() == 0
        assert np.isinf(depth).sum() == 0
    for mask in masks.values():
        assert np.isnan(mask).sum() == 0
        assert np.isinf(mask).sum() == 0

    assert (depth_mins<=0) == 0
    assert (depth_maxs<=depth_mins) == 0


class MVSDataset(Dataset):
    def __init__(self, datapath, listfile, split, nviews, img_wh=(768, 576), robust_train=True):
        
        super(MVSDataset, self).__init__()
        self.levels = 4 
        self.datapath = datapath
        self.split = split
        self.listfile = listfile
        self.robust_train = robust_train
        assert self.split in ['train', 'val', 'all'], \
            'split must be either "train", "val" or "all"!'

        self.img_wh = img_wh
        if img_wh is not None:
            assert img_wh[0]%32==0 and img_wh[1]%32==0, \
                'img_wh must both be multiples of 32!'
        self.nviews = nviews
        self.scale_factors = {} # depth scale factors for each scan
        self.scale_factor = 0 # depth scale factors for each scan
        self.build_metas()

        self.color_augment = T.ColorJitter(brightness=0.5, contrast=0.5)

    def build_metas(self):
        self.metas = []
        with open(self.listfile) as f:
            self.scans = [line.rstrip() for line in f.readlines()]
        for scan in self.scans:
            with open(os.path.join(self.datapath, scan, "cams/pair.txt")) as f:
                num_viewpoint = int(f.readline())
                for _ in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    if len(src_views) >= self.nviews-1:
                        self.metas += [(scan, ref_view, src_views)]

    def read_cam_file(self, scan, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        depth_min = float(lines[11].split()[0])
        depth_max = float(lines[11].split()[-1])

        if scan not in self.scale_factors:
            self.scale_factors[scan] = 100.0 / depth_min
        depth_min *= self.scale_factors[scan]
        depth_max *= self.scale_factors[scan]
        extrinsics[:3, 3] *= self.scale_factors[scan]

        return intrinsics, extrinsics, depth_min, depth_max

    def read_depth_mask(self, scan, filename, depth_min, depth_max, scale):
        depth = np.array(read_pfm(filename)[0], dtype=np.float32)
        # depth = (depth * self.scale_factor) * scale
        depth = (depth * self.scale_factors[scan]) * scale
        # depth = depth * scale
        # depth = np.squeeze(depth,2)

        mask = (depth>=depth_min) & (depth<=depth_max)
        assert mask.sum() > 0
        mask = mask.astype(np.float32)
        if self.img_wh is not None:
            depth = cv2.resize(depth, self.img_wh,
                                 interpolation=cv2.INTER_NEAREST)
        h, w = depth.shape
        depth_ms = {}
        mask_ms = {}

        for i in range(4):
            depth_cur = cv2.resize(depth, (w//(2**i), h//(2**i)), interpolation=cv2.INTER_NEAREST)
            mask_cur = cv2.resize(mask, (w//(2**i), h//(2**i)), interpolation=cv2.INTER_NEAREST)

            depth_ms[f"stage{4-i}"] = depth_cur
            mask_ms[f"stage{4-i}"] = mask_cur

        return depth_ms, mask_ms


    def read_img(self, filename):
        img = Image.open(filename)
        # img = self.color_augment(img)
        # scale 0~255 to 0~1
        np_img = np.array(img, dtype=np.float32) / 255.
        return np_img

    def __len__(self):
        return len(self.metas)
    
    def mark(self,flag,k,b,t,x,y,thresh=1,inv=0):
        H,W = flag.shape
        if inv:
            x_pred = k*y+b
            delta = (abs(x-x_pred) < thresh).reshape(H,W)
        else:    
            y_pred = k*x+b
            delta = (abs(y-y_pred) < thresh).reshape(H,W)
        idx = np.where(delta)
        idx_y,idx_x = idx
        if not len(idx_y):
            return flag
        flag[idx_y,idx_x] = t
        return flag


    def epipolar_search(self,rot,trans,H,W):
        t1,t2,t3 = trans[0],trans[1],trans[2]
        f =np.zeros((H,W,2))
        x,y = np.meshgrid(np.arange(W),np.arange(H))
        x,y = x.reshape(-1), y.reshape(-1)
        grid = np.stack((x,y,np.ones_like(x)),axis=0)
        A =  np.matmul(rot,grid).reshape(3,H,W)
        inv = 0
        f[:,:,0] = (A[1,:]*t3 - A[2,:]*t2) / (A[0,:]*t3 - A[2,:]*t1+1e-10)
        f[:,:,1] = (t2/(t3+1e-10))-f[:,:,0]*(t1/(t3+1e-10))
        if abs(f[:,:,0]).max()>=5:
            f[:,:,0] = 1./(f[:,:,0]+1e-10)
            f[:,:,1] = (t1/(t3+1e-10))-f[:,:,0]*(t2/(t3+1e-10))
            inv = 1
        f[:,:,0] = 0.1*np.floor(10*f[:,:,0])
        f[:,:,1] = 10*np.floor(0.1*f[:,:,1])
        k_all = np.unique(f[:,:,0])
        b_all = np.unique(f[:,:,1])
        ref_flag = np.zeros((H,W))
        src_flag = np.zeros((H,W)) 
        t=1

        if len(k_all) > 500 or len(b_all) > 500:
            return ref_flag,src_flag
            
        for k in k_all:
            for b in b_all:
                idx = np.where(np.logical_and(f[:,:,0]==k,f[:,:,1]==b))
                if len(idx[0]):
                    src_flag = self.mark(src_flag,k,b,t,x,y,inv=inv)
                    idx_y,idx_x = idx
                    ref_flag[idx_y,idx_x]=t
                    t+=1
        
        return ref_flag,src_flag


    def __getitem__(self, idx):
        meta = self.metas[idx]
        scan, ref_view, src_views = meta
        
        if self.robust_train:
            num_src_views = len(src_views)
            index = random.sample(range(num_src_views), self.nviews - 1)
            view_ids = [ref_view] + [src_views[i] for i in index]
            scale = random.uniform(0.8, 1.25)

        else:
            view_ids = [ref_view] + src_views[:self.nviews - 1]
            scale = 1

        imgs = []
        mask = None
        depth = None
        depth_min = None
        depth_max = None

        proj={}
        proj_matrices_0 = []
        proj_matrices_1 = []
        proj_matrices_2 = []
        proj_matrices_3 = []


        for i, vid in enumerate(view_ids):
            img_filename = os.path.join(self.datapath, '{}/blended_images/{:0>8}.jpg'.format(scan, vid))
            depth_filename = os.path.join(self.datapath, '{}/rendered_depth_maps/{:0>8}.pfm'.format(scan, vid))
            proj_mat_filename = os.path.join(self.datapath, '{}/cams/{:0>8}_cam.txt'.format(scan, vid))

            img = self.read_img(img_filename)
            imgs.append(img.transpose(2,0,1))

            intrinsics, extrinsics, depth_min_, depth_max_ = self.read_cam_file(scan, proj_mat_filename)


            proj_mat_0 = np.zeros(shape=(2, 4, 4), dtype=np.float32)
            proj_mat_1 = np.zeros(shape=(2, 4, 4), dtype=np.float32)
            proj_mat_2 = np.zeros(shape=(2, 4, 4), dtype=np.float32)
            proj_mat_3 = np.zeros(shape=(2, 4, 4), dtype=np.float32)
            extrinsics[:3, 3] *= scale
            intrinsics[:2,:] *= 0.125
            proj_mat_0[0,:4,:4] = extrinsics.copy()
            proj_mat_0[1,:3,:3] = intrinsics.copy()

            intrinsics[:2,:] *= 2
            proj_mat_1[0,:4,:4] = extrinsics.copy()
            proj_mat_1[1,:3,:3] = intrinsics.copy()

            intrinsics[:2,:] *= 2
            proj_mat_2[0,:4,:4] = extrinsics.copy()
            proj_mat_2[1,:3,:3] = intrinsics.copy()

            intrinsics[:2,:] *= 2
            proj_mat_3[0,:4,:4] = extrinsics.copy()
            proj_mat_3[1,:3,:3] = intrinsics.copy()  

            proj_matrices_0.append(proj_mat_0)
            proj_matrices_1.append(proj_mat_1)
            proj_matrices_2.append(proj_mat_2)
            proj_matrices_3.append(proj_mat_3)

            if i == 0:  # reference view
                depth_min = depth_min_ * scale
                depth_max = depth_max_ * scale
                depth, mask = self.read_depth_mask(scan, depth_filename, depth_min, depth_max, scale)
                for l in range(self.levels):
                    mask[f'stage{l+1}'] = mask[f'stage{l+1}'] # np.expand_dims(mask[f'stage{l+1}'],2)
                    depth[f'stage{l+1}'] = depth[f'stage{l+1}']

        proj['stage1'] = np.stack(proj_matrices_0)
        proj['stage2'] = np.stack(proj_matrices_1)
        proj['stage3'] = np.stack(proj_matrices_2)
        proj['stage4'] = np.stack(proj_matrices_3)



        poses = deepcopy(proj['stage1'])
        ref_proj, src_projs = poses[0], poses[1:]
        ref_proj_new = ref_proj[0].copy()
        ref_proj_new[:3, :4] = np.matmul(ref_proj[1, :3, :3], ref_proj[0, :3, :4])
        H,W = imgs[0].shape[-2:]
        for i,src_proj in enumerate(src_projs):
            src_proj_new = src_proj[0].copy()
            src_proj_new[:3, :4] = np.matmul(src_proj[1, :3, :3], src_proj[0, :3, :4])
            pro = np.matmul(src_proj_new, np.linalg.inv(ref_proj_new))
            rot = pro[:3, :3]  
            trans = pro[:3, 3:4].reshape(-1)
            ref_flag,src_flag = self.epipolar_search(rot,trans,H//8,W//8)
            flag_rs = np.stack((ref_flag,src_flag),axis=0)[None,:,:,:]
            if i==0:
                flag = flag_rs
            else:
                flag = np.concatenate((flag,flag_rs),axis=0)

        # check_invalid_input(imgs, depth, mask, depth_min, depth_max)
        # data is numpy array
        return {"imgs": imgs,                   # [Nv, 3, H, W]
                "proj_matrices": proj,          # [N,2,4,4]
                "depth": depth,                 # [1, H, W]
                "depth_values": np.array([depth_min, depth_max], dtype=np.float32),
                "mask": mask,
                'flag':flag}                   # [1, H, W]
        