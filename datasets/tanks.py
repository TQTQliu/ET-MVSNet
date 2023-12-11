from torch.utils.data import Dataset
from datasets.data_io import *
import os
import numpy as np
import cv2
from PIL import Image
from copy import deepcopy


class MVSDataset(Dataset):
    def __init__(self, datapath, n_views=7, split='intermediate'):
        self.levels = 4
        self.datapath = datapath
        self.split = split
        self.build_metas()
        self.n_views = n_views

    def build_metas(self):
        self.metas = []
        if self.split == 'intermediate':
            self.scans = ['Family', 'Horse','Playground', 'Francis',  'Train', 'Lighthouse', 'M60', 'Panther']
            # self.scans = ['Family','Playground', 'Francis',  'Train', 'Lighthouse', 'M60', 'Panther']
            # self.scans = ['Horse']

            
        elif self.split == 'advanced':
            self.scans = ['Auditorium', 'Ballroom', 'Courtroom',
                          'Museum', 'Palace', 'Temple']

        for scan in self.scans:
            with open(os.path.join(self.datapath, self.split, scan, 'pair.txt')) as f:
                num_viewpoint = int(f.readline())
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    if len(src_views) != 0:
                        self.metas += [(scan, -1, ref_view, src_views)]
   

    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))

        depth_min = float(lines[11].split()[0])
        depth_max = float(lines[11].split()[1])

        return intrinsics, extrinsics, depth_min, depth_max

    def read_img(self, filename):
        img = Image.open(filename)
        np_img = np.array(img, dtype=np.float32) / 255.
        return np_img

    def scale_input(self, intrinsics, img):
        """
        intrinsics: 3x3
        img: W H C
        """
        intrinsics[1,2] =  intrinsics[1,2] - 28  # 1080 -> 1024
        img = img[28:1080-28, :, :]
        return intrinsics, img

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
        
    def scale_mvs_input(self, img, intrinsics, max_w, max_h, base=64):
        h, w = img.shape[:2]
        if h > max_h or w > max_w:
            scale = 1.0 * max_h / h
            if scale * w > max_w:
                scale = 1.0 * max_w / w
            new_w, new_h = scale * w // base * base, scale * h // base * base
        else:
            new_w, new_h = 1.0 * w // base * base, 1.0 * h // base * base
        scale_w = 1.0 * new_w / w
        scale_h = 1.0 * new_h / h
        intrinsics[0, :] *= scale_w
        intrinsics[1, :] *= scale_h

        img = cv2.resize(img, (int(new_w), int(new_h)))

        return img, intrinsics


    def __getitem__(self, idx):
        scan, _, ref_view, src_views = self.metas[idx]
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.n_views-1]
        imgs = []

        # depth = None
        depth_min = None
        depth_max = None

        proj_matrices_0 = []
        proj_matrices_1 = []
        proj_matrices_2 = []
        proj_matrices_3 = []

        for i, vid in enumerate(view_ids):
            img_filename = os.path.join(self.datapath, self.split, scan, f'images/{vid:08d}.jpg')
            proj_mat_filename = os.path.join(self.datapath, self.split, scan, f'cams_1/{vid:08d}_cam.txt')

            img = self.read_img(img_filename)

            intrinsics, extrinsics, depth_min_, depth_max_ = self.read_cam_file(proj_mat_filename)
            if scan == 'Panther' or scan == 'Playground':
                img, intrinsics = self.scale_mvs_input(img, intrinsics, 1216, 896)
            else:
                intrinsics, img = self.scale_input(intrinsics, img)
            imgs.append(img.transpose(2,0,1))

            proj_mat_0 = np.zeros(shape=(2, 4, 4), dtype=np.float32)
            proj_mat_1 = np.zeros(shape=(2, 4, 4), dtype=np.float32)
            proj_mat_2 = np.zeros(shape=(2, 4, 4), dtype=np.float32)
            proj_mat_3 = np.zeros(shape=(2, 4, 4), dtype=np.float32)

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
                depth_min =  depth_min_
                depth_max = depth_max_


        # proj_matrices: N*4*4
        proj={}
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


        return {"imgs": imgs, # N*3*H0*W0
                "proj_matrices": proj, # N*4*4
                "depth_values": np.array([depth_min, depth_max], dtype=np.float32),
                "filename": scan + '/{}/' + '{:0>8}'.format(view_ids[0]) + "{}",
                'flag':flag}  
