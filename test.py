


import os
import cv2
import json
import imageio
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
#from utils import Datasetprovider, NeRFDataset, Embedder, ViewDependentHead, NoViewDirHead
#from utils import NeRFDataset, Embedder, ViewDependentHead, NoViewDirHead
#from utils import NeRF



def sample_rays(ray_directions, ray_origins, sample_z_vals):

    nrays = len(ray_origins)
    sample_z_vals = sample_z_vals.repeat(nrays, 1)
    rays = ray_origins[:, None, :] + ray_directions[:, None, :] * sample_z_vals[..., None]
    return rays, sample_z_vals

def sample_viewdirs(ray_directions): #归一化平面上xyz的值
    return ray_directions /torch.norm(ray_directions, dim = -1,keepdim=True)
 
        
class Datasetprovider:
    def __init__(self, root, transforms_file, half_resolution=False):#half_resolution 进行半分辨率采样，处理空白信息
        self.meta = json.load(open(os.path.join(root, transforms_file),"r"))       
        self.root = root
        self.frames = self.meta['frames']
        self.images = []
        self.poses = []
        self.camera_angle_x = self.meta['camera_angle_x']


        for frame in self.frames:                                            #读取照片
            image_file = os.path.join(self.root,frame["file_path"]+'.png')
            image = imageio.imread(image_file)
            if half_resolution:
                image = cv2.resize(image, dsize = None, fx = 0.5,interpolation = cv2.INTER_AREA)
            self.images.append(image)
            self.poses.append(frame['transform_matrix'])

        self.poses = np.stack(self.poses) #将self.poses, self.images转成数组
        self.images = (np.stack(self.images)/255.).astype(np.float32) #将self.images转成数组 （/255）归一化 定义数据类型
        self.width = self.images.shape[2]
        self.height = self.images.shape[1]
        
        self.focal = 0.5*self.width / np.tan(0.5*self.camera_angle_x) #求焦距
        alpha = self.images[..., [3]] #取透明度
        rgb = self.images[..., :3]
        self.images = rgb * alpha + (1-alpha) 

class NeRFDataset:
    def __init__(self, provider : Datasetprovider, batch_size = 1024, device = 'cpu'):

        self.images = provider.images
        self.poses = provider.poses
        self.focal = provider.focal
        self.width = provider.width
        self.height = provider.height
        self.batch_size = batch_size
        self.num_images = len(self.images)
        self.precrop_iters = 500     #进行500轮训练 主要拿到中间物体的坐标
        self.precrop_frac = 0.5
        self.niter = 0 
        self.device = device
        self.initialize()

    def initialize(self): #生成每一张画布的像素坐标
        warange = torch.arange(self.width, dtype = torch.float32, device = self.device)
        harange = torch.arange(self.height, dtype = torch.float32, device = self.device)
        y, x = torch.meshgrid(harange,warange) #生成像素坐标


        #像素坐标转相机坐标
        self.transformed_x = (x - self.width*0.5) / self.focal
        self.transformed_y = (y - self.height*0.5) / self.focal

        self.precrop_index = torch.arange(self.width*self.height).view(self.height,self.width)
        dH = int(self.height // 2 * self.precrop_frac) #定义增量
        dW = int(self.width // 2 * self.precrop_frac)  
        self.precrop_index = self.precrop_index[
            self.height // 2 - dH : self.height // 2 + dH,
            self.width // 2 - dW : self.width // 2 + dW,
        ].reshape(-1)

        poses = torch.FloatTensor(self.poses, device = self.device)
        #构造射线（r = O + td）
        all_ray_dirs,all_ray_origins = [],[]        #两个空链表来储存所有射线的方向和原点

        for i in range(len(self.images)):

            ray_dirs, ray_origins = self.make_rays(self.transformed_x, self.transformed_y, poses[i])  #make_rays返回每张照片像素点在世界坐标系下的坐标，以及相机坐标系的原点在世界坐标系下的坐标
            all_ray_dirs.append(ray_dirs)
            all_ray_origins.append(ray_origins)


        self.all_ray_dirs = torch.stack(all_ray_dirs, dim = 0)
        self.all_ray_origins = torch.stack(all_ray_dirs, dim = 0)
        self.images = torch.FloatTensor(self.images, device = self.device).view(self.num_images, -1, 3)
        pass


    def make_rays(self, x, y, pose): 
        directions = torch.stack([x, -y, -torch.ones_like(x)], dim = -1)
        camera_matrix = pose[:3, :3] #取出3*3的旋转矩阵R

 
        ray_dirs = directions.reshape(-1,3) @ camera_matrix.T #相机坐标系转世界坐标系   #.reshape(-1,3) 生成3列行数自动确定
        ray_origin = pose[:3, 3].view(1,3).repeat(len(ray_dirs),1) #相机坐标系原点在世界坐标系下的位置
        return ray_dirs, ray_origin
     
    def __len__(self): #内置的魔法函数 求trainset的长度
        return self.num_images

    def __getitem__(self, index):
        self.niter += 1
        ray_dirs = self.all_ray_dirs[index]
        ray_oris = self.all_ray_origins[index]
        img_pixels = self.images[index]

        if self.niter < self.precrop_iters: #控制在500轮以内
            ray_dirs = ray_dirs[self.precrop_index]
            ray_oris = ray_oris[self.precrop_index]
            img_pixels = img_pixels[self.precrop_index]
        
        nrays = self.batch_size
        select_inds = np.random.choice(ray_dirs.shape[0], size=[nrays], replace = False)
        ray_dirs = ray_dirs[select_inds]
        ray_oris = ray_oris[select_inds]
        img_pixels = img_pixels[select_inds]
        return ray_dirs, ray_oris, img_pixels


class NeRF(nn.Module):
    def __init__(self, x_pedim = 10, nwidth = 256, ndepth = 8, view_pedim = 4):
        super().__init__()
        xdim = (x_pedim * 2 + 1) * 3
        layers = []
        layers_in = [nwidth] * ndepth
        layers_in[0] = xdim
        layers_in[5] = nwidth + xdim

        for i in range(ndepth):
            layers.append(nn.Linear(layers_in[i],nwidth))

        if view_pedim > 0:
            view_dim = (view_pedim * 2 + 1) * 3
            self.view_emded = Embedder(view_pedim)  #positiona encoding
            self.head = ViewDependentHead(nwidth, view_dim)
        else:
            self.head = NoViewDirHead(nwidth, 4)

        self.xembed = Embedder(x_pedim)
        self.layers = nn.Sequential(*layers)

    
    def forward(self, x, view_dirs):
        
        xshape = x.shape
        x = self.xembed(x)
        if self.view_embed is not None:
            view_dirs = view_dirs[:, None].expand(xshape)
            view_dirs = self.view_embed(view_dirs)

        raw_x = x
        for i, layer in enumerate(self.layers):
            x = torch.relu(layer(x))
            
            if i == 4:
                x = torch.cat([x, raw_x], axis=-1)

        return self.head(x, view_dirs)


class ViewDependentHead(nn.Module):
    def __init__(self, ninput, nview):
        super().__init__()
        self.feature = nn.Linear(ninput, ninput)
        self.alpha = nn.Linear(ninput,1)   #第9层网络结构
        self.view_fc = nn.Linear(ninput + nview, ninput // 2)  #10
        self.rgb = nn.Linear(ninput // 2, 3) #11

    def forward(self, x, view_dirs): #正向传播过程
        feature = self.feature(x)
        sigma = self.alpha(x).relu()
        feature = torch.cat([feature, view_dirs], dim = -1)
        feature = self.view_fc(feature).relu()
        rgb = self.rgb(feature).sigmoid()
        return sigma, rgb

class Embedder(nn.Module):
    def __init__(self, encoding_dim):
        self.encoding_dim = encoding_dim
    
    def forword(self, x):
        res = [x]
        for i in range(self.encoding_dim):    #positional encoding
            for fn in [torch.sin, torch.cos]:
                res.append(fn((2.**i) * x))
        return torch.cat(res, dim = -1)



class NoViewDirHead(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.head = nn.Linear(ninput, noutput)
    
    def forward(self, x, view_dirs):
        
        x = self.head(x)
        rgb   = x[..., :3].sigmoid()
        sigma = x[..., 3].relu()
        return sigma, rgb

if __name__ == '__main__':
    root = 'data/nerf_synthetic/lego'
    transforms_file = 'transforms_train.json'
    half_resolution = False

    provider = Datasetprovider(root, transforms_file, half_resolution)
    batch_size=1024
    device = 'cpu'
    trainset = NeRFDataset(provider, batch_size, device)

    x_pedim = 10
    view_pedim = 4
    coarse = NeRF(x_pedim = x_pedim, view_pedim = view_pedim).to(device)
    params = list(coarse.parameters())
    fine = NeRF(x_pedim = x_pedim, view_pedim = view_pedim).to(device)
    params.extend(list(fine.parameters()))
    #ray_dirs, ray_oris, img_pixels = trainset[0]

    lrate_decay = 500 * 1000
    lrate = 5e-4
    optimizer = optim.Adam(params, lrate)

    ray_dirs, ray_oris, img_pixels = trainset[0]

    num_samples = 64
    sample_z_vals           = torch.linspace(2.0, 6.0, num_samples, device=device).view(1, num_samples) #初采样   每根射线上有64个采样点
    rays, z_vals =  sample_rays(ray_dirs, ray_oris, sample_z_vals)
    view_dirs = sample_viewdirs(ray_dirs)



    pass









