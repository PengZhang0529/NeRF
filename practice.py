
import os
import json
import cv2
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class Datasetprovider: 
    def __init__(self, root, transforms_file, half_resolution = False):
        self.meta = json.load(open(os.path.join(root,transforms_file),"r"))
        self.root = root
        self.frames = self.meta['frames']
        self.images = []
        self.poses = []
        self.camera_angle_x = self.meta['camera_angle_x']


        for frame in self.frames:
            image_file = os.path.join(self.root,frame['file_path']+'.png')
            image = imageio.imread(image_file)
            if half_resolution:
                image = cv2.resize(image, dsize = None, fx = 0.5, interpolation = cv2.INTER_AREA)
            self.images.append(image)
            self.poses.append(frame['transform_matrix'])
    

        self.poses = np.stack(self.poses)
        self.images = (np.stack(self.images)/255.).astype(np.float32)
        self.width = self.images.shape[2]
        self.height = self.images.shape[1]

        self.focal = 0.5*self.width / np.tan(0.5*self.camera_angle_x)
        alpha = self.images[...,[3]] #取透明度
        rgb = self.images[...,:3]
        self.images = rgb * alpha + (1-alpha)  #???


class NeRFDataset:
    def __init__(self, provider : Datasetprovider, batch_size=1024, device='cpu'):
        self.images = provider.images
        self.poses = provider.poses
        self.focal = provider.focal
        self.width = provider.width
        self.height = provider.height
        self.batch_size = batch_size
        self.num_image = len(self.images)
        self.precrop_iters = 500
        self.precrop_frac = 0.5
        self.niter = 0
        self.device = device
        self.initialize()

    def initialize(self):
        warange = torch.arange(self.width, dtype = torch.float32, device=self.device) #返回一个一维张量
        harange = torch.arange(self.height, dtype = torch.float32, device=self.device)
        y, x = torch.meshgrid(harange,warange)

        #像素坐标转相机坐标
        self.transformed_x = (x - self.width*0.5) / self.focal
        self.transformed_y = (y - self.height*0.5) / self.focal

        self.precrop_index = torch.arange(self.width*self.height).view(self.height,self.width)
        dH = int(self.height // 2 * self.precrop_frac)
        dW = int(self.width // 2 * self.precrop_frac)
        self.precrop_index = self.precrop_index[
            self.height // 2 -dH : self.height // 2 + dH,
            self.width // 2 -dH : self.width // 2 +dW, 
        ].reshape(-1)
        

        poses = torch.FloatTensor(self.poses, device = self.device)
        all_ray_dirs, all_ray_origins = [],[]   #两个空链表来储存所有射线的方向和原点

        #每张图片对应一条射线
        for i in range(len(self.images)):
            ray_dirs, ray_origins = self.make_rays(self.transformed_x,self.transformed_y, poses[i])
            all_ray_dirs.append(ray_dirs)
            all_ray_origins.append(ray_origins)


        self.all_ray_dirs = torch.stack(all_ray_dirs, dim=0)
        self.all_ray_origins = torch.stack(all_ray_origins, dim=0)
        self.images = torch.FloatTensor(self.images, device = self.device).view(self.num_image,-1,3)
    

    def make_rays(self,x,y,pose):
        directions = torch.stack([x,-y,-torch.ones_like(x)],dim=-1)
        camera_matrix = pose[:3, :3]


        ray_dirs = directions.reshape(-1,3) @ camera_matrix.T
        ray_origin = pose[:3, 3].view(1,3).repeat(len(ray_dirs),1)
        return ray_dirs, ray_origin

    def get_test_item(self, index=0):

        ray_dirs   = self.all_ray_dirs[index]
        ray_oris   = self.all_ray_origins[index]
        img_pixels = self.images[index]

        for i in range(0, len(ray_dirs), self.batch_size):
            yield ray_dirs[i:i+self.batch_size], ray_oris[i:i+self.batch_size], img_pixels[i:i+self.batch_size] #生成器生成数据
    
    

    def __len__(self): #内置的魔法函数 求trainset的长度
        return self.num_image

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
        select_inds = np.random.choice(ray_dirs.shape[0], size=[nrays], replace = False) #随机选择batch_size个 组成一个列表
        ray_dirs = ray_dirs[select_inds]
        ray_oris = ray_oris[select_inds]
        img_pixels = img_pixels[select_inds]
        return ray_dirs, ray_oris, img_pixels
    

class NeRF(nn.Module):
    def __init__(self,x_pedim = 10, nwidth = 256, ndepth = 8, view_pedim = 4):
        super().__init__()
        xdim = (x_pedim * 2 + 1) * 3 #位置编码 63
        layers = []
        layers_in = [nwidth] * ndepth
        layers_in[0] = xdim
        layers_in[5] =  nwidth + xdim
        
        for i in range(ndepth):
            layers.append(nn.Linear(layers_in[i],nwidth))

        if view_pedim > 0:
            view_dim = (view_pedim * 2 + 1) * 3
            self.view_embed = Embedder(view_pedim)   #位置编码 27
            self.head = ViewDependentHead(nwidth, view_dim)
        else:
            self.head = NoViewDirHead(nwidth,4)
        
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
        self.feature = nn.Linear(ninput,ninput)
        self.alpha = nn.Linear(ninput,1)
        self.view_fc = nn.Linear(ninput + nview, ninput // 2)
        self.rgb = nn.Linear(ninput //2, 3)
    
    def forward(self, x, view_dirs):
        feature = self.feature(x)
        sigma = self.alpha(x).relu()
        feature = torch.cat([feature,view_dirs], dim = -1)
        feature = self.view_fc(feature).relu()
        rgb = self.rgb(feature).sigmoid()
        return sigma, rgb


class Embedder(nn.Module):
    def __init__(self, encodeing_dim):
        super().__init__()
        self.encoding_dim = encodeing_dim
    
    def forward(self, x):
        res = [x]
        for i in range(self.encoding_dim):
            for fn in [torch.sin, torch.cos]:
                res.append(fn((2.**i)*x))
        
        return torch.cat(res, dim = -1)


class NoViewDirHead(nn.Module):
    def __init__(self,ninput,noutput):
        super().__init__()

        self.head = nn.Linear(ninput,noutput)
    
    def forward(self, x, viewdirs):

        x = self.head(x)
        rgb = x[...,:3].sigmoid()
        sigma = x[...,3].relu()
        return sigma, rgb


def sample_rays(ray_directions, ray_origins, sample_z_vals):

    nrays = len(ray_origins)
    sample_z_vals =sample_z_vals.repeat(nrays,1)
    rays = ray_origins[:, None, :] + ray_directions[:, None, :] * sample_z_vals[...,None] #射线r = 0+td
    return rays, sample_z_vals 

def sample_viewdirs(ray_directions):  #???
    return ray_directions / torch.norm(ray_directions, dim = -1, keepdim = True)



def predict_to_rgb(sigma, rgb, z_vals, raydirs, white_background=False):

    device         = sigma.device
    delta_prefix   = z_vals[..., 1:] - z_vals[..., :-1] #求采样点之间的距离 得到63个维度 缺一个维度
    delta_addition = torch.full((z_vals.size(0), 1), 1e10, device=device) #添加一个维度  在比较远的地方
    delta = torch.cat([delta_prefix, delta_addition], dim=-1)
    delta = delta * torch.norm(raydirs[..., None, :], dim=-1) #通过样本点的长度缩放距离

    alpha    = 1.0 - torch.exp(-sigma * delta)
    exp_term = 1.0 - alpha
    epsilon  = 1e-10
    exp_addition = torch.ones(exp_term.size(0), 1, device=device)  #光线衰减 越远的衰减越严重 近的地方设置为1
    exp_term = torch.cat([exp_addition, exp_term + epsilon], dim=-1)
    transmittance = torch.cumprod(exp_term, axis=-1)[..., :-1]  #torch.cumprod 连乘

    weights       = alpha * transmittance
    rgb           = torch.sum(weights[..., None] * rgb, dim=-2)
    depth         = torch.sum(weights * z_vals, dim=-1) #计算每个像素的深度值
    acc_map       = torch.sum(weights, -1) #每个像素累计权重 表示每个像素的光线数

    if white_background:
        rgb       = rgb + (1.0 - acc_map[..., None])
    return rgb, depth, acc_map, weights



def sample_pdf(bins, weights, N_samples, det=False):
    # Get pdf
    device = weights.device
    weights = weights + 1e-5 # prevent nans 避免出现全0张量
    pdf = weights / torch.sum(weights, -1, keepdim=True)  #权重归一化得到概率函数（把权重变成一个概率分布）
    cdf = torch.cumsum(pdf, -1)  #求和
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins)) 处理最小值情况
    
    # Take uniform samples
    if det: #采取统一均匀采样
        u = torch.linspace(0., 1., steps=N_samples, device=device)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else: #随机采样
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Invert CDF
    u = u.contiguous() #重新排列
    inds = torch.searchsorted(cdf, u, right=True) #查找小于等于u的最大索引位置
    below = torch.max(torch.zeros_like(inds-1), inds-1) #下限最大索引位置
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds) #上限索引位置
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2) #并为张量

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g) #通过gather函数索引cdf
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g) #索引区间边界

    #计算样本权重值
    denom = (cdf_g[...,1]-cdf_g[...,0])  #计算样本区间长度
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom) #避免分母等于0 小于极小值 分母设置为1
    #进行差值得到精确的u值
    t = (u-cdf_g[...,0]) / denom 
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])
    return samples





def render_rays(coarse, fine, raydirs, rayoris, sample_z_vals, importance=0, white_background=False):

    rays, z_vals = sample_rays(raydirs, rayoris, sample_z_vals)
    view_dirs    = sample_viewdirs(raydirs)

    sigma, rgb = coarse(rays, view_dirs)
    sigma      = sigma.squeeze(dim=-1)
    rgb1, depth1, acc_map1, weights1 = predict_to_rgb(sigma, rgb, z_vals, raydirs, white_background)

    # 使用weights1进行重采样
    z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1]) #计算中间值
    z_samples  = sample_pdf(z_vals_mid, weights1[...,1:-1], num_samples2, det=True) #概率密度
    z_samples  = z_samples.detach()

    z_vals, _  = torch.sort(torch.cat([z_vals, z_samples], -1), -1) #原始深度z和重采样深度拼接并排序
    rays       = rayoris[...,None,:] + raydirs[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3] #生成新的射线
    sigma, rgb = fine(rays, view_dirs)
    sigma      = sigma.squeeze(dim=-1)

    # 第二次重采样的预测才是最终结果，这是论文中，分层采样环节（Hierarchical sampling）
    rgb2, depth2, acc_map2, weights2 = predict_to_rgb(sigma, rgb, z_vals, raydirs, white_background)
    return rgb1, rgb2




def train():

    pbar     = tqdm(range(1, maxiters))
    for global_step in pbar:

        idx   = np.random.randint(0, len(trainset))
        raydirs, rayoris, imagepixels = trainset[idx]

        rgb1, rgb2 = render_rays(coarse, fine, raydirs, rayoris, sample_z_vals, num_samples2, white_background)
        loss1 = ((rgb1 - imagepixels)**2).mean() #均方误差
        loss2 = ((rgb2 - imagepixels)**2).mean()
        psnr  = -10. * torch.log(loss2.detach()) / np.log(10.) #图像的峰值信噪比
        loss  = loss1 + loss2 #总的loss
        #更新参数
        optimizer.zero_grad() #优化器的梯度清零
        loss.backward() #进行反向传播  反向传播计算得到每个参数的梯度值
        optimizer.step() #优化器更新模型参数 通过梯度下降执行一步参数更新
        pbar.set_description(f"{global_step} / {maxiters}, Loss: {loss.item():.6f}, PSNR: {psnr.item():.6f}")


        decay_rate = 0.1  
        new_lrate  = lrate * (decay_rate ** (global_step / lrate_decay)) #随着迭代次数增大 学习率减少
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        if global_step % 5000 == 0 or global_step == 500:

            imgpath = f"imgs/{global_step:02d}.png" #定义存储照片路径
            pthpath = f"ckpt/{global_step:02d}.pth" #定义存储模型参数路径
            coarse.eval()
            with torch.no_grad():
                rgbs, imgpixels = [], []
                for raydirs, rayoris, imagepixels in trainset.get_test_item():

                    rgb1, rgb2  = render_rays(coarse, fine, raydirs, rayoris, sample_z_vals, importance, white_background)
                    rgbs.append(rgb2)
                    imgpixels.append(imagepixels)

                rgb       = torch.cat(rgbs, dim=0)
                imgpixels = torch.cat(imgpixels, dim=0)
                loss      = ((rgb - imgpixels)**2).mean()
                psnr      = -10. * torch.log(loss) / np.log(10.)

                print(f"Save image {imgpath}, Loss: {loss.item():.6f}, PSNR: {psnr.item():.6f}")
            coarse.train()
            #把图片像素值从（0.1）范围映射到（0，255）的范围 并且定位int8的数据类型 并转化RGB的顺序
            temp_image = (rgb.view(provider.height, provider.width, 3).cpu().numpy() * 255).astype(np.uint8)
            cv2.imwrite(imgpath, temp_image[..., ::-1]) #将颜色通道转换成bgr形式
            torch.save([coarse.state_dict(), fine.state_dict()], pthpath) #将模型保存到指定路径
 

        


def make_video360():

    cstate, fstate = torch.load(ckpt = '300000.pth', map_location="cpu") #map_location 将模型加载到cpu
    coarse.load_state_dict(cstate)
    fine.load_state_dict(fstate)
    
    coarse.eval()  #不进行训练只进行前向传播
    fine.eval()
    imagelist = []

    #遍历全景图片，渲染并保存每个视角的图片  
    for i, gfn in tqdm(enumerate(trainset.get_rotate_360_rays()), desc="Rendering"):
        
        with torch.no_grad():#设置不需要梯度更新
            rgbs = []
            for raydirs, rayoris in gfn():
                rgb1, rgb2 = render_rays(coarse, fine, raydirs, rayoris, sample_z_vals, importance, white_background)
                rgbs.append(rgb2)

            rgb = torch.cat(rgbs, dim=0)
        
        rgb  = (rgb.view(provider.height, provider.width, 3).cpu().numpy() * 255).astype(np.uint8)
        file = f"rotate360/{i:03d}.png"

        print(f"Rendering to {file}") #输出当前图像的路径
        cv2.imwrite(file, rgb[..., ::-1]) # 保存图像 rgb转bgr
        imagelist.append(rgb)

    

    #把所有图片合成为360度全景视频
    video_file = f"videos/rotate360.mp4"
    print(f"Write imagelist to video file {video_file}") #打印信息
    imageio.mimwrite(video_file, imagelist, fps=30, quality=10) #







if __name__ == '__main__':
    root = 'data/nerf_synthetic/lego'
    transforms_file = 'transforms_train.json'
    half_resolution = False

    provider = Datasetprovider(root, transforms_file, half_resolution) #数据提取


    batch_size =1024
    device = 'cpu'
    trainset = NeRFDataset(provider, batch_size, device)

    x_pedim = 10
    view_pedim = 4
    coarse = NeRF(x_pedim = x_pedim, view_pedim = view_pedim).to(device)
    params = list(coarse.parameters())  # len(params) = 24?
    fine = NeRF(x_pedim = x_pedim, view_pedim = view_pedim).to(device)
    params.extend(list(fine.parameters())) # len(params) = 48?

    lrate_decay = 500 * 1000
    lrate = 5e-4
    optimizer = optim.Adam(params, lrate)

    ray_dirs, ray_oris, img_pixels = trainset[0]
    
    num_samples1 = 64
    num_samples2 = 128
    white_background = True
    sample_z_vals = torch.linspace(2.0, 6.0, num_samples1, device = device).view(1, num_samples1) #初采样 每根射线上有64个采样点  为什么是2-6
    

    #rgb1, rgb2 = render_rays() 
    maxiters = 100000 + 1
    make_video360()
    ckpt = '300000.pth'

    pass





