import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import cv2
import os
import json
import argparse
import imageio

# 主要知识点
# 1. 位置编码，Positional Encoding
#    - 对于输入的x、y、z坐标，因为是连续的无法进行区分，因此采用ff特征，即傅立叶特征进行编码
#    - 编码为cos、sin不同频率的叠加，使得连续值可以具有足够的区分性
# 2. 视图独立性，View Dependent
#    - 输入不仅仅是光线采样点的x、y、z坐标，加上了视图依赖，即x、y、z、theta、pi，5d输入，此时多了射线所在视图
# 3. 分层采样，Hierarchical sampling
#    - 将渲染分为两级，由于第一级别的模型是均匀采样，而实际会有很多无效的采样（即对颜色没有贡献的区域会占比太多），在模型
#       中看来，就是某些点的梯度为0，对模型训练没有贡献
#    - 因此采用两级模型，model、fine，model模型使用均匀采样，推断后得到weights的分布，通过对weights分布进行重采样，使得采样点
#       更加集中在更重要的区域，今儿使得参与训练的点大都是有效的点。所以model作为一级推理，fine则推理重采样后的点
#
# x. 拓展，对于射线的方向和原点的理解，需要具有基本的3d变换知识，建议看GAMES101的前5章补充知识
#    PSNR是峰值信噪比，表示重建的逼真程度
# 这三个环节有了，效果就会非常逼真，但是某些细节上还是存在不足。另外训练时间非常关键

class BlenderProvider:
    def __init__(self, root, transforms_file, half_resolution=True):

        self.meta            = json.load(open(os.path.join(root, transforms_file), "r")) #读取.json文件，meta是一个字典
        self.root            = root
        self.frames          = self.meta["frames"]
        self.images          = [] #构建一个空链表来存储100张图片的（800,800,4）的矩阵
        self.poses           = [] #构建一个空链表来存储100张图片的（4,4）的旋转矩阵
        self.camera_angle_x  = self.meta["camera_angle_x"]
        
        for frame in self.frames:
            image_file = os.path.join(self.root, frame["file_path"] + ".png") #读取图片路径
            image      = imageio.imread(image_file) #读取每张图片 arry image.shape=(800,800,4)

            if half_resolution:
                image  = cv2.resize(image, dsize=None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

            self.images.append(image)
            self.poses.append(frame["transform_matrix"])

        self.poses  = np.stack(self.poses) #将列表self.poses转化成数组arry（100,4,4）
        self.images = (np.stack(self.images) / 255.0).astype(np.float32) #将列表self.images转化成数组arry（100,800,800,4） 并归一化处理
        self.width  = self.images.shape[2] #获取图像的宽 800
        self.height = self.images.shape[1] #获取图像的高 800
        self.focal  = 0.5 * self.width / np.tan(0.5 * self.camera_angle_x) #计算焦距
        alpha       = self.images[..., [3]] #取出图像透明度 shape=（100,800,800,1）
        rgb         = self.images[..., :3]  #取出图像rgb shape=（100,800,800,3）
        self.images = rgb * alpha + (1 - alpha) #图像透明度转rgb shape = （100,800,800,3）


class NeRFDataset:
    def __init__(self, provider, batch_size=1024, device="cpu"):

        self.images        = provider.images
        self.poses         = provider.poses
        self.focal         = provider.focal
        self.width         = provider.width
        self.height        = provider.height
        self.batch_size    = batch_size
        self.num_image     = len(self.images)
        self.precrop_iters = 500 #参数的用处？？
        self.precrop_frac  = 0.5 #参数的用处？？
        self.niter         = 0   #参数的用处？？
        self.device        = device
        self.initialize()


    def initialize(self):

        warange = torch.arange(self.width,  dtype=torch.float32, device=self.device)
        harange = torch.arange(self.height, dtype=torch.float32, device=self.device)
        y, x = torch.meshgrid(harange, warange) # torch.meshgrid的功能是生成网格，可以用于生成坐标。函数输入两个数据类型相同的一维张量，两个输出张量的行数为第一个输入张量的元素个数，列数为第二个输入张量的元素个数，当两个输入张量数据类型不同或维度不是一维时会报错。

        #像素坐标转相机坐标
        self.transformed_x = (x - self.width * 0.5) / self.focal
        self.transformed_y = (y - self.height * 0.5) / self.focal

        # pre center crop
        self.precrop_index = torch.arange(self.width * self.height).view(self.height, self.width)

        dH = int(self.height // 2 * self.precrop_frac)
        dW = int(self.width  // 2 * self.precrop_frac)
        #定义半采样的范围和索引
        self.precrop_index = self.precrop_index[
            self.height // 2 - dH:self.height // 2 + dH, 
            self.width  // 2 - dW:self.width  // 2 + dW
        ].reshape(-1)

        poses = torch.FloatTensor(self.poses, device=self.device)
        all_ray_dirs, all_ray_origins = [], [] #构建两个空列表放置射线的方向和原点

        for i in range(len(self.images)): #构造每张图片的射线
            ray_dirs, ray_origins = self.make_rays(self.transformed_x, self.transformed_y, poses[i])
            all_ray_dirs.append(ray_dirs)
            all_ray_origins.append(ray_origins)

        self.all_ray_dirs    = torch.stack(all_ray_dirs, dim=0) #将all_ray_dirs列表stack起来转成tensor (100,640000,3)
        self.all_ray_origins = torch.stack(all_ray_origins, dim=0)
        self.images          = torch.FloatTensor(self.images, device=self.device).view(self.num_image, -1, 3) #将图像信息转化成同self.all_ray_origins相同shap
        

    def __getitem__(self, index):
        self.niter += 1

        ray_dirs   = self.all_ray_dirs[index]
        ray_oris   = self.all_ray_origins[index]
        img_pixels = self.images[index] #像素点的真实值
        if self.niter < self.precrop_iters: #？？
            ray_dirs   = ray_dirs[self.precrop_index]
            ray_oris   = ray_oris[self.precrop_index]
            img_pixels = img_pixels[self.precrop_index]

        nrays          = self.batch_size
        select_inds    = np.random.choice(ray_dirs.shape[0], size=[nrays], replace=False) #随机选取1024个不重复的索引，
        ray_dirs       = ray_dirs[select_inds]
        ray_oris       = ray_oris[select_inds]
        img_pixels     = img_pixels[select_inds]

        # dirs是指：direction
        # ori是指： origin
        return ray_dirs, ray_oris, img_pixels


    def __len__(self):
        return self.num_image


    def make_rays(self, x, y, pose):

        # 100, 100, 3
        # 坐标系在-y，-z方向上
        directions    = torch.stack([x, -y, -torch.ones_like(x)], dim=-1)
        camera_matrix = pose[:3, :3]
        
        # 10000 x 3
        #相机坐标转世界坐标
        ray_dirs = directions.reshape(-1, 3) @ camera_matrix.T
        ray_origin = pose[:3, 3].view(1, 3).repeat(len(ray_dirs), 1)
        return ray_dirs, ray_origin


    def get_test_item(self, index=0):

        ray_dirs   = self.all_ray_dirs[index]
        ray_oris   = self.all_ray_origins[index]
        img_pixels = self.images[index]

        for i in range(0, len(ray_dirs), self.batch_size):
            yield ray_dirs[i:i+self.batch_size], ray_oris[i:i+self.batch_size], img_pixels[i:i+self.batch_size]


    def get_rotate_360_rays(self):
        def trans_t(t):
            return np.array([
                [1,0,0,0],
                [0,1,0,0],
                [0,0,1,t],
                [0,0,0,1],
            ], dtype=np.float32)

        def rot_phi(phi):
            return np.array([
                [1,0,0,0],
                [0,np.cos(phi),-np.sin(phi),0],
                [0,np.sin(phi), np.cos(phi),0],
                [0,0,0,1],
            ], dtype=np.float32)

        def rot_theta(th) : 
            return np.array([
                [np.cos(th),0,-np.sin(th),0],
                [0,1,0,0],
                [np.sin(th),0, np.cos(th),0],
                [0,0,0,1],
            ], dtype=np.float32)

        def pose_spherical(theta, phi, radius):
            c2w = trans_t(radius)
            c2w = rot_phi(phi/180.*np.pi) @ c2w
            c2w = rot_theta(theta/180.*np.pi) @ c2w
            c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
            return c2w

        for th in np.linspace(-180., 180., 41, endpoint=False):
            pose = torch.FloatTensor(pose_spherical(th, -30., 4.), device=self.device)

            def genfunc():
                ray_dirs, ray_origins = self.make_rays(self.transformed_x, self.transformed_y, pose)
                for i in range(0, len(ray_dirs), 1024):
                    yield ray_dirs[i:i+1024], ray_origins[i:i+1024]

            yield genfunc


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False):
    # Get pdf
    device = weights.device
    weights = weights + 1e-5 # prevent nans 避免出现全0张量
    pdf = weights / torch.sum(weights, -1, keepdim=True) # 累加不一样 为了能够求除法   #权重归一化得到概率函数（把权重变成一个概率分布）
    cdf = torch.cumsum(pdf, -1) #torch.cumsum() 累和不改变shape
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  #(batch, len(bins)) 处理最小值情况
    
    # Take uniform samples
    if det: #采取统一均匀采样
        u = torch.linspace(0., 1., steps=N_samples, device=device)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples]) #扩展到shape=(1024,128)
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples]) #返回了一个张量，包含了从0-1的均匀分布中抽取的一组随机数

    # Invert CDF
    u = u.contiguous() #重新排列
    inds = torch.searchsorted(cdf, u, right=True)  #查找cdf中大于等于u的最大索引位置  就是返回一个和u一样大小的tensor,其中的元素是在cdf中大于u中值的索引，(right=False:大于,right=True:大于等于)
    below = torch.max(torch.zeros_like(inds-1), inds-1)  #下限最大索引位置  按维度dim 返回最大值以及最大值的索引。
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds) #上限索引位置 根据inds索引进行取值 
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2) #并为张量

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g) #通过gather函数索引cdf  .unsqueeze(1) 在指定维度增加一个维度  
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g) #索引区间边界

    denom = (cdf_g[...,1]-cdf_g[...,0]) #计算样本权重值 #计算样本区间长度
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)  #避免分母等于0 小于极小值 分母设置为1   # cc = torch.where(a>0,a,b) 合并a,b两个tensor，如果a中元素大于0，则c中与a对应的位置取a的值，否则取b的值
    t = (u-cdf_g[...,0])/denom  #进行差值得到精确的u值
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])
    return samples


def sample_rays(ray_directions, ray_origins, sample_z_vals):

    nrays = len(ray_origins)
    sample_z_vals = sample_z_vals.repeat(nrays, 1)  #sample_z_vals.shape=（1024,64）
    rays = ray_origins[:, None, :] + ray_directions[:, None, :] * sample_z_vals[..., None] #r = o+td Tensor与列向量做*乘法的结果是每行乘以列向量对应行的值（相当于把列向量的列复制，成为与lhs维度相同
    return rays, sample_z_vals


def sample_viewdirs(ray_directions):
    return ray_directions / torch.norm(ray_directions, dim=-1, keepdim=True)
    

def predict_to_rgb(sigma, rgb, z_vals, raydirs, white_background=False):

    device         = sigma.device
    delta_prefix   = z_vals[..., 1:] - z_vals[..., :-1] #求相邻采样点之间的距离 得到63个维度 缺一个维度
    delta_addition = torch.full((z_vals.size(0), 1), 1e10, device=device) #添加一个维度  在比较远的地方
    delta = torch.cat([delta_prefix, delta_addition], dim=-1) 
    delta = delta * torch.norm(raydirs[..., None, :], dim=-1) #通过样本点的长度缩放距离

    alpha    = 1.0 - torch.exp(-sigma * delta)
    exp_term = 1.0 - alpha
    epsilon  = 1e-10
    exp_addition = torch.ones(exp_term.size(0), 1, device=device) ##光线衰减 越远的衰减越严重 近的地方设置为1
    exp_term = torch.cat([exp_addition, exp_term + epsilon], dim=-1) #为什么要加epsilon
    transmittance = torch.cumprod(exp_term, axis=-1)[..., :-1]

    weights       = alpha * transmittance
    rgb           = torch.sum(weights[..., None] * rgb, dim=-2) #对第2维采样点数进行累加
    depth         = torch.sum(weights * z_vals, dim=-1)  #对第2维采样点数进行累加  #计算每个像素的深度值
    acc_map       = torch.sum(weights, -1)  #每个像素累计权重 表示每个像素的光线数

    if white_background:
        rgb       = rgb + (1.0 - acc_map[..., None])
    return rgb, depth, acc_map, weights


def render_rays(model, fine, raydirs, rayoris, sample_z_vals, importance=0, white_background=False):

    rays, z_vals = sample_rays(raydirs, rayoris, sample_z_vals)
    view_dirs    = sample_viewdirs(raydirs)

    sigma, rgb = model(rays, view_dirs) #进行正向传播
    sigma      = sigma.squeeze(dim=-1)
    rgb1, depth1, acc_map1, weights1 = predict_to_rgb(sigma, rgb, z_vals, raydirs, white_background)

    # 使用weights1进行重采样
    z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1]) #???为什么要取中间值
    z_samples  = sample_pdf(z_vals_mid, weights1[...,1:-1], importance, det=True) #概率密度  
    z_samples  = z_samples.detach()

    z_vals, _  = torch.sort(torch.cat([z_vals, z_samples], -1), -1)  #原始深度z和重采样深度拼接并排序
    rays       = rayoris[...,None,:] + raydirs[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3] #生成新的射线
    sigma, rgb = fine(rays, view_dirs) #进入细模型
    sigma      = sigma.squeeze(dim=-1)

    # 第二次重采样的预测才是最终结果，这是论文中，分层采样环节（Hierarchical sampling）
    rgb2, depth2, acc_map2, weights2 = predict_to_rgb(sigma, rgb, z_vals, raydirs, white_background)
    return rgb1, rgb2

# 无视图独立性的head
class NoViewDirHead(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.head = nn.Linear(ninput, noutput)
    
    def forward(self, x, view_dirs):
        
        x = self.head(x)
        rgb   = x[..., :3].sigmoid()
        sigma = x[..., 3].relu()
        return sigma, rgb

# 视图独立性的head
class ViewDenepdentHead(nn.Module):
    def __init__(self, ninput, nview):
        super().__init__()

        self.feature = nn.Linear(ninput, ninput)
        self.view_fc = nn.Linear(ninput + nview, ninput // 2)
        self.alpha = nn.Linear(ninput, 1)
        self.rgb = nn.Linear(ninput // 2, 3)
    
    def forward(self, x, view_dirs):
        
        feature = self.feature(x)
        sigma   = self.alpha(x).relu()
        feature = torch.cat([feature, view_dirs], dim=-1)
        feature = self.view_fc(feature).relu()
        rgb     = self.rgb(feature).sigmoid()
        return sigma, rgb

# 位置编码实现
class Embedder(nn.Module):
    def __init__(self, positional_encoding_dim):
        super().__init__()
        self.positional_encoding_dim = positional_encoding_dim

    def forward(self, x):
        
        positions = [x]
        for i in range(self.positional_encoding_dim):
            for fn in [torch.sin, torch.cos]:
                positions.append(fn((2.0 ** i) * x)) #位置编码

        return torch.cat(positions, dim=-1)

# 基本模型结构
class NeRF(nn.Module):
    def __init__(self, x_pedim=10, nwidth=256, ndepth=8, view_pedim=4):
        super().__init__()
        
        xdim         = (x_pedim * 2 + 1) * 3

        layers       =  []
        layers_in    = [nwidth] * ndepth
        layers_in[0] = xdim
        layers_in[5] = nwidth + xdim

        # 模型中特定层[5]会存在concat
        for i in range(ndepth):
            layers.append(nn.Linear(layers_in[i], nwidth))
        
        if view_pedim > 0:
            view_dim = (view_pedim * 2 + 1) * 3
            self.view_embed = Embedder(view_pedim)
            self.head = ViewDenepdentHead(nwidth, view_dim)
        else:
            self.head = NoViewDirHead(nwidth, 4)
        
        self.xembed = Embedder(x_pedim)
        self.layers = nn.Sequential(*layers) #形参——单个星号代表这个位置接收任意多个非关键字参数，转化成元组方式。实参——如果*号加在了是实参上，代表的是将输入迭代器拆成一个个元素。


    
    def forward(self, x, view_dirs):
        
        xshape = x.shape
        x = self.xembed(x)
        if self.view_embed is not None:
            view_dirs = view_dirs[:, None].expand(xshape)  #其将单个维度扩大成更大维度，返回一个新的tensor
            view_dirs = self.view_embed(view_dirs)

        raw_x = x
        
        for i, layer in enumerate(self.layers):
            x = torch.relu(layer(x))
            
            if i == 4:
                x = torch.cat([x, raw_x], axis=-1)

        return self.head(x, view_dirs)


def train():

    pbar     = tqdm(range(1, maxiters))
    for global_step in pbar:

        idx   = np.random.randint(0, len(trainset)) #返回一个随机整型数 0-100中随机选取一个
        raydirs, rayoris, imagepixels = trainset[idx] 

        rgb1, rgb2 = render_rays(model, fine, raydirs, rayoris, sample_z_vals, importance, white_background)
        loss1 = ((rgb1 - imagepixels)**2).mean()
        loss2 = ((rgb2 - imagepixels)**2).mean()
        psnr  = -10. * torch.log(loss2.detach()) / np.log(10.)  #图像的峰值信噪比
        loss  = loss1 + loss2  #总的loss
        
        optimizer.zero_grad() #优化器的梯度清零
        loss.backward() #进行反向传播
        optimizer.step() #优化器更新模型参数
        pbar.set_description(f"{global_step} / {maxiters}, Loss: {loss.item():.6f}, PSNR: {psnr.item():.6f}") #

        decay_rate = 0.1
        new_lrate  = lrate * (decay_rate ** (global_step / lrate_decay)) #随着迭代次数增大 学习率减少
        
        #optimizer.param_groups： 是一个list，其中的元素为字典；
        #optimizer.param_groups[0]：长度为7的字典，包括[‘params’, ‘lr’, ‘betas’, ‘eps’, ‘weight_decay’, ‘amsgrad’, ‘maximize’]这7个参数；
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        #if global_step % 5000 == 0 or global_step == 500: 
        if global_step % 1 == 0 or global_step == 1: 
            imgpath = f"imgs/{global_step:02d}.png" #定义存储照片路径
            pthpath = f"ckpt/{global_step:02d}.pth" #定义存储模型参数路径
            model.eval() #不启用 BatchNormalization 和 Dropout 训练完train样本后，生成的模型model要用来测试样本。在model(test)之前，需要加上model.eval()，否则的话，有输入数据，即使不训练，它也会改变权值。这是model中含有batch normalization层所带来的的性质。
            with torch.no_grad():
                rgbs, imgpixels = [], []
                for raydirs, rayoris, imagepixels in trainset.get_test_item():

                    rgb1, rgb2  = render_rays(model, fine, raydirs, rayoris, sample_z_vals, importance, white_background)
                    rgbs.append(rgb2)
                    imgpixels.append(imagepixels)

                rgb       = torch.cat(rgbs, dim=0)
                imgpixels = torch.cat(imgpixels, dim=0)
                loss      = ((rgb - imgpixels)**2).mean()
                psnr      = -10. * torch.log(loss) / np.log(10.)

                print(f"Save image {imgpath}, Loss: {loss.item():.6f}, PSNR: {psnr.item():.6f}")
            model.train()   #启用 BatchNormalization 和 Dropout 
            
            temp_image = (rgb.view(height, width, 3).cpu().numpy() * 255).astype(np.uint8) #把图片像素值从（0.1）范围映射到（0，255）的范围 并且定位int8的数据类型 并转化RGB的顺序
            cv2.imwrite(imgpath, temp_image[..., ::-1]) #将颜色通道转换成bgr形式
            torch.save([model.state_dict(), fine.state_dict()], pthpath) #将模型保存到指定路径


def make_video360():

    mstate, fstate = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(mstate)
    fine.load_state_dict(fstate)
    model.eval()  #不进行训练只进行前行传播
    fine.eval()
    imagelist = []

    for i, gfn in tqdm(enumerate(trainset.get_rotate_360_rays()), desc="Rendering"):

        with torch.no_grad():
            rgbs = []
            for raydirs, rayoris in gfn():
                rgb1, rgb2 = render_rays(model, fine, raydirs, rayoris, sample_z_vals, importance, white_background)
                rgbs.append(rgb2)

            rgb = torch.cat(rgbs, dim=0)
        
        rgb  = (rgb.view(height, width, 3).cpu().numpy() * 255).astype(np.uint8)
        file = f"rotate360/{i:03d}.png"

        print(f"Rendering to {file}")
        '''
        b = a[i:j:s]这种格式呢，i,j与上面的一样，但s表示步进，缺省为1.所以a[i:j:1]相当于a[i:j]
        当s<0时：i缺省时，默认为-1； j缺省时，默认为-len(a)-1
        所以a[::-1]相当于 a[-1:-len(a)-1:-1]，也就是从最后一个元素到第一个元素复制一遍。
        '''
        cv2.imwrite(file, rgb[..., ::-1])
        imagelist.append(rgb)

    video_file = f"videos/rotate360.mp4"
    print(f"Write imagelist to video file {video_file}")
    imageio.mimwrite(video_file, imagelist, fps=30, quality=10)


if __name__ == "__main__":

    parser = argparse.ArgumentParser() #创建一个解析对象
    #然后向该对象中添加要关注的命令行参数和选项，每一个add_argument方法对应一个要关注的参数或选项；
    parser.add_argument("--datadir", type=str, default='data/nerf_synthetic/lego', help='input data directory') 
    parser.add_argument("--make-video360", action="store_true", help="make 360 rotation video")
    parser.add_argument("--half-resolution", action="store_true", help="use half resolution")
    parser.add_argument("--ckpt", default="300000.pth", type=str, help="model file used to make 360 rotation video")
    args = parser.parse_args() #调用parse_args()方法进行解析，解析成功之后即可使用。

    device      = "cpu"
    maxiters    = 100000 + 1
    batch_size  = 1024
    lrate_decay = 500 * 1000
    lrate       = 5e-4
    importance  = 128              #射线重采样的次数
    num_samples = 64                    # 每个光线的初采样数量
    positional_encoding_dim = 10        # 位置编码维度
    view_encoding_dim       = 4         # View Dependent对应的位置编码维度
    white_background        = True      # 图片背景是白色的
    half_resolution         = args.half_resolution    # 只进行一半分辨率的重建(400x400)，False表示(800x800)分辨率
    sample_z_vals           = torch.linspace(2.0, 6.0, num_samples, device=device).view(1, num_samples) #初采样   每根射线上有64个采样点

    model = NeRF(
        x_pedim    = positional_encoding_dim,
        view_pedim = view_encoding_dim
    ).to(device)
    params = list(model.parameters()) #list()将任何可迭代的数据转换成列表类型,model.parameters()保存的是Weights和Bais参数的值。
    
    # 使用model产生的权重进行重采样，然后再推理，所以这个才是效果更好的模型
    fine = NeRF(
        x_pedim    = positional_encoding_dim,
        view_pedim = view_encoding_dim
    ).to(device)
    params.extend(list(fine.parameters())) 

    optimizer = optim.Adam(params, lrate) #模型参数优化
    os.makedirs("imgs",      exist_ok=True) #递归创建目录
    os.makedirs("rotate360", exist_ok=True)
    os.makedirs("videos",    exist_ok=True)
    os.makedirs("ckpt",      exist_ok=True)

    print(model)

    provider = BlenderProvider("data/nerf_synthetic/lego", "transforms_train.json", half_resolution)
    trainset = NeRFDataset(provider, batch_size, device)
    width    = trainset.width
    height   = trainset.height

    if args.make_video360:
        make_video360()
    else:
        train()

    print("Program done.")