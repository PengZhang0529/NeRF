import numpy as np

import numpy as np
import torch

import time
from tqdm import *

'''
for i in tqdm(range(100)):
    time.sleep(1)

d=np.array([[[ 0,  1,  2],
        [ 2,  3,  4],
        [ 4,  5,  6]],
 
       [[ 8,  9, 10],
        [10, 11, 12],
        [12, 13, 14]],
 
       [[16, 17, 18],
        [18, 19, 20],
        [20, 21, 22]]])

print(d[...,1])
print(d[...,[2]])
'''

'''
a = torch.randn(2,1)
print(a)
print(a.view(-1))
'''
'''
import torch
in_=torch.tensor([[2., 4., 6.], [1., 3., 5.]])
print(in_)
out_prod = torch.cumprod(in_,dim=0)#竖着累积
print("cumulative product:", out_prod)
out_prod = torch.cumprod(in_,dim=1)#横着累积
print("cumulative product:", out_prod)

a = torch.tensor([[0.0349,  0.0670, -0.0612, 0.0280, -0.0222,  0.0422],
         [-1.6719,  0.1242, -0.6488, 0.3313, -1.3965, -0.0682],
         [-1.3419,  0.4485, -0.6589, 0.1420, -0.3260, -0.4795]])
b = torch.tensor([[-0.0658, -0.1490, -0.1684, 0.7188,  0.3129, -0.1116],
         [-0.2098, -0.2980,  0.1126, 0.9666, -0.0178,  0.1222],
         [ 0.1179, -0.4622, -0.2112, 1.1151,  0.1846,  0.4283]])
cc = torch.where(a>0,a,b)     #合并a,b两个tensor，如果a中元素大于0，则c中与a对应的位置取a的值，否则取b的值
print(cc)
'''

a = torch.randint(0,9,(2,2,2))
print(a)
b=a[...,0]
c=a[0,...]
print(b)
print(b.shape)
print(c)
print(c.shape)

