import torch
from time import time

device = torch.device('cuda:0')
device_name = torch.cuda.get_device_name()

def measure_mm_time(x):
    '''
    `x` is assumed to be a N by N tensor
    '''
    print(f'Measuring MM: x.shape is {x.shape}, x.dtype is {x.dtype}, x.device is {x.device}')
    torch.cuda.synchronize()
    st = time()
    y = x @ x
    torch.cuda.synchronize()
    et = time()
    time_spent = et - st
    return y, time_spent

def measure_bmm_time(x):
    '''
    `x` is assumed to be a 1 by N by N tensor
    '''
    print(f'Measuring BMM: x.shape is {x.shape}, x.dtype is {x.dtype}, x.device is {x.device}')
    torch.cuda.synchronize()
    st = time()
    y = torch.bmm(x, x)
    torch.cuda.synchronize()
    et = time()
    time_spent = et - st
    return y, time_spent
