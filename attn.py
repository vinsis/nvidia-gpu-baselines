import torch
from time import time

device = torch.device('cuda:0')

def measure_attn(x, enable_flash, enable_math, enable_mem, num_trials=10):
    '''
    `x` has shape: (seq_len, batch_size, embed_dim)
    '''
    print(f'x.shape is {x.shape}, x.dtype is {x.dtype}, x.device is {x.device}')
    seq_len, bs, embed_dim = x.size(0), x.size(1), x.size(2)
    assert embed_dim % 8 == 0
    num_heads = 8
    attn = torch.nn.MultiheadAttention(embed_dim, num_heads).to(device).half()
    _ = attn.eval()
    with torch.backends.cuda.sdp_kernel(enable_flash=enable_flash, enable_math=enable_math, enable_mem_efficient=enable_mem):
        print(f'flash: {torch.backends.cuda.flash_sdp_enabled()}, math: {torch.backends.cuda.math_sdp_enabled()}, memory efficient: {torch.backends.cuda.mem_efficient_sdp_enabled()}')
        torch.cuda.synchronize()
        st = time()
        for _ in range(num_trials):
            _ = attn(x,x,x)
        torch.cuda.synchronize()
        et = time()
    time_taken = (et - st)
    return time_taken

def measure_attn_no_context(x, num_trials=10):
    '''
    `x` has shape: (seq_len, batch_size, embed_dim)
    '''
    print(f'x.shape is {x.shape}, x.dtype is {x.dtype}, x.device is {x.device}')
    seq_len, bs, embed_dim = x.size(0), x.size(1), x.size(2)
    assert embed_dim % 8 == 0
    num_heads = 8
    attn = torch.nn.MultiheadAttention(embed_dim, num_heads).to(device).half()
    _ = attn.eval()
    # with torch.backends.cuda.sdp_kernel(enable_flash=enable_flash, enable_math=enable_math, enable_mem_efficient=enable_mem):
    print(f'flash: {torch.backends.cuda.flash_sdp_enabled()}, math: {torch.backends.cuda.math_sdp_enabled()}, memory efficient: {torch.backends.cuda.mem_efficient_sdp_enabled()}')
    torch.cuda.synchronize()
    st = time()
    for _ in range(num_trials):
        _ = attn(x,x,x)
    torch.cuda.synchronize()
    et = time()
    time_taken = (et - st)
    return time_taken

if __name__ == '__main__':
    torch.cuda.empty_cache()
    x = torch.randn(3000, 32, 64 * 8).half().cuda()
    t_no_context = measure_attn_no_context(x, 200)
    print(f'no context took {t_no_context} seconds for 200 trials')
    torch.cuda.empty_cache()
    t_fff = measure_attn(x, False, False, False, 200)
    print(f'fff took {t_fff} seconds for {200} trials')
    torch.cuda.empty_cache()
    t_tff = measure_attn(x, True, False, False, 200)
    print(f'tff took {t_tff} seconds for {200} trials')
    torch.cuda.empty_cache()
    t_ftf = measure_attn(x, False, True, False, 200)
    print(f'ftf took {t_ftf} seconds for 200 trials')
    torch.cuda.empty_cache()
    t_fft = measure_attn(x, False, False, True, 200)
    print(f'fft took {t_fft} seconds for 200 trials')
    torch.cuda.empty_cache()
