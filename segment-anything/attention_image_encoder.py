import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Type
from time import time


class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x
    
    def forward_sdp_kernel(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B, nHead, H * W, C)
        q, k, v = qkv.reshape(3, B, self.num_heads, H * W, -1).unbind(0)

        x = F.scaled_dot_product_attention(q, k, v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x
    
if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'Using device {device}')
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    bs = 16
    h, w = (32,32) if torch.cuda.is_available() else (16, 16)
    num_heads = 16
    embed_dim = 128 * num_heads # maximum shape allowed by sdp kernel for flash attention
    x = torch.randn(bs, h, w, embed_dim, dtype=dtype, device=device)
    attn = Attention(embed_dim, num_heads).to(device).eval()
    if torch.cuda.is_available():
        attn = attn.half()
    
    y1 = attn.forward(x)
    y2 = attn.forward_sdp_kernel(x)
    print((y1 - y2).abs().max().item())
    atol = 1e-3 if torch.cuda.is_available() else 1e-6
    print(torch.allclose(y1, y2, atol=atol))

    if torch.cuda.is_available():
        num_trials = 100
        torch.cuda.synchronize()
        st = time()
        for _ in range(num_trials):
            _ = attn.forward(x)
        torch.cuda.synchronize()
        et = time()
        print(f'Forward took {et - st} seconds for {num_trials} trials')

        torch.cuda.synchronize()
        st = time()
        for _ in range(num_trials):
            _ = attn.forward_sdp_kernel(x)
        torch.cuda.synchronize()
        et = time()
        print(f'Forward sdp kernel took {et - st} seconds for {num_trials} trials')

        with torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_math=False, enable_mem_efficient=False
        ):
            if embed_dim//num_heads > 128:
                print('Skipping flash attention because embed_dim > 128')
            elif x.ndim != 4:
                print('Skipping flash attention because x.ndim != 4')
            else:
                torch.cuda.synchronize()
                st = time()
                for _ in range(num_trials):
                    _ = attn.forward_sdp_kernel(x)
                torch.cuda.synchronize()
                et = time()
                print(f'Forward flash took {et - st} seconds for {num_trials} trials')

        with torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_math=True, enable_mem_efficient=False
        ):
            torch.cuda.synchronize()
            st = time()
            for _ in range(num_trials):
                _ = attn.forward_sdp_kernel(x)
            torch.cuda.synchronize()
            et = time()
            print(f'Forward math took {et - st} seconds for {num_trials} trials')

        with torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_math=False, enable_mem_efficient=True
        ):
            torch.cuda.synchronize()
            st = time()
            for _ in range(num_trials):
                _ = attn.forward_sdp_kernel(x)
            torch.cuda.synchronize()
            et = time()
            print(f'Forward memory efficient took {et - st} seconds for {num_trials} trials')
