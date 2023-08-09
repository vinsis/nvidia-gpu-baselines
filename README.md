# nvidia-gpu-baselines
Some experiments on NVIDIA GPUs

## Flash attention
- Tensor needs to have the shape `B, num_heads, seq_len, embed_dim`
- UserWarning: Flash attention requires q,k,v to have the same last dimension and to be a multiple of 8 and less than or equal to 128. Got Query.size(-1): 160, Key.size(-1): 160, Value.size(-1): 160 instead. (Triggered internally at /opt/conda/conda-bld/pytorch_1682343970094/work/aten/src/ATen/native/transformers/cuda/sdp_utils.h:253.)