"""GPU/CPU KV-cache storage units used by ContextManager.

Split out of kv_cache_manager.py unchanged; kv_cache_manager re-exports these
names so existing imports keep working.
"""
import math
import os
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange


# Allocate a fixed-size block of GPU memory specifically for storing the KV-Cache of the local_window.
class CudaCache:
    def __init__(self, num_units, unit_size, dtype):
        self.num_units = num_units  # n_block
        self.unit_size = unit_size  # block_size * hidden_dim * 2
        self.dtype = dtype
        self.data = torch.empty(
            (num_units, unit_size),
            device="cuda",
            dtype=dtype
        )
        self.idle_set = set(list(range(num_units)))

    def update_cache_size(self, new_unit_size):
        self.unit_size = new_unit_size
        self.data = torch.empty(
            (self.num_units, new_unit_size),
            device="cuda",
            dtype=self.dtype
        )

    def alloc(self):
        assert len(self.idle_set) > 0
        idx = self.idle_set.pop()
        return self.data[idx], idx

    def delete(self, idx):
        assert idx not in self.idle_set
        self.idle_set.add(idx)


# The KV-Cache management unit supports data transfer between the CPU and GPU.
class MemoryUnit:
    # Initialize the KV-Cache management unit and store it on the CPU.
    def __init__(
        self, 
        kv: Tuple[torch.Tensor, torch.Tensor], 
        cache: CudaCache, 
        load_to_cache: bool = False, 
        pin_memory: bool = False,
    ):
        self.cache = cache

        if kv[0].is_cuda:
            # vm.max_map_count fix: this copy MUST stay non_blocking=False, because
            # non_blocking=True pins the CPU destination (~0.5 /dev/zero mappings per block)
            # and long videos then hit the 65530-mapping wall. The .pin_memory() call below is
            # a second, independent door to the same crash — closing only one door does
            # nothing, so long runs also need --no_pin_memory or REKV_DISABLE_PIN=1. Pinning
            # and copy synchronicity change transfer speed only, never values.
            cpu_data = tuple(_t.contiguous().to("cpu", non_blocking=False) for _t in kv)
        else:
            cpu_data = tuple(_t.contiguous() for _t in kv)

        # REKV_DISABLE_PIN bypasses the --no_pin_memory plumbing chain (several hops are
        # positional) at the only place that matters; it is honored for every backbone.
        # Each .pin_memory() is a cudaHostAlloc that costs ~2 MB /dev/zero mappings toward
        # vm.max_map_count=65530.
        if pin_memory and os.environ.get("REKV_DISABLE_PIN") != "1":
            cpu_data = tuple(_t.pin_memory() for _t in cpu_data)

        if load_to_cache:
            gpu_data, gpu_data_id = cache.alloc()
            gpu_data = gpu_data.view((2,) + kv[0].shape)
            gpu_data[0].copy_(kv[0], non_blocking=True)
            gpu_data[1].copy_(kv[1], non_blocking=True)
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream())
        else:
            gpu_data, gpu_data_id = None, None
            event = None

        self.cpu_data = cpu_data
        self.gpu_data = gpu_data
        self.gpu_data_id = gpu_data_id
        self.event = event

    # Load data from the CPU to the GPU and copy it to 'target' when necessary.
    # target: 2x (n_head, n_token, head_dim), on GPU
    def load(self, target: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> bool:
        if self.gpu_data is not None:
            if target is not None:
                target[0].copy_(self.gpu_data[0], non_blocking=True)
                target[1].copy_(self.gpu_data[1], non_blocking=True)
                target_event = torch.cuda.Event()
                target_event.record(torch.cuda.current_stream())
            else:
                target_event = None

            return False, target_event

        gpu_data, gpu_data_id = self.cache.alloc()
        gpu_data = gpu_data.view((2,) + self.cpu_data[0].shape)

        if target is not None:
            target[0].copy_(self.cpu_data[0], non_blocking=True)
            target[1].copy_(self.cpu_data[1], non_blocking=True)
            target_event = torch.cuda.Event()
            target_event.record(torch.cuda.current_stream())
            gpu_data[0].copy_(target[0], non_blocking=True)
            gpu_data[1].copy_(target[1], non_blocking=True)

        else:
            gpu_data[0].copy_(self.cpu_data[0], non_blocking=True)
            gpu_data[1].copy_(self.cpu_data[1], non_blocking=True)

        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream())
        self.event = event
        self.gpu_data = gpu_data
        self.gpu_data_id = gpu_data_id

        return True, target_event

    # Get the KV-Cache stored on GPU
    def get(self):
        assert self.gpu_data is not None
        self.event.wait()
        return self.gpu_data

    # Clear the KV-Cache stored on GPU
    def offload(self):
        assert self.gpu_data is not None
        self.event.wait()
        self.gpu_data = None
        self.cache.delete(self.gpu_data_id)
        self.gpu_data_id = None

    def calculate_cpu_memory(self):
        return len(self.cpu_data) * self.cpu_data[0].numel() * self.cpu_data[0].element_size()


# A dynamically growing vector cache on the GPU, used to store representative vectors of video frames.
class VectorTensor:
    # Initialize an empty cache of size (16, hidden_dim) on the GPU.
    def __init__(
        self, 
        hidden_size,
        element_dtype,
        device
    ):
        init_cached_size = 16
        self.data = torch.empty(
            (init_cached_size, hidden_size),
            dtype=element_dtype,
            device=device
        )
        self.length = 0
        self.cache_size = init_cached_size
        self.hidden_size = hidden_size

    # Double the size of the cache.
    def append_cache(self):
        new_cache_size = self.cache_size * 2
        data_shape = self.data.shape
        new_data = torch.empty(
            (new_cache_size,) + data_shape[1:],
            device=self.data.device,
            dtype=self.data.dtype
        )
        new_data[:self.cache_size,...].copy_(self.data)
        self.data = new_data
        self.cache_size = new_cache_size

    # Append a frame vector to the cache, and expand the cache if it exceeds the current cache size.
    def append(self, tensor: torch.Tensor):
        assert tensor.dtype == self.data.dtype
        assert tensor.size(1) == self.hidden_size, f'{tensor.size(1)}, {self.hidden_size}'
        assert tensor.is_contiguous()

        append_l = tensor.size(0)

        while self.length + append_l > self.cache_size:
            self.append_cache()

        self.data[self.length: self.length+append_l, ...].copy_(tensor)

        self.length += append_l

    # Get the cached frame vectors
    def get_data(self):
        return self.data[:self.length, ...]

    def get_blockwise_cosine_similarity(self):
        query = self.data[:self.length].float()  # (T, D), convert to fp32 to prevent numerical overflow
        key = self.data[:self.length].float()  # (T, D), convert to fp32 to prevent numerical overflow
        logits = torch.matmul(query, key.T)  # (T,T)
        norms = query.norm(dim=1)
        logits = logits / (norms[:, None] * norms[None, :])
        return logits

    def get_cosine_similarity(self, tensor: torch.Tensor):
        assert tensor.dim() == 1 and tensor.size(0) == self.hidden_size, f'{tensor.size(0)}, {self.hidden_size}'
        key = self.data[:self.length].float()  # (T, D), convert to fp32 to prevent numerical overflow
        query = tensor[None, :].float()  # (1, D)

        logits = torch.matmul(query, key.T)[0]  # (T,)
        key_norms = key.norm(dim=1)
        query_norm = query.norm(dim=1)

        logits = logits / (key_norms * query_norm)

        assert logits.dim() == 1 and logits.size(0) == self.length
        return logits

    def get_cosine_similarity_tokenwise(self, tensor: torch.Tensor, with_softmax: bool = False, head_idx: int = -1): 
        """ tokenwise similarity function """

        assert tensor.dim() == 3, f"tensor must be 3D, got {tensor.dim()}" 
        h, n, d = tensor.shape 
        assert h * d == self.hidden_size, f'{h * d} vs. {self.hidden_size}'        
        
        key = self.data[:self.length].float() # (T, D)
        query = tensor.float() # (Nq, D) 
        
        # extract heads from key 
        key = rearrange(key, "n (h d) -> h n d", h=h, d=d)

        # multi-head similarity 
        if head_idx != -1:
            key = key[head_idx].unsqueeze(0)
            query = query[head_idx].unsqueeze(0)

        logits = torch.einsum("hnd,hqd->hqn", key, query) # (H, Nq, T) 

        if with_softmax: 
            temperature = 1.0 
            logits = logits / (math.sqrt(d) * temperature)
            logits = F.softmax(logits, dim=-1) # (H, Nq, T)
        
        logits = logits.mean(1, keepdim=False) # (H, T) 
        logits = logits.mean(0, keepdim=False) # (T,)
        # assert logits.dim() == 1 and logits.size(0) == self.length 

        return logits 


    def get_cosine_similarity_with_external(self, tensor: torch.Tensor, **ext_kwargs):

        assert tensor.dim() == 1 and tensor.size(0) == self.hidden_size, f'{tensor.size(0)}, {self.hidden_size}'

        internal_key = self.data[:self.length].float()
        internal_key = internal_key / internal_key.norm(dim=1, keepdim=True)

        external_key = ext_kwargs['ext_video_features'].float().to(internal_key.device)
        external_key = external_key / external_key.norm(dim=1, keepdim=True)
        
        assert external_key.shape[0] == self.length, f'{external_key.shape[0]} vs. {self.length}'

        internal_query = tensor[None, :].float()
        internal_query = internal_query / internal_query.norm(dim=1, keepdim=True)
        external_query = ext_kwargs['ext_question_features'].float().to(internal_query.device)
        external_query = external_query / external_query.norm(dim=1, keepdim=True)

        assert internal_query.shape[0] == 1 and external_query.shape[0] == 1, f'{internal_query.shape}, {external_query.shape}'

        ext_weights = ext_kwargs['ext_weights']

        # Compute cosine similarities separately, then weight the logits
        # This avoids the squaring effect that occurs when weighting vectors before concatenation
        internal_logits = torch.matmul(internal_query, internal_key.T)[0]  # (length,)
        external_logits = torch.matmul(external_query, external_key.T)[0]  # (length,)

        logits = ext_weights[0] * internal_logits + ext_weights[1] * external_logits

        assert logits.dim() == 1 and logits.size(0) == self.length 
        return logits

    def __len__(self):
        return self.length
