import torch 
from logzero import logger

class Abstract_Ext_ReKV:

    """ 
    Abstract class for External ReKV models (e.g., SigLIP2, PECore, LanguageBind, etc.)  
    """ 

    def __init__(self, model_config, topk, chunk_size, encode_chunk_size=8, sample_fps=0.5, temporal_patch_size=1.0):
        self.model_config = model_config
        self.topk = topk
        self.chunk_size = chunk_size 
        self.encode_chunk_size = encode_chunk_size

        self.similarity = None 

        self.block_sims = None
        self.adjacent_sims = None  
        
        self.retrieved_block_indices = None  
        self.sample_fps = sample_fps
        self.temporal_patch_size = temporal_patch_size 

    def clear_cache(self):
        pass 

    def _encode_video_chunk(self, video_chunk):
        pass 
    
    @torch.inference_mode()
    def encode_video(self, video, **kwargs): 
        # Accept kwargs for compatibility (e.g., streaming)
        pass 
    
    @torch.inference_mode()
    def encode_question(self, question): 
        pass 
    
    def calc_memory_usage(self):
        """Calculate CPU memory usage of kv_cache (list of CPU tensors) in bytes."""
        total_memory = 0
        if hasattr(self, 'kv_cache') and self.kv_cache is not None:
            for tensor in self.kv_cache:
                if isinstance(tensor, torch.Tensor):
                    total_memory += tensor.element_size() * tensor.numel()
        return total_memory 