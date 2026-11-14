import os 
import torch
from PIL import Image
from tqdm import tqdm
from logzero import logger

from  transformers import AutoProcessor, AutoModel
from model.abstract_ext_rekv import Abstract_Ext_ReKV


def _unwrap_features(out):
    """Return the projected feature tensor from get_{image,text}_features.

    transformers <= 4.5x returned the projected tensor directly; 5.x returns the full
    BaseModelOutputWithPooling with the projection stored in `pooler_output`.
    """
    if isinstance(out, torch.Tensor):
        return out
    return out.pooler_output


class CLIP_ReKV(Abstract_Ext_ReKV):
    """CLIP model integration for ReKV - supports the following 
        - CLIP: openai/clip-vit-large-patch14
        - SigLIP: google/siglip-so400m-patch14-384 
        - SigLIP2: google/siglip2-so400m-patch14-384
    """

    kv_cache = None 

    def __init__(self, model_config, topk, chunk_size, encode_chunk_size=8, sample_fps=0.5, temporal_patch_size=1.0):
        Abstract_Ext_ReKV.__init__(self, model_config, topk, chunk_size, encode_chunk_size, sample_fps, temporal_patch_size)

        self.model, self.processor = self.load_clip(model_config)

    def clear_cache(self):
        self.kv_cache = [] 

    def load_clip(self, model_config): 
        model = AutoModel.from_pretrained(model_config, device_map='auto')
        processor = AutoProcessor.from_pretrained(model_config) 
        model.requires_grad_(False)
        model.eval()
        return model, processor

    @torch.inference_mode()
    def _encode_video_chunk(self, video_chunk):
        frm_feats = _unwrap_features(self.model.get_image_features(pixel_values=video_chunk))
        video_features = frm_feats.reshape(video_chunk.shape[0], -1) # [chunk_size, D]
        # TODO: either keep video_features on GPU or CPU 
        self.kv_cache.append(video_features.cpu())
    
    def encode_video(self, video, target_frames=None, **kwargs): 

        """  Encode video frames into features. 
            
        Args:
            video: (Nv, H, W, 3)
            target_frames: (int) if specified, sample the video to the target number of frames
            **kwargs: Additional arguments (e.g., streaming) for compatibility

        Returns:
            video_kwargs: (dict) containing the sample fps
        """

        # encode chunk by chunk
        num_frames = video.shape[0]

         # Sample frames if target_frames is specified
        if target_frames is not None and target_frames < num_frames:
            # Uniformly sample frames
            indices = torch.linspace(0, num_frames - 1, target_frames).long()
            video = video[indices]
            
            # Adjust FPS proportionally
            original_fps = self.sample_fps 
            new_fps = original_fps * target_frames / num_frames
            
            print(f"Sampled video input: {video.shape} (from {num_frames} to {target_frames} frames)")
            print(f"Adjusted FPS: {new_fps:.2f} (from {original_fps})")
            num_frames = target_frames
        else:
            new_fps = self.sample_fps 
            print(f"Using all {num_frames} frames")

        num_chunks = num_frames // self.encode_chunk_size

        assert isinstance(video, torch.Tensor)

        # Convert tensor to numpy if needed, then preprocess each frame
        if video.is_cuda:
            video_np = video.cpu().numpy()
        else:
            video_np = video.numpy()
        
        pil_images = [Image.fromarray(frame) for frame in video_np]
        inputs = self.processor(images=pil_images, return_tensors="pt", )
        video = inputs['pixel_values'].to(self.device)

        for chunk_idx in tqdm(range(num_chunks), desc="Encoding CLIPvideo chunks..."):
            start_idx = chunk_idx * self.encode_chunk_size
            end_idx = start_idx + self.encode_chunk_size
            chunk_video = video[start_idx:end_idx]
            self._encode_video_chunk(chunk_video) 

        # Handle remaining frames
        remaining_frames = num_frames % self.encode_chunk_size
        if remaining_frames > 0:
            start_idx = num_chunks * self.encode_chunk_size
            end_idx = start_idx + remaining_frames
            remaining_video = video[start_idx:end_idx]
            self._encode_video_chunk(remaining_video)

        video_kwargs = {
            "sample_fps": new_fps 
        }

        return video_kwargs 
        
    @torch.inference_mode()
    def encode_question(self, question):
        # Get max_position_embeddings from model config to handle truncation
        max_length = getattr(self.model.config, 'max_position_embeddings', None)
        if max_length is None:
            # Fallback: try to get from text model config if it exists
            if hasattr(self.model, 'text_model') and hasattr(self.model.text_model.config, 'max_position_embeddings'):
                max_length = self.model.text_model.config.max_position_embeddings
            else:
                # Default CLIP context length is 77
                max_length = 77
        
        # https://github.com/google-research/big_vision/issues/186 
        # https://github.com/google-research/big_vision/issues/166#issuecomment-2730192450
        # SigLIP2 max_length is 64 and requires lower-case input 
        if "siglip2" in self.model_config:
            max_length = 64 
            question = question.lower() 
            inputs = self.processor(
                text=question, 
                return_tensors="pt", 
                padding="max_length",
                truncation=True,
            ).to(self.device) 
        else:
            inputs = self.processor(
                text=question, 
                return_tensors="pt", 
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(self.device) 
        
        question_features = _unwrap_features(self.model.get_text_features(**inputs))
        return question_features

    def get_cosine_similarity(self, query_features, key_features):
        logits = query_features @ key_features.T # (n, k)
        query_norms = query_features.norm(dim=-1, keepdim=True)
        key_norms = key_features.norm(dim=-1, keepdim=True)
        logits = logits / (query_norms * key_norms + 1e-8)
        return logits

    @torch.inference_mode()
    def calc_topk(self, question): 

        """ Calculate the top-k retrieved blocks for a question.
        
        Args:
            question: (str) the question to encode
            temporal_patch_size: (float) the number of temporal patches to merge frames into a single feature

        Returns:
            ret: (list) the top-k retrieved blocks
        """


        kv_cache_features = torch.cat(self.kv_cache, dim=0) # (num_frames, D) 
        
        num_frames = kv_cache_features.shape[0] 
        dim = kv_cache_features.shape[1]

        merge_size = max(1, (self.chunk_size * self.temporal_patch_size))
        remainder_size = int(num_frames % merge_size)
        logger.info(f"Cache Size: {kv_cache_features.shape} - Remainder Size: {remainder_size}")
        
        cropped_size = num_frames - remainder_size
        chunked_kv_cache_features = kv_cache_features[:cropped_size].reshape(-1, merge_size, dim).mean(1) # num_chunks, dim         
        if remainder_size > 0: 
            remainder_kv_cache_features = kv_cache_features[-remainder_size:].mean(dim=0, keepdim=True) # (1, D) 
            chunked_kv_cache_features = torch.cat([chunked_kv_cache_features, remainder_kv_cache_features], dim=0) # (num_chunks + 1, D) 
        
        question_features = self.encode_question(question).to(kv_cache_features.device) # (1, D) 
        logits =  self.get_cosine_similarity(chunked_kv_cache_features, question_features) # (num_frames, 1)
        self.similarity = logits.clone().detach().cpu()

        # Provide full sorted rankings (chunk-level) instead of truncating to top-k
        chunk_scores = logits.squeeze(-1)
        sorted_chunk_indices = torch.argsort(chunk_scores, dim=0, descending=True)
        if sorted_chunk_indices.dim() == 0:
            sorted_chunk_indices = sorted_chunk_indices.unsqueeze(0)
        self.retrieved_block_indices = [sorted_chunk_indices.cpu().tolist()]

        return question_features, chunked_kv_cache_features
    
    def calc_memory_usage(self):
        # Placeholder for memory calculation
        if self.kv_cache is None or len(self.kv_cache) == 0:
            return 0
        
        total_elements = sum(f.numel() for f in self.kv_cache)
        total_bytes = total_elements * self.kv_cache[0].element_size()
        logger.info(f"KV Cache Memory Usage: {total_bytes / (1024**2):.2f} MB")
        return total_bytes

    @property
    def device(self):
        assert hasattr(self, 'model') and self.model is not None, "Model not loaded"
        return self.model.device

def load_model(model_config='openai/clip-vit-large-patch14',
               topk=64, chunk_size=1, encode_chunk_size=8, **kwargs):
    model = CLIP_ReKV(
        model_config, 
        topk, chunk_size, encode_chunk_size, 
        sample_fps=kwargs.get("sample_fps", 0.5), 
        temporal_patch_size=kwargs.get("temporal_patch_size", 1.0)
    )
    processor = model.processor
    return model, processor