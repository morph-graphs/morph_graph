import torch
import torch.nn.functional as F
from logzero import logger
from tqdm import tqdm

class Abstract_ReKV:
    processor = None
    kv_cache = None

    def __init__(self, processor, n_frame_tokens, init_prompt_ids, n_local, topk, encode_chunk_size, chunk_size):
        self.processor = processor
        self.n_frame_tokens = n_frame_tokens
        self.init_prompt_ids = init_prompt_ids
        self.n_local = n_local
        self.topk = topk
        self.encode_chunk_size = encode_chunk_size
        self.chunk_size = chunk_size

    def clear_cache(self):
        if self.kv_cache is not None:
            for layer_kv in self.kv_cache:
                reset_fn = getattr(layer_kv, "reset_position_embedding", None)
                if callable(reset_fn):
                    reset_fn(self.n_local)
        self.kv_cache = None
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


    @torch.inference_mode()
    def encode_init_prompt(self):
        if not isinstance(self.init_prompt_ids, torch.Tensor):
            self.init_prompt_ids = torch.as_tensor([self.init_prompt_ids], device=self.device)
        output = self.language_model(input_ids=self.init_prompt_ids, use_cache=True, return_dict=True)
        self.kv_cache = output.past_key_values

    def get_tokens_per_frame(self, video_slice): 
        pass 

    def _get_video_features(self, pixel_values_videos):
        pass

    def _encode_video_chunk(self, video_chunk):
        pixel_values_videos = self.processor.video_processor(video_chunk, return_tensors="pt").pixel_values_videos.to(self.device, self.dtype)  # (1, Nv, 3, H, W)
        video_features = self._get_video_features(pixel_values_videos)  # (1, Nv*196, D)
        assert self.n_local >= video_features.shape[1], f'n_local: {self.n_local}, video_features: {video_features.shape[1]}'
        output = self.language_model.model(inputs_embeds=video_features, past_key_values=self.kv_cache, use_cache=True, return_dict=True)
        self.kv_cache = output.past_key_values 

    @torch.inference_mode()
    def encode_video(self, video, **kwargs):  # video: (Nv, H, W, 3)
        # Accept kwargs for compatibility (e.g., streaming, target_frames)
        # Base implementation ignores these, subclasses can override
        
        # encode chunk by chunk
        num_frames = video.shape[0]
        encode_chunk_size = self.encode_chunk_size
        
        num_chunks = num_frames // encode_chunk_size

        for chunk_idx in tqdm(range(num_chunks), desc="Encoding video chunks"):
            start_idx = chunk_idx * encode_chunk_size
            end_idx = start_idx + encode_chunk_size
            chunk_video = video[start_idx:end_idx]
            self._encode_video_chunk(chunk_video)
            logger.debug(f'KV-Cache RAM usage: {self.calc_memory_usage() / (1024**3):.1f} GB')

        # Handle remaining frames
        remaining_frames = num_frames % encode_chunk_size
        if remaining_frames > 0:
            start_idx = num_chunks * encode_chunk_size
            end_idx = start_idx + remaining_frames
            remaining_video = video[start_idx:end_idx]
            self._encode_video_chunk(remaining_video)
        
        logger.debug(f'KV-Cache RAM usage: {self.calc_memory_usage() / (1024**3):.1f} GB')

    @torch.inference_mode()
    def question_answering(self, input_text, max_new_tokens=128):
        pass

    def calc_memory_usage(self):
        n_layers = len(self.kv_cache)
        memory = n_layers * self.kv_cache[0].calculate_cpu_memory()
        return memory

    # === Similarity Tracking Helper Methods ===
    
    def _init_similarity_tracking(self):
        """Initialize data structures for tracking adjacent frame similarities."""
        self.visual_buffer = [] 
        self.last_frame_repr = None
        self.adjacent_vis_similarities = []
        self.running_threshold = None
    
    def _reset_similarity_tracking(self):
        """Reset all similarity tracking state."""
        self.visual_buffer = []
        self.last_frame_repr = None
        self.adjacent_vis_similarities = []
        self.running_threshold = None
    
    def _update_similarity_tracking(self, video_embeds):
        """
        Update similarity tracking with new video embeddings.
        
        Args:
            video_embeds: Video embeddings tensor of shape (1, Nt*N, hidden_dim)
        """
        video_embeds_lst = video_embeds.split(self.n_frame_tokens, dim=1)  # list([1, Nt, 3584])
        if video_embeds_lst:
            frame_reprs = torch.stack([v.cpu().mean(dim=1) for v in video_embeds_lst], dim=1)
            sims = []
            
            # Compute similarity between last frame of previous chunk and first frame of current chunk
            if self.last_frame_repr is not None:
                sims.append(F.cosine_similarity(self.last_frame_repr, frame_reprs[:, 0, :], dim=-1))
                sims[0] = sims[0][None, :]
            
            # Compute similarities between consecutive frames within this chunk
            if frame_reprs.size(1) > 1:
                sims.append(F.cosine_similarity(frame_reprs[:, :-1, :], frame_reprs[:, 1:, :], dim=-1))
            
            # Update last frame representation
            self.last_frame_repr = frame_reprs[:, -1, :]
            
            # Update running similarities and threshold
            if sims:
                sims = torch.cat(sims, dim=-1).flatten()
                self.adjacent_vis_similarities.extend(sims.tolist())
                sims_tensor = torch.tensor(self.adjacent_vis_similarities, device=sims.device)
                self.running_threshold = [
                    torch.quantile(sims_tensor[:i + 1], 0.10).item() for i in range(sims_tensor.numel())
                ]
        
        self.visual_buffer.extend(video_embeds_lst)
    
    def _finalize_similarity_tracking(self):
        """Finalize similarity computation after all frames have been processed."""
        self.visual_buffer = torch.stack(self.visual_buffer, dim=1)  # (1, N, Nt, 3584)
        self.adjacent_vis_similarities = F.cosine_similarity(
            self.visual_buffer.mean(dim=2)[:, :-1, :], 
            self.visual_buffer.mean(dim=2)[:, 1:, :], 
            dim=-1
        )

    # === End Similarity Tracking Helper Methods ===
