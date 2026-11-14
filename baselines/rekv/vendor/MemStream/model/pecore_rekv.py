import os 
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from logzero import logger

# PECore libraries 
import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms

from model.abstract_ext_rekv import Abstract_Ext_ReKV

class PECore_ReKV(Abstract_Ext_ReKV):

    device = 'cuda' 
    kv_cache = None 

    def __init__(self, model_config, topk, chunk_size, encode_chunk_size=8, sample_fps=0.5, temporal_patch_size=1.0):
        Abstract_Ext_ReKV.__init__(self, model_config, topk, chunk_size, encode_chunk_size, sample_fps, temporal_patch_size)

        self.model, self.preprocess, self.tokenizer = self.load_pecore(model_config)

    def clear_cache(self):
        self.kv_cache = []

    def load_pecore(self, model_config):
        if os.path.exists("/opt/tiger/MemStream/model_zoo/PE-Core-L14-336/PE-Core-L14-336.pt"): 
            checkpoint_path = "/opt/tiger/MemStream/model_zoo/PE-Core-L14-336/PE-Core-L14-336.pt"
            model = pe.CLIP.from_config(model_config, pretrained=True, checkpoint_path=checkpoint_path)
        else:
            model = pe.CLIP.from_config(model_config, pretrained=True)
        preprocess = transforms.get_image_transform(model.image_size)
        tokenizer = transforms.get_text_tokenizer(model.context_length)
        model = model.to(self.device) 
        model.eval()
        return model, preprocess, tokenizer 

    @torch.inference_mode()
    def _encode_video_chunk(self, video_chunk):
        frm_feats = self.model.encode_image(video_chunk, normalize=False) 
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
        
        video = [self.preprocess(Image.fromarray(frame)) for frame in video_np]
        video = torch.stack(video, dim=0)

        for chunk_idx in tqdm(range(num_chunks), desc="Encoding PECore video chunks..."):
            start_idx = chunk_idx * self.encode_chunk_size
            end_idx = start_idx + self.encode_chunk_size
            chunk_video = video[start_idx:end_idx].to(self.device)
            self._encode_video_chunk(chunk_video) 

        # Handle remaining frames
        remaining_frames = num_frames % self.encode_chunk_size
        if remaining_frames > 0:
            start_idx = num_chunks * self.encode_chunk_size
            end_idx = start_idx + remaining_frames
            remaining_video = video[start_idx:end_idx].to(self.device)
            self._encode_video_chunk(remaining_video)

        video_kwargs = {
            "sample_fps": new_fps 
        }

        return video_kwargs 
        
    @torch.inference_mode()
    def encode_question(self, question):
        input_ids = self.tokenizer(question).to(self.device) 
        question_features = self.model.encode_text(input_ids) 
        return question_features

    def get_cosine_similarity(self, query_features, key_features):
        logits = query_features @ key_features.T # (n, k)
        query_norms = query_features.norm(dim=-1, keepdim=True)
        key_norms = key_features.norm(dim=-1, keepdim=True)
        logits = logits / (query_norms * key_norms+ 1e-8)
        return logits

    def get_block_and_adjacent_similarities(self):
        kv_cache_features = torch.cat(self.kv_cache, dim=0) # (num_frames, D)
        dim = kv_cache_features.shape[1]
        remainder_size = int(kv_cache_features.shape[0] % (self.chunk_size * self.temporal_patch_size))        
        cropped_size = kv_cache_features.shape[0] - remainder_size
        chunked_kv_cache_features = kv_cache_features[:cropped_size].reshape(-1, self.chunk_size * self.temporal_patch_size, dim).mean(1) # num_chunks, dim
        # Drop the trailing partial group, matching calc_topk and the QA model's truncation. Keeping
        # it would make these dumps span one more block than the retrieval indexing, so block i in
        # the similarity figures would not be block i in the cache.
        self.block_sims = self.get_cosine_similarity(chunked_kv_cache_features, chunked_kv_cache_features).clone().detach().cpu().numpy() 
        self.adjacent_sims = F.cosine_similarity(chunked_kv_cache_features[:-1], chunked_kv_cache_features[1:], dim=-1)
        return self.block_sims, self.adjacent_sims

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
        # DROP the tail rather than appending it as an extra block. The QA model discards trailing
        # frames that do not fill a temporal group (qwen3_vl_rekv.process_video truncates to a
        # multiple of temporal_factor), so those frames are never in the KV cache. Appending them
        # here yields ceil(T/merge_size) external blocks against floor(T/merge_size) internal ones,
        # which trips `assert len(current_rankings) == len(ext_rankings)` in _calc_block_topk for
        # every odd-T video -- and, where the lengths happened to agree, would have made the RRF
        # fusion index a block the cache does not hold. Only backbones with temporal_patch_size > 1
        # (Qwen3-VL) can have a non-zero remainder here.
        if remainder_size > 0:
            logger.info(
                f"dropping {remainder_size} trailing frame(s) to match the QA model's "
                f"temporal-group truncation ({num_frames} -> {cropped_size} frames, "
                f"{chunked_kv_cache_features.shape[0]} blocks)"
            )

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
        pass 

def load_model(model_config='PE-Core-L14-336',
               topk=64, chunk_size=1, encode_chunk_size=8, **kwargs):
    model = PECore_ReKV(
        model_config, 
        topk, chunk_size, 
        encode_chunk_size, 
        sample_fps=kwargs.get("sample_fps", 0.5), 
        temporal_patch_size=kwargs.get("temporal_patch_size", 1.0)
    )
    processor = model.preprocess
    return model, processor