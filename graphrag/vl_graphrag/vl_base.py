from dataclasses import dataclass, field
from typing import Literal

import numpy as np

@dataclass
class VLQueryParam:
    # different strategies for query retrieval
    mode: Literal["local", "global", "naive"] = "naive"
    # which key embedding index to query
    key_source: Literal["caption", "video"] = "caption"
    # return only the retrieved context without generating the response 
    only_need_context: bool = False 
    # TODO: Implement this response type 
    response_type: str = "Single Sentence"
    # leiden community depth  
    level: int = 2 
    # Vector Database seed count (how many nodes to seed for retrieval) 
    top_k: int = 3 

    # max frames/segments retrieved for QA 
    max_segments_for_qa: int = 5 
    max_frames_for_video_unit: int = 128 



    