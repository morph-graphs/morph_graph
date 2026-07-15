"""Qwen3-VL helpers for MCQ answering during eval."""

from __future__ import annotations

from pathlib import Path

import torch
from ezcolorlog import root_logger as logger
from transformers import AutoModelForImageTextToText, AutoProcessor

MCQ_POST_PROMPT = "Answer with the option's letter from the given choices directly."
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32


def model_path_to_key(model_path: str) -> str:
    """Short key for results filenames."""
    p = model_path.lower().replace("\\", "/")
    if "qwen3-vl-8b" in p:
        return "qwen3_vl_8b"
    if "qwen3-vl-30b-a3b" in p:
        return "qwen3_vl_30b_a3b"
    if "qwen3-vl-30b" in p:
        return "qwen3_vl_30b"
    return Path(model_path.rstrip("/")).name.lower().replace("-", "_")


def load_model(model_path: str):
    logger.info(f"Loading model: {model_path}")
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=DTYPE,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(model_path)
    return model.eval(), processor


def _mcq_suffix(question: str, options: list[str]) -> str:
    return f"{question.strip()}\nOptions:\n" + "\n".join(options) + f"\n{MCQ_POST_PROMPT}"


def build_messages_from_caption_context(
    caption_context: str,
    question: str,
    options: list[str],
) -> list[dict]:
    """Text-only MCQ prompt from retrieved segment captions (default QA path)."""
    content = (
        "Here is a frame-by-frame description of a video (rooms, scene boundaries, "
        "and object sightings):\n\n"
        f"{caption_context}\n\nAnswer the following question based only on this description:\n\n"
        f"{_mcq_suffix(question, options)}"
    )
    return [{"role": "user", "content": content}]


def build_messages_video(
    video_path: str,
    question: str,
    options: list[str],
    *,
    fps: float = 1.0,
    max_frames: int = 2048,
    min_pixels: int = 256 * 32 * 32,
    n_frames: int | None = None,
) -> list[dict]:
    """Chat messages for a pre-decoded video tensor passed via `run_inference_on_video_tensor`."""
    video: dict = {"type": "video", "video": video_path, "min_pixels": min_pixels}
    if n_frames is not None:
        video["nframes"] = n_frames
    else:
        video["fps"] = fps
        video["max_frames"] = max_frames
    return [
        {
            "role": "user",
            "content": [
                video,
                {"type": "text", "text": _mcq_suffix(question, options)},
            ],
        }
    ]


def build_messages_open_qa(question: str) -> list[dict]:
    """Chat messages for open-ended QA on a stitched video tensor."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": question.strip()},
            ],
        }
    ]


def build_messages_video_tensor(
    question: str,
    options: list[str],
) -> list[dict]:
    """Chat messages for a pre-decoded video tensor passed via `run_inference_on_video_tensor`."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": _mcq_suffix(question, options)},
            ],
        }
    ]


def run_inference_on_video_tensor(
    model,
    processor,
    video_tensor,
    messages: list[dict],
    *,
    video_metadata: dict | None = None,
    max_new_tokens: int = 32,
) -> str:
    """Run MCQ inference on an in-memory video tensor (e.g. stitched RAG segments)."""
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    processor_kwargs = dict(
        text=[text],
        videos=[video_tensor],
        padding=True,
        return_tensors="pt",
    )
    if video_metadata is not None:
        processor_kwargs["video_metadata"] = [video_metadata]
    inputs = processor(**processor_kwargs).to(DEVICE)
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


def run_inference(model, processor, messages: list[dict], max_new_tokens: int = 32) -> str:
    image_inputs, video_inputs, video_metadata, video_kwargs = None, None, None, {}
    try:
        from qwen_vl_utils import process_vision_info

        patch_size = getattr(getattr(processor, "image_processor", None), "patch_size", 16) or 16
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=patch_size,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        if video_inputs is not None:
            video_inputs, video_metadata = zip(*video_inputs)
            video_inputs, video_metadata = list(video_inputs), list(video_metadata)
    except ImportError:
        pass

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    processor_kwargs = dict(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    if video_inputs is not None:
        processor_kwargs["video_metadata"] = video_metadata
        processor_kwargs["do_resize"] = False
        processor_kwargs.update(video_kwargs)

    inputs = processor(**processor_kwargs).to(DEVICE)
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
