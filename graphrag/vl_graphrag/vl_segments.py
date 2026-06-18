import re
from pathlib import Path
from typing import Any


_SEGMENT_LINE_RE = re.compile(
    r"^-\s*\[(?P<label>.+?),\s*frames\s*(?P<start>\d+)-(?P<end>\d+)\]\s*(?P<caption>.*)$"
)


def _video_id_from_path(video_path: str) -> str:
    if not video_path:
        return "unknown"
    path = Path(video_path)
    return f"{path.parent.name}/{path.stem}"


def _node_id(video_id: str, segment_idx: int) -> str:
    return f"{video_id}::seg_{segment_idx:04d}"


def _build_key_text(segment: dict[str, Any]) -> str:
    key_parts = []
    scene_label = segment.get("scene_label")
    frame_range = segment.get("frame_range")
    content = str(segment.get("content", "")).strip()

    if scene_label is not None:
        key_parts.append(str(scene_label))
    if frame_range is not None:
        key_parts.append(f"frames {frame_range[0]}-{frame_range[1]}")

    prefix = f"[{', '.join(key_parts)}] " if key_parts else ""
    return f"{prefix}{content}".strip()


def parse_caption_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Parse one bullet-caption row into generic segment nodes."""
    video_path = str(row.get("video_path", ""))
    video_id = str(row.get("video_id") or _video_id_from_path(video_path))
    caption_blob = str(row.get("caption", ""))

    segments = []
    for segment_idx, line in enumerate(caption_blob.splitlines()):
        match = _SEGMENT_LINE_RE.match(line.strip())
        if match is None:
            continue

        scene_label = match.group("label").strip()
        start_frame = int(match.group("start"))
        end_frame = int(match.group("end"))
        caption = match.group("caption").strip()
        node_id = _node_id(video_id, segment_idx)
        segment = {
            "node_id": node_id,
            "source_id": node_id,
            "video_id": video_id,
            "video_path": video_path,
            "segment_idx": segment_idx,
            "scene_label": scene_label,
            "frame_range": [start_frame, end_frame],
            "caption": caption,
            "content": caption,
        }
        segment["key_text"] = _build_key_text(segment)
        segments.append(segment)
    return segments


def normalize_text_segments(data: Any) -> list[dict[str, Any]]:
    """Normalize supported generic segment inputs into segment nodes.

    A `caption` field is treated as a per-video bullet-caption blob in the
    format emitted by `per_video_segment_captions_*.json`. Generic single
    segment records should provide `key_text` or `content`.
    """
    if isinstance(data, str):
        return parse_caption_row({"caption": data})

    if isinstance(data, dict):
        if "caption" in data:
            return parse_caption_row(data)
        segment = dict(data)
        video_id = str(segment.get("video_id") or _video_id_from_path(str(segment.get("video_path", ""))))
        segment_idx = int(segment.get("segment_idx", 0))
        node_id = str(segment.get("node_id") or _node_id(video_id, segment_idx))
        key_text = str(segment.get("key_text") or _build_key_text(segment))
        if not key_text:
            raise ValueError(
                "Text segment records need `key_text` or `content` for caption-key indexing"
            )
        segment.update(
            {
                "node_id": node_id,
                "source_id": str(segment.get("source_id", node_id)),
                "video_id": video_id,
                "segment_idx": segment_idx,
                "key_text": key_text,
            }
        )
        if "frame_range" in segment and segment["frame_range"] is not None:
            segment["frame_range"] = list(segment["frame_range"])
        return [segment]

    if isinstance(data, list):
        segments: list[dict[str, Any]] = []
        for item in data:
            segments.extend(normalize_text_segments(item))
        return segments

    raise TypeError(f"Unsupported text segment input type: {type(data)!r}")


def _build_vision_key(segment: dict[str, Any]) -> Any:
    if "vision_key" in segment:
        return segment["vision_key"]
    if "content" in segment:
        return segment["content"]

    key = {
        k: segment[k]
        for k in ["video_path", "clip_path", "frame_range", "sampled_frames"]
        if k in segment
    }
    return key or None


def normalize_vision_segments(data: Any) -> list[dict[str, Any]]:
    """Normalize supported generic segment inputs for video-key indexing.

    Each segment must provide a `vision_key`, `content`, or enough video
    metadata (`video_path`/`clip_path` plus optional frame fields) for the
    injected embedding function to load/embed the clip.
    """
    if isinstance(data, dict):
        if "caption" in data:
            segments = parse_caption_row(data)
            for segment in segments:
                segment["vision_key"] = _build_vision_key(segment)
            return segments

        segment = dict(data)
        video_id = str(segment.get("video_id") or _video_id_from_path(str(segment.get("video_path", ""))))
        segment_idx = int(segment.get("segment_idx", 0))
        node_id = str(segment.get("node_id") or _node_id(video_id, segment_idx))
        vision_key = _build_vision_key(segment)
        if vision_key is None:
            raise ValueError(
                "Vision segment records need `vision_key`, `content`, `video_path`, or `clip_path` for video-key indexing"
            )
        segment.update(
            {
                "node_id": node_id,
                "source_id": str(segment.get("source_id", node_id)),
                "video_id": video_id,
                "segment_idx": segment_idx,
                "vision_key": vision_key,
            }
        )
        if "frame_range" in segment and segment["frame_range"] is not None:
            segment["frame_range"] = list(segment["frame_range"])
        return [segment]

    if isinstance(data, list):
        segments: list[dict[str, Any]] = []
        for item in data:
            segments.extend(normalize_vision_segments(item))
        return segments

    raise TypeError(f"Unsupported vision segment input type: {type(data)!r}")
