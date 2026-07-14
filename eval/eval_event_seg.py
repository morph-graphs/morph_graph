"""Load a video, sample frames, and encode them with a vision model."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from ezcolorlog import root_logger as logger
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "event_chunk"

DEFAULT_VIDEO = (
    REPO_ROOT / "data" / "wearable-ai" / "egolongqa" / "val" / "00cdb8ca10c069f8.mp4"
)

# DEFAULT_VIDEO = ( 
#     REPO_ROOT / "data"/ "vsi_morph" / "vsi_constructed" / "Cat" / "0000.mp4"
# )

class EventSeg:
    @staticmethod
    def _adj_similarities(features: torch.Tensor) -> np.ndarray:
        features = features.detach().cpu().float()
        return torch.nn.functional.cosine_similarity(features[:-1], features[1:], dim=1).numpy()

    @staticmethod
    def _seg_result(
        sims: np.ndarray,
        boundaries: np.ndarray,
        thresholds: np.ndarray,
    ) -> dict[str, np.ndarray]:
        boundary_frames = np.flatnonzero(boundaries) + 1
        return {
            "similarities": sims,
            "thresholds": thresholds,
            "boundaries": boundaries,
            "boundary_frames": boundary_frames,
        }

    @staticmethod
    def adj_threshold_chunking(
        features: torch.Tensor,
        threshold: float,
    ) -> dict[str, np.ndarray]:
        """Mark a boundary wherever adjacent similarity falls below a fixed threshold."""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")

        sims = EventSeg._adj_similarities(features)
        boundaries = sims < threshold
        thresholds = np.full(len(sims), threshold, dtype=np.float32)
        return EventSeg._seg_result(sims, boundaries, thresholds)

    @staticmethod
    def adj_percentile_chunking(
        features: torch.Tensor,
        percentile: float,
        *,
        min_history: int = 2,
    ) -> dict[str, np.ndarray]:
        """Detect event boundaries with a causal running percentile threshold.

        For each adjacent similarity s_t, compare against the p-th percentile of
        all previous similarities {s_0, ..., s_{t-1}}. If s_t is below that
        running threshold, mark frame t+1 as an event boundary.
        """
        if not 0.0 <= percentile <= 100.0:
            raise ValueError("percentile must be in [0, 100]")

        sims = EventSeg._adj_similarities(features)
        n = sims.shape[0]
        thresholds = np.full(n, np.nan, dtype=np.float32)
        boundaries = np.zeros(n, dtype=bool)
        history: list[float] = []

        for i, sim in enumerate(sims):
            if len(history) >= min_history:
                thresh = float(np.percentile(history, percentile))
                thresholds[i] = thresh
                boundaries[i] = sim < thresh
            history.append(float(sim))

        return EventSeg._seg_result(sims, boundaries, thresholds)

    @staticmethod
    def chunk(
        features: torch.Tensor,
        strategy: str,
        *,
        percentile: float = 25.0,
        threshold: float = 0.75,
        min_history: int = 2,
    ) -> dict[str, np.ndarray]:
        if strategy == "threshold":
            return EventSeg.adj_threshold_chunking(features, threshold)
        if strategy == "percentile":
            return EventSeg.adj_percentile_chunking(
                features,
                percentile,
                min_history=min_history,
            )
        raise ValueError(f"Unknown chunking strategy: {strategy}")

    @staticmethod
    def otsu_threshold(values: torch.Tensor, nbins: int = 128, fallback_to_median: bool = False) -> float | None:
        if values.numel() == 0:
            return None

        v = values.detach().float().clamp(0.0, 2.0)
        if fallback_to_median and float(v.var().item()) <= 1e-12:
            return float(torch.median(v).item())

        hist = torch.histc(v, bins=nbins, min=0.0, max=2.0)
        total = float(hist.sum().item())
        if total <= 0:
            return float(torch.median(v).item()) if fallback_to_median else None

        p = hist / total
        step = 2.0 / nbins
        centers = (torch.arange(nbins, dtype=torch.float32, device=v.device) + 0.5) * step
        omega = torch.cumsum(p, 0)
        mu_k = torch.cumsum(p * centers, 0)
        mu_total = mu_k[-1]
        denom = (omega * (1.0 - omega)).clamp_min(1e-12)
        sigma_b2 = (mu_total * omega - mu_k) ** 2 / denom
        sigma_b2[omega < 1e-6] = -1
        sigma_b2[(1.0 - omega) < 1e-6] = -1
        threshold_index = int(torch.argmax(sigma_b2).item())
        if float(sigma_b2[threshold_index].item()) <= 0.0:
            return float(torch.median(v).item()) if fallback_to_median else None
        return float((threshold_index + 0.5) * step)


def load_video(video_path: str):
    from torchcodec.decoders import VideoDecoder    
    return VideoDecoder(video_path)


def get_frames(decoder, fps: float) -> torch.Tensor:
    frames, _ = get_frames_with_indices(decoder, fps)
    return frames


def sample_frame_indices(decoder, fps: float) -> list[int]:
    metadata = decoder.metadata
    total_frames = metadata.num_frames
    video_fps = metadata.average_fps

    if total_frames <= 0:
        raise ValueError("No frames available in video")
    if fps <= 0:
        raise ValueError("fps must be positive")

    if video_fps is None or video_fps <= 0:
        return list(range(total_frames))

    nframes = max(1, min(int(round(total_frames / video_fps * fps)), total_frames))
    if nframes == 1:
        return [0]
    return torch.linspace(0, total_frames - 1, nframes).round().long().tolist()


def get_frames_with_indices(decoder, fps: float) -> tuple[torch.Tensor, list[int]]:
    indices = sample_frame_indices(decoder, fps)
    batch = decoder.get_frames_at(indices)
    frames = batch.data if hasattr(batch, "data") else batch
    logger.info(f"Sampled {len(indices)} frames from {decoder.metadata.num_frames} total frames")
    return frames, indices


def events_from_boundaries(boundary_frames: np.ndarray, n_frames: int) -> list[tuple[int, int]]:
    cuts = [0, *[int(frame) for frame in boundary_frames], n_frames]
    return [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]


def frame_to_pil(frames: torch.Tensor, index: int) -> Image.Image:
    frame = frames[index].detach().cpu()
    if frame.dtype != torch.uint8:
        frame = frame.clamp(0, 255).to(torch.uint8)
    if frame.ndim == 3 and frame.shape[0] in (1, 3, 4):
        frame = frame.permute(1, 2, 0)
    arr = frame.numpy()
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L").convert("RGB")
    return Image.fromarray(arr, mode="RGB")


def load_model_and_processor(ckpt: str = "google/siglip2-base-patch16-224"):
    model = AutoModel.from_pretrained(ckpt, device_map="auto").eval()
    processor = AutoImageProcessor.from_pretrained(ckpt)
    logger.info(f"Loaded model and processor from {ckpt}")
    return model, processor


def encode_frames(
    model,
    processor,
    frames: torch.Tensor,
    batch_size: int = 32,
) -> torch.Tensor:
    frames = frames.detach().cpu()
    if frames.dtype != torch.uint8:
        frames = frames.clamp(0, 255).to(torch.uint8)
    if frames.ndim == 4 and frames.shape[1] in (1, 3, 4):
        frames = frames.permute(0, 2, 3, 1)

    features: list[torch.Tensor] = []
    for start in tqdm(range(0, len(frames), batch_size)):
        batch = frames[start : start + batch_size]
        images = []
        for arr in batch.numpy():
            if arr.ndim == 2:
                images.append(Image.fromarray(arr, mode="L").convert("RGB"))
            else:
                images.append(Image.fromarray(arr, mode="RGB"))

        inputs = processor(images=images, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.get_image_features(**inputs)
            if not isinstance(out, torch.Tensor):
                out = out.pooler_output
        features.append(out.cpu())

    return torch.cat(features, dim=0)


def adjacent_similarity(features: torch.Tensor) -> np.ndarray:
    features = features.detach().cpu().float()
    return torch.nn.functional.cosine_similarity(features[:-1], features[1:], dim=1).numpy()


def compute_sim_matrix(features: torch.Tensor, batch_size: int = 32) -> np.ndarray:
    features = torch.nn.functional.normalize(features.detach().cpu().float(), dim=1)
    n = features.shape[0]
    sims = torch.zeros((n, n))

    for i_start in tqdm(range(0, n, batch_size), desc="Similarity matrix"):
        i_batch = features[i_start : i_start + batch_size]
        for j_start in range(0, n, batch_size):
            j_batch = features[j_start : j_start + batch_size]
            block = i_batch @ j_batch.T
            sims[
                i_start : i_start + i_batch.shape[0],
                j_start : j_start + j_batch.shape[0],
            ] = block

    return sims.numpy()


def save_event_seg_figure(
    features: torch.Tensor,
    output_path: Path,
    *,
    fps: float | None = None,
    batch_size: int = 32,
    boundary_frames: np.ndarray | None = None,
    thresholds: np.ndarray | None = None,
) -> Path:
    """Save adjacent similarity (top) and self-similarity matrix (bottom) to a PDF."""
    adj = adjacent_similarity(features)
    sims = compute_sim_matrix(features, batch_size=batch_size)
    n = sims.shape[0]

    fig, (ax_adj, ax_sim) = plt.subplots(
        2,
        1,
        figsize=(12, 10),
        gridspec_kw={"height_ratios": [1, 2.2]},
    )

    x = np.arange(len(adj))
    ax_adj.plot(x, adj, color="#2563eb", linewidth=1.5, marker="o", markersize=3, label="adjacent sim")
    ax_adj.set_ylim(0.0, 1.05)
    ax_adj.set_xlabel("Frame index")
    ax_adj.set_ylabel("Cosine similarity")
    ax_adj.set_title("Adjacent frame similarity")
    ax_adj.grid(True, alpha=0.25)

    if thresholds is not None:
        valid = ~np.isnan(thresholds)
        if valid.any():
            ax_adj.plot(
                x[valid],
                thresholds[valid],
                color="#ea580c",
                linewidth=1.5,
                linestyle="--",
                label="running threshold",
            )

    if boundary_frames is not None:
        for frame in boundary_frames:
            ax_adj.axvline(
                frame - 1,
                color="#dc2626",
                linewidth=1.2,
                alpha=0.7,
                linestyle=":",
            )

    legend_handles = []
    if thresholds is not None and (~np.isnan(thresholds)).any():
        legend_handles.append(
            Line2D([0], [0], color="#ea580c", linewidth=1.5, linestyle="--", label="running threshold")
        )
    if boundary_frames is not None and len(boundary_frames) > 0:
        legend_handles.append(
            Line2D([0], [0], color="#dc2626", linewidth=1.2, linestyle=":", label="event boundary")
        )
    if legend_handles:
        ax_adj.legend(handles=legend_handles, loc="lower left", fontsize=9)

    if fps is not None:
        ax_time = ax_adj.twiny()
        ax_time.set_xlim(ax_adj.get_xlim())
        ax_time.set_xticks(ax_adj.get_xticks())
        ax_time.set_xticklabels([f"{tick / fps:.1f}" for tick in ax_adj.get_xticks()])
        ax_time.set_xlabel("Time (s)")

    im = ax_sim.imshow(sims, cmap="viridis", vmin=0.0, vmax=1.0, aspect="equal")
    ax_sim.set_xlabel("Frame index")
    ax_sim.set_ylabel("Frame index")
    ax_sim.set_title("Frame self-similarity")
    if n <= 20:
        ax_sim.set_xticks(range(n))
        ax_sim.set_yticks(range(n))

    if boundary_frames is not None:
        for frame in boundary_frames:
            ax_sim.axvline(frame - 0.5, color="#dc2626", linewidth=1.0, alpha=0.7, linestyle=":")
            ax_sim.axhline(frame - 0.5, color="#dc2626", linewidth=1.0, alpha=0.7, linestyle=":")

    cbar = fig.colorbar(im, ax=ax_sim, fraction=0.046, pad=0.04)
    cbar.set_label("Cosine similarity")
    fig.tight_layout()

    output_path = Path(output_path)
    if output_path.suffix.lower() != ".pdf":
        output_path = output_path.with_suffix(".pdf")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Wrote figure to {output_path}")
    return output_path


def plot_adj_similarity(features: torch.Tensor, *, fps: float | None = None) -> None:
    """Plot cosine similarity between consecutive frame features."""
    y = adjacent_similarity(features)
    x = range(len(y))

    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(x, y, color="#2563eb", linewidth=1.5, marker="o", markersize=3)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Cosine similarity")
    ax.set_title("Adjacent frame similarity")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    if fps is not None:
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(ax.get_xticks())
        ax2.set_xticklabels([f"{tick / fps:.1f}" for tick in ax.get_xticks()])
        ax2.set_xlabel("Time (s)")

    plt.show()


def plot_sim_matrix(features: torch.Tensor, batch_size: int = 32) -> None:
    """Plot cosine similarity matrix between all frame features."""
    sims_np = compute_sim_matrix(features, batch_size=batch_size)
    n = sims_np.shape[0]

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(sims_np, cmap="viridis", vmin=0.0, vmax=1.0, aspect="equal")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Frame index")
    ax.set_title("Frame feature cosine similarity")
    if n <= 20:
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cosine similarity")
    fig.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode sampled video frames.")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--ckpt", default="google/siglip2-base-patch16-224")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--strategy",
        choices=["percentile", "threshold"],
        default="percentile",
        help="Boundary detection strategy",
    )
    parser.add_argument("--percentile", type=float, default=25.0)
    parser.add_argument("--threshold", type=float, default=0.75, help="Used when --strategy=threshold")
    args = parser.parse_args()

    video_path = args.video.resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    decoder = load_video(str(video_path))
    frames = get_frames(decoder, args.fps)
    model, processor = load_model_and_processor(args.ckpt)
    features = encode_frames(model, processor, frames, batch_size=args.batch_size)

    print(f"Video:    {video_path}")
    print(f"Frames:   {frames.shape}")
    print(f"Features: {features.shape}")

    seg = EventSeg.chunk(
        features,
        args.strategy,
        percentile=args.percentile,
        threshold=args.threshold,
    )
    boundary_frames = seg["boundary_frames"]
    print(f"Strategy: {args.strategy}")
    print(f"Boundary frames: {boundary_frames.tolist()}")

    output_path = args.output_dir / f"{video_path.stem}.pdf"
    save_event_seg_figure(
        features,
        output_path,
        fps=args.fps,
        batch_size=args.batch_size,
        boundary_frames=boundary_frames,
        thresholds=seg["thresholds"],
    )


if __name__ == "__main__":
    main()
