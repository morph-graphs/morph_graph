"""Smoke test for the Qwen3.8 env.

Usage:
  python smoke_qwen38.py --skip-model          # imports + kernels only
  python smoke_qwen38.py                       # + text generation
  python smoke_qwen38.py --video /path/clip.mp4
  python smoke_qwen38.py --attn sdpa           # compare against flash_attention_2
"""
import argparse
import importlib
import time
import torch
from ezcolorlog import root_logger as logger 

QWEN_MODEL_PATH="/home/vatsalag/scratch/model_hub/Qwen3.8-27B"
VIDEO_PATH="/scratch/zt1/project/abhinav2-prj/user/vatsalag/datasets/HD-EPIC/videos/P09/P09-20240625-103827.mp4"

def check_imports():
    print(f"torch {torch.__version__}  cuda={torch.version.cuda}  "
          f"available={torch.cuda.is_available()}  gpus={torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"  device0: {torch.cuda.get_device_name(0)}  bf16={torch.cuda.is_bf16_supported()}")
    for name in ["transformers", "accelerate", "huggingface_hub", "flash_attn",
                 "fla", "causal_conv1d", "torchcodec", "qwen_vl_utils"]:
        try:
            mod = importlib.import_module(name)
            print(f"  ok   {name} {getattr(mod, '__version__', '?')}")
        except Exception as e:  # ImportError, or undefined-symbol errors from ABI mismatch
            print(f"  FAIL {name}: {type(e).__name__}: {e}")

    # Kernel calls, not just imports: ABI mismatches often only surface here.
    if not torch.cuda.is_available():
        return

    def flash_attn_check():
        from flash_attn import flash_attn_func
        q = torch.randn(1, 128, 8, 64, device="cuda", dtype=torch.bfloat16)
        flash_attn_func(q, q, q, causal=True)

    def causal_conv1d_check():
        from causal_conv1d import causal_conv1d_fn
        x = torch.randn(1, 64, 128, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(64, 4, device="cuda", dtype=torch.bfloat16)
        causal_conv1d_fn(x, w)

    for name, check in [("flash_attn", flash_attn_check), ("causal_conv1d", causal_conv1d_check)]:
        try:
            check()
            print(f"  ok   {name} kernel ran")
        except Exception as e:
            print(f"  FAIL {name} kernel: {type(e).__name__}: {e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=QWEN_MODEL_PATH)
    p.add_argument("--attn", default="flash_attention_2")
    p.add_argument("--use_video", action="store_true")
    p.add_argument("--skip-model", action="store_true")
    args = p.parse_args()

    check_imports()
    if args.skip_model:
        return

    from transformers import AutoModelForMultimodalLM, AutoProcessor

    # Watch the output of this step for a "fast path is not available" warning.
    # If you see it, the Gated DeltaNet layers are running on the slow torch fallback.
    t0 = time.time()

    logger.info("Imports finished!")

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation=args.attn, device_map="auto",
    )
    logger.info("Loaded model!")

    model.eval()
    logger.info(f"\nloaded in {time.time() - t0:.0f}s  attn={model.config._attn_implementation}")
    if torch.cuda.is_available():
        logger.info(f"  mem allocated: {torch.cuda.memory_allocated() / 1e9:.1f} GB (device 0)")

    def run(content, label):
        messages = [{"role": "user", "content": content}]
        logger.info(f"Messages: {messages}")
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device)
        n_in = inputs["input_ids"].shape[1]
        t0 = time.time()
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        dt = time.time() - t0
        text = processor.batch_decode(out[:, n_in:], skip_special_tokens=True)[0]
        n_out = out.shape[1] - n_in
        logger.info(f"\n[{label}] in={n_in} tok  out={n_out} tok  {dt:.1f}s ({n_out / dt:.1f} tok/s)")
        logger.info(text)

    run([{"type": "text", "text": "In one sentence, what is a hash map?"}], "text")

    if args.use_video:
        run([{"type": "video", "video": VIDEO_PATH},
             {"type": "text", "text": "Describe what happens in this video in two sentences."}],
            "video")


if __name__ == "__main__":
    main()
