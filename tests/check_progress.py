import os
import json
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Check progress of batch jobs")
    parser.add_argument("--out_dir", type=str, default="out", help="Directory where answers are saved")
    parser.add_argument("--log_dir", type=str, default="slurm_logs", help="Directory where chunk_tasks_*.json are saved")
    args = parser.parse_args()
    
    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir)
    
    chunk_files = sorted(log_dir.glob("chunk_tasks_*.json"))
    if not chunk_files:
        print(f"No chunk files found in {log_dir}")
        return
        
    total_tasks = 0
    total_processed = 0
    
    print("--- Progress Report ---")
    for chunk_file in chunk_files:
        chunk_name = chunk_file.stem  # chunk_tasks_0
        chunk_idx = chunk_name.split('_')[-1]
        
        with open(chunk_file, 'r') as f:
            tasks = json.load(f)
            num_tasks = len(tasks)
            
        ans_file = out_dir / f"answers_chunk_{chunk_idx}.json"
        processed = 0
        if ans_file.exists():
            try:
                with open(ans_file, 'r') as f:
                    answers = json.load(f)
                    processed = len(answers)
            except Exception:
                pass # Might be writing to it exactly now
                
        total_tasks += num_tasks
        total_processed += processed
        
        status = "DONE" if processed == num_tasks else "RUNNING"
        print(f"Chunk {chunk_idx}: {processed}/{num_tasks} tasks ({status})")
        
    pct = (total_processed / total_tasks * 100) if total_tasks > 0 else 0
    print("-" * 23)
    print(f"Total Progress: {total_processed} / {total_tasks} ({pct:.1f}%)")

if __name__ == "__main__":
    main()
