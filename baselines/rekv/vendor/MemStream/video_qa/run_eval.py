import os
import argparse
import subprocess
import multiprocessing
from functools import partial 
import logzero 
import json 
from ezcolorlog import root_logger as logger 

if not os.environ.get('ENABLE_LOGGING'): 
    print("Turning off logger")
    # disable logfile for logzero
    logzero.logfile(None)
    # Turn off logging by setting level to CRITICAL (only shows critical errors)
    logzero.loglevel(logzero.CRITICAL)

    logger.disabled = True 

# rename to avoid shadowing built-in
def run_cmd(cmd, sub=False, device=None, shell=False):
    print(f'exec: {cmd}')
    if isinstance(cmd, list) and shell:
        cmd = ' '.join(cmd)
    # If a string command is provided without explicit shell, enable shell to support redirection, pipes, etc.
    if isinstance(cmd, str) and not shell:
        shell = True
    env = None
    if sub:
        env = os.environ.copy()
        if device is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(device)
    subprocess.run(cmd, env=env, shell=shell, check=True)

def get_gpu_devices(chunk_index, gpus_per_chunk):
    """Generate GPU device string for a given chunk.
    
    Args:
        chunk_index: The index of the chunk (0-based)
        gpus_per_chunk: Number of GPUs to allocate per chunk
    
    Returns:
        Comma-separated string of GPU IDs (e.g., "0,1,2,3")
    """
    start_gpu = chunk_index * gpus_per_chunk
    gpu_ids = [str(start_gpu + i) for i in range(gpus_per_chunk)]
    return ','.join(gpu_ids)

def eval_morph_vsi(args, gpus_per_chunk, save_dir, common_cmd_extensions):
    """Morph-VSI evaluation (multiple-choice, offline).

    Expects converted annotations (VSI-style list of videos), produced by
    data/morph/convert.py -> data/morph/morph_vsi.json.
    """
    num_chunks = args.num_chunks

    anno_path = args.morph_anno_path or "data/morph/morph_vsi.json"

    logger.info(f"save_dir: {save_dir}")

    solver = "rekv_offline_vqa"
    if not args.only_eval:
        if os.path.exists(f'{save_dir}/results.csv') and args.resume:
            print(f'Results file already exists: {save_dir}/results.csv')
        else:
            processes = []
            idxs = [args.chunk_idx] if args.chunk_idx is not None else list(range(0, num_chunks))

            for i, idx in enumerate(idxs):
                cmd = ["python", "-m", f"video_qa.{solver}",
                       "--model",               args.model,
                       "--prefill_method",      args.prefill_method,
                       "--retrieval_method",    args.retrieval_method,
                       "--sample_fps",          str(args.sample_fps),
                       "--n_local",             str(args.n_local),
                       "--encode_chunk_size",   str(args.encode_chunk_size),
                       "--retrieve_size",       str(args.retrieve_size),
                       "--retrieve_chunk_size", str(args.retrieve_chunk_size),
                       "--save_dir",            save_dir,
                       "--anno_path",           anno_path,
                       "--debug",               args.debug,
                       "--num_chunks",          str(num_chunks),
                       "--chunk_idx",           str(idx)]
                cmd.extend(common_cmd_extensions)

                device_str = get_gpu_devices(i, gpus_per_chunk)
                p = multiprocessing.Process(target=run_cmd, args=(cmd, True, device_str))
                processes.append(p)
                p.start()
            for p in processes:
                p.join()

            all_files_exist = [os.path.exists(f"{save_dir}/{num_chunks}_{idx}.csv") for idx in range(num_chunks)]
            if not all(all_files_exist):
                missing = [f"{save_dir}/{num_chunks}_{idx}.csv" for idx in range(num_chunks)
                           if not os.path.exists(f"{save_dir}/{num_chunks}_{idx}.csv")]
                raise RuntimeError(f"Some files do not exist: {missing}")

    # Merge chunk CSVs
    if args.chunk_idx is None:
        run_cmd(f"> {save_dir}/results.csv", shell=True)
        for idx in range(num_chunks):
            if idx == 0:
                run_cmd(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv", shell=True)
            run_cmd(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv", shell=True)

        run_cmd(["python", "video_qa/eval/eval_multiple_choice.py", "--save_dir", save_dir], shell=False)


def eval_egolongqa(args, gpus_per_chunk, save_dir, common_cmd_extensions):
    """EgoLongQA multiple-choice evaluation (offline)."""
    num_chunks = args.num_chunks
    source_path = "data/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl"
    anno_path = "data/egolongqa/egolongqa_val_700_mc.json"

    if not args.only_eval and (
        not os.path.exists(anno_path) or os.path.getmtime(anno_path) < os.path.getmtime(source_path)
    ):
        run_cmd(
            ["python", "data/egolongqa/convert.py", "--input", source_path, "--output", anno_path],
            shell=False,
        )

    logger.info(f"save_dir: {save_dir}")

    solver = "rekv_offline_vqa"
    if not args.only_eval:
        if os.path.exists(f'{save_dir}/results.csv') and args.resume:
            print(f'Results file already exists: {save_dir}/results.csv')
        else:
            processes = []
            idxs = [args.chunk_idx] if args.chunk_idx is not None else list(range(0, num_chunks))

            for i, idx in enumerate(idxs):
                cmd = ["python", "-m", f"video_qa.{solver}",
                       "--model",               args.model,
                       "--prefill_method",      args.prefill_method,
                       "--retrieval_method",    args.retrieval_method,
                       "--sample_fps",          str(args.sample_fps),
                       "--n_local",             str(args.n_local),
                       "--encode_chunk_size",   str(args.encode_chunk_size),
                       "--retrieve_size",       str(args.retrieve_size),
                       "--retrieve_chunk_size", str(args.retrieve_chunk_size),
                       "--save_dir",            save_dir,
                       "--anno_path",           anno_path,
                       "--debug",               args.debug,
                       "--num_chunks",          str(num_chunks),
                       "--chunk_idx",           str(idx)]
                cmd.extend(common_cmd_extensions)

                device_str = get_gpu_devices(i, gpus_per_chunk)
                p = multiprocessing.Process(target=run_cmd, args=(cmd, True, device_str))
                processes.append(p)
                p.start()
            for p in processes:
                p.join()

            all_files_exist = [os.path.exists(f"{save_dir}/{num_chunks}_{idx}.csv") for idx in range(num_chunks)]
            if not all(all_files_exist):
                missing = [f"{save_dir}/{num_chunks}_{idx}.csv" for idx in range(num_chunks)
                           if not os.path.exists(f"{save_dir}/{num_chunks}_{idx}.csv")]
                raise RuntimeError(f"Some files do not exist: {missing}")

    if args.chunk_idx is None:
        run_cmd(f"> {save_dir}/results.csv", shell=True)
        for idx in range(num_chunks):
            if idx == 0:
                run_cmd(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv", shell=True)
            run_cmd(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv", shell=True)

        run_cmd(["python", "video_qa/eval/eval_multiple_choice.py", "--save_dir", save_dir], shell=False)


def eval_vsi_super_recall(args, gpus_per_chunk, save_dir, common_cmd_extensions):
    """VSI-SuperRecall evaluation (multiple-choice, offline)."""
    num_chunks = args.num_chunks

    subset = args.vsi_subset
    anno_map = {
        "10mins":  "data/vsi_super_recall/test_10mins_converted.json",
        "30mins":  "data/vsi_super_recall/test_30mins_converted.json",
        "60mins":  "data/vsi_super_recall/test_60mins_converted.json",
        "120mins": "data/vsi_super_recall/test_120mins_converted.json",
        "240mins": "data/vsi_super_recall/test_240mins_converted.json",
    }
    if subset not in anno_map:
        raise ValueError(f"vsi_super_recall only supports subsets: {list(anno_map.keys())}; got '{subset}'")

    anno_path = anno_map[subset]
    save_dir = os.path.join(save_dir, subset)

    logger.info(f"save_dir: {save_dir}")

    solver = "rekv_offline_vqa"
    if not args.only_eval:
        if os.path.exists(f'{save_dir}/results.csv') and args.resume:
            print(f'Results file already exists: {save_dir}/results.csv')
        else:
            processes = []
            idxs = [args.chunk_idx] if args.chunk_idx is not None else list(range(0, num_chunks))

            for i, idx in enumerate(idxs):
                cmd = ["python", "-m", f"video_qa.{solver}",
                       "--model",               args.model,
                       "--prefill_method",      args.prefill_method,
                       "--retrieval_method",    args.retrieval_method,
                       "--sample_fps",          str(args.sample_fps),
                       "--n_local",             str(args.n_local),
                       "--encode_chunk_size",   str(args.encode_chunk_size),
                       "--retrieve_size",       str(args.retrieve_size),
                       "--retrieve_chunk_size", str(args.retrieve_chunk_size),
                       "--save_dir",            save_dir,
                       "--anno_path",           anno_path,
                       "--debug",               args.debug,
                       "--num_chunks",          str(num_chunks),
                       "--chunk_idx",           str(idx)]
                cmd.extend(common_cmd_extensions)

                device_str = get_gpu_devices(i, gpus_per_chunk)
                p = multiprocessing.Process(target=run_cmd, args=(cmd, True, device_str))
                processes.append(p)
                p.start()
            for p in processes:
                p.join()

            all_files_exist = [os.path.exists(f"{save_dir}/{num_chunks}_{idx}.csv") for idx in range(num_chunks)]
            if not all(all_files_exist):
                missing = [f"{save_dir}/{num_chunks}_{idx}.csv" for idx in range(num_chunks)
                           if not os.path.exists(f"{save_dir}/{num_chunks}_{idx}.csv")]
                raise RuntimeError(f"Some files do not exist: {missing}")

    # Merge chunk CSVs
    if args.chunk_idx is None:
        run_cmd(f"> {save_dir}/results.csv", shell=True)
        for idx in range(num_chunks):
            if idx == 0:
                run_cmd(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv", shell=True)
            run_cmd(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv", shell=True)

        run_cmd(["python", "video_qa/eval/eval_multiple_choice.py", "--save_dir", save_dir], shell=False)


def eval_timescope(args, gpus_per_chunk, save_dir, common_cmd_extensions):
    num_chunks = args.num_chunks
    # No dataset-specific save_dir modifications needed
    
    logger.info(f"save_dir: {save_dir}")
    
    solver = "rekv_offline_vqa"
    if not args.only_eval:
        # If results file already exists and resume is enabled, skip the QA process and return
        if os.path.exists(f'{save_dir}/results.csv') and args.resume:
            print(f'Results file already exists: {save_dir}/results.csv')
        else:
            # QA
            processes = []
            idxs = [args.chunk_idx] if args.chunk_idx is not None else list(range(0, num_chunks))
            
            for i, idx in enumerate(idxs):
                cmd = ["python", "-m", f"video_qa.{solver}",
                        "--model", args.model,
                        "--prefill_method", args.prefill_method,
                        "--retrieval_method", args.retrieval_method, 
                        "--sample_fps", str(args.sample_fps),
                        "--n_local", str(args.n_local),
                        "--encode_chunk_size", str(args.encode_chunk_size),
                        "--retrieve_size", str(args.retrieve_size),
                        "--retrieve_chunk_size", str(args.retrieve_chunk_size),
                        "--save_dir", save_dir,
                        "--anno_path", "data/timescope/full_data.json",
                        "--debug", args.debug,
                        "--num_chunks", str(num_chunks),
                        "--chunk_idx", str(idx)]
                
                # Add common extensions
                cmd.extend(common_cmd_extensions)
                
                device_str = get_gpu_devices(i, gpus_per_chunk)
                p = multiprocessing.Process(target=run_cmd, args=(cmd, True, device_str))
                processes.append(p)
                p.start()
            # wait for all processes
            for p in processes:
                p.join()
                
            # check if all files exist prior to merging 
            all_files_exist = [os.path.exists(f"{save_dir}/{num_chunks}_{idx}.csv") for idx in range(num_chunks)]
            if not all(all_files_exist):
                missing_files = [f"{save_dir}/{num_chunks}_{idx}.csv" for idx in range(num_chunks) if not os.path.exists(f"{save_dir}/{num_chunks}_{idx}.csv")]
                raise RuntimeError(f"Some files do not exist: {missing_files}")
                
    # merge results
    run_cmd(f"> {save_dir}/results.csv", shell=True)
    for idx in range(num_chunks):
        if idx == 0:
            run_cmd(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv", shell=True)
        run_cmd(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv", shell=True)
    
    # eval
    run_cmd(
        ["python", "video_qa/eval/eval_multiple_choice.py", "--save_dir", save_dir],
        shell=False,
    )

def eval_lvbench(args, gpus_per_chunk, save_dir, common_cmd_extensions): 
    
    num_chunks = args.num_chunks
    # No dataset-specific save_dir modifications needed
    
    logger.info(f"save_dir: {save_dir}")

    solver = "rekv_offline_vqa"
    if not args.only_eval:
        # If results file already exists and resume is enabled, skip the QA process and return
        if os.path.exists(f'{save_dir}/results.csv') and args.resume:
            print(f'Results file already exists: {save_dir}/results.csv')
        else:
            # QA
            processes = []
            idxs = [args.chunk_idx] if args.chunk_idx is not None else list(range(0, num_chunks))

            for i, idx in enumerate(idxs):
                cmd = ["python", "-m", f"video_qa.{solver}",
                        "--model", args.model,
                        "--prefill_method", args.prefill_method,
                        "--retrieval_method", args.retrieval_method, 
                        "--sample_fps", str(args.sample_fps),
                        "--n_local", str(args.n_local),
                        "--encode_chunk_size", str(args.encode_chunk_size),
                        "--retrieve_size", str(args.retrieve_size),
                        "--retrieve_chunk_size", str(args.retrieve_chunk_size),
                        "--save_dir", save_dir,
                        "--anno_path", "data/lvbench/full_data.json",
                        "--debug", args.debug,
                        "--num_chunks", str(num_chunks),
                        "--chunk_idx", str(idx)]

                # Add common extensions
                cmd.extend(common_cmd_extensions)
                
                device_str = get_gpu_devices(i, gpus_per_chunk)
                p = multiprocessing.Process(target=run_cmd, args=(cmd, True, device_str))
                processes.append(p)
                p.start()
            # wait for all processes
            for p in processes:
                p.join()

            # check if all files exist prior to merging 
            # Only meaningful when this process ran every chunk. An array task owns one
            # chunk, so the others legitimately do not exist yet -- without this the task
            # fails here even though its own chunk was written fine, which masked a real
            # CUDA OOM in the worker.
            if args.chunk_idx is None:
                all_files_exist = [os.path.exists(f"{save_dir}/{num_chunks}_{idx}.csv") for idx in range(num_chunks)]
                if not all(all_files_exist):
                    missing_files = [f"{save_dir}/{num_chunks}_{idx}.csv" for idx in range(num_chunks) if not os.path.exists(f"{save_dir}/{num_chunks}_{idx}.csv")]
                    raise RuntimeError(f"Some files do not exist: {missing_files}")

    # A single array task owns one chunk; merging/scoring belongs to the launcher. Without this
    # guard the task writes its chunk fine, then fails on `head -n 1 {N}_0.csv`, and slurm marks
    # the whole task FAILED even though the chunk succeeded.
    if args.chunk_idx is not None:
        logger.info(f'chunk {args.chunk_idx} written; skipping merge/score '
                    f'(re-run with --only_eval once all chunks are done)')
        return

    # merge results
    run_cmd(f"> {save_dir}/results.csv", shell=True)
    for idx in range(num_chunks):
        if idx == 0:
            run_cmd(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv", shell=True)
        run_cmd(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv", shell=True)
    
    # eval
    run_cmd(
        ["python", "video_qa/eval/eval_multiple_choice.py", "--save_dir", save_dir],
        shell=False,
    )

def eval_mlvu(args, gpus_per_chunk, save_dir, common_cmd_extensions):
    num_chunks = args.num_chunks
    # No dataset-specific save_dir modifications needed
    
    logger.info(f"save_dir: {save_dir}")

    solver = "rekv_offline_vqa"
    if not args.only_eval:
        # If results file already exists and resume is enabled, skip the QA process and return
        if os.path.exists(f'{save_dir}/results.csv') and args.resume:
            print(f'Results file already exists: {save_dir}/results.csv')
        else:
            # QA
            processes = []
            idxs = [args.chunk_idx] if args.chunk_idx is not None else list(range(0, num_chunks))

            for i, idx in enumerate(idxs):
                cmd = ["python", "-m", f"video_qa.{solver}",
                        "--model", args.model,
                        "--prefill_method", args.prefill_method,
                        "--retrieval_method", args.retrieval_method, 
                        "--sample_fps", str(args.sample_fps),
                        "--n_local", str(args.n_local),
                        "--encode_chunk_size", str(args.encode_chunk_size),
                        "--retrieve_size", str(args.retrieve_size),
                        "--retrieve_chunk_size", str(args.retrieve_chunk_size),
                        "--save_dir", save_dir,
                        "--anno_path", "data/mlvu/dev_debug_mc.json",
                        "--debug", args.debug,
                        "--num_chunks", str(num_chunks),
                        "--chunk_idx", str(idx)]

                # Add common extensions
                cmd.extend(common_cmd_extensions)
                
                device_str = get_gpu_devices(i, gpus_per_chunk)
                p = multiprocessing.Process(target=run_cmd, args=(cmd, True, device_str))
                processes.append(p)
                p.start()
            # wait for all processes
            for p in processes:
                p.join()

            # check if all files exist prior to merging 
            all_files_exist = [os.path.exists(f"{save_dir}/{num_chunks}_{idx}.csv") for idx in range(num_chunks)]
            if not all(all_files_exist):
                missing_files = [f"{save_dir}/{num_chunks}_{idx}.csv" for idx in range(num_chunks) if not os.path.exists(f"{save_dir}/{num_chunks}_{idx}.csv")]
                raise RuntimeError(f"Some files do not exist: {missing_files}")

    # merge results
    run_cmd(f"> {save_dir}/results.csv", shell=True)
    for idx in range(num_chunks):
        if idx == 0:
            run_cmd(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv", shell=True)
        run_cmd(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv", shell=True)
    
    # eval
    run_cmd(
        ["python", "video_qa/eval/eval_multiple_choice.py", "--save_dir", save_dir],
        shell=False,
    )

def eval_qaego4d(args, gpus_per_chunk, save_dir, common_cmd_extensions):
    num_chunks = args.num_chunks
    # No dataset-specific save_dir modifications needed
    
    logger.info(f"save_dir: {save_dir}")

    solver = "rekv_offline_vqa"
    if not args.only_eval:
        # If results file already exists and resume is enabled, skip the QA process and return
        if os.path.exists(f'{save_dir}/results.csv') and args.resume:
            print(f'Results file already exists: {save_dir}/results.csv')
        else:
            # QA
            processes = []
            idxs = [args.chunk_idx] if args.chunk_idx is not None else list(range(0, num_chunks))

            for i, idx in enumerate(idxs):
                cmd = ["python", "-m", f"video_qa.{solver}",
                        "--model", args.model,
                        "--prefill_method", args.prefill_method,
                        "--retrieval_method", args.retrieval_method,
                        "--sample_fps", str(args.sample_fps),
                        "--n_local", str(args.n_local),
                        "--encode_chunk_size", str(args.encode_chunk_size),
                        "--retrieve_size", str(args.retrieve_size),
                        "--retrieve_chunk_size", str(args.retrieve_chunk_size),
                        "--save_dir", save_dir,
                        "--anno_path", "data/qaego4d/test_mc.json",
                        "--debug", args.debug,
                        "--num_chunks", str(num_chunks),
                        "--chunk_idx", str(idx)]

                # Add common extensions
                cmd.extend(common_cmd_extensions)
                
                device_str = get_gpu_devices(i, gpus_per_chunk)
                p = multiprocessing.Process(target=run_cmd, args=(cmd, True, device_str))
                processes.append(p)
                p.start()
            # wait for all processes
            for p in processes:
                p.join()

            # check if all files exist prior to merging 
            all_files_exist = [os.path.exists(f"{save_dir}/{num_chunks}_{idx}.csv") for idx in range(num_chunks)]
            if not all(all_files_exist):
                missing_files = [f"{save_dir}/{num_chunks}_{idx}.csv" for idx in range(num_chunks) if not os.path.exists(f"{save_dir}/{num_chunks}_{idx}.csv")]
                raise RuntimeError(f"Some files do not exist: {missing_files}")

    # merge results
    run_cmd(f"> {save_dir}/results.csv", shell=True)
    for idx in range(num_chunks):
        if idx == 0:
            run_cmd(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv", shell=True)
        run_cmd(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv", shell=True)
    
    # eval
    run_cmd(
        ["python", "video_qa/eval/eval_multiple_choice.py", "--save_dir", save_dir],
        shell=False,
    )

def eval_egoschema(args, gpus_per_chunk, save_dir, common_cmd_extensions):
    num_chunks = args.num_chunks
    # No dataset-specific save_dir modifications needed
    
    solver = "rekv_offline_vqa"
    if not args.only_eval:
        # If results file already exists and resume is enabled, skip the QA process and return
        if os.path.exists(f'{save_dir}/results.csv') and args.resume:
            print(f'Results file already exists: {save_dir}/results.csv')
        else:
            # QA
            processes = []
            idxs = [args.chunk_idx] if args.chunk_idx is not None else range(0, num_chunks)
            
            for i, idx in enumerate(idxs):
                cmd = ["python", "-m", f"video_qa.{solver}",
                        "--model", args.model,
                        "--prefill_method", args.prefill_method,
                        "--retrieval_method", args.retrieval_method,
                        "--sample_fps", str(args.sample_fps),
                        "--n_local", str(args.n_local),
                        "--encode_chunk_size", str(args.encode_chunk_size),
                        "--retrieve_size", str(args.retrieve_size),
                        "--retrieve_chunk_size", str(args.retrieve_chunk_size),
                        "--save_dir", save_dir,
                        "--anno_path", "data/egoschema/full.json",
                        "--debug", args.debug,
                        "--num_chunks", str(num_chunks),
                        "--chunk_idx", str(idx)]
                
                # Add common extensions
                cmd.extend(common_cmd_extensions)
                
                device_str = get_gpu_devices(i, gpus_per_chunk)
                p = multiprocessing.Process(target=run_cmd, args=(cmd, True, device_str))
                processes.append(p)
                p.start()
            for p in processes:
                p.join()

            # check if all files exist prior to merging 
            all_files_exist = [os.path.exists(f"{save_dir}/{num_chunks}_{idx}.csv") for idx in range(num_chunks)]
            if not all(all_files_exist):
                missing_files = [f"{save_dir}/{num_chunks}_{idx}.csv" for idx in range(num_chunks) if not os.path.exists(f"{save_dir}/{num_chunks}_{idx}.csv")]
                raise RuntimeError(f"Some files do not exist: {missing_files}")

    # merge results
    run_cmd(f"> {save_dir}/results.csv")
    for idx in range(num_chunks):
        if idx == 0:
            run_cmd(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
        run_cmd(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
    
    # eval
    run_cmd(f"python video_qa/eval/eval_egoschema.py --save_dir {save_dir}")

def eval_activitynet_qa(args, gpus_per_chunk, save_dir, common_cmd_extensions):
    num_chunks = args.num_chunks
    # No dataset-specific save_dir modifications needed
    
    solver = "rekv_offline_vqa"
    if not args.only_eval:
        # QA
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", "-m", f"video_qa.{solver}",
                    "--model", args.model,
                    "--prefill_method", args.prefill_method,
                    "--retrieval_method", args.retrieval_method,
                    "--sample_fps", str(args.sample_fps),
                    "--n_local", str(args.n_local),
                    "--encode_chunk_size", str(args.encode_chunk_size),
                    "--retrieve_size", str(args.retrieve_size),
                    "--save_dir", save_dir,
                    "--anno_path", "data/activitynet_qa/test.json",
                    "--debug", args.debug,
                    "--num_chunks", str(num_chunks),
                    "--chunk_idx", str(idx)]
            
            # Add common extensions
            cmd.extend(common_cmd_extensions)
            
            device_str = get_gpu_devices(idx, gpus_per_chunk)
            p = multiprocessing.Process(target=run_cmd, args=(cmd, True, device_str))
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        # merge results
        run_cmd(f"> {save_dir}/results.csv")
        run_cmd(f"rm -rf {save_dir}/tmp")
        for idx in range(num_chunks):
            if idx == 0:
                run_cmd(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
            run_cmd(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
            run_cmd(f"rm {save_dir}/{num_chunks}_{idx}.csv")
    # eval
    run_cmd(f"python video_qa/eval/eval_open_ended.py --pred_path {save_dir}/results.csv --output_dir {save_dir}/tmp --output_json {save_dir}/results.json")

def eval_rvs_ego(args, gpus_per_chunk, save_dir, common_cmd_extensions):
    num_chunks = args.num_chunks
    # No dataset-specific save_dir modifications needed
    
    solver = "rekv_stream_vqa"
    if not args.only_eval:
        # QA
        processes = []
        idxs = [args.chunk_idx] if args.chunk_idx is not None else list(range(0, num_chunks))
        
        for i, idx in enumerate(idxs):
            cmd = ["python", "-m", f"video_qa.{solver}",
                    "--model", args.model,
                    "--prefill_method", args.prefill_method,
                    "--retrieval_method", args.retrieval_method,
                    "--sample_fps", str(args.sample_fps),
                    "--n_local", str(args.n_local),
                    "--encode_chunk_size", str(args.encode_chunk_size),
                    "--retrieve_size", str(args.retrieve_size),
                    "--save_dir", save_dir,
                    "--anno_path", "data/rvs/ego/ego4d_oe.json",
                    "--debug", args.debug,
                    "--num_chunks", str(num_chunks),
                    "--chunk_idx", str(idx)]
            
            # Add common extensions
            cmd.extend(common_cmd_extensions)
            
            device_str = get_gpu_devices(i, gpus_per_chunk)
            p = multiprocessing.Process(target=run_cmd, args=(cmd, True, device_str))
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        # merge results (only if we ran all chunks)
        if args.chunk_idx is None:
            run_cmd(f"> {save_dir}/results.csv")
            run_cmd(f"rm -rf {save_dir}/tmp")
            for idx in range(num_chunks):
                chunk_file = f"{save_dir}/{num_chunks}_{idx}.csv"
                if os.path.exists(chunk_file):
                    if idx == 0:
                        run_cmd(f"head -n 1 {chunk_file} > {save_dir}/results.csv")
                    run_cmd(f"tail -n +2 {chunk_file} >> {save_dir}/results.csv")
                    run_cmd(f"rm {chunk_file}")
    # eval (only if results.csv exists)
    if args.chunk_idx is None and os.path.exists(f"{save_dir}/results.csv"):
        run_cmd(f"python video_qa/eval/eval_open_ended.py --pred_path {save_dir}/results.csv --output_dir {save_dir}/tmp --output_json {save_dir}/results.json")

def eval_rvs_movie(args, gpus_per_chunk, save_dir, common_cmd_extensions):
    num_chunks = args.num_chunks
    # No dataset-specific save_dir modifications needed
    
    solver = "rekv_stream_vqa"
    if not args.only_eval:
        # QA
        processes = []
        idxs = [args.chunk_idx] if args.chunk_idx is not None else list(range(0, num_chunks))
        
        for i, idx in enumerate(idxs):
            cmd = ["python", "-m", f"video_qa.{solver}",
                    "--model", args.model,
                    "--prefill_method", args.prefill_method,
                    "--retrieval_method", args.retrieval_method,
                    "--sample_fps", str(args.sample_fps),
                    "--n_local", str(args.n_local),
                    "--encode_chunk_size", str(args.encode_chunk_size),
                    "--retrieve_size", str(args.retrieve_size),
                    "--save_dir", save_dir,
                    "--anno_path", "data/rvs/movie/movienet_oe.json",
                    "--debug", args.debug,
                    "--num_chunks", str(num_chunks),
                    "--chunk_idx", str(idx)]
            
            # Add common extensions
            cmd.extend(common_cmd_extensions)
            
            device_str = get_gpu_devices(i, gpus_per_chunk)
            p = multiprocessing.Process(target=run_cmd, args=(cmd, True, device_str))
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        # merge results (only if we ran all chunks)
        if args.chunk_idx is None:
            run_cmd(f"> {save_dir}/results.csv")
            run_cmd(f"rm -rf {save_dir}/tmp")
            for idx in range(num_chunks):
                chunk_file = f"{save_dir}/{num_chunks}_{idx}.csv"
                if os.path.exists(chunk_file):
                    if idx == 0:
                        run_cmd(f"head -n 1 {chunk_file} > {save_dir}/results.csv")
                    run_cmd(f"tail -n +2 {chunk_file} >> {save_dir}/results.csv")
                    run_cmd(f"rm {chunk_file}")
    # eval (only if results.csv exists)
    if args.chunk_idx is None and os.path.exists(f"{save_dir}/results.csv"):
        run_cmd(f"python video_qa/eval/eval_open_ended.py --pred_path {save_dir}/results.csv --output_dir {save_dir}/tmp --output_json {save_dir}/results.json")

def eval_videomme(args, gpus_per_chunk, save_dir, common_cmd_extensions):
    num_chunks = args.num_chunks
    # No dataset-specific save_dir modifications needed
    
    logger.info(f"save_dir: {save_dir}")

    solver = "rekv_offline_vqa"
    if not args.only_eval:
        # If results file already exists and resume is enabled, skip the QA process and return
        if os.path.exists(f'{save_dir}/results.csv') and args.resume:
            print(f'Results file already exists: {save_dir}/results.csv')
        else:
            # QA
            processes = []
            idxs = [args.chunk_idx] if args.chunk_idx is not None else list(range(0, num_chunks))

            for i, idx in enumerate(idxs):
                cmd = ["python", "-m", f"video_qa.{solver}",
                        "--model", args.model,
                        "--prefill_method", args.prefill_method,
                        "--retrieval_method", args.retrieval_method, 
                        "--sample_fps", str(args.sample_fps),
                        "--n_local", str(args.n_local),
                        "--encode_chunk_size", str(args.encode_chunk_size),
                        "--retrieve_size", str(args.retrieve_size),
                        "--retrieve_chunk_size", str(args.retrieve_chunk_size),
                        "--save_dir", save_dir,
                        "--anno_path", "data/videomme/full_data.json",
                        "--debug", args.debug,
                        "--num_chunks", str(num_chunks),
                        "--chunk_idx", str(idx)]

                # Add common extensions
                cmd.extend(common_cmd_extensions)
                
                device_str = get_gpu_devices(i, gpus_per_chunk)
                p = multiprocessing.Process(target=run_cmd, args=(cmd, True, device_str))
                processes.append(p)
                p.start()
            # wait for all processes
            for p in processes:
                p.join()

            # check if all files exist prior to merging 
            all_files_exist = [os.path.exists(f"{save_dir}/{num_chunks}_{idx}.csv") for idx in range(num_chunks)]
            if not all(all_files_exist):
                missing_files = [f"{save_dir}/{num_chunks}_{idx}.csv" for idx in range(num_chunks) if not os.path.exists(f"{save_dir}/{num_chunks}_{idx}.csv")]
                raise RuntimeError(f"Some files do not exist: {missing_files}")

    # merge results
    run_cmd(f"> {save_dir}/results.csv", shell=True)
    for idx in range(num_chunks):
        if idx == 0:
            run_cmd(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv", shell=True)
        run_cmd(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv", shell=True)
    
    # eval
    run_cmd(
        ["python", "video_qa/eval/eval_multiple_choice.py", "--save_dir", save_dir],
        shell=False,
    )

def eval_cgbench(args, gpus_per_chunk, save_dir, common_cmd_extensions):
    """ handles both cgbench and cgbench_clue_grounded """
    num_chunks = args.num_chunks
    
    if args.cgbench_task_mode == "random_clue_acc": 
        assert args.dataset == 'cgbench_clue_grounded'
        save_dir = f"{save_dir}/random-windows"
    if args.cgbench_task_mode == "long_with_clue_acc":
        save_dir = f"{save_dir}/long_with_clue_acc"
    
    logger.info(f"save_dir: {save_dir}")

    solver = "rekv_offline_vqa"
    if not args.only_eval:

        # If results file already exists and resume is enabled, skip the QA process and return
        if os.path.exists(f'{save_dir}/results.csv') and args.resume:
            print(f'Results file already exists: {save_dir}/results.csv')
        else:
            # QA
            processes = []
            idxs = [args.chunk_idx] if args.chunk_idx is not None else range(0, num_chunks)
                
            for i,idx in enumerate(idxs):
                cmd = ["python", "-m", f"video_qa.{solver}",
                        "--model", args.model,
                        "--prefill_method", args.prefill_method, 
                        "--retrieval_method", args.retrieval_method, 
                        "--sample_fps", str(args.sample_fps),
                        "--n_local", str(args.n_local),
                        "--encode_chunk_size", str(args.encode_chunk_size),
                        "--retrieve_chunk_size", str(args.retrieve_chunk_size),
                        "--retrieve_size", str(args.retrieve_size),
                        "--save_dir", save_dir,
                        "--anno_path", "data/cgbench/full_mc.json",
                        "--debug", args.debug,
                        "--num_chunks", str(num_chunks),
                        "--chunk_idx", str(idx)]
                
                # Add common extensions
                cmd.extend(common_cmd_extensions)
                
                # Add cgbench-specific flags
                if args.dataset == 'cgbench_clue_grounded':
                    cmd.append("--use_temporal_windows")
                if args.cgbench_task_mode == "random_clue_acc": 
                    cmd.append("--set_random_temporal_windows")
                elif args.cgbench_task_mode == "long_with_clue_acc":
                    cmd.append("--retrieve_from_temporal_windows")

                device_str = get_gpu_devices(i, gpus_per_chunk)
                p = multiprocessing.Process(target=run_cmd, args=(cmd, True, device_str))
                processes.append(p)
                p.start()
            for p in processes:
                p.join()

            # check if all files exist prior to merging 
            # Only meaningful when this process ran every chunk. An array task owns one
            # chunk, so the others legitimately do not exist yet -- without this the task
            # fails here even though its own chunk was written fine, which masked a real
            # CUDA OOM in the worker.
            if args.chunk_idx is None:
                all_files_exist = [os.path.exists(f"{save_dir}/{num_chunks}_{idx}.csv") for idx in range(num_chunks)]
                if not all(all_files_exist):
                    raise RuntimeError(f"Some files do not exist: {save_dir}/{num_chunks}_{idx}.csv")
            
    # A single array task owns one chunk; merging/scoring belongs to the launcher. Without this
    # guard the task writes its chunk fine, then fails on `head -n 1 {N}_0.csv`, and slurm marks
    # the whole task FAILED even though the chunk succeeded.
    if args.chunk_idx is not None:
        logger.info(f'chunk {args.chunk_idx} written; skipping merge/score '
                    f'(re-run with --only_eval once all chunks are done)')
        return

    # merge results
    run_cmd(f"> {save_dir}/results.csv")
    for idx in range(num_chunks):
        if idx == 0:
            run_cmd(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
        run_cmd(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
    
    # eval
    run_cmd(f"python video_qa/eval/eval_multiple_choice.py --save_dir {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llava_ov_7b", choices=['llava_ov_0.5b', 'llava_ov_7b', 'llava_ov_72b', 
                                                                             'video_llava_7b', 'longva_7b', 
                                                                             'qwen2_vl_7b', 'qwen2_5_vl_3b', 'qwen2_5_vl_7b', 'qwen2_5_vl_32b',
                                                                             'qwen3_vl_8b', 'qwen3_vl_4b',
                                                                             'internvl3_8b', 'internvl3_5_8b'])
    parser.add_argument("--dataset", type=str, default=None, choices=['mlvu', 'qaego4d', 'egoschema', 'activitynet_qa', 
                                                                      'rvs_ego', 'rvs_movie', 'cgbench', 'cgbench_clue_grounded', 'videomme', 'lvbench',
                                                                      'morph_vsi', 'egolongqa', 'vsi_super_recall'])
    parser.add_argument("--vsi-subset", type=str, default="10mins",
                        choices=["10mins", "30mins", "60mins", "120mins", "240mins"],
                        help="For --dataset vsi_super_recall: which converted annotation split to use")
    parser.add_argument("--morph-anno-path", type=str, default=None,
                        help="For --dataset morph_vsi: path to VSI-style JSON (default: data/morph/morph_vsi.json)")

    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=None) # run one chunk only if specified
    parser.add_argument("--only_eval", action="store_true")
    parser.add_argument("--sample_fps", type=float, default=1)
    parser.add_argument("--target_frames", type=int, default=None)
    parser.add_argument("--n_local", type=int, default=15000)
    parser.add_argument("--retrieve_size", type=int, default=64)
    parser.add_argument("--retrieve_chunk_size", type=int, default=1) 
    parser.add_argument("--debug", type=str, default='false')
    parser.add_argument("--resume", action="store_true") 
    parser.add_argument("--encode_chunk_size", type=int, default=64)
    parser.add_argument("--enable_similarity", action="store_true",
                        help="Enable similarity logging during evaluation")

    # Per-frame token budget for tokens-per-frame sweeps (Qwen2.5-VL / Qwen3-VL). Leave unset and
    # each model keeps its own default (Qwen2.5-VL 200/256, Qwen3-VL 448/512).
    parser.add_argument("--min_tokens", type=int, default=None)
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--no_pin_memory", dest="pin_memory", action="store_false",
                        help="Do not page-lock offloaded KV blocks. Needed above ~115 GiB of "
                             "KV per video; transfer-speed only, results are unchanged.")


    # append on save dir 
    parser.add_argument("--save_dir_suffix", type=str, default=None)
    parser.add_argument("--i_am_pulkit", action='store_true', default=False)

    # extenal ReKV args
    parser.add_argument("--ext_model", type=str, default=None, choices=["pecore_l14_336", "clip_vit_large_patch14", 
                                                                        "siglip_so400m_patch14_384", "siglip2_so400m_patch14_384"])
    parser.add_argument("--ext_merge_method", type=str, default="rrf", choices=['rrf', 'l2_concat'])
    parser.add_argument("--ext_weights",  nargs='+', type=float, default=[0.5, 0.5], help='Weights for each external model')
    parser.add_argument("--ext_rank_passing", type=int, default=None, help="share ranks across layers")
    parser.add_argument("--ext_rrf_k", type=int, default=60, help="RRF constant k for reciprocal rank fusion")

    # load retrieved indices from folder 
    parser.add_argument("--retrieved_indices_folder", type=str, default=None)
    parser.add_argument("--retrieved_sims_folder", type=str, default=None)

    # prefill and retrieval methods 
    parser.add_argument("--prefill_method", type=str, default="swa")
    parser.add_argument("--retrieval_method", type=str, default="rekv")

    # cg-bench specific evaluation functions 
    parser.add_argument("--cgbench_task_mode", type=str, default=None, 
        choices=['long_acc', 'clue_acc', 'miou', 'random_clue_acc', 'long_with_clue_acc'] 
    ) 
    
    # GPU allocation
    parser.add_argument("--gpus_per_chunk", type=int, default=None,
                        help="Number of GPUs to use per chunk. Default: 1 for most models, 4 for llava_ov_72b")
    
    args = parser.parse_args()
    
    # Determine GPUs per chunk
    if args.gpus_per_chunk is not None:
        gpus_per_chunk = args.gpus_per_chunk
    elif args.model == 'llava_ov_72b':
        gpus_per_chunk = 4
    else:
        gpus_per_chunk = 1

    # Build base save_dir
    base_results_dir = "results-pulkit" if args.i_am_pulkit else "results"

    save_dir = f"{base_results_dir}/{args.model}/{args.dataset}/{args.retrieve_size}-{args.sample_fps}"    
    if args.target_frames is not None:
        save_dir = f"{save_dir}-target_frames={args.target_frames}"
    if args.max_tokens is not None:
        # keep token budgets in separate trees, otherwise a tpf sweep overwrites itself
        save_dir = f"{save_dir}-tpf={args.min_tokens}_{args.max_tokens}"

    save_dir = f"{save_dir}/{args.prefill_method}-{args.retrieval_method}"
    if args.retrieve_chunk_size != 1: 
        save_dir = f"{save_dir}/ret_chunk={args.retrieve_chunk_size}"
    
    if args.ext_rank_passing is not None:
        save_dir = f"{save_dir}/cross-layer-rank-passing={args.ext_rank_passing}-{'-'.join([str(w) for w in args.ext_weights])}"
    
    if args.ext_model is not None:
        save_dir = f"{save_dir}/{args.ext_model}-{args.ext_merge_method}-{'-'.join([str(w) for w in args.ext_weights])}"
            
    if args.save_dir_suffix is not None:
        save_dir = f"{save_dir}-{args.save_dir_suffix}"
    
    print(f"Using {gpus_per_chunk} GPU(s) per chunk")
    
    # Build common command extensions
    common_cmd_extensions = []
    
    if args.retrieved_indices_folder is not None:
        common_cmd_extensions.extend(["--retrieved_indices_folder", args.retrieved_indices_folder])
    if args.retrieved_sims_folder is not None:
        common_cmd_extensions.extend(["--retrieved_sims_folder", args.retrieved_sims_folder])
    
    # Only add ext_model related args if ext_model is not None
    if args.ext_model is not None:
        common_cmd_extensions.extend(["--ext_model", args.ext_model,
                                      "--ext_merge_method", args.ext_merge_method,
                                      "--ext_weights"] + [str(w) for w in args.ext_weights])
    
    if args.ext_rank_passing is not None: 
        common_cmd_extensions.extend(["--ext_rank_passing", str(args.ext_rank_passing)])
        common_cmd_extensions.extend(["--ext_weights"] + [str(w) for w in args.ext_weights])

    if args.ext_model is not None or args.ext_rank_passing is not None:
        common_cmd_extensions.extend(["--ext_rrf_k", str(args.ext_rrf_k)])
    
    if args.target_frames is not None:
        common_cmd_extensions.extend(["--target_frames", str(args.target_frames)])

    # Per-frame token budget (Qwen2.5-VL / Qwen3-VL only). Encoded into save_dir below so two
    # budgets cannot silently overwrite each other's results.
    if args.min_tokens is not None or args.max_tokens is not None:
        if args.min_tokens is None or args.max_tokens is None:
            raise ValueError("pass both --min_tokens and --max_tokens, or neither")
        common_cmd_extensions.extend(["--min_tokens", str(args.min_tokens),
                                      "--max_tokens", str(args.max_tokens)])

    # Deliberately NOT part of save_dir: pinning changes transfer speed, not values, so pinned
    # and unpinned chunks of the same arm belong in the same results directory.
    if not args.pin_memory:
        common_cmd_extensions.append("--no_pin_memory")

    if args.resume:
        common_cmd_extensions.append("--resume")
    
    if args.enable_similarity:
        common_cmd_extensions.append("--enable_similarity")
    
    func_dic = {
        'mlvu': eval_mlvu,
        'qaego4d': eval_qaego4d,
        'egoschema': eval_egoschema,
        'activitynet_qa': eval_activitynet_qa,
        'rvs_ego': eval_rvs_ego,
        'rvs_movie': eval_rvs_movie,
        'cgbench': eval_cgbench,
        'cgbench_clue_grounded': eval_cgbench, 
        'videomme': eval_videomme,
        'lvbench': eval_lvbench,
        'morph_vsi': eval_morph_vsi,
        'egolongqa': eval_egolongqa,
        'vsi_super_recall': eval_vsi_super_recall,
    }
    if args.dataset in func_dic:
        print(f'Execute {args.dataset} evaluation')
        func_dic[args.dataset](args, gpus_per_chunk, save_dir, common_cmd_extensions)