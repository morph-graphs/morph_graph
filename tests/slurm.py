import os
import json
import argparse
from pathlib import Path
import subprocess


def get_gpu_info(args, remove_nodes=None):
    if args.gpus_to_use == 'a4+':
        gpu_type = ['a6000', 'a5000', 'a4000']
    elif args.gpus_to_use == 'a4':
        gpu_type = ['a4000']
    elif args.gpus_to_use == 'a5':
        gpu_type = ['a5000']
    elif args.gpus_to_use == 'a6':
        gpu_type = ['a6000']
    elif args.gpus_to_use == 'a5+':
        gpu_type = ['a5000', 'a6000']
    elif args.gpus_to_use == 'a6+':
        gpu_type = ['a6000']
    elif args.gpus_to_use == 'h100+':
        gpu_type = ['h100', 'h200']
    elif args.gpus_to_use == 'ti':
        gpu_type = ['gtx1080ti', 'rtx2080ti']
    else:
        raise ValueError("Invalid gpu type")

    if args.qos == 'scav' or getattr(args, 'scav', False):
        if remove_nodes is None:
            remove_nodes = []
        # remove_nodes.extend(['cbcb', 'gamma', 'vulcan'])
        
    def run(cmd, print_err=True):
        try:
            return subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT).decode('UTF-8').splitlines()
        except subprocess.CalledProcessError as e:
            if print_err:
                print("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
            return [cmd.split()[-1]]
            
    gpudata = run('sinfo -O nodehost,gres -h')
    new_gpu_data = []
    for gpu in gpu_type:
        new_gpu_data += [line.split(' ')[0] for line in gpudata if gpu in line]
    if remove_nodes is not None:
        for node in remove_nodes:
            new_gpu_data = [gpu_node for gpu_node in new_gpu_data if node not in gpu_node]
    assert len(new_gpu_data) > 0, 'No GPU found'
    return ','.join(new_gpu_data)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', type=str, required=True, help="Path to JSON or JSONL file with questions")
    parser.add_argument('--script_module', type=str, default="morph_graph.tests.debug_event_graph", help="Python module to run")
    parser.add_argument('--video_dir', type=str, required=True, help="Directory containing the videos")
    parser.add_argument('--out_dir', type=str, required=True, help="Directory for the outputs")
    parser.add_argument('--log_dir', type=str, default='slurm_logs', help="Directory for slurm logs and scripts")
    parser.add_argument('--qos', type=str, default='high')
    parser.add_argument('--scav', action='store_true')
    parser.add_argument('--vulcan', action='store_true')
    parser.add_argument('--cml', action='store_true')
    parser.add_argument('--gpus_to_use', default='a6', type=str, help='GPU type to use')
    parser.add_argument('--num_jobs', type=int, default=10, help="Number of slurm jobs to distribute tasks into")
    parser.add_argument('--dryrun', action='store_true', help="Do not submit, just generate scripts")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Extract all tasks
    all_tasks = []
    
    # Handle both .json and .jsonl
    data_list = []
    if args.json_file.endswith('.jsonl'):
        with open(args.json_file, 'r') as f:
            for line in f:
                if line.strip():
                    data_list.append(json.loads(line))
    else:
        with open(args.json_file, 'r') as f:
            data_list = json.load(f)
            
    for data in data_list:
        if 'video_path' in data:
            vid = data['video_path']
        elif 'video_uid' in data and 'object' in data:
            vid = f"{data['object'].replace(' ', '_')}/{data['video_uid']}.mp4"
        else:
            continue
            
        q = data.get('question', '')
        if not q:
            continue
            
        task_dict = {
            'video': os.path.join(args.video_dir, vid),
            'query': q,
            'qid': data.get('qid', data.get('id', None))
        }
        
        options = data.get('mcq_options', data.get('options'))
        if options:
            if isinstance(options, dict):
                task_dict['mcq_options'] = [f"{k}. {v}" for k, v in options.items()]
            else:
                task_dict['mcq_options'] = options
                
        all_tasks.append(task_dict)
        
    # Sort tasks by video so consecutive queries for same video can reuse cache in script
    all_tasks.sort(key=lambda x: x['video'])
    
    # Chunk tasks
    num_jobs = min(args.num_jobs, len(all_tasks))
    if num_jobs == 0:
        print("No tasks found.")
        return
        
    chunk_size = (len(all_tasks) + num_jobs - 1) // num_jobs
    chunks = [all_tasks[i:i + chunk_size] for i in range(0, len(all_tasks), chunk_size)]
    num_jobs = len(chunks)

    # Create commands and scripts per chunk
    commands_file = os.path.join(args.log_dir, 'commands.txt')
    with open(commands_file, 'w') as f:
        for i, chunk in enumerate(chunks):
            chunk_file = os.path.abspath(os.path.join(args.log_dir, f'chunk_tasks_{i}.json'))
            with open(chunk_file, 'w') as cf:
                json.dump(chunk, cf, indent=2)
                
            out_json = os.path.abspath(os.path.join(args.out_dir, f'answers_chunk_{i}.json'))
            out_work_dir = os.path.abspath(os.path.join(args.out_dir, f'working_dir_chunk_{i}'))
            
            cmd = (
                f"python -m {args.script_module} \\\n"
                f"    --tasks-json {chunk_file} \\\n"
                f"    --output-json {out_json} \\\n"
                f"    --tokens-per-frame 256 --fps 1.0 --sim-threshold 0.9 \\\n"
                f"    --aligned-model pe_core_vit_l \\\n"
                f"    --qa \\\n"
                f"    --working-dir {out_work_dir}"
            )
            
            # Create a bash script that runs this chunk
            chunk_script = os.path.abspath(os.path.join(args.log_dir, f"run_chunk_{i}.sh"))
            with open(chunk_script, 'w') as sh_f:
                sh_f.write("#!/bin/bash\n")
                sh_f.write(f"echo 'Processing chunk {i} with {len(chunk)} tasks'\n")
                sh_f.write(f"{cmd}\n")
            
            os.chmod(chunk_script, 0o755)
            f.write(f"{chunk_script}\n")

    # num_jobs is already updated above to match len(chunks)
    slurm_script = os.path.join(args.log_dir, 'submit.slurm')
    
    with open(slurm_script, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write(f"#SBATCH --array=1-{num_jobs}%500\n") # cap max running array jobs to 500
        f.write(f"#SBATCH --output={os.path.abspath(args.log_dir)}/%A_%a.log\n")
        f.write(f"#SBATCH --error={os.path.abspath(args.log_dir)}/%A_%a.err\n")
        f.write("#SBATCH --nodes=1\n")
        if args.scav:
            if args.vulcan:
                f.write("#SBATCH --account=vulcan-abhinav\n")
                f.write("#SBATCH --partition=vulcan-scavenger\n")
                f.write("#SBATCH --qos=vulcan-scavenger\n")
            else:
                f.write("#SBATCH --account=scavenger\n")
                f.write("#SBATCH --partition=scavenger\n")
                # remove_nodes = ['cbcb', 'clip', 'gamma']
                remove_nodes = None
                nodes_to_use = get_gpu_info(args, remove_nodes)
                # breakpoint()
                f.write(f'#SBATCH --nodelist={nodes_to_use}\n')
        else:
            f.write(f"#SBATCH --qos={args.qos}\n")

        f.write("#SBATCH --time=4:00:00\n")
        f.write("#SBATCH --gres=gpu:4\n")
        f.write("#SBATCH --cpus-per-task=8\n")
        f.write("#SBATCH --mem=32G\n")
        f.write("\n")
        f.write("source /etc/profile.d/modules.sh\n")
        f.write('module load cuda/12.6.3\n')
        f.write('module load ffmpeg/7.1\n')
        f.write('cd /fs/cfar-projects/mem_open_vocab/morph_graphs/anubhav\n')
        f.write('eval "$(/fs/cfar-projects/mem_open_vocab/miniconda3/bin/conda shell.bash hook)"\n')
        f.write("conda activate morph_graphs_qwen3_cu126\n") # Use env specified in debug_event_graph.py
        f.write("export PYTHONPATH=$(pwd):$PYTHONPATH\n")
        f.write("\n")
        f.write(f"CMD=$(head -n $SLURM_ARRAY_TASK_ID {os.path.abspath(commands_file)} | tail -n 1)\n")
        f.write("srun $CMD\n")

    print(f"Generated {num_jobs} jobs across {len(all_tasks)} total queries.")
    print(f"Slurm script created at: {slurm_script}")
    
    if not args.dryrun:
        print("Submitting jobs...")
        os.system(f"sbatch {slurm_script}")
    else:
        print(f"Dry-run mode. To submit, run: sbatch {slurm_script}")

if __name__ == '__main__':
    main()
