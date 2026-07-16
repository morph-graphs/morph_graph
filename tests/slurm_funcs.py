import numpy as np
import os
from datetime import datetime
import argparse
import time
import pandas as pd
import socket
import subprocess
import yaml
import wandb
import getpass
import os
import socket
import yaml


def check_qos(args):
    qos_dict = {"sailon" : {"nhrs" : 2, "cores": 16, "mem":128},
            "scav" : {"nhrs" : 72, "cores": 92, "mem":500},
            "vulc_scav" : {"nhrs" : 72, "cores": 32, "mem":220},
            "vulc_exe" : {"nhrs" : 24*7, "cores": 32, "mem":220},

            "zara" : {"nhrs" : 72, "cores": 92, "mem":500},
            
            "cml_scav" : {"nhrs" : 72, "cores": 16, "mem":128}, 

            "high" : {"gpu":4, "cores": 16, "mem":200, "nhrs": 36},
            "medium" : {"gpu":2, "cores": 8, "mem":64, "nhrs": 72},
            "default" : {"gpu":1, "cores": 4, "mem":32, "nhrs": 168},
            "tron" : {"gpu":1, "cores": 4, "mem":32, "nhrs": 168}}
    
    for key, max_value in qos_dict[args.qos].items():
        val_from_args = getattr(args, key)
        if val_from_args != None:
            if val_from_args > max_value:
                raise ValueError("Invalid paramter for {} for {}".format(key, args.qos))
        else:
            setattr(args, key, max_value)
    return args

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
    elif args.gpus_to_use == 'h100':
        gpu_type = ['h100']
    elif args.gpus_to_use == 'ti':
        gpu_type = ['gtx1080ti', 'rtx2080ti']
    else:
        raise ValueError("Invalid gpu type")

    if args.qos == 'scav':
        remove_nodes.append('cbcb')
        remove_nodes.append('gamma')
        remove_nodes.append('vulcan')
    def run(cmd, print_err=True):
        try:
            return subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT).decode('UTF-8').splitlines()
        except subprocess.CalledProcessError as e:
            # raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
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


def get_transfer_commands(data_paths, destination_dir, dataset, touch_file_path):
    transfer_commands = []
    len_check_commands = []
    touch_file_check_command = f'[ ! -f "{touch_file_path}" ]'
    for key, path in data_paths.items():
        if 'split' in key:
            transfer_commands.append(f'cp -r  {path} {destination_dir}')
        else:
            if dataset in {'epickitchens'}:
                if path[-1]!='/':
                    path += '/'
                transfer_commands.append('ln -s {} {}'.format(path, destination_dir))
            else:
                if 'point' in key:

                    condition_check_command = f'[ "${key}" -eq 1 ] || {touch_file_check_command} &&'
                else:
                    condition_check_command = f'[ "${key}" -eq 1 ] &&'
                transfer_commands.append(f'{condition_check_command} {base_sync_command} {path} {destination_dir}')
                folder_name = path.split('/')[-1]
                dest_dir = os.path.join(destination_dir, folder_name)
                len_check_commands.append(base_len_check_command.format(path, dest_dir, key))
                print(key)
    return transfer_commands, len_check_commands


def get_slurm_init_commands(args, num_exp):
    slurm_init_commands = []
    slurm_init_commands.append("#!/bin/bash\n")
    slurm_init_commands.append(f"#SBATCH --array=1-{num_exp}\n")
    #slurmfile.write(f"#SBATCH --array=1-10\n")
    slurm_init_commands.append("#SBATCH --output=/dev/null\n")
    slurm_init_commands.append("#SBATCH --error=/dev/null\n")
    slurm_init_commands.append("#SBATCH --requeue\n")
    slurm_init_commands.append("#SBATCH --nodes=1\n")
    
    return slurm_init_commands

def args_specific_commands(args, remove_nodes=None, zara=False):
    args = check_qos(args)
    qos_specific_commands = []
    if zara:
        qos_specific_commands.append("#SBATCH --account=abhinav2-prj-cmsc\n")
        qos_specific_commands.append("#SBATCH --partition gpu\n")
    elif args.qos == "scav":
        qos_specific_commands.append("#SBATCH --account=scavenger\n")
        qos_specific_commands.append("#SBATCH --qos scavenger\n")
        qos_specific_commands.append("#SBATCH --partition scavenger\n")
    
    elif args.qos == "vulc_scav":
        qos_specific_commands.append("#SBATCH --account=vulcan-abhinav\n")
        qos_specific_commands.append("#SBATCH --qos vulcan-scavenger\n")
        qos_specific_commands.append("#SBATCH --partition vulcan-scavenger\n")
    elif args.qos == 'cml_scav':
        qos_specific_commands.append("#SBATCH --account=cml-scavenger\n")
        qos_specific_commands.append("#SBATCH --qos cml-scavenger\n")
        qos_specific_commands.append("#SBATCH --partition cml-scavenger\n")
    elif args.qos == 'vulc_exe':
        qos_specific_commands.append("#SBATCH --account=vulcan-abhinav\n")
        qos_specific_commands.append("#SBATCH --qos vulcan-exempt\n")
        qos_specific_commands.append("#SBATCH --partition vulcan-ampere\n")
    qos_specific_commands.append("#SBATCH --time=%d:00:00\n" % args.nhrs)
    qos_specific_commands.append("#SBATCH --cpus-per-task=%d\n" % args.cores)
    qos_specific_commands.append("#SBATCH --mem=%dG\n" % args.mem)
    if not args.gpu is None:
        if zara:
            qos_specific_commands.append(f'#SBATCH --gres=gpu:h100:{args.gpu}\n')
        else:
            nodes_to_use = get_gpu_info(args, remove_nodes)
            qos_specific_commands.append(f'#SBATCH --gres=gpu:{args.gpu}\n')
            qos_specific_commands.append(f'#SBATCH --nodes=1\n')
            qos_specific_commands.append(f'#SBATCH --nodelist={nodes_to_use}\n')

    return qos_specific_commands


def main_slurm_run_commands(args, log_files_path=None, data_dir=None, zara=False):
    commands = []
    commands.append("cd " + os.getcwd() + '\n')
    commands.append("export PYTHONPATH=$(pwd):$PYTHONPATH\n")
    commands.append("module load ffmpeg\n")
    commands.append("export MKL_THREADING_LAYER=GNU\n")
    commands.append("source /fs/cfar-projects/actionloc/new_miniconda/bin/activate\n")
    commands.append("conda activate pips2\n")
    if zara:
        commands.append("export SCRATCH_DIR=tmp\n")
        commands.append("module load singularity\n")
        commands.append("export MKL_THREADING_LAYER=GNU\n")
        commands.append('export WANDB_MODE=offline\n')
        torch_home = '/scratch/zt1/project/abhinav2-prj/user/pulkit'

    else:
        commands.append("export SCRATCH_DIR\n")
        commands.append('[ -d "/scratch1" ] && SCRATCH_DIR="scratch1" || SCRATCH_DIR="scratch0"\n')
        torch_home = '/fs/cfar-projects/actionloc'
    commands.append(f'export TORCH_HOME={torch_home}\n')
    log_file_path = os.path.join(log_files_path, 'log.txt')
    err_file_path = os.path.join(log_files_path, 'err.txt')
    now_file_path = os.path.join(log_files_path, 'commands.txt')
    if data_dir is not None:
        commands += transfer_commands(args, data_dir, zara)
    commands.append(f"srun --export=ALL,SCRATCH_DIR=$SCRATCH_DIR --output=$(head -n $SLURM_ARRAY_TASK_ID {log_file_path} | tail -n 1) --error=$(head -n $SLURM_ARRAY_TASK_ID {err_file_path} | tail -n 1)  $(head -n $SLURM_ARRAY_TASK_ID {now_file_path} | tail -n 1)\n")
    return commands

def prepare_command_for_transfer(args, path, destination_dir):
    if args.transfer_type == 'ln' or args.dataset == 'k400':
        if path[-1] != '/':
            path += '/'
        return f'ln -s {path} {destination_dir} \n'
    elif args.transfer_type == 'rsync':
        if path[-1] == '/':
            path = path[:-1]
        return f'~/msrsync/msrsync3 -p {args.cores} -P {path} {destination_dir} \n'

def transfer_commands(args, data_dir, zara=False):
    transfer_commands = []
    transfer_commands.append(f'mkdir -p {data_dir}\n')
    hostname = socket.gethostname()
    
    with open('paths.yaml', 'r') as f:
        paths = yaml.safe_load(f)
    if zara:
        all_paths = paths[args.dataset]['zaratan']
    else:
        all_paths = paths[args.dataset]['nexus']
    video_path = all_paths['videos']
    transfer_commands.append(prepare_command_for_transfer(args, video_path, data_dir))
    if zara:
        split_data = '/scratch/zt1/project/abhinav2-prj/user/pulkit/few_shot_split/*'
    else:
        split_data = '/fs/cfar-projects/actionloc/bounce_back/few_shot_split/*'
    split_transfer_command = 'cp -r {} {} \n'.format(split_data, data_dir)
    transfer_commands.append(split_transfer_command)
    point_info_path = os.path.join(all_paths['points_info'] , args.point_info_name)
    if args.dataset != 'ssv2':
        point_info_path += '_fps_10'
    transfer_commands.append(prepare_command_for_transfer(args, point_info_path, data_dir))

    return transfer_commands