# Sourced at the top of every baselines/<b>/run_<b>.sh, before any Python starts. Every baseline here runs as plain
# processes on the GPUs it was given, never as a torch.distributed job, but some clusters start a job's container with
# torchrun's variables set (Max's H100 nodes do), and libraries then act as if they were one rank of a launched job:
#   - transformers 4.50-4.53 turns from_pretrained(device_map="auto") into tensor-parallel loading and starts NCCL on
#     MASTER_PORT: EGAgent's SigLIP 2 fails to load ("too many values to unpack"); M3-Agent's memorization model loads,
#     but through that path;
#   - accelerate's Accelerator() (VST) starts an NCCL group on MASTER_ADDR:MASTER_PORT, so a second job on the node
#     with the same port fails (EADDRINUSE);
#   - with TORCHELASTIC_USE_AGENT_STORE=True, EgoRAG's one-process gloo group fails (it looks for torchrun's agent).
# Without them each baseline runs as it does everywhere else. Nothing here launches with torchrun, so nothing needs them.
unset WORLD_SIZE RANK LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK GROUP_WORLD_SIZE ROLE_NAME ROLE_RANK ROLE_WORLD_SIZE \
      MASTER_ADDR MASTER_PORT TORCHELASTIC_RUN_ID TORCHELASTIC_RESTART_COUNT TORCHELASTIC_MAX_RESTARTS \
      TORCHELASTIC_USE_AGENT_STORE TORCHELASTIC_ERROR_FILE
