#!/bin/bash
# The first line of every copy of a job-board task's Slurm job (worker.py --queues submits one copy per queue that
# fits the task, all under one job name). The first copy to start takes the task: it creates the marker file
# atomically (noclobber) and cancels the copies still pending. A copy that starts while another holds the marker
# exits 3, and the job then ends with 0 without doing any work. The same job restarting after a requeue keeps the task.
#   bash coord/slurm_guard.sh <marker file> || exit 0
# A marker, not only squeue, decides: two copies starting in the same second both see the other RUNNING in squeue,
# and a copy submitted after the first one already started (and finished) would see no one at all.
marker=$1
me=$SLURM_JOB_ID
if ! ( set -C; echo "$me" > "$marker" ) 2>/dev/null && [ "$(cat "$marker" 2>/dev/null)" != "$me" ]; then
    echo "job $me: copy $(cat "$marker" 2>/dev/null) of $SLURM_JOB_NAME has this task; exiting without work"
    exit 3
fi
rm -f "${marker%.ran}.rc"   # the exit code of this copy's earlier, requeued start is not this run's
echo "job $me ($SLURM_JOB_PARTITION) runs $SLURM_JOB_NAME; cancelling its pending copies"
squeue -h -u "$USER" -n "$SLURM_JOB_NAME" -t PENDING -o %i | grep -vx "$me" | xargs -r scancel
exit 0
