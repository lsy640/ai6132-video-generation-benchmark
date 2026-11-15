#!/bin/bash
# Login/head-node shell submission only. Both allocations are submitted before release.
set -euo pipefail
: "${LONGCAT_ROOT:?set runtime}"
BATCH="${1:?usage: submit_batch.sh /absolute/new-batch}"
[[ -f "$BATCH/config.json" && -f "$BATCH/code/wave.slurm" && -d "$BATCH/job_ids" ]] || exit 2
cd "$BATCH"
exec 9>submission.lock
flock -n 9 || exit 0
if [ ! -f job_ids/short ]; then
    first=$(sbatch --hold --parsable --job-name=lc16-short18 --chdir="$BATCH" --export=ALL,LONGCAT_BATCH="$BATCH",LONGCAT_WAVE=short code/wave.slurm)
    printf '%s\n' "${first%%;*}" > job_ids/short
fi
first=$(cat job_ids/short)
if [ ! -f job_ids/finish ]; then
    last=$(sbatch --hold --parsable --job-name=lc16-long5 --dependency=afterany:"$first" --chdir="$BATCH" --export=ALL,LONGCAT_BATCH="$BATCH",LONGCAT_WAVE=finish code/wave.slurm)
    printf '%s\n' "${last%%;*}" > job_ids/finish
fi
last=$(cat job_ids/finish)
case "$first$last" in *[!0-9]*|'') exit 2;; esac
printf '{"short":%s,"finish":%s,"dependency":"afterany:%s","client_required":false}\n' "$first" "$last" "$first" > jobs.json
scontrol release "$first" 2>/dev/null || true
scontrol release "$last" 2>/dev/null || true
cat jobs.json
