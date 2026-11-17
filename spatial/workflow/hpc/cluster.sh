# Cluster settings shared by all SLURM jobs: partitions, QOS and GPU types.
# Sourced by run_all.sh and deconvolution_submit.sh. To submit a single job by hand:
#   source workflow/hpc/cluster.sh
#   sbatch "${SBATCH_CPU[@]}" workflow/hpc/atlas_scores.sbatch
# Every value can be overridden from the environment. HHCA_SBATCH_EXTRA adds sbatch options to every job,
# e.g. HHCA_SBATCH_EXTRA="--exclude=node07 --account=mylab".
# shellcheck disable=SC2034  # SBATCH_CPU and SBATCH_GPU are read by the scripts that source this file

CPU_PARTITION="${HHCA_CPU_PARTITION:-cpu_p}"
CPU_QOS="${HHCA_CPU_QOS:-cpu_normal}"
GPU_PARTITION="${HHCA_GPU_PARTITION:-gpu_p}"
GPU_QOS="${HHCA_GPU_QOS:-gpu_normal}"
# cell2location, stereoscope and spotiphy run on torch 2.3.1 + CUDA 12.1, which supports V100 and A100 cards
GPU_CONSTRAINT="${HHCA_GPU_CONSTRAINT:-v100_32gb|a100_20gb|a100_40gb|a100_80gb}"

read -r -a SBATCH_EXTRA <<< "${HHCA_SBATCH_EXTRA:-}"
SBATCH_CPU=(--partition="$CPU_PARTITION" --qos="$CPU_QOS" ${SBATCH_EXTRA[@]+"${SBATCH_EXTRA[@]}"})
SBATCH_GPU=(--partition="$GPU_PARTITION" --qos="$GPU_QOS" --gres=gpu:1 --constraint="$GPU_CONSTRAINT"
            ${SBATCH_EXTRA[@]+"${SBATCH_EXTRA[@]}"})
