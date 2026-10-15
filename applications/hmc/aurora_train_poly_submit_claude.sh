#!/bin/bash
# Submit the FA-polynomial training + trained U-turn job (aurora_train_poly_claude.pbs) on an Aurora login node.
# Usage:
#   bash aurora_train_poly_submit_claude.sh [NSTEP_N2] [NSTEP_N8] [NTRAJ] [WALLTIME] [MAXEVAL_N2] [MAXEVAL_N8]
#   defaults:                                120        60         4       02:00:00   200          300
#   (N=2 longitudinal at nstep 60 gave dH 1.4-4.1 in PBS 8933335, hence 120)
# Billed to GenesisHackathonOct26 on reservation R8914313 (until 2026-10-16 22:00 UTC); 1 node.
# Outputs: ${RUN}/tr_<jobid>_full_N{2,8}_claude_{coeffs.py,history.dat,train.log,uturn.dat,uturn.log}
set -u

GPT=${GPT:-/flare/GenesisHackathonOct26/nmatsum/gpt}
RUN=${RUN:-/flare/GenesisHackathonOct26/nmatsum/run_decorr_claude}
NSTEP_N2=${1:-120}
NSTEP_N8=${2:-60}
NTRAJ=${3:-4}
WALLTIME=${4:-02:00:00}
MAXEVAL_N2=${5:-200}
MAXEVAL_N8=${6:-300}
PBS_SCRIPT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/aurora_train_poly_claude.pbs

[ -f ${PBS_SCRIPT} ] || { echo "missing ${PBS_SCRIPT}: commit + git pull"; exit 1; }
[ -f ${RUN}/therm_8888_b.cfg ] || { echo "missing ${RUN}/therm_8888_b.cfg"; exit 1; }
[ -f ${GPT}/lib/cgpt/build/cgpt.so ] || { echo "missing cgpt.so under ${GPT}"; exit 1; }
[ -f ${GPT}/applications/hmc/train_poly_claude.py ] || { echo "gpt checkout lacks train_poly_claude.py: git pull"; exit 1; }
grep -q -- '--coeffs_file' ${GPT}/applications/hmc/decorrelation_claude.py \
    || { echo "gpt checkout lacks --coeffs_file in decorrelation_claude.py: git pull"; exit 1; }

echo "submitting: NSTEP_N2=${NSTEP_N2} NSTEP_N8=${NSTEP_N8} NTRAJ=${NTRAJ} walltime=${WALLTIME} maxeval=${MAXEVAL_N2}/${MAXEVAL_N8}"
cd ${RUN}
qsub -l walltime=${WALLTIME} -o ${RUN} \
     -v GPT=${GPT},RUN=${RUN},NSTEP_N2=${NSTEP_N2},NSTEP_N8=${NSTEP_N8},NTRAJ=${NTRAJ},MAXEVAL_N2=${MAXEVAL_N2},MAXEVAL_N8=${MAXEVAL_N8} \
     ${PBS_SCRIPT}
