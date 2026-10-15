#!/bin/bash
# Submit the polyfull N=2 vs N=8 U-turn test (aurora_polyfull_uturn_claude.pbs) on an Aurora login node.
# Usage:
#   bash aurora_polyfull_uturn_submit_claude.sh [NSTEP] [NTRAJ] [WALLTIME]
#   defaults:                                    60      4       01:30:00
# Billed to GenesisHackathonOct26 on reservation R8914313 (until 2026-10-16 22:00 UTC); 1 node.
# Outputs and logs: ${RUN}/pf_<jobid>_*_claude.{log,dat}; PBS stdout: ${RUN}/polyfull_uturn.o<jobid>
set -u

GPT=${GPT:-/flare/GenesisHackathonOct26/nmatsum/gpt}
RUN=${RUN:-/flare/GenesisHackathonOct26/nmatsum/run_decorr_claude}
NSTEP=${1:-60}
NTRAJ=${2:-4}
WALLTIME=${3:-01:30:00}
PBS_SCRIPT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/aurora_polyfull_uturn_claude.pbs

[ -f ${RUN}/therm_8888_b.cfg ] || { echo "missing ${RUN}/therm_8888_b.cfg"; exit 1; }
[ -f ${GPT}/lib/cgpt/build/cgpt.so ] || { echo "missing cgpt.so under ${GPT}"; exit 1; }
grep -q -- '--longitudinal' ${GPT}/applications/hmc/decorrelation_claude.py \
    || { echo "gpt checkout lacks --longitudinal in decorrelation_claude.py: git pull"; exit 1; }
[ -f ${GPT}/applications/hmc/eval_ml_loss_claude.py ] || { echo "gpt checkout lacks eval_ml_loss_claude.py: git pull"; exit 1; }

echo "submitting: NSTEP=${NSTEP} NTRAJ=${NTRAJ} walltime=${WALLTIME}"
cd ${RUN}
qsub -l walltime=${WALLTIME} -o ${RUN} -v GPT=${GPT},RUN=${RUN},NSTEP=${NSTEP},NTRAJ=${NTRAJ} ${PBS_SCRIPT}
