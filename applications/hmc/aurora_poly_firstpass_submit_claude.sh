#!/bin/bash
# Submit the U-turn first pass (aurora_poly_firstpass_claude.pbs) on an Aurora login node.
# Usage:
#   bash aurora_poly_firstpass_submit_claude.sh [NSTEP_POLY] [NSTEP_MASKED] [NTRAJ] [WALLTIME]
#   defaults:                                    60           200            1       01:00:00
# Example (3 trajectories each, 2 h):  bash aurora_poly_firstpass_submit_claude.sh 60 200 3 02:00:00
# Billed to GenesisHackathonOct26 on reservation R8914313 (until 2026-10-16 22:00 UTC); 1 node.
# Outputs and logs: ${RUN}/fp_<jobid>_*_claude.{log,dat}; PBS stdout: ${RUN}/poly_firstpass.o<jobid>
set -u

GPT=${GPT:-/flare/GenesisHackathonOct26/nmatsum/gpt}
RUN=${RUN:-/flare/GenesisHackathonOct26/nmatsum/run_decorr_claude}
NSTEP_POLY=${1:-60}
NSTEP_MASKED=${2:-200}
NTRAJ=${3:-1}
WALLTIME=${4:-01:00:00}
PBS_SCRIPT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/aurora_poly_firstpass_claude.pbs

[ -f ${RUN}/therm_8888_b.cfg ] || { echo "missing ${RUN}/therm_8888_b.cfg"; exit 1; }
[ -f ${GPT}/lib/cgpt/build/cgpt.so ] || { echo "missing cgpt.so under ${GPT}"; exit 1; }
[ -f ${GPT}/applications/hmc/poly_kernel_claude.py ] || { echo "gpt checkout lacks poly_kernel_claude.py: git pull"; exit 1; }

echo "submitting: NSTEP_POLY=${NSTEP_POLY} NSTEP_MASKED=${NSTEP_MASKED} NTRAJ=${NTRAJ} walltime=${WALLTIME}"
cd ${RUN}
qsub -l walltime=${WALLTIME} -o ${RUN} \
     -v GPT=${GPT},RUN=${RUN},NSTEP_POLY=${NSTEP_POLY},NSTEP_MASKED=${NSTEP_MASKED},NTRAJ=${NTRAJ} \
     ${PBS_SCRIPT}
