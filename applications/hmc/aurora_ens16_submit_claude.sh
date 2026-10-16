#!/bin/bash
# Submit the 16^4 ensemble job (aurora_ens16_claude.pbs) on an Aurora login node.
# Usage:
#   bash aurora_ens16_submit_claude.sh [NCHAIN] [NCHUNK] [NPER] [NSTEP] [WALLTIME]
#   defaults:                           8        6        5      80      01:30:00
# Billed to GenesisHackathonOct26 on reservation R8914313 (until 2026-10-16 22:00 UTC); 1 node.
set -u

GPT=${GPT:-/flare/GenesisHackathonOct26/nmatsum/gpt}
RUN=${RUN:-/flare/GenesisHackathonOct26/nmatsum/run_decorr_claude}
NCHAIN=${1:-8}
NCHUNK=${2:-6}
NPER=${3:-5}
NSTEP=${4:-80}
WALLTIME=${5:-01:30:00}
PBS_SCRIPT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/aurora_ens16_claude.pbs

[ -f ${PBS_SCRIPT} ] || { echo "missing ${PBS_SCRIPT}: commit + git pull"; exit 1; }
[ -f ${RUN}/therm_8888_b.cfg ] || { echo "missing ${RUN}/therm_8888_b.cfg"; exit 1; }
[ -f ${GPT}/lib/cgpt/build/cgpt.so ] || { echo "missing cgpt.so under ${GPT}"; exit 1; }
[ -f ${GPT}/applications/hmc/replicate_cfg_claude.py ] || { echo "gpt checkout lacks replicate_cfg_claude.py: git pull"; exit 1; }
[ ${NCHAIN} -le 12 ] || { echo "NCHAIN must be <= 12 (one PVC tile each)"; exit 1; }

echo "submitting: NCHAIN=${NCHAIN} NCHUNK=${NCHUNK} NPER=${NPER} NSTEP=${NSTEP} walltime=${WALLTIME}"
cd ${RUN}
qsub -l walltime=${WALLTIME} -o ${RUN} \
     -v GPT=${GPT},RUN=${RUN},NCHAIN=${NCHAIN},NCHUNK=${NCHUNK},NPER=${NPER},NSTEP=${NSTEP} ${PBS_SCRIPT}
