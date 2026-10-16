#!/bin/bash
# Submit the hold-out U-turn job (aurora_holdout_claude.pbs) on an Aurora login node.
# Usage:
#   bash aurora_holdout_submit_claude.sh <L: 8|16> [NSTEP_N8] [NSTEP_N2] [WALLTIME]
#   defaults:  L=8:  60  120  01:00:00     L=16:  120  120  02:00:00
# Billed to GenesisHackathonOct26 on reservation R8914313 (until 2026-10-16 22:00 UTC); 1 node.
# Outputs: ${RUN}/ho<L>_<jobid>_*_claude.{dat,log}; starts for L=8 in ${RUN}/holdout8/.
set -u

GPT=${GPT:-/flare/GenesisHackathonOct26/nmatsum/gpt}
RUN=${RUN:-/flare/GenesisHackathonOct26/nmatsum/run_decorr_claude}
LSIZE=${1:?usage: $0 <8|16> [NSTEP_N8] [NSTEP_N2] [WALLTIME]}
if [ ${LSIZE} = 16 ]; then D8=120; DW=02:00:00; else D8=60; DW=01:00:00; fi
NSTEP_N8=${2:-${D8}}
NSTEP_N2=${3:-120}
WALLTIME=${4:-${DW}}
PBS_SCRIPT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/aurora_holdout_claude.pbs

[ ${LSIZE} = 8 ] || [ ${LSIZE} = 16 ] || { echo "L must be 8 or 16"; exit 1; }
[ -f ${PBS_SCRIPT} ] || { echo "missing ${PBS_SCRIPT}: commit + git pull"; exit 1; }
[ -f ${GPT}/lib/cgpt/build/cgpt.so ] || { echo "missing cgpt.so under ${GPT}"; exit 1; }
grep -q -i traceless ${GPT}/applications/hmc/lambda_min_claude.py || { echo "lambda_min_claude.py lacks the trace-sector fix: git pull"; exit 1; }
for N in 2 8; do
  [ -f ${RUN}/tr_8933429_full_N${N}_claude_coeffs.py ] || { echo "missing trained N=${N} coeffs"; exit 1; }
done
if [ ${LSIZE} = 8 ]; then
  [ -f ${RUN}/therm_8888_b.cfg ] || { echo "missing ${RUN}/therm_8888_b.cfg"; exit 1; }
else
  for c in 0 1 2 3; do [ -f ${RUN}/ens16/chain${c}_6.cfg ] || { echo "missing ens16/chain${c}_6.cfg"; exit 1; }; done
fi

echo "submitting: L=${LSIZE} NSTEP_N8=${NSTEP_N8} NSTEP_N2=${NSTEP_N2} walltime=${WALLTIME}"
cd ${RUN}
qsub -N holdout${LSIZE} -l walltime=${WALLTIME} -o ${RUN} \
     -v GPT=${GPT},RUN=${RUN},LSIZE=${LSIZE},NSTEP_N8=${NSTEP_N8},NSTEP_N2=${NSTEP_N2} ${PBS_SCRIPT}
