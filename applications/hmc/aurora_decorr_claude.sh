#!/bin/bash
# Aurora runs for the RMHMC momentum-decorrelation study, 8^4 Iwasaki beta = 2.95
# (spec: genesis/rmhmc_trainable_kernel_impl_plan_claude.md, "Implementation log" entry (9)).
# Runs on ONE Aurora compute node, one python process per PVC tile (ZE_AFFINITY_MASK), side by side.
# NOT EXACT: the implicit leapfrog with a U-dependent kernel is not volume preserving
# (N. H. Christ, L.-C. Jin, C. Lehner, E. Lundstrum, N. Matsumoto, arXiv:2412.19904).
#
# Usage (on the compute node; outputs and logs go to ${RUN}):
#   bash aurora_decorr_claude.sh tests
#        the 4 validation tests, tiles 0-3
#   bash aurora_decorr_claude.sh therm <ntraj> <in.cfg|none> <out.cfg> [nstep]
#        eta = 0 HMC chunk on tile 0 (tau 1, nstep default 80); chain chunks via in/out to fit the walltime.
#        (spec's nstep 20 gave acc 0.05 on 8^4 beta 2.95; nstep 80 gave 0.83, scan 2026-10-15)
#   bash aurora_decorr_claude.sh tune <in.cfg> "<niters>" "<nsteps>" [ntraj]
#        one run per (niter, nstep) pair, eta = 0.05, measurements off; niter 0 = eta = 0 reference
#   bash aurora_decorr_claude.sh summary
#        last lines / pass markers of all logs in ${RUN}
set -u

GPT=${GPT:-/flare/GenesisHackathonOct26/nmatsum/gpt}
HMC=${GPT}/applications/hmc
RUN=${RUN:-/flare/GenesisHackathonOct26/nmatsum/run_decorr_claude}
mkdir -p ${RUN}
cd ${RUN}

# lmod scripts reference unset variables, so relax `set -u` around module loads.
set +u
module load frameworks            # python3.12 + numpy, as used to build cgpt
set -u
source ${GPT}/lib/cgpt/build/source.sh
export SYCL_PROGRAM_COMPILE_OPTIONS="-ze-opt-large-register-file"
export ZE_FLAT_DEVICE_HIERARCHY=FLAT   # 12 tiles visible as 12 devices
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}   # 12 x 8 < 104 host cores

COMMON="--L 8,8,8,8 --action iwasaki --beta 2.95 --tol 1e-10 --mpi 1.1.1.1"

on_tile() {  # on_tile <tile> <log> <cmd...>: start in background, pinned to one tile
  local tile=$1 log=$2; shift 2
  echo "==== [$(date +%T)] tile ${tile} -> ${log}: $*"
  ZE_AFFINITY_MASK=${tile} "$@" > ${RUN}/${log} 2>&1 &
}

summary() {  # summary [log...]: pass markers, errors and last trajectory lines
  local logs=${*:-${RUN}/*_claude.log}
  for f in ${logs}; do
    echo "---- $(basename ${f})"
    grep -h -E "ALL PASS|SOME FAILED|Traceback|Error|saved|loaded" ${f} | tail -4
    grep -h -E "TRAJ|^GPT.* [0-9]+ +[0-9.]+ +[-0-9.e+]+ " ${f} | tail -3
  done
}

run_tests() {
  local tile=0
  for t in test_transverse_smear test_transverse_kinetic test_momentum_filter test_implicit_leapfrog; do
    on_tile ${tile} ${t}_claude.log python3 ${HMC}/${t}_claude.py --mpi 1.1.1.1
    tile=$((tile + 1))
  done
  wait
  summary ${RUN}/test_*_claude.log
}

run_therm() {  # <ntraj> <in.cfg|none> <out.cfg> [nstep]
  local nt=$1 in=$2 outcfg=$3 ns=${4:-80} load=""
  [ "${in}" != "none" ] && load="--load ${in}"
  # on_tile 0 ${outcfg%.cfg}_claude.log python3 ${HMC}/hmc_transverse_claude.py ${COMMON} \
  #     --eta 0.0 --niter 1 --tau 1.0 --nstep 20 --n ${nt} ${load} --save ${outcfg} --seed ${outcfg%.cfg}
  on_tile 0 ${outcfg%.cfg}_claude.log python3 ${HMC}/hmc_transverse_claude.py ${COMMON} \
      --eta 0.0 --niter 1 --tau 1.0 --nstep ${ns} --n ${nt} ${load} --save ${outcfg} --seed ${outcfg%.cfg}
  wait
  summary ${RUN}/${outcfg%.cfg}_claude.log
}

run_tune() {  # <in.cfg> "<niters>" "<nsteps>" [ntraj]
  local in=$1 niters=$2 nsteps=$3 nt=${4:-2} tile=0 logs=""
  for ni in ${niters}; do
    for ns in ${nsteps}; do
      if [ ${tile} -ge 12 ]; then echo "more than 12 (niter, nstep) pairs: skipping the rest"; break 2; fi
      local eta=0.05 nl=${ni}
      if [ ${ni} = 0 ]; then eta=0.0; nl=1; fi   # niter 0 = eta = 0 reference
      local log=tune_e${eta}_n${nl}_ns${ns}_claude.log
      on_tile ${tile} ${log} python3 ${HMC}/decorrelation_claude.py ${COMMON} \
          --eta ${eta} --niter ${nl} --stot 3.0 --nstep ${ns} --n ${nt} --meas_every 0 \
          --load ${in} --seed tune_e${eta}_n${nl}_ns${ns}
      logs="${logs} ${RUN}/${log}"
      tile=$((tile + 1))
    done
  done
  wait
  summary ${logs}
}

stage=${1:-}
shift
case ${stage} in
  tests)   run_tests ;;
  therm)   run_therm "$@" ;;
  tune)    run_tune "$@" ;;
  summary) summary ;;
  *) echo "usage: $0 tests | therm <ntraj> <in|none> <out> | tune <in> \"<niters>\" \"<nsteps>\" [ntraj] | summary"; exit 1 ;;
esac
echo "==== [$(date +%T)] done: ${stage}"
