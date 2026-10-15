#!/bin/bash
# Build Grid (lehner/Grid, branch feature/gpt) + GPT (cgpt) on macOS / Apple silicon, CPU only.
# Grid has no Metal backend (accelerators: cuda|sycl|hip|none), so this is a NEON build:
# --enable-simd=GEN with 16-byte (128-bit NEON) vectors, OpenMP threads, no MPI.
# Adapted from NM's ~/build_grid_mac_gpt.sh and gpt's scripts/bootstrap/macos-catalina.clang.avx2.no-mpi.
# Sources:
#   GPT:  C. Lehner et al., https://github.com/lehner/gpt (master; NM fork nobuyukimatsumoto68/gpt)
#   Grid: P. Boyle et al., arXiv:1512.03487; fork https://github.com/lehner/Grid (feature/gpt)
# Note: no `rm` anywhere (house rule); rebuilds overwrite in place.
#
# Prerequisites (Homebrew): gcc (g++-15), gmp, mpfr, openssl@3, fftw, autoconf, automake, libtool, wget.
# Layout (dir_base = parent of the gpt checkout, e.g. ~/Documents/genesis):
#   ${dir_base}/gpt         this checkout (cgpt is built here)
#   ${dir_base}/Grid        lehner/Grid feature/gpt, build dir Grid/build (= ../Grid/build for cgpt)
#   ${dir_base}/opt/lime    c-lime install
#   ${dir_base}/.venv-gpt   python venv with numpy
#   ${dir_base}/log_mac     per-step logs
#
# Usage:  bash mac_build_claude.sh [stage]
#   stage = lime | venv | grid | gpt | test | all (default all)
set -u

stage=${1:-all}
NJ=${NJ:-4}                       # make parallelism (house cap: 4 cores)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

gpt_root="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." >/dev/null 2>&1 && pwd )"
[ -d ${gpt_root}/lib/cgpt ] || { echo "run this script from inside a gpt checkout (scripts/bootstrap/)"; exit 1; }
dir_base=${DIR_BUILD:-$(dirname ${gpt_root})}
dir_opt=${dir_base}/opt
dir_log=${dir_base}/log_mac
dir_venv=${dir_base}/.venv-gpt
mkdir -p ${dir_opt} ${dir_log}

# ---------------- environment ----------------
BREW=$(brew --prefix)
export CC=${BREW}/bin/gcc-15
export CXX=${BREW}/bin/g++-15
GMP=$(brew --prefix gmp)
MPFR=$(brew --prefix mpfr)
OPENSSL=$(brew --prefix openssl@3)
FFTW=$(brew --prefix fftw)
PYTHON=${BREW}/bin/python3        # must match ${BREW}/bin/python3-config, which cgpt's make uses

run() {  # run <logname> <cmd...>; abort on failure
  local log=$1; shift
  echo "==== [$(date +%T)] ${log}: $*"
  ( time "$@" ) > ${dir_log}/${log}.log 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then echo "FAILED (${rc}): see ${dir_log}/${log}.log"; tail -20 ${dir_log}/${log}.log; exit $rc; fi
}

check_prereq() {
  local miss=0
  for f in ${CC} ${CXX} ${PYTHON} ${BREW}/bin/python3-config; do [ -x $f ] || { echo "missing: $f"; miss=1; }; done
  for d in ${GMP} ${MPFR} ${OPENSSL} ${FFTW}; do [ -d $d/include ] || { echo "missing: $d (brew install $(basename $d))"; miss=1; }; done
  [ -f ${FFTW}/lib/libfftw3.a ] || [ -f ${FFTW}/lib/libfftw3.dylib ] || { echo "missing double-precision fftw in ${FFTW}"; miss=1; }
  [ $miss -eq 0 ] || exit 1
}

# ---------------- c-lime ----------------
build_lime() {
  cd ${dir_opt}
  [ -d c-lime ] || run lime_clone git clone https://github.com/usqcd-software/c-lime
  cd c-lime
  run lime_autogen ./autogen.sh
  run lime_configure ./configure --prefix=${dir_opt}/lime CC=${CC} CFLAGS="-O2 -fPIC"
  run lime_make make -j ${NJ}
  run lime_install make install
}

# ---------------- python venv (numpy for cgpt) ----------------
build_venv() {
  [ -x ${dir_venv}/bin/python3 ] || run venv_create ${PYTHON} -m venv ${dir_venv}
  run venv_pip ${dir_venv}/bin/python3 -m pip install --upgrade pip numpy
}

# ---------------- Grid (lehner fork, feature/gpt) ----------------
build_grid() {
  cd ${dir_base}
  [ -d Grid ] || run grid_clone git clone -b feature/gpt https://github.com/lehner/Grid
  cd Grid
  # downloads Eigen; skip if already done. bootstrap.sh calls `sha256sum --check` (GNU); macOS's
  # /sbin/sha256sum is BSD and rejects it, so put Homebrew coreutils' GNU tools first on PATH.
  # [ -d Eigen ] || run bootstrap ./bootstrap.sh
  [ -d Eigen ] || run bootstrap env PATH=${BREW}/opt/coreutils/libexec/gnubin:${PATH} ./bootstrap.sh
  mkdir -p ${dir_base}/Grid/build; cd ${dir_base}/Grid/build
  # build dir is Grid/build so that cgpt's `make` finds it as ../Grid/build relative to gpt/
  run configure ../configure --prefix=${dir_base}/Grid/install \
      --enable-simd=GEN \
      --enable-gen-simd-width=16 \
      --enable-comms=none \
      --disable-gparity \
      --disable-fermion-reps \
      --with-hdf5=no \
      --with-gmp=${GMP} \
      --with-mpfr=${MPFR} \
      --with-openssl=${OPENSSL} \
      --with-fftw=${FFTW} \
      --with-lime=${dir_opt}/lime \
      CC=${CC} CXX=${CXX} \
      CXXFLAGS="-fPIC -std=c++17 -O3 -fopenmp" \
      LDFLAGS="-fopenmp"
  # Nc=3 (Grid default)
  run grid_make make -j ${NJ}
}

# ---------------- GPT / cgpt ----------------
build_gpt() {
  source ${dir_venv}/bin/activate                  # python3 (numpy) for cgpt's make
  cd ${gpt_root}/lib/cgpt
  # Homebrew python3-config --ldflags has no -lpython; resolve symbols at import time
  export CGPT_EXTRA_LDFLAGS="-undefined dynamic_lookup"
  run cgpt_make ./make ${dir_base}/Grid/build ${NJ}
  [ -f ${gpt_root}/lib/cgpt/build/cgpt.so ] || { echo "cgpt.so not built"; exit 1; }
  echo "To use:  source ${dir_venv}/bin/activate; source ${gpt_root}/lib/cgpt/build/source.sh"
}

# ---------------- smoke test ----------------
run_test() {
  source ${dir_venv}/bin/activate
  source ${gpt_root}/lib/cgpt/build/source.sh
  cd ${gpt_root}/tests/qcd
  run test_gauge python3 gauge.py
  tail -5 ${dir_log}/test_gauge.log
}

check_prereq
case ${stage} in
  lime) build_lime ;;
  venv) build_venv ;;
  grid) build_grid ;;
  gpt)  build_gpt ;;
  test) run_test ;;
  all)  build_lime; build_venv; build_grid; build_gpt; run_test ;;
  *) echo "unknown stage ${stage}"; exit 1 ;;
esac
echo "==== [$(date +%T)] done: ${stage}"
