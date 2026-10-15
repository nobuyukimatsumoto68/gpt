#!/bin/bash
# Build Grid (lehner/Grid, branch feature/gpt) + GPT (cgpt) on ALCF Aurora (Intel PVC, SYCL).
# Adapted from NM's LLNL Tuolumne (HIP) Grid build script; configure flags follow
# lehner/Grid systems/Aurora/config-command (SYCL). Sources:
#   GPT:  C. Lehner et al., https://github.com/lehner/gpt (master; NM fork nobuyukimatsumoto68/gpt)
#   Grid: P. Boyle et al., arXiv:1512.03487; fork https://github.com/lehner/Grid (feature/gpt)
# Note: the commented `rm` / `find -delete` lines of the Tuolumne script are dropped (house no-rm rule).
#
# Usage (on an Aurora login node):  bash aurora_build_claude.sh [stage]
#   stage = lime | grid | gpt | all (default all). Each stage logs to ${dir_log}/<step>.log.
#   grid is incremental once configured; FORCE_CONFIGURE=1 bash aurora_build_claude.sh grid reconfigures.
#   Grid's bootstrap.sh runs only if Grid/configure is missing (fresh clone); FORCE_BOOTSTRAP=1 forces it.
set -u

stage=${1:-all}
NJ=${NJ:-16}                      # make parallelism (login node: keep modest)

# gpt checkout containing this script (scripts/bootstrap/ -> ../..); empty if run standalone
gpt_root="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." >/dev/null 2>&1 && pwd )"
[ -d ${gpt_root}/lib/cgpt ] || gpt_root=""
# default: build next to the gpt checkout, so the layout is <dir>/gpt, <dir>/Grid, <dir>/log, <dir>/install
if [ -n "${gpt_root}" ]; then dir_build_default=$(dirname ${gpt_root}); else dir_build_default=/flare/GenesisHackathonOct26/${USER}; fi
# dir_build=${DIR_BUILD:-/flare/GenesisHackathonOct26/${USER}/gpt_fthmc}
dir_build=${DIR_BUILD:-${dir_build_default}}
# dir_opt=${dir_build}/opt          # gmp/mpfr install prefix
dir_opt=${dir_build}/opt            # c-lime (-fPIC) install prefix
dir_log=${dir_build}/log
# mkdir -p ${dir_build} ${dir_opt} ${dir_log}
# mkdir -p ${dir_build} ${dir_log}
mkdir -p ${dir_build} ${dir_opt} ${dir_log}

# ---------------- environment ----------------
# oneapi + mpich are loaded by default on Aurora.
# c-lime: built from source with -fPIC (lime stage). The system spack c-lime ships only a non-PIC
# liblime.a, which cannot be linked into the shared cgpt.so ("relocation R_X86_64_32 ... recompile with -fPIC").
# lmod scripts reference unset variables, so relax `set -u` around module loads.
set +u
# module load c-lime autoconf automake
module load autoconf automake
module load frameworks            # python3.12 + numpy for cgpt
set -u
# export CLIME=${C_LIME_ROOT}
export CLIME=${dir_opt}/lime
export SYCL_PROGRAM_COMPILE_OPTIONS="-ze-opt-large-register-file"
# github access from login nodes goes through the ALCF proxy
export HTTP_PROXY=http://proxy.alcf.anl.gov:3128
export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
export http_proxy=${HTTP_PROXY}
export https_proxy=${HTTPS_PROXY}

# gmp/mpfr: use Aurora's system spack installs (headers + libs; no lmod module, so not in `module avail`)
AURORA_SPACK=/opt/aurora/26.181.0/spack/unified/1.1.1/install/linux-x86_64
GMP=${AURORA_SPACK}/gmp-6.3.0-a3afskm
MPFR=${AURORA_SPACK}/mpfr-4.2.1-2l5fivm
# GMP_VER=6.3.0
# MPFR_VER=4.2.1

run() {  # run <logname> <cmd...>; abort on failure
  local log=$1; shift
  echo "==== [$(date +%T)] ${log}: $*"
  ( time "$@" ) > ${dir_log}/${log}.log 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then echo "FAILED (${rc}): see ${dir_log}/${log}.log"; tail -20 ${dir_log}/${log}.log; exit $rc; fi
}

# ---------------- deps: gmp + mpfr (not needed: Aurora's spack has them, see GMP/MPFR above) ----------------
# build_deps() {
#   cd ${dir_opt}
#   [ -f gmp-${GMP_VER}.tar.xz ]   || run gmp_fetch  curl -sSLO https://ftp.gnu.org/gnu/gmp/gmp-${GMP_VER}.tar.xz
#   [ -d gmp-${GMP_VER} ]          || tar xf gmp-${GMP_VER}.tar.xz
#   cd gmp-${GMP_VER}
#   run gmp_configure ./configure --prefix=${dir_opt}/gmp CC=gcc
#   run gmp_make make -j ${NJ}
#   run gmp_install make install
#   cd ${dir_opt}
#   [ -f mpfr-${MPFR_VER}.tar.xz ] || run mpfr_fetch curl -sSLO https://ftp.gnu.org/gnu/mpfr/mpfr-${MPFR_VER}.tar.xz
#   [ -d mpfr-${MPFR_VER} ]        || tar xf mpfr-${MPFR_VER}.tar.xz
#   cd mpfr-${MPFR_VER}
#   run mpfr_configure ./configure --prefix=${dir_opt}/mpfr --with-gmp=${dir_opt}/gmp CC=gcc
#   run mpfr_make make -j ${NJ}
#   run mpfr_install make install
# }

# ---------------- c-lime (USQCD), static but -fPIC so it can go into cgpt.so ----------------
build_lime() {
  cd ${dir_opt}
  [ -d c-lime ] || run lime_clone git clone https://github.com/usqcd-software/c-lime
  cd c-lime
  run lime_autogen ./autogen.sh
  run lime_configure ./configure --prefix=${dir_opt}/lime CC=icx CFLAGS="-O2 -fPIC"
  run lime_make make -j ${NJ}
  run lime_install make install
}

# ---------------- Grid (lehner fork, feature/gpt) ----------------
build_grid() {
  cd ${dir_build}
  [ -d Grid ] || run grid_clone git clone -b feature/gpt https://github.com/lehner/Grid
  cd Grid
  # Incremental by default: once Grid/build is configured (config.status exists), skip configure and
  # only run make (rebuilds just what is missing/stale). FORCE_CONFIGURE=1 reruns configure, e.g. after
  # changing configure flags below. A change that alters Grid/Config.h means a full recompile; a
  # library path change (e.g. --with-lime) does not, since config.status keeps an unchanged Config.h.
  # Grid's bootstrap.sh (re-downloads Eigen, regenerates configure) runs only when needed.
  if [ "${FORCE_CONFIGURE:-0}" = "1" ] || [ ! -f ${dir_build}/Grid/build/config.status ]; then
  # run bootstrap ./bootstrap.sh
  if [ "${FORCE_BOOTSTRAP:-0}" = "1" ] || [ ! -f ${dir_build}/Grid/configure ]; then
    run bootstrap ./bootstrap.sh
  fi
  mkdir -p ${dir_build}/Grid/build; cd ${dir_build}/Grid/build
  # build dir is Grid/build so that cgpt's `make` finds it as ../Grid/build relative to gpt/
  # LDFLAGS: dropped -fsycl-device-lib=all from lehner's config-command (icpx 2026.1 rejects it;
  # device libs are linked by default).
  run configure ../configure --prefix=${dir_build}/install \
      --enable-simd=GPU \
      --enable-gen-simd-width=64 \
      --enable-comms=mpi-auto \
      --disable-gparity \
      --disable-fermion-reps \
      --enable-shm=nvlink \
      --enable-checksum-comms=no \
      --enable-log-views=yes \
      --enable-accelerator=sycl \
      --enable-accelerator-aware-mpi=no \
      --enable-unified=no \
      --with-lime=${CLIME} \
      --with-gmp=${GMP} \
      --with-mpfr=${MPFR} \
      MPICXX=mpicxx \
      CXX=icpx \
      LDFLAGS="-fiopenmp -fsycl -fsycl-device-code-split=per_kernel -lze_loader -L${MKLROOT}/lib -qmkl=parallel -lsycl -fsycl-max-parallel-link-jobs=16 -fno-sycl-rdc -lnuma -Wl,-rpath,${GMP}/lib -Wl,-rpath,${MPFR}/lib" \
      CXXFLAGS="-fPIC -fiopenmp -fsycl-unnamed-lambda -fsycl -Wno-tautological-compare -qmkl=parallel"
  # Nc=3 (Grid default); the Tuolumne build used --enable-Nc=4.
  else
    echo "==== Grid/build already configured: skipping bootstrap + configure (FORCE_CONFIGURE=1 to redo)"
  fi
  cd ${dir_build}/Grid/build
  run grid_make make -j ${NJ}
  run grid_install make install
}

# ---------------- GPT / cgpt ----------------
build_gpt() {
  # Build the gpt checkout this script lives in (NM's fork, branch master),
  # as gpt's own scripts/bootstrap/* do. Fall back to cloning the fork if run standalone.
  local root=${gpt_root}
  if [ -z "${root}" ]; then
    cd ${dir_build}
    # [ -d gpt ] || run gpt_clone git clone -b 2024-10 https://github.com/lehner/gpt
    [ -d gpt ] || run gpt_clone git clone -b master https://github.com/nobuyukimatsumoto68/gpt
    root=${dir_build}/gpt
  fi
  echo "==== building cgpt in ${root}"
  cd ${root}/lib/cgpt
  run cgpt_make ./make ${dir_build}/Grid/build ${NJ}
  echo "source ${root}/lib/cgpt/build/source.sh   # sets PYTHONPATH for gpt"
}

case ${stage} in
  # deps) build_deps ;;
  lime) build_lime ;;
  grid) build_grid ;;
  gpt)  build_gpt ;;
  # all)  build_deps; build_grid; build_gpt ;;
  # all)  build_grid; build_gpt ;;
  all)  build_lime; build_grid; build_gpt ;;
  *) echo "unknown stage ${stage}"; exit 1 ;;
esac
echo "==== [$(date +%T)] done: ${stage}"
