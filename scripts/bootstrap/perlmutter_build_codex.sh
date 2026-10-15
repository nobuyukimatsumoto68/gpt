#!/bin/bash
# Build lehner/Grid feature/gpt with CUDA sm_80 and NM's GPT cgpt on Perlmutter.
# Sources: Grid, P. Boyle et al., arXiv:1512.03487; GPT, C. Lehner et al.
# This script does not submit jobs. Run the test stage only inside an approved GPU allocation.
#
# Usage: bash perlmutter_build_codex.sh [lime|grid|gpt|test|all]
# Storage: set DIR_BUILD to a project-CFS directory, or use the default below.
# The expected layout is DIR_BUILD/{gpt,Grid,opt,install,log}.
set -euo pipefail

stage=${1:-all}
NJ=${NJ:-8}

gpt_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." >/dev/null 2>&1 && pwd)"
if [ ! -d "${gpt_root}/lib/cgpt" ]; then
    echo "Run this script from an NM gpt checkout containing lib/cgpt" >&2
    exit 1
fi

dir_build=${DIR_BUILD:-/global/cfs/cdirs/m5429/nmatsum/genesis}
dir_opt=${dir_build}/opt
dir_log=${dir_build}/log
grid_root=${dir_build}/Grid
grid_build=${grid_root}/build
grid_install=${dir_build}/install
mkdir -p "${dir_build}" "${dir_opt}" "${dir_log}"

# cpe-cuda must follow PrgEnv-gnu and cudatoolkit. This combination selects
# the tested CUDA 12.9 / GCC 12.3 pair on the current Perlmutter software stack.
set +u
module load PrgEnv-gnu
module load cudatoolkit/12.9
module load cpe-cuda/25.09
module load gcc-native/12.3
module load python/3.12-26.1.0
module load cray-fftw/3.3.10.11
set -u

export CC=${CC:-cc}
export CXX=${CXX:-nvcc}
export GMP=${GMP:-${CONDA_PREFIX}}
export MPFR=${MPFR:-${CONDA_PREFIX}}
if [ -z "${FFTW:-}" ]; then
    FFTW=${FFTW_DIR%/lib}
fi
export FFTW

run() {
    local log_name=$1
    shift
    echo "==== [$(date +%T)] ${log_name}: $*"
    (time "$@") >"${dir_log}/${log_name}.log" 2>&1
    local return_code=$?
    if [ "${return_code}" -ne 0 ]; then
        echo "FAILED (${return_code}): see ${dir_log}/${log_name}.log" >&2
        tail -20 "${dir_log}/${log_name}.log" >&2 || true
        exit "${return_code}"
    fi
}

require_file() {
    if [ ! -e "$1" ]; then
        echo "Missing required path: $1" >&2
        exit 1
    fi
}

check_dependencies() {
    require_file "${GMP}/include/gmp.h"
    require_file "${MPFR}/include/mpfr.h"
    require_file "${FFTW}/include/fftw3.h"
    command -v "${CXX}" >/dev/null
    command -v python3 >/dev/null
    command -v python3-config >/dev/null
    python3 -c 'import numpy' >/dev/null
}

build_lime() {
    module unload craype-accel-nvidia80 || true
    unset CRAY_ACCEL_TARGET
    cd "${dir_opt}"
    if [ ! -d c-lime ]; then
        run lime_clone git clone https://github.com/usqcd-software/c-lime c-lime
    fi
    cd c-lime
    if [ ! -f config.status ]; then
        run lime_autogen ./autogen.sh
        run lime_configure ./configure --prefix="${dir_opt}/lime" CC="${CC}" CFLAGS="-O2 -fPIC"
    fi
    run lime_make make -j "${NJ}"
    run lime_install make install
}

build_grid() {
    module load craype-accel-nvidia80
    export CRAY_ACCEL_TARGET=nvidia80
    cd "${dir_build}"
    if [ ! -d "${grid_root}" ]; then
        run grid_clone git clone -b feature/gpt https://github.com/lehner/Grid Grid
    fi
    cd "${grid_root}"
    if [ ! -f configure ]; then
        run grid_bootstrap ./bootstrap.sh
    fi
    mkdir -p "${grid_build}"
    cd "${grid_build}"
    if [ "${FORCE_CONFIGURE:-0}" = "1" ] || [ ! -f config.status ]; then
        run grid_configure ../configure \
            --prefix="${grid_install}" \
            --enable-comms=mpi \
            --enable-simd=GPU \
            --enable-gen-simd-width=64 \
            --enable-shm=nvlink \
            --with-gmp="${GMP}" \
            --with-mpfr="${MPFR}" \
            --with-fftw="${FFTW}" \
            --with-lime="${dir_opt}/lime" \
            --enable-accelerator=cuda \
            --disable-fermion-reps \
            --disable-unified \
            --disable-gparity \
            CXX="${CXX}" \
            LDFLAGS="-cudart shared -Xcompiler -fPIC -Wl,-rpath,${GMP}/lib -Wl,-rpath,${MPFR}/lib -Wl,-rpath,${dir_opt}/lime/lib" \
            CXXFLAGS="-ccbin CC -gencode arch=compute_80,code=sm_80 -std=c++17 -cudart shared -Xcompiler -fPIC"
    else
        echo "==== Grid/build already configured; use FORCE_CONFIGURE=1 to reconfigure"
    fi
    run grid_make make -j "${NJ}"
    run grid_install make install
}

build_gpt() {
    module load craype-accel-nvidia80
    export CRAY_ACCEL_TARGET=nvidia80
    check_dependencies
    cd "${gpt_root}/lib/cgpt"
    if [ -z "${CGPT_EXTRA_LDFLAGS:-}" ]; then
        export CGPT_EXTRA_LDFLAGS="-Wl,-rpath,${GMP}/lib -Wl,-rpath,${MPFR}/lib -Wl,-rpath,${dir_opt}/lime/lib"
    fi
    run cgpt_make ./make "${grid_build}" "${NJ}"
    require_file "${gpt_root}/lib/cgpt/build/cgpt.so"
    echo "source ${gpt_root}/lib/cgpt/build/source.sh"
}

run_gpu_tests() {
    if [ -z "${SLURM_JOB_ID:-}" ]; then
        echo "test stage requires an already-approved GPU allocation; no job was submitted" >&2
        exit 2
    fi
    if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
        echo "test stage requires CUDA_VISIBLE_DEVICES inside a GPU allocation" >&2
        exit 2
    fi
    source "${gpt_root}/lib/cgpt/build/source.sh"
    cd "${gpt_root}/tests/qcd"
    run gauge_test python3 gauge.py
    cd "${gpt_root}/applications/hmc"
    run pure_gauge_test python3 pure_gauge.py --ntherm 1 --n 1 --nwrite 1
}

case "${stage}" in
    lime) build_lime ;;
    grid) build_grid ;;
    gpt) build_gpt ;;
    test) run_gpu_tests ;;
    all) build_lime; build_grid; build_gpt ;;
    *) echo "unknown stage ${stage}; use lime, grid, gpt, test, or all" >&2; exit 1 ;;
esac
echo "==== [$(date +%T)] done: ${stage}"
