#!/bin/bash

set -e

cd "$(dirname "$0")"

echo "configuration ==========================================================="
JOBDATE="$(date '+%Y-%m-%d')"
echo "JOBDATE ${JOBDATE}"

JOBNAME="$(basename -s .sh "$0")"
echo "JOBNAME ${JOBNAME}"

JOBPROJECT="$(basename -s .git "$(git remote get-url origin)")"
echo "JOBPROJECT ${JOBPROJECT}"

NOTEBOOK_NAME="2026-07-30-exploratory-edge-sparsity"
echo "NOTEBOOK_NAME ${NOTEBOOK_NAME}"
NOTEBOOK_PATH="bindle/${NOTEBOOK_NAME}.py"
echo "NOTEBOOK_PATH ${NOTEBOOK_PATH}"

# A coarse L1 x L2 regularization-strength grid, independent of each other
# (NOT constrained to l1_scale + l2_scale = 1 like every other sweep in this
# project) -- each axis spans roughly 3 orders of magnitude in half-decade
# steps, to map out how L1 and L2 magnitude separately (and jointly)
# affect where the sparsity-driven "phenotype composition collapse" seen
# in bindle/2026-07-31-exploratory-edge-sparsity-analysis.py (and in
# ad-hoc exploration of osf.io/6vrxc, osf.io/ykp74 -- similar independent
# L1/L2 grids, not yet backed by a committed slurm script) sits. Only 5
# n_zero_edges points are sampled (not the usual 99-101-point sweep),
# deliberately spanning that collapse: 0 (dense baseline), 100 (just
# before/at the collapse onset observed around n~=95-110), and
# 125/150/175 (progressively past it).
#
# Fixed (not swept) for this script: n_classes=3 (this project's original
# training set {S1, S2, S3}) and doubled_class left at the notebook's
# default (uniform presentation, --doubled-class is not passed at all --
# see slurm/2026-07-30/2026-07-30-exploratory-edge-sparsity.sh for why
# omitting the flag entirely, rather than passing --doubled-class -1, is
# the safe way to get the uniform default: a LITERAL "-1" argument value
# is misparsed as a new flag by marimo's CLI arg parsing unless written
# --doubled-class=-1 with an equals sign).
# Swept across:
#   - n_zero_edges in {0, 100, 125, 150, 175}                          (5)
#     density = 1 - n_zero_edges/256 for each (exact in binary floating
#     point, since 256 = 2^8 -- see the base sweep script for the same
#     reasoning applied to a denser range).
#   - l1_scale in {0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0}
#                                                                       (9)
#   - l2_scale in {0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01}
#                                                                       (7)
# crossed with 3 replicate seeds per (n_zero_edges, l1_scale, l2_scale)
# combination (lighter than this project's usual 10 -- this is a coarse
# exploratory grid, not a production sweep):
#   - seed in {1, 2, 3}                                                (3)
# total = 5 * 9 * 7 * 3 = 945 replicates.
#
# Generations vs. epochs: same convention as every other sweep in this
# project -- NUM_EPOCH = round(500e6 / 3600) = 138889, giving 3600 *
# 138889 = 500,000,400 generations per replicate (~100-120 minutes
# single-threaded, comfortably inside the 4-hour time limit below).
#
# The cluster caps a job array at 1000 queued tasks. 945 replicates fits
# under that cap with NO packing needed at all -- CHUNK=1, one replicate
# per array task, --cpus-per-task=1 exactly matching the single
# concurrently-launched replicate in every task (945 array tasks total).
L1_SCALES=(0.0001 0.0003 0.001 0.003 0.01 0.03 0.1 0.3 1.0)
L2_SCALES=(0.00001 0.00003 0.0001 0.0003 0.001 0.003 0.01)
N_CLASSES=3
DENSITIES=(1.0 0.609375 0.51171875 0.4140625 0.31640625)  # n_zero_edges = 0, 100, 125, 150, 175
N_ZERO_LABELS=(0 100 125 150 175)
SEEDS=(1 2 3)
N_NZERO=${#DENSITIES[@]}
N_L1=${#L1_SCALES[@]}
N_L2=${#L2_SCALES[@]}
N_SEED=${#SEEDS[@]}
N_TASKS=$((N_NZERO * N_L1 * N_L2 * N_SEED))
CHUNK=1
N_ARRAY_TASKS=$(((N_TASKS + CHUNK - 1) / CHUNK))
NUM_EPOCH=138889
echo "N_CLASSES=${N_CLASSES} (fixed, not swept; doubled_class left at notebook default, uniform)"
echo "N_NZERO=${N_NZERO} N_ZERO_LABELS=${N_ZERO_LABELS[*]} DENSITIES=${DENSITIES[*]}"
echo "N_L1=${N_L1} L1_SCALES=${L1_SCALES[*]}"
echo "N_L2=${N_L2} L2_SCALES=${L2_SCALES[*]}"
echo "N_SEED=${N_SEED} SEEDS=${SEEDS[*]}"
echo "N_TASKS=${N_TASKS} CHUNK=${CHUNK} N_ARRAY_TASKS=${N_ARRAY_TASKS}"
echo "NUM_EPOCH=${NUM_EPOCH} (total generations per replicate = 3600 * NUM_EPOCH = $((3600 * NUM_EPOCH)))"

SOURCE_REVISION="$(git rev-parse HEAD)"
echo "SOURCE_REVISION ${SOURCE_REVISION}"
SOURCE_REMOTE_URL="$(git config --get remote.origin.url)"
echo "SOURCE_REMOTE_URL ${SOURCE_REMOTE_URL}"

echo "initialization telemetry ==============================================="
echo "date $(date)"
echo "hostname $(hostname)"
echo "PWD ${PWD}"
echo "SLURM_JOB_ID ${SLURM_JOB_ID:-nojid}"
echo "SLURM_ARRAY_TASK_ID ${SLURM_ARRAY_TASK_ID:-notid}"
module purge || :
module load Python/3.10.8 || :
echo "python3.10 $(which python3.10)"
echo "python3.10 --version $(python3.10 --version)"

echo "setup HOME dirs ========================================================"
mkdir -p "${HOME}/joblatest"
mkdir -p "${HOME}/joblog"
mkdir -p "${HOME}/jobscript"
if ! [ -e "${HOME}/scratch" ]; then
    if [ -e "/mnt/scratch/${USER}" ]; then
        ln -s "/mnt/scratch/${USER}" "${HOME}/scratch" || :
    else
        mkdir -p "${HOME}/scratch" || :
    fi
fi

echo "setup BATCHDIR =========================================================="
BATCHDIR="${HOME}/scratch/${JOBPROJECT}/${JOBNAME}/${JOBDATE}"
if [ -e "${BATCHDIR}" ]; then
    echo "BATCHDIR ${BATCHDIR} exists, clearing it"
fi
rm -rf "${BATCHDIR}"
mkdir -p "${BATCHDIR}"
echo "BATCHDIR ${BATCHDIR}"

echo "symlinking latest"
LATESTDIR="${HOME}/scratch/${JOBPROJECT}/${JOBNAME}/latest"
echo "${BATCHDIR} > ${LATESTDIR}"
ln -sfn "${BATCHDIR}" "${LATESTDIR}"

BATCHDIR_JOBLOG="${BATCHDIR}/joblog"
echo "BATCHDIR_JOBLOG ${BATCHDIR_JOBLOG}"
mkdir -p "${BATCHDIR_JOBLOG}"

BATCHDIR_JOBRESULT="${BATCHDIR}/jobresult"
echo "BATCHDIR_JOBRESULT ${BATCHDIR_JOBRESULT}"
mkdir -p "${BATCHDIR_JOBRESULT}"

BATCHDIR_JOBSCRIPT="${BATCHDIR}/jobscript"
echo "BATCHDIR_JOBSCRIPT ${BATCHDIR_JOBSCRIPT}"
mkdir -p "${BATCHDIR_JOBSCRIPT}"

BATCHDIR_JOBSOURCE="${BATCHDIR}/_jobsource"
echo "BATCHDIR_JOBSOURCE ${BATCHDIR_JOBSOURCE}"
if [[ $* == *--dirty* ]]; then
    cp -r "$(git rev-parse --show-toplevel)" "${BATCHDIR_JOBSOURCE}"
else
    mkdir -p "${BATCHDIR_JOBSOURCE}"
    for attempt in {1..5}; do
        rm -rf "${BATCHDIR_JOBSOURCE}/.git"
        git -C "${BATCHDIR_JOBSOURCE}" init \
        && git -C "${BATCHDIR_JOBSOURCE}" remote add origin "${SOURCE_REMOTE_URL}" \
        && git -C "${BATCHDIR_JOBSOURCE}" fetch origin "${SOURCE_REVISION}" --depth=1 \
        && git -C "${BATCHDIR_JOBSOURCE}" reset --hard FETCH_HEAD \
        && break || echo "failed to clone, retrying..."
        if [ $attempt -eq 5 ]; then
            echo "failed to clone, failing"
            exit 1
        fi
        sleep 5
    done
fi

BATCHDIR_ENV="${BATCHDIR}/_jobenv"
python3.10 -m venv --system-site-packages "${BATCHDIR_ENV}"
source "${BATCHDIR_ENV}/bin/activate"
echo "python3.10 $(which python3.10)"
echo "python3.10 --version $(python3.10 --version)"
for attempt in {1..5}; do
    python3.10 -m pip install --upgrade pip 'setuptools<75' wheel || :
    python3.10 -m pip install --upgrade uv \
    && python3.10 -m uv pip install joinem==0.11.1 \
    && python3.10 -m uv pip install \
        -r "${BATCHDIR_JOBSOURCE}/requirements.txt" \
    && break || echo "pip install attempt ${attempt} failed"
    if [ ${attempt} -eq 5 ]; then
        echo "pip install failed"
        exit 1
    fi
done

echo "setup dependencies ========================================== ${SECONDS}"
source "${BATCHDIR_ENV}/bin/activate"
python3.10 -m uv pip freeze

echo "sbatch preamble ========================================================="
JOB_PREAMBLE=$(cat << EOF
set -e
shopt -s globstar

# adapted from https://unix.stackexchange.com/a/504829
handlefail() {
    echo ">>>error<<<" || :
    awk 'NR>L-4 && NR<L+4 { printf "%-5d%3s%s\n",NR,(NR==L?">>>":""),\$0 }' L=\$1 \$0 || :
    ln -sfn "\${JOBSCRIPT}" "\${HOME}/joblatest/jobscript.failed" || :
    ln -sfn "\${JOBLOG}" "\${HOME}/joblatest/joblog.failed" || :
    $(which scontrol || which echo) requeuehold "${SLURM_JOBID:-nojid}"
}
trap 'handlefail $LINENO' ERR

echo "initialization telemetry ------------------------------------ \${SECONDS}"
echo "SOURCE_REVISION ${SOURCE_REVISION}"
echo "BATCHDIR ${BATCHDIR}"

echo "cc SLURM script --------------------------------------------- \${SECONDS}"
JOBSCRIPT="\${HOME}/jobscript/\${SLURM_JOB_ID:-nojid}"
echo "JOBSCRIPT \${JOBSCRIPT}"
cp "\${0}" "\${JOBSCRIPT}"
chmod +x "\${JOBSCRIPT}"
cp "\${JOBSCRIPT}" "${BATCHDIR_JOBSCRIPT}/\${SLURM_JOB_ID:-nojid}"
ln -sfn "\${JOBSCRIPT}" "${HOME}/joblatest/jobscript.launched"

echo "cc job log -------------------------------------------------- \${SECONDS}"
JOBLOG="\${HOME}/joblog/\${SLURM_JOB_ID:-nojid}"
echo "JOBLOG \${JOBLOG}"
touch "\${JOBLOG}"
ln -sfn "\${JOBLOG}" "${BATCHDIR_JOBLOG}/\${SLURM_JOB_ID:-nojid}"
ln -sfn "\${JOBLOG}" "\${HOME}/joblatest/joblog.launched"

echo "setup JOBDIR ------------------------------------------------ \${SECONDS}"
JOBDIR="${BATCHDIR}/__\${SLURM_ARRAY_TASK_ID:-\${SLURM_JOB_ID:-\${RANDOM}}}"
echo "JOBDIR \${JOBDIR}"
if [ -e "\${JOBDIR}" ]; then
    echo "JOBDIR \${JOBDIR} exists, clearing it"
fi
rm -rf "\${JOBDIR}"
mkdir -p "\${JOBDIR}"
cd "\${JOBDIR}"
echo "PWD \${PWD}"

echo "job telemetry ----------------------------------------------- \${SECONDS}"
echo "source SLURM_JOB_ID ${SLURM_JOB_ID:-nojid}"
echo "current SLURM_JOB_ID \${SLURM_JOB_ID:-nojid}"
echo "SLURM_ARRAY_TASK_ID \${SLURM_ARRAY_TASK_ID:-notid}"
echo "hostname \$(hostname)"
echo "date \$(date)"

echo "module setup ------------------------------------------------ \${SECONDS}"
module purge || :
module load Python/3.10.8 || :
echo "python3.10 \$(which python3.10)"
echo "python3.10 --version \$(python3.10 --version)"

echo "setup dependencies- ----------------------------------------- \${SECONDS}"
source "${BATCHDIR_ENV}/bin/activate"
python3.10 -m uv pip freeze

EOF
)

echo "create sbatch file: work ==============================================="

SBATCH_FILE="$(mktemp)"
echo "SBATCH_FILE ${SBATCH_FILE}"

###############################################################################
# WORK ---------------------------------------------------------------------- #
###############################################################################
cat > "${SBATCH_FILE}" << EOF
#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CHUNK}
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output="/mnt/home/%u/joblog/%j"
#SBATCH --mail-user=mawni4ah2o@pomail.net
#SBATCH --mail-type=FAIL,TIME_LIMIT,ARRAY_TASKS
#SBATCH --account=ecode
#SBATCH --requeue
#SBATCH --array=0-$((N_ARRAY_TASKS - 1))

${JOB_PREAMBLE}

echo "lscpu ------------------------------------------------------- \${SECONDS}"
lscpu || :

echo "cpuinfo ----------------------------------------------------- \${SECONDS}"
cat /proc/cpuinfo || :

echo "marimo notebook source -------------------------------------- \${SECONDS}"
echo "notebook source: ${BATCHDIR_JOBSOURCE}/${NOTEBOOK_PATH}"
cat "${BATCHDIR_JOBSOURCE}/${NOTEBOOK_PATH}" || :

echo "task assignment --------------------------------------------- \${SECONDS}"
DENSITIES=(${DENSITIES[*]})
N_ZERO_LABELS=(${N_ZERO_LABELS[*]})
L1_SCALES=(${L1_SCALES[*]})
L2_SCALES=(${L2_SCALES[*]})
SEEDS=(${SEEDS[*]})
TASK_ID=\${SLURM_ARRAY_TASK_ID:-0}
echo "TASK_ID=\${TASK_ID} CHUNK=${CHUNK}"
echo "owns global replicate indices \$((TASK_ID * ${CHUNK})) .. \$((TASK_ID * ${CHUNK} + ${CHUNK} - 1))"

# Each of the CHUNK replicates runs as its own single-threaded process so
# all CHUNK share the array task's CPUs (--cpus-per-task=${CHUNK}) without
# oversubscribing: pin the numeric libraries to one thread each.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Run one (n_zero_edges, l1_scale, l2_scale, seed) trial on CPU --
# n_classes=${N_CLASSES} is fixed for every replicate in this script, and
# doubled_class is left at the notebook's default (uniform presentation --
# see the comment above --doubled-class isn't passed at all). Each
# replicate runs in its own working dir \${JOBDIR}/r<gid> and the notebook
# writes one timeseries csv plus G/B snapshot npz stores to that dir's
# dd_trial_outputs/, self-describing via keyname (every run option as a
# key=value segment) with a uuid replicate identifier stamped on every
# timeseries row.
run_replicate() {
    local gid="\$1"
    local seed_idx=\$((gid % ${N_SEED}))
    local rem1=\$((gid / ${N_SEED}))
    local l2_idx=\$((rem1 % ${N_L2}))
    local rem2=\$((rem1 / ${N_L2}))
    local l1_idx=\$((rem2 % ${N_L1}))
    local n_idx=\$((rem2 / ${N_L1}))

    local density="\${DENSITIES[\${n_idx}]}"
    local nzerolabel="\${N_ZERO_LABELS[\${n_idx}]}"
    local l1="\${L1_SCALES[\${l1_idx}]}"
    local l2="\${L2_SCALES[\${l2_idx}]}"
    local seed="\${SEEDS[\${seed_idx}]}"
    local repdir="\${JOBDIR}/r\${gid}"
    mkdir -p "\${repdir}"
    cd "\${repdir}"

    # Export from a PRIVATE per-replicate copy of the notebook. Every
    # replicate --- within this task and across all array tasks sharing
    # the single _jobsource clone on the network filesystem --- would
    # otherwise run marimo against the same notebook file. Under that
    # concurrency marimo clobbers the shared source to an empty default
    # stub (marimo.App() with one empty cell), after which every later
    # export yields a blank notebook with no outdata. A private copy
    # removes the shared-file race. (This notebook is fully self-contained
    # -- no 'from pylib import ...' -- so no sibling pylib symlink is
    # needed, unlike notebooks that pull in project modules.)
    local nbdir="\${repdir}/_nb"
    mkdir -p "\${nbdir}"
    cp "${BATCHDIR_JOBSOURCE}/${NOTEBOOK_PATH}" "\${nbdir}/${NOTEBOOK_NAME}.py"
    echo "  [gid=\${gid}] n_zero_edges=\${nzerolabel} density=\${density} l1=\${l1} l2=\${l2} seed=\${seed} repdir=\${repdir}"
    python3.10 -m marimo export ipynb \
        --include-outputs --sort topological -f \
        "\${nbdir}/${NOTEBOOK_NAME}.py" \
        -o "\${repdir}/${NOTEBOOK_NAME}.ipynb" \
        -- \
        --seed "\${seed}" \
        --density "\${density}" \
        --l1-scale "\${l1}" \
        --l2-scale "\${l2}" \
        --n-classes ${N_CLASSES} \
        --num-epoch ${NUM_EPOCH}

    # Fail loudly on a blank/failed export. marimo can exit 0 while
    # producing a notebook whose cells never executed --- and the run
    # cell is what writes dd_trial_outputs/ --- so a "successful" export
    # with no outputs would otherwise sail through to >>>complete<<<.
    # Require both a non-trivial exported notebook and the run cell's
    # csv + npz outputs, else fail the replicate (return 1, caught by
    # the wait loop below). Timeseries output is CSV, not parquet ---
    # written progressively during the run so a SLURM timeout still
    # leaves partial data on disk --- and gets converted to parquet only
    # at collation time (joinem infers CSV in / parquet out from the
    # file extensions), rather than every replicate separately producing
    # its own parquet file.
    local nb_out="\${repdir}/${NOTEBOOK_NAME}.ipynb"
    local outdata_dir="\${repdir}/dd_trial_outputs"
    local nb_bytes
    nb_bytes=\$(wc -c < "\${nb_out}" 2>/dev/null || echo 0)
    if [ "\${nb_bytes}" -lt 10000 ]; then
        echo "ERROR [gid=\${gid}]: exported notebook \${nb_out} missing or trivial (\${nb_bytes} bytes)"
        return 1
    fi
    local n_csv
    n_csv=\$(find "\${outdata_dir}" -maxdepth 1 -name '*ext=.csv' 2>/dev/null | wc -l)
    if [ "\${n_csv}" -lt 1 ]; then
        echo "ERROR [gid=\${gid}]: no timeseries csv under \${outdata_dir}"
        return 1
    fi
    local n_npz
    n_npz=\$(find "\${outdata_dir}" -maxdepth 1 -name '*ext=.npz' 2>/dev/null | wc -l)
    if [ "\${n_npz}" -lt 2 ]; then
        echo "ERROR [gid=\${gid}]: expected 2 (G, B) snapshot npz files under \${outdata_dir}, found \${n_npz}"
        return 1
    fi
    echo "  [gid=\${gid}] export OK: \${nb_bytes} byte notebook, \${n_csv} csv + \${n_npz} npz in dd_trial_outputs"
}

echo "do work (CHUNK=${CHUNK} replicates concurrently) ------------ \${SECONDS}"
declare -A REP_PID
for j in \$(seq 0 \$((${CHUNK} - 1))); do
    GID=\$((TASK_ID * ${CHUNK} + j))
    if [ "\${GID}" -ge ${N_TASKS} ]; then
        echo "  skipping global index \${GID} (>= ${N_TASKS}, partial final chunk)"
        continue
    fi
    run_replicate "\${GID}" &
    REP_PID[\${GID}]=\$!
done

echo "launched \${#REP_PID[@]} replicate(s); waiting -------------- \${SECONDS}"
WORK_FAIL=0
for gid in "\${!REP_PID[@]}"; do
    if wait "\${REP_PID[\${gid}]}"; then
        echo "  replicate gid=\${gid} ok"
    else
        echo "  replicate gid=\${gid} FAILED (pid \${REP_PID[\${gid}]})"
        WORK_FAIL=1
    fi
done
if [ "\${WORK_FAIL}" -ne 0 ]; then
    echo "one or more replicates failed; failing array task"
    exit 1
fi

echo "finalization telemetry -------------------------------------- \${SECONDS}"
ls -lR "\${JOBDIR}" | head -200
du -sh "\${JOBDIR}"
ln -sfn "\${JOBSCRIPT}" "${HOME}/joblatest/jobscript.finished"
ln -sfn "\${JOBLOG}" "${HOME}/joblatest/joblog.finished"
echo "SECONDS \${SECONDS}"
echo '>>>complete<<<'

EOF
###############################################################################
# --------------------------------------------------------------------------- #
###############################################################################


echo "submit sbatch file ====================================================="
$(which sbatch && echo --job-name="${JOBNAME}" || which bash) "${SBATCH_FILE}"

echo "create sbatch file: collate ============================================"

SBATCH_FILE="$(mktemp)"
echo "SBATCH_FILE ${SBATCH_FILE}"

###############################################################################
# COLLATE ------------------------------------------------------------------- #
###############################################################################
cat > "${SBATCH_FILE}" << EOF
#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output="/mnt/home/%u/joblog/%j"
#SBATCH --mail-user=mawni4ah2o@pomail.net
#SBATCH --mail-type=ALL
#SBATCH --account=ecode
#SBATCH --requeue

${JOB_PREAMBLE}

echo "BATCHDIR ${BATCHDIR}"
ls -l "${BATCHDIR}"

echo "finalize ---------------------------------------------------- \${SECONDS}"
echo "   - archive job dir"
pushd "${BATCHDIR}/.."
    tar czf \
    "${BATCHDIR_JOBRESULT}/a=jobarchive+date=${JOBDATE}+job=${JOBNAME}+ext=.tar.gz" \
    "\$(basename "${BATCHDIR}")"/__*
popd

echo "   - join per-replicate timeseries csvs across all conditions, as parquet"
# Each replicate writes one self-describing timeseries csv (density,
# nzeroedges, nclasses, seed, l1scale, l2scale, numepoch, replicate
# columns stamped by the notebook's run cell) under r<gid>/dd_trial_outputs/,
# so a straight concatenation yields a collated frame spanning the whole
# sweep. Per-replicate output is CSV (written progressively during the run,
# so a SLURM timeout still leaves partial data on disk) rather than
# parquet; joinem infers CSV input / parquet output from the file
# extensions below, so the conversion to parquet happens once here rather
# than once per replicate. The G/B snapshot npz stores are NOT tabular
# (they're keyed arrays), so they aren't joined here -- they ride along in
# the jobarchive tarball above instead.
out_path="${BATCHDIR_JOBRESULT}/a=trace+date=${JOBDATE}+job=${JOBNAME}+ext=.pqt"
ls -1 "${BATCHDIR}"/__*/**/dd_trial_outputs/*ext=.csv 2>/dev/null \
    | tee /dev/stderr \
    | python3.10 -m joinem --progress "\${out_path}" \
    || echo "no timeseries files to join"
ls -l "${BATCHDIR_JOBRESULT}"
du -h "${BATCHDIR_JOBRESULT}"

echo "   - archive joblog"
pushd "${BATCHDIR}"
    tar czf \
    "${BATCHDIR_JOBRESULT}/a=joblog+date=${JOBDATE}+job=${JOBNAME}+ext=.tar.gz" \
    -h "\$(basename "${BATCHDIR_JOBLOG}")"
popd

echo "   - archive jobscript"
pushd "${BATCHDIR}"
    tar czf \
    "\$(basename "${BATCHDIR_JOBRESULT}")/a=jobscript+date=${JOBDATE}+job=${JOBNAME}+ext=.tar.gz" \
    -h "\$(basename ${BATCHDIR_JOBSCRIPT})"
popd

ls -l "${BATCHDIR}"

echo "cleanup ----------------------------------------------------- \${SECONDS}"
cd "${BATCHDIR}"
for f in _*; do
    echo "tar and rm \$f"
    tar cf "\${f}.tar" -h "\${f}"
    rm -rf "\${f}"
done
cd
ls -l "${BATCHDIR}"

echo "finalization telemetry -------------------------------------- \${SECONDS}"
ln -sfn "\${JOBSCRIPT}" "\${HOME}/joblatest/jobscript.completed"
ln -sfn "\${JOBLOG}" "\${HOME}/joblatest/joblog.completed"
ln -sfn "${BATCHDIR_JOBRESULT}" "\${HOME}/joblatest/jobresult.completed"
echo "SECONDS \${SECONDS}"
echo '>>>complete<<<'

EOF
###############################################################################
# --------------------------------------------------------------------------- #
###############################################################################

echo "submit sbatch file ====================================================="
$(which sbatch && echo --job-name="${JOBNAME}" --dependency=singleton || which bash) "${SBATCH_FILE}"

echo "finalization telemetry ================================================="
echo "BATCHDIR ${BATCHDIR}"
echo "SECONDS ${SECONDS}"
echo '>>>complete<<<'
