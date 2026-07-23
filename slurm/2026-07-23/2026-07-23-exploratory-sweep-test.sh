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

NOTEBOOK_NAME="2026-07-23-exploratory"
echo "NOTEBOOK_NAME ${NOTEBOOK_NAME}"
NOTEBOOK_PATH="bindle/${NOTEBOOK_NAME}.py"
echo "NOTEBOOK_PATH ${NOTEBOOK_PATH}"

# TEST VARIANT of slurm/2026-07-23/2026-07-23-exploratory-sweep.sh: IDENTICAL
# sweep definitions, index decomposition, and CHUNK=3 packing as the full
# production script -- the only difference is the #SBATCH --array line
# below, which submits 10 explicit array TASK indices (ARRAY_INDICES)
# instead of the full 0-983 range.
#
# Production already packs each array task's CHUNK=3 concurrent replicates
# as the SAME (blip_freq, l1/l2 mix, zero_init, v, seed) condition run
# under all 3 schedule_mode values side by side (schedule_idx is the
# fastest-varying index, exactly matching CHUNK) -- there is no way for a
# single array task to run fewer than all 3 schedule_mode values, since
# CHUNK=3 always spans one full none/local/global triple. So every task
# index below independently (not just once, collectively) exercises all
# 3 modes -- this test doubles as a check that concurrently running all 3
# modes on the 3 CPUs of one job doesn't clobber each other's output
# files (each replicate writes to its own ${JOBDIR}/r<gid>/dd_trial_outputs/,
# keyed by its own global index, so same-job concurrency is exactly as
# isolated as cross-job concurrency).
#
# The 10 selected task indices were picked (via the SAME decomposition
# used in production) to land on v in {0, 4, 8, 12, 16, 20}, one CHUNK=3
# triple (i.e. one replicate under each of none/local/global) each;
# blip_freq/l1-l2-mix/seed are whatever the decomposition naturally
# assigns at that task (arbitrary, per request -- not swept here). 10
# tasks * 3 replicates = 30 replicates total:
#   TASK_ID= 0 -> gid  0.. 2 -> v=0  zero_init=True   (blip=0.66 l1/l2=1.0/0.0 seed=1)
#   TASK_ID= 1 -> gid  3.. 5 -> v=0  zero_init=False  (blip=0.66 l1/l2=1.0/0.0 seed=1)
#   TASK_ID=26 -> gid 78..80 -> v=4  zero_init=True   (blip=0.66 l1/l2=1.0/0.0 seed=1)
#   TASK_ID=30 -> gid 90..92 -> v=8  zero_init=True   (blip=0.66 l1/l2=1.0/0.0 seed=1)
#   TASK_ID=31 -> gid 93..95 -> v=8  zero_init=False  (blip=0.66 l1/l2=1.0/0.0 seed=1)
#   TASK_ID=34 -> gid102..104-> v=12 zero_init=True   (blip=0.66 l1/l2=1.0/0.0 seed=1)
#   TASK_ID=38 -> gid114..116-> v=16 zero_init=True   (blip=0.66 l1/l2=1.0/0.0 seed=1)
#   TASK_ID=39 -> gid117..119-> v=16 zero_init=False  (blip=0.66 l1/l2=1.0/0.0 seed=1)
#   TASK_ID=42 -> gid126..128-> v=20 zero_init=True   (blip=0.66 l1/l2=1.0/0.0 seed=1)
#   TASK_ID=43 -> gid129..131-> v=20 zero_init=False  (blip=0.66 l1/l2=1.0/0.0 seed=1)
# each task's 3-gid range covers schedule_mode = none, local, global (in
# that order). v=4 and v=12 (zero_init=True only) are the two added task
# indices, broadening coverage beyond the original v in {0, 8, 16, 20}.
#
# Sweep: the single-trial elastic-net GRN notebook (v1..v20 double-descent
# model) across:
#   - blip_freq     in {0.66, 0.63, 0.6, 0.5}                       (4)
#   - (l1_scale, l2_scale) in {(1.0, 0.0), (0.995, 0.005), (0.9933, 0.0067)}
#     i.e. pure-L1, the notebook's default mix, and a near-pure-L1 mix (3)
#   - zero_init     in {True, False}                                (2)
#   - schedule_mode in {none, local, global}                        (3)
#     controls how ties are broken in the environment-presentation
#     schedule (the notebook's --schedule-mode flag).
# crossed with an UNEVEN v/seed split (v=0 gets fewer replicates than the
# rest, so this isn't one uniform Cartesian product):
#   - v = 0                      -> 1 replicate  (seed 1 only)       (1 v x 1 seed)
#   - v in {2, 4, ..., 20} (even, excluding 0) -> 4 replicates each
#     (seeds 1..4)                                                   (10 v x 4 seed)
# total = 4 * 3 * 2 * 3 * (1*1 + 10*4) = 4 * 3 * 2 * 3 * 41 = 2952 replicates.
#
# Generations vs. epochs: the notebook's SSWM loop runs
# TOTAL_BLOCKS (fixed at 3600 inside the notebook, not CLI-configurable)
# blocks of K generations each, where K is set by the "--num-epoch" CLI
# flag (the notebook's internal name for this per-block generation count --
# distinct from the "epoch" column in its output parquet, which is
# actually the snapshot index, 0..48). To hit a ~500,000,000-generation
# budget: NUM_EPOCH = round(500e6 / 3600) = 138889, giving an actual total
# of 3600 * 138889 = 500,000,400 generations per replicate (400 over the
# round-number target -- 3600 does not divide 500e6 evenly).
#
# The cluster caps a job array at 1000 queued tasks, so we pack CHUNK=3
# replicates into each array task and run those 3 *concurrently* (one CPU
# each, see --cpus-per-task below) rather than sequentially. schedule_mode
# is deliberately the FASTEST-varying (innermost) dimension in the index
# decomposition below, exactly matching CHUNK=3, so each array task's 3
# concurrent replicates are the SAME (blip_freq, l1/l2 mix, zero_init, v,
# seed) condition run under all 3 schedule_mode values side by side --
# this divides 2952 replicates evenly into 2952 / 3 = 984 array tasks
# (only 8 of which are actually submitted here, see ARRAY_INDICES).
#
# Global replicate index r in [0, N_TASKS) is split into two contiguous
# blocks rather than one uniform Cartesian product, since v=0 and the
# rest of the v values don't share the same seed count. Both blocks
# decompose fastest-varying first, starting with schedule_idx so it
# aligns with CHUNK:
#   - r < N_TASKS_V0: the v=0 block (single seed).
#     schedule_idx = r % N_SCHEDULE;
#     zero_idx = (r / N_SCHEDULE) % N_ZERO;
#     mix_idx = (r / N_SCHEDULE / N_ZERO) % N_MIX;
#     blip_idx = r / N_SCHEDULE / N_ZERO / N_MIX.
#   - r >= N_TASKS_V0: the "rest" block (v in {2,4,...,20}, 4 seeds
#     each), re-based to r' = r - N_TASKS_V0.
#     schedule_idx = r' % N_SCHEDULE;
#     zero_idx = (r' / N_SCHEDULE) % N_ZERO;
#     v_idx = (r' / N_SCHEDULE / N_ZERO) % N_V_REST;
#     mix_idx = (r' / N_SCHEDULE / N_ZERO / N_V_REST) % N_MIX;
#     seed_idx = (r' / N_SCHEDULE / N_ZERO / N_V_REST / N_MIX) % N_REST_SEED;
#     blip_idx = r' / N_SCHEDULE / N_ZERO / N_V_REST / N_MIX / N_REST_SEED.
# N_TASKS_V0 (72) is itself a multiple of CHUNK=3, so no CHUNK-triple
# straddles the v0/rest block boundary. Array task t owns the CHUNK
# consecutive indices r = t * CHUNK + j for j in [0, CHUNK) (each
# launched as a background job).
#
# Benchmarked the notebook's core SSWM loop single-threaded (non-cluster
# hardware) at ~83,000 generations/sec, so one 500M-generation replicate
# takes ~100 minutes -- comfortably inside the 4-hour job time limit below
# even allowing for slower cluster CPUs.
BLIP_FREQS=(0.66 0.63 0.6 0.5)
L1_SCALES=(1.0 0.995 0.9933)
L2_SCALES=(0.0 0.005 0.0067)
ZERO_INITS=(True False)
SCHEDULE_MODES=(none local global)
V0_SEEDS=(1)
REST_SEEDS=(1 2 3 4)
V_REST=(2 4 6 8 10 12 14 16 18 20)
N_BLIP=${#BLIP_FREQS[@]}
N_MIX=${#L1_SCALES[@]}
N_ZERO=${#ZERO_INITS[@]}
N_SCHEDULE=${#SCHEDULE_MODES[@]}
N_V0_SEED=${#V0_SEEDS[@]}
N_REST_SEED=${#REST_SEEDS[@]}
N_V_REST=${#V_REST[@]}
N_TASKS_V0=$((N_BLIP * N_MIX * N_ZERO * N_SCHEDULE * N_V0_SEED))
N_TASKS_REST=$((N_BLIP * N_MIX * N_ZERO * N_SCHEDULE * N_REST_SEED * N_V_REST))
N_TASKS=$((N_TASKS_V0 + N_TASKS_REST))
CHUNK=3
N_ARRAY_TASKS=$(((N_TASKS + CHUNK - 1) / CHUNK))
NUM_EPOCH=138889
# explicit array TASK indices to submit (see banner above for what each
# decodes to -- each covers a CHUNK=3 triple, i.e. all 3 schedule_mode
# values, for one (blip_freq, mix, zero_init, v, seed) condition)
ARRAY_INDICES="0,1,26,30,31,34,38,39,42,43"
echo "N_BLIP=${N_BLIP} BLIP_FREQS=${BLIP_FREQS[*]}"
echo "N_MIX=${N_MIX} L1_SCALES=${L1_SCALES[*]} L2_SCALES=${L2_SCALES[*]}"
echo "N_ZERO=${N_ZERO} ZERO_INITS=${ZERO_INITS[*]}"
echo "N_SCHEDULE=${N_SCHEDULE} SCHEDULE_MODES=${SCHEDULE_MODES[*]}"
echo "N_V0_SEED=${N_V0_SEED} V0_SEEDS=${V0_SEEDS[*]} (v=0 replicate count)"
echo "N_REST_SEED=${N_REST_SEED} REST_SEEDS=${REST_SEEDS[*]} N_V_REST=${N_V_REST} V_REST=${V_REST[*]}"
echo "N_TASKS_V0=${N_TASKS_V0} N_TASKS_REST=${N_TASKS_REST} N_TASKS=${N_TASKS} CHUNK=${CHUNK} N_ARRAY_TASKS=${N_ARRAY_TASKS}"
echo "ARRAY_INDICES=${ARRAY_INDICES} (only these 10 of ${N_ARRAY_TASKS} array indices are actually submitted; 10*3=30 replicates)"
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
#SBATCH --array=${ARRAY_INDICES}

${JOB_PREAMBLE}

echo "lscpu ------------------------------------------------------- \${SECONDS}"
lscpu || :

echo "cpuinfo ----------------------------------------------------- \${SECONDS}"
cat /proc/cpuinfo || :

echo "marimo notebook source -------------------------------------- \${SECONDS}"
echo "notebook source: ${BATCHDIR_JOBSOURCE}/${NOTEBOOK_PATH}"
cat "${BATCHDIR_JOBSOURCE}/${NOTEBOOK_PATH}" || :

echo "task assignment --------------------------------------------- \${SECONDS}"
BLIP_FREQS=(${BLIP_FREQS[*]})
L1_SCALES=(${L1_SCALES[*]})
L2_SCALES=(${L2_SCALES[*]})
ZERO_INITS=(${ZERO_INITS[*]})
SCHEDULE_MODES=(${SCHEDULE_MODES[*]})
V0_SEEDS=(${V0_SEEDS[*]})
REST_SEEDS=(${REST_SEEDS[*]})
V_REST=(${V_REST[*]})
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

# Run one (blip_freq, seed, l1/l2 mix, v, zero_init, schedule_mode) trial
# on CPU. Each replicate runs in its own working dir \${JOBDIR}/r<gid> and
# the notebook writes one timeseries parquet plus G/B snapshot npz stores
# to that dir's dd_trial_outputs/, self-describing via keyname (every run
# option as a key=value segment) with a uuid replicate identifier
# stamped on every timeseries row.
run_replicate() {
    local gid="\$1"
    local v seed

    if [ "\${gid}" -lt "${N_TASKS_V0}" ]; then
        # v=0 block: single seed, so no v/seed indexing needed.
        local schedule_idx=\$((gid % ${N_SCHEDULE}))
        local rem1=\$((gid / ${N_SCHEDULE}))
        local zero_idx=\$((rem1 % ${N_ZERO}))
        local rem2=\$((rem1 / ${N_ZERO}))
        local mix_idx=\$((rem2 % ${N_MIX}))
        local rem3=\$((rem2 / ${N_MIX}))
        local blip_idx=\$((rem3 % ${N_BLIP}))
        v=0
        seed="\${V0_SEEDS[0]}"
    else
        # "rest" block (v in {2,4,...,20}), re-based to start at 0.
        local rgid=\$((gid - ${N_TASKS_V0}))
        local schedule_idx=\$((rgid % ${N_SCHEDULE}))
        local rem1=\$((rgid / ${N_SCHEDULE}))
        local zero_idx=\$((rem1 % ${N_ZERO}))
        local rem2=\$((rem1 / ${N_ZERO}))
        local v_idx=\$((rem2 % ${N_V_REST}))
        local rem3=\$((rem2 / ${N_V_REST}))
        local mix_idx=\$((rem3 % ${N_MIX}))
        local rem4=\$((rem3 / ${N_MIX}))
        local seed_idx=\$((rem4 % ${N_REST_SEED}))
        local blip_idx=\$((rem4 / ${N_REST_SEED}))
        v="\${V_REST[\${v_idx}]}"
        seed="\${REST_SEEDS[\${seed_idx}]}"
    fi

    local blip="\${BLIP_FREQS[\${blip_idx}]}"
    local l1="\${L1_SCALES[\${mix_idx}]}"
    local l2="\${L2_SCALES[\${mix_idx}]}"
    local zeroinit="\${ZERO_INITS[\${zero_idx}]}"
    local schedulemode="\${SCHEDULE_MODES[\${schedule_idx}]}"
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
    echo "  [gid=\${gid}] blip_freq=\${blip} seed=\${seed} l1=\${l1} l2=\${l2} v=\${v} zero_init=\${zeroinit} schedule_mode=\${schedulemode} repdir=\${repdir}"
    python3.10 -m marimo export ipynb \
        --include-outputs --sort topological -f \
        "\${nbdir}/${NOTEBOOK_NAME}.py" \
        -o "\${repdir}/${NOTEBOOK_NAME}.ipynb" \
        -- \
        --seed "\${seed}" \
        --v "\${v}" \
        --zero-init "\${zeroinit}" \
        --l1-scale "\${l1}" \
        --l2-scale "\${l2}" \
        --blip-freq "\${blip}" \
        --schedule-mode "\${schedulemode}" \
        --num-epoch ${NUM_EPOCH}

    # Fail loudly on a blank/failed export. marimo can exit 0 while
    # producing a notebook whose cells never executed --- and the run
    # cell is what writes dd_trial_outputs/ --- so a "successful" export
    # with no outputs would otherwise sail through to >>>complete<<<.
    # Require both a non-trivial exported notebook and the run cell's
    # parquet + npz outputs, else fail the replicate (return 1, caught by
    # the wait loop below).
    local nb_out="\${repdir}/${NOTEBOOK_NAME}.ipynb"
    local outdata_dir="\${repdir}/dd_trial_outputs"
    local nb_bytes
    nb_bytes=\$(wc -c < "\${nb_out}" 2>/dev/null || echo 0)
    if [ "\${nb_bytes}" -lt 10000 ]; then
        echo "ERROR [gid=\${gid}]: exported notebook \${nb_out} missing or trivial (\${nb_bytes} bytes)"
        return 1
    fi
    local n_pqt
    n_pqt=\$(find "\${outdata_dir}" -maxdepth 1 -name '*ext=.pqt' 2>/dev/null | wc -l)
    if [ "\${n_pqt}" -lt 1 ]; then
        echo "ERROR [gid=\${gid}]: no timeseries parquet under \${outdata_dir}"
        return 1
    fi
    local n_npz
    n_npz=\$(find "\${outdata_dir}" -maxdepth 1 -name '*ext=.npz' 2>/dev/null | wc -l)
    if [ "\${n_npz}" -lt 2 ]; then
        echo "ERROR [gid=\${gid}]: expected 2 (G, B) snapshot npz files under \${outdata_dir}, found \${n_npz}"
        return 1
    fi
    echo "  [gid=\${gid}] export OK: \${nb_bytes} byte notebook, \${n_pqt} parquet + \${n_npz} npz in dd_trial_outputs"
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

echo "   - join per-replicate timeseries parquets across all conditions"
# Each replicate writes one self-describing timeseries parquet (v, seed,
# zeroinit, l1scale, l2scale, blipfreq, numepoch, schedulemode, replicate
# columns stamped by the notebook's run cell) under r<gid>/dd_trial_outputs/,
# so a straight concatenation yields a collated frame spanning the whole
# sweep. The G/B snapshot npz stores are NOT tabular (they're keyed
# arrays), so they aren't joined here -- they ride along in the
# jobarchive tarball above instead.
out_path="${BATCHDIR_JOBRESULT}/a=trace+date=${JOBDATE}+job=${JOBNAME}+ext=.pqt"
ls -1 "${BATCHDIR}"/__*/**/dd_trial_outputs/*ext=.pqt 2>/dev/null \
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
