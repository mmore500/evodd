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

# Reproduces the same (l1_scale=1.0, l2_scale=0.0, schedule_mode in
# {none, local}, zero_init in {True, False}) condition slice of
# slurm/2026-07-23/2026-07-23-exploratory-sweep.sh this project has already
# probed at blip_freq=0.5 (bindle/teeplots/2026-07-25-exploratory-sweep-and
# -noblip's "blipfreq=0.5+...+l1scale-1.0+l2scale-0.0+schedulemode-
# {local,none}+zeroinit={True,False}" condition) -- same uneven v/seed
# split, same fixed L1/L2 mix, "global" schedule_mode dropped as in every
# recent sibling -- but with changes to the blip mechanism itself:
#
#   1. blip_mode is fixed at "fixed" (deterministic single-bit flip per
#      training pattern, not blip_mode="bitflip"'s random per-replicate
#      draw), with the 3 flip sites now swept directly via the notebook's
#      new --blip-sites flag instead of being hardcoded:
#        - blip_sites = 0,4,8  -- one distinct site per pattern (this is
#          exactly the notebook's original, pre-existing blip_mode="fixed"
#          behavior, now reachable as an explicit condition)
#        - blip_sites = 0,0,0  -- all three patterns blipped at the SAME
#          site instead of three different ones
#      (2 conditions -- see bindle/2026-07-23-exploratory.py's
#      build_schedule cell, "make_blips")
#
#   2. --blip-release-prob (new notebook flag, swept over 3 values here):
#      each time a blip is presented, WITH PROBABILITY blip_release_prob
#      (freshly drawn per blip occurrence, not fixed per replicate) that
#      occurrence's flip site is "released" from selection against the
#      blip's fixed (flipped) value -- instead of comparing to the
#      flipped target, that site is scored against whatever value the
#      organism ITSELF currently expresses there (see benefit's
#      release_mask), so it always counts as a match: selection then
#      actively rewards confidently expressing *some* value at that site
#      without caring which one, rather than the notebook's default of
#      rewarding a match to the flipped value. With the complementary
#      probability, that occurrence is scored normally (exactly the
#      original notebook's behavior). Three conditions:
#        - blip_release_prob = 0.0  -- never released: every blip
#          occurrence scored normally against the flipped value, exactly
#          the notebook's original (pre-existing) behavior. Included here
#          as the control arm rather than assumed from a prior sweep.
#        - blip_release_prob = 1.0  -- always released: every blip
#          occurrence uses the organism's own expressed value.
#        - blip_release_prob = 0.5  -- released roughly half the time,
#          independently drawn per blip occurrence (per K-generation
#          block, not per generation -- see run_sswm's per-block Bernoulli
#          draw): some occurrences of a given blip are scored normally,
#          others are released, within the SAME replicate.
#      Implemented via a new release_masks array (one mask per
#      training-set row, giving the release mask to apply IF an
#      occurrence releases) plus the per-block Bernoulli draw itself,
#      threaded through benefit()/fitness_output_masked_*_elastic()/
#      run_sswm_*_elastic(). Backward compatible: blip_release_prob
#      defaults to 0.0, and the coin-flip draw is skipped entirely
#      (short-circuited) whenever it's 0.0, reproducing the pre-existing
#      notebook's behavior bit-for-bit (verified against the prior
#      notebook revision for a matched seed/condition, aside from
#      wall-clock timing).
#
# blip_freq is swept over a new range extending past this project's usual
# {0.5, 0.6, 0.63, 0.66} values up past 1 -- recall blip_freq is a
# blip:true pattern-count RATIO (see build_schedule), not a probability, so
# values above 1 (blips presented MORE often than true patterns) are
# meaningful:
#   blip_freq in {0.5, 0.6, 0.75, 1.0, 1.2, 1.5}                        (6)
# crossed with:
#   blip_release_prob in {0.0, 1.0, 0.5}                                (3)
#   blip_sites in {"0,4,8", "0,0,0"}                                    (2)
#   zero_init in {True, False}                                         (2)
#   schedule_mode in {none, local}                                     (2)
#   l1_scale=1.0, l2_scale=0.0 (fixed, not swept)
#   blip_mode=fixed (fixed, not swept)
# and the same uneven v/seed split as the base sweep:
#   v = 0                       -> 1 replicate  (seed 1 only)      (1 v x 1 seed)
#   v in {2, 4, ..., 20} (even) -> 4 replicates each (seeds 1..4)  (10 v x 4 seed)
# total = 6 * 3 * 2 * 2 * 2 * (1*1 + 10*4) = 6 * 3 * 2 * 2 * 2 * 41
#       = 5904 replicates.
#
# Generations vs. epochs: the notebook's SSWM loop runs
# TOTAL_BLOCKS (fixed at 3600 inside the notebook, not CLI-configurable)
# blocks of K generations each, where K is set by the "--num-epoch" CLI
# flag (the notebook's internal name for this per-block generation count --
# distinct from the "epoch" column in its output parquet, which is
# actually the snapshot index, 0..48). To hit a ~500,000,000-generation
# budget (matching every prior sweep in this project, for comparability):
# NUM_EPOCH = round(500e6 / 3600) = 138889, giving an actual total of
# 3600 * 138889 = 500,000,400 generations per replicate (400 over the
# round-number target -- 3600 does not divide 500e6 evenly).
#
# The cluster caps a job array at 1000 queued tasks; at 5904 total
# replicates this job needs packing well beyond a single naturally-size-2
# axis to fit. schedule_mode (2) and blip_release_prob (3) are packed
# together as the FASTEST-varying (innermost) pair of dimensions in the
# index decomposition below, giving CHUNK = 2 * 3 = 6, so each array
# task's 6 concurrent replicates are the SAME (zero_init, blip_sites, v,
# seed, blip_freq) condition run under all 2 schedule_mode x 3
# blip_release_prob combinations side by side -- this divides 5904
# replicates evenly into 5904 / 6 = 984 array tasks, comfortably under
# the cap (and matching this project's other sweeps' array-task count).
#
# Global replicate index r in [0, N_TASKS) is split into two contiguous
# blocks rather than one uniform Cartesian product, since v=0 and the
# rest of the v values don't share the same seed count. Both blocks
# decompose fastest-varying first, starting with schedule_idx and
# release_idx (together) so they align with CHUNK:
#   - r < N_TASKS_V0: the v=0 block (single seed).
#     schedule_idx = r % N_SCHEDULE;
#     release_idx = (r / N_SCHEDULE) % N_RELEASE;
#     zero_idx = (r / N_SCHEDULE / N_RELEASE) % N_ZERO;
#     sites_idx = (r / N_SCHEDULE / N_RELEASE / N_ZERO) % N_SITES;
#     blip_idx = r / N_SCHEDULE / N_RELEASE / N_ZERO / N_SITES.
#   - r >= N_TASKS_V0: the "rest" block (v in {2,4,...,20}, 4 seeds
#     each), re-based to r' = r - N_TASKS_V0.
#     schedule_idx = r' % N_SCHEDULE;
#     release_idx = (r' / N_SCHEDULE) % N_RELEASE;
#     zero_idx = (r' / N_SCHEDULE / N_RELEASE) % N_ZERO;
#     sites_idx = (r' / N_SCHEDULE / N_RELEASE / N_ZERO) % N_SITES;
#     v_idx = (r' / N_SCHEDULE / N_RELEASE / N_ZERO / N_SITES) % N_V_REST;
#     seed_idx = (r' / N_SCHEDULE / N_RELEASE / N_ZERO / N_SITES / N_V_REST) % N_REST_SEED;
#     blip_idx = r' / N_SCHEDULE / N_RELEASE / N_ZERO / N_SITES / N_V_REST / N_REST_SEED.
# N_TASKS_V0 (144) is itself a multiple of CHUNK=6, so no CHUNK-group
# straddles the v0/rest block boundary. Array task t owns the CHUNK
# consecutive indices r = t * CHUNK + j for j in [0, CHUNK) (each
# launched as a background job).
#
# Benchmarked the notebook's core SSWM loop single-threaded (non-cluster
# hardware) at ~83,000 generations/sec, so one 500M-generation replicate
# takes ~100 minutes -- comfortably inside the 4-hour job time limit below
# even allowing for slower cluster CPUs.
BLIP_FREQS=(0.5 0.6 0.75 1.0 1.2 1.5)
BLIP_MODE=fixed
BLIP_RELEASE_PROBS=(0.0 1.0 0.5)
L1_SCALE=1.0
L2_SCALE=0.0
BLIP_SITES=("0,4,8" "0,0,0")
ZERO_INITS=(True False)
SCHEDULE_MODES=(none local)
V0_SEEDS=(1)
REST_SEEDS=(1 2 3 4)
V_REST=(2 4 6 8 10 12 14 16 18 20)
N_BLIP=${#BLIP_FREQS[@]}
N_RELEASE=${#BLIP_RELEASE_PROBS[@]}
N_SITES=${#BLIP_SITES[@]}
N_ZERO=${#ZERO_INITS[@]}
N_SCHEDULE=${#SCHEDULE_MODES[@]}
N_V0_SEED=${#V0_SEEDS[@]}
N_REST_SEED=${#REST_SEEDS[@]}
N_V_REST=${#V_REST[@]}
N_TASKS_V0=$((N_SCHEDULE * N_RELEASE * N_ZERO * N_SITES * N_BLIP * N_V0_SEED))
N_TASKS_REST=$((N_SCHEDULE * N_RELEASE * N_ZERO * N_SITES * N_V_REST * N_REST_SEED * N_BLIP))
N_TASKS=$((N_TASKS_V0 + N_TASKS_REST))
CHUNK=$((N_SCHEDULE * N_RELEASE))
N_ARRAY_TASKS=$(((N_TASKS + CHUNK - 1) / CHUNK))
NUM_EPOCH=138889
echo "N_BLIP=${N_BLIP} BLIP_FREQS=${BLIP_FREQS[*]}"
echo "BLIP_MODE=${BLIP_MODE} (fixed, not swept)"
echo "N_RELEASE=${N_RELEASE} BLIP_RELEASE_PROBS=${BLIP_RELEASE_PROBS[*]}"
echo "L1_SCALE=${L1_SCALE} L2_SCALE=${L2_SCALE} (fixed, not swept)"
echo "N_SITES=${N_SITES} BLIP_SITES=${BLIP_SITES[*]}"
echo "N_ZERO=${N_ZERO} ZERO_INITS=${ZERO_INITS[*]}"
echo "N_SCHEDULE=${N_SCHEDULE} SCHEDULE_MODES=${SCHEDULE_MODES[*]}"
echo "N_V0_SEED=${N_V0_SEED} V0_SEEDS=${V0_SEEDS[*]} (v=0 replicate count)"
echo "N_REST_SEED=${N_REST_SEED} REST_SEEDS=${REST_SEEDS[*]} N_V_REST=${N_V_REST} V_REST=${V_REST[*]}"
echo "N_TASKS_V0=${N_TASKS_V0} N_TASKS_REST=${N_TASKS_REST} N_TASKS=${N_TASKS} CHUNK=${CHUNK} N_ARRAY_TASKS=${N_ARRAY_TASKS}"
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
BLIP_FREQS=(${BLIP_FREQS[*]})
BLIP_RELEASE_PROBS=(${BLIP_RELEASE_PROBS[*]})
BLIP_SITES=(${BLIP_SITES[*]})
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

# Run one (zero_init, blip_sites, v, seed, blip_freq, schedule_mode,
# blip_release_prob) trial on CPU, with l1_scale=${L1_SCALE},
# l2_scale=${L2_SCALE}, blip_mode=${BLIP_MODE} held fixed (reproducing the
# specific condition this job targets rather than sweeping them). Each
# replicate runs in its own working dir \${JOBDIR}/r<gid> and the notebook
# writes one timeseries csv plus G/B snapshot npz stores to that dir's
# dd_trial_outputs/, self-describing via keyname (every run option as a
# key=value segment) with a uuid replicate identifier stamped on every
# timeseries row.
run_replicate() {
    local gid="\$1"
    local v seed

    if [ "\${gid}" -lt "${N_TASKS_V0}" ]; then
        # v=0 block: single seed, so no v/seed indexing needed.
        local schedule_idx=\$((gid % ${N_SCHEDULE}))
        local rem1=\$((gid / ${N_SCHEDULE}))
        local release_idx=\$((rem1 % ${N_RELEASE}))
        local rem2=\$((rem1 / ${N_RELEASE}))
        local zero_idx=\$((rem2 % ${N_ZERO}))
        local rem3=\$((rem2 / ${N_ZERO}))
        local sites_idx=\$((rem3 % ${N_SITES}))
        local rem4=\$((rem3 / ${N_SITES}))
        local blip_idx=\$((rem4 % ${N_BLIP}))
        v=0
        seed="\${V0_SEEDS[0]}"
    else
        # "rest" block (v in {2,4,...,20}), re-based to start at 0.
        local rgid=\$((gid - ${N_TASKS_V0}))
        local schedule_idx=\$((rgid % ${N_SCHEDULE}))
        local rem1=\$((rgid / ${N_SCHEDULE}))
        local release_idx=\$((rem1 % ${N_RELEASE}))
        local rem2=\$((rem1 / ${N_RELEASE}))
        local zero_idx=\$((rem2 % ${N_ZERO}))
        local rem3=\$((rem2 / ${N_ZERO}))
        local sites_idx=\$((rem3 % ${N_SITES}))
        local rem4=\$((rem3 / ${N_SITES}))
        local v_idx=\$((rem4 % ${N_V_REST}))
        local rem5=\$((rem4 / ${N_V_REST}))
        local seed_idx=\$((rem5 % ${N_REST_SEED}))
        local rem6=\$((rem5 / ${N_REST_SEED}))
        local blip_idx=\$((rem6 % ${N_BLIP}))
        v="\${V_REST[\${v_idx}]}"
        seed="\${REST_SEEDS[\${seed_idx}]}"
    fi

    local zeroinit="\${ZERO_INITS[\${zero_idx}]}"
    local sites="\${BLIP_SITES[\${sites_idx}]}"
    local schedulemode="\${SCHEDULE_MODES[\${schedule_idx}]}"
    local blipfreq="\${BLIP_FREQS[\${blip_idx}]}"
    local releaseprob="\${BLIP_RELEASE_PROBS[\${release_idx}]}"
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
    echo "  [gid=\${gid}] blip_freq=\${blipfreq} blip_mode=${BLIP_MODE} blip_sites=\${sites} blip_release_prob=\${releaseprob} seed=\${seed} l1=${L1_SCALE} l2=${L2_SCALE} v=\${v} zero_init=\${zeroinit} schedule_mode=\${schedulemode} repdir=\${repdir}"
    python3.10 -m marimo export ipynb \
        --include-outputs --sort topological -f \
        "\${nbdir}/${NOTEBOOK_NAME}.py" \
        -o "\${repdir}/${NOTEBOOK_NAME}.ipynb" \
        -- \
        --seed "\${seed}" \
        --v "\${v}" \
        --zero-init "\${zeroinit}" \
        --l1-scale ${L1_SCALE} \
        --l2-scale ${L2_SCALE} \
        --blip-freq "\${blipfreq}" \
        --blip-mode ${BLIP_MODE} \
        --blip-sites "\${sites}" \
        --blip-release-prob "\${releaseprob}" \
        --schedule-mode "\${schedulemode}" \
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
# Each replicate writes one self-describing timeseries csv (v, seed,
# zeroinit, l1scale, l2scale, blipfreq, blipmode, blipsites, bliprelprob,
# numepoch, replicate columns stamped by the notebook's run cell) under
# r<gid>/dd_trial_outputs/, so a straight concatenation yields a collated
# frame spanning the whole sweep.
# Per-replicate output is CSV (written progressively during the run, so a
# SLURM timeout still leaves partial data on disk) rather than parquet;
# joinem infers CSV input / parquet output from the file extensions below,
# so the conversion to parquet happens once here rather than once per
# replicate. The G/B snapshot npz stores are NOT tabular (they're keyed
# arrays), so they aren't joined here -- they ride along in the jobarchive
# tarball above instead.
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
