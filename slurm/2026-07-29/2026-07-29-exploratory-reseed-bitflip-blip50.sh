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

# Reproduces/reseeds the single condition behind the teeplot artifact
#   blipfreq=0.5+blipmode=bitflip+dataset=exploratory-sweep-randomblip
#   +l1scale=0.9966+l2scale=0.003+schedulemode=local+viz=make-compound-plot
#   +zeroinit=False+ext=.pdf
# (from bindle/2026-07-28-exploratory-sweep-randomblip.py, sourced from
# slurm/2026-07-27/2026-07-27-exploratory-sweep-randomblip.sh's OSF output,
# osf.io/j3frh) -- fixing blip_freq=0.5 and blip_mode=bitflip (the defining
# attributes of that artifact) as constants rather than swept dimensions,
# and running a fresh, smaller sweep on top with 5 NEW seeds and 3 NEW
# (l1_scale, l2_scale) mixes not used by that original job.
#
# Verified via git history (comparing the commit that added
# slurm/2026-07-27/2026-07-27-exploratory-sweep-randomblip.sh against the
# commit that added slurm/2026-07-28/2026-07-28-exploratory-sweep-bitflip.sh,
# the two commits bracketing every change to bindle/2026-07-23-exploratory.py
# since the randomblip job ran) that the actual simulation trajectory logic
# -- develop/mutate/benefit, l1_cost/l2_cost, both
# fitness_output_masked_*_elastic functions, both
# run_sswm_output_masked_scheduled_traced_*_elastic loops,
# build_schedule/make_blips (including the blip-drawing RNG and
# rejection-sampling mechanics under blip_mode=bitflip), sample_G,
# make_visible_mask, all CLI-flag wiring, and the TOTAL_BLOCKS/NUM_EPOCH
# generation-budget constants -- is byte-identical between the two commits.
# The only diff in that window is additive, post-hoc measurement/output-schema
# code (a new "bitflip" phenotype bucket -- bitflip_frac/bitflip_chi2 columns
# -- carved out of "other", spanning all 48 single-bit transformations of the
# 3 training patterns). So this job -- necessarily run against current HEAD,
# per this repo's SOURCE_REVISION-at-launch-time convention -- reproduces the
# same underlying G/B evolutionary dynamics as the original randomblip job
# for matching parameters; its output parquet just carries two extra
# columns the original didn't have.
#
# Sweep: the single-trial elastic-net GRN notebook (v1..v20 double-descent
# model), with blip_freq=0.5 and blip_mode=bitflip held fixed, across:
#   - (l1_scale, l2_scale) in {(0.995, 0.005), (0.9933, 0.0067),
#     (0.925, 0.075)} -- l2_scale = 1 - l1_scale throughout, matching the
#     exact precedent set by slurm/2026-07-23/*.sh's first two mixes
#     (0.995/0.005 and 0.9933/0.0067 are identical to those scripts'
#     L1_SCALES/L2_SCALES entries); 0.925/0.075 extends the same rule to a
#     new, more strongly-regularized mix none of this project's prior
#     sweeps have tried.                                             (3)
#   - zero_init     in {True, False}                                (2)
#   - schedule_mode in {none, local, global}                         (3)
#     all three are already-supported values of the notebook's
#     --schedule-mode flag (see weighted_interleave in
#     bindle/2026-07-23-exploratory.py); prior sweep scripts only ever
#     exercised {none, local} -- this is the first to also cover "global".
# crossed with an UNEVEN v/seed split, matching the original randomblip
# job's convention (v=0 gets fewer replicates than the rest):
#   - v = 0                      -> 1 replicate  (seed 5 only)       (1 v x 1 seed)
#   - v in {4, 8, 12, 16, 20}    -> 5 replicates each (seeds 5..9)    (5 v x 5 seed)
#     seeds 5..9 are NEW -- every prior sweep in this project (including
#     the original randomblip job) only ever used seeds 1..4.
# total = 3 * 2 * 3 * (1*1 + 5*5) = 3 * 2 * 3 * 26 = 468 replicates
# (18 for the v=0 block + 450 for the v-in-{4,8,12,16,20} block), i.e. 468
# (l1/l2 mix, zero_init, schedule_mode, v, seed) CONDITIONS -- there's no
# separate "run under both/all X values" outer crossing here, since (unlike
# the base sweeps) schedule_mode is swept directly within this same total
# rather than doubling/tripling a smaller per-schedule-mode replicate count.
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
# The cluster caps a job array at 1000 queued tasks; at 468 total
# replicates this job is comfortably under that cap even packed at
# CHUNK=2 (234 array tasks), so -- unlike the base sweeps, which need
# CHUNK=2 just to fit -- packing here is purely a throughput convenience
# (2 concurrent replicates per array task, one CPU each, see
# --cpus-per-task below) rather than a hard requirement. zero_init is
# deliberately the FASTEST-varying (innermost) dimension in the index
# decomposition below, exactly matching CHUNK=2 (it's the one swept axis
# that's naturally size-2, playing the role schedule_mode played in the
# base sweeps before schedule_mode itself grew to 3 values here) -- so
# each array task's 2 concurrent replicates are the SAME (l1/l2 mix,
# schedule_mode, v, seed) condition run under both zero_init values side
# by side. This divides 468 replicates evenly into 468 / 2 = 234 array
# tasks.
#
# Global replicate index r in [0, N_TASKS) is split into two contiguous
# blocks rather than one uniform Cartesian product, since v=0 and the
# rest of the v values don't share the same seed count. Both blocks
# decompose fastest-varying first, starting with zero_idx so it aligns
# with CHUNK:
#   - r < N_TASKS_V0: the v=0 block (single seed).
#     zero_idx = r % N_ZERO;
#     schedule_idx = (r / N_ZERO) % N_SCHEDULE;
#     mix_idx = r / N_ZERO / N_SCHEDULE.
#   - r >= N_TASKS_V0: the "rest" block (v in {4,8,12,16,20}, 5 seeds
#     each), re-based to r' = r - N_TASKS_V0.
#     zero_idx = r' % N_ZERO;
#     schedule_idx = (r' / N_ZERO) % N_SCHEDULE;
#     v_idx = (r' / N_ZERO / N_SCHEDULE) % N_V_REST;
#     mix_idx = (r' / N_ZERO / N_SCHEDULE / N_V_REST) % N_MIX;
#     seed_idx = r' / N_ZERO / N_SCHEDULE / N_V_REST / N_MIX.
# N_TASKS_V0 (18) is itself a multiple of CHUNK=2, so no CHUNK-pair
# straddles the v0/rest block boundary. Array task t owns the CHUNK
# consecutive indices r = t * CHUNK + j for j in [0, CHUNK) (each
# launched as a background job).
#
# Benchmarked the notebook's core SSWM loop single-threaded (non-cluster
# hardware) at ~83,000 generations/sec, so one 500M-generation replicate
# takes ~100 minutes -- comfortably inside the 4-hour job time limit below
# even allowing for slower cluster CPUs.
BLIP_FREQ=0.5
BLIP_MODE=bitflip
L1_SCALES=(0.995 0.9933 0.925)
L2_SCALES=(0.005 0.0067 0.075)
ZERO_INITS=(True False)
SCHEDULE_MODES=(none local global)
V0_SEEDS=(5)
REST_SEEDS=(5 6 7 8 9)
V_REST=(4 8 12 16 20)
N_MIX=${#L1_SCALES[@]}
N_ZERO=${#ZERO_INITS[@]}
N_SCHEDULE=${#SCHEDULE_MODES[@]}
N_V0_SEED=${#V0_SEEDS[@]}
N_REST_SEED=${#REST_SEEDS[@]}
N_V_REST=${#V_REST[@]}
N_TASKS_V0=$((N_ZERO * N_SCHEDULE * N_MIX * N_V0_SEED))
N_TASKS_REST=$((N_ZERO * N_SCHEDULE * N_V_REST * N_MIX * N_REST_SEED))
N_TASKS=$((N_TASKS_V0 + N_TASKS_REST))
CHUNK=2
N_ARRAY_TASKS=$(((N_TASKS + CHUNK - 1) / CHUNK))
NUM_EPOCH=138889
echo "BLIP_FREQ=${BLIP_FREQ} BLIP_MODE=${BLIP_MODE} (fixed, not swept)"
echo "N_MIX=${N_MIX} L1_SCALES=${L1_SCALES[*]} L2_SCALES=${L2_SCALES[*]}"
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

# Run one (l1/l2 mix, zero_init, schedule_mode, v, seed) trial on CPU,
# with blip_freq=${BLIP_FREQ} and blip_mode=${BLIP_MODE} held fixed
# (reproducing the specific condition this job targets rather than
# sweeping them). Each replicate runs in its own working dir
# \${JOBDIR}/r<gid> and the notebook writes one timeseries parquet plus
# G/B snapshot npz stores to that dir's dd_trial_outputs/, self-describing
# via keyname (every run option as a key=value segment) with a uuid
# replicate identifier stamped on every timeseries row.
run_replicate() {
    local gid="\$1"
    local v seed

    if [ "\${gid}" -lt "${N_TASKS_V0}" ]; then
        # v=0 block: single seed, so no v/seed indexing needed.
        local zero_idx=\$((gid % ${N_ZERO}))
        local rem1=\$((gid / ${N_ZERO}))
        local schedule_idx=\$((rem1 % ${N_SCHEDULE}))
        local rem2=\$((rem1 / ${N_SCHEDULE}))
        local mix_idx=\$((rem2 % ${N_MIX}))
        v=0
        seed="\${V0_SEEDS[0]}"
    else
        # "rest" block (v in {4,8,12,16,20}), re-based to start at 0.
        local rgid=\$((gid - ${N_TASKS_V0}))
        local zero_idx=\$((rgid % ${N_ZERO}))
        local rem1=\$((rgid / ${N_ZERO}))
        local schedule_idx=\$((rem1 % ${N_SCHEDULE}))
        local rem2=\$((rem1 / ${N_SCHEDULE}))
        local v_idx=\$((rem2 % ${N_V_REST}))
        local rem3=\$((rem2 / ${N_V_REST}))
        local mix_idx=\$((rem3 % ${N_MIX}))
        local rem4=\$((rem3 / ${N_MIX}))
        local seed_idx=\$((rem4 % ${N_REST_SEED}))
        v="\${V_REST[\${v_idx}]}"
        seed="\${REST_SEEDS[\${seed_idx}]}"
    fi

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
    echo "  [gid=\${gid}] blip_freq=${BLIP_FREQ} blip_mode=${BLIP_MODE} seed=\${seed} l1=\${l1} l2=\${l2} v=\${v} zero_init=\${zeroinit} schedule_mode=\${schedulemode} repdir=\${repdir}"
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
        --blip-freq ${BLIP_FREQ} \
        --blip-mode ${BLIP_MODE} \
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
# zeroinit, l1scale, l2scale, blipfreq, blipmode, numepoch, replicate
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
