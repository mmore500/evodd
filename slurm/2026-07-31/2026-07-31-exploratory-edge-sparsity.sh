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

# Sweep: the single-trial elastic-net GRN notebook
# (bindle/2026-07-30-exploratory-edge-sparsity.py), a from-scratch variant
# of the v1..v20 double-descent model family that replaces BOTH the v
# (visible-gene count) axis AND the blip machinery entirely:
#   - No blips: every training block presents one of the actual training
#     patterns (round-robin), never a blip-target substitute -- there's no
#     blip_freq/blip_mode axis at all.
#   - No v/visible-gene masking (and so no zero_init axis either): all 16
#     genes are always visible. "Model size" is instead controlled by
#     zeroing out a random subset of the GRN's N*N = 16*16 = 256 possible
#     regulatory edges (the notebook's --density flag), fixed for the
#     whole replicate -- see the notebook's "Edge sparsity" section.
# Swept across:
#   - density     in 99 evenly-spaced values across (0, 1]               (99)
#     (fraction of the 256 edges RETAINED; density=1 -> dense/no edges
#     zeroed). density=0 (all 256 edges zeroed) is deliberately EXCLUDED:
#     at that point B is permanently all-zero (edge_mask zeroes every
#     entry regardless of seed), so l1_cost/l2_cost are always 0 --
#     making the l1/l2 mix axis a no-op -- and the GRN can't discriminate
#     any pattern regardless of n_classes either, so all 40 replicates
#     that condition would occupy (10 seeds x 2 mixes x 2 n_classes) are
#     degenerate/uninformative. The remaining 99 points keep the same
#     1/99 spacing as a full 100-point [0, 1] linspace, just missing that
#     one endpoint.
#   - (l1_scale, l2_scale) in {(1.0, 0.0), (0.0, 0.0)}
#     "with L1" (pure L1, l2_scale=0 so the L2 term contributes nothing)
#     vs. "without L1" (NEITHER term contributes -- l1_scale=l2_scale=0,
#     i.e. no regularization at all, not a swap to pure L2)               (2)
#   - n_classes   in {3, 5}
#     3-class is this project's original training set {S1, S2, S3}
#     (CLASS_8[[0,3,6]]); 5-class adds CLASS_8[[1,7]] (see the notebook's
#     "Training set" section for the full mapping)                       (2)
# crossed with 10 replicate seeds per density (uniform -- unlike the
# v-sweep scripts, there's no uneven "v=0 gets fewer replicates" special
# case here, since density doesn't have an analogous degenerate-value
# asymmetry once density=0 itself is excluded):
#   - seed in {1..10}                                                    (10)
# total = 99 * 2 * 2 * 10 = 3960 replicates.
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
# The cluster caps a job array at 1000 queued tasks, so we pack
# CHUNK=4 (= N_MIX * N_NCLASSES, the fastest-varying full cross of the
# two size-2 axes) replicates into each array task and run those 4
# *concurrently* (one CPU each, so --cpus-per-task below always equals
# the actual number of concurrently-launched replicates) rather than
# sequentially -- each array task's 4 concurrent replicates are the SAME
# (density, seed) pair run under all 4 (l1/l2 mix, n_classes)
# combinations side by side. This divides 3960 replicates evenly into
# 3960 / 4 = 990 array tasks (no partial final chunk), comfortably under
# the 1000-task cap.
#
# Global replicate index r in [0, N_TASKS) decomposes fastest-varying
# first, matching CHUNK:
#   nclasses_idx = r % N_NCLASSES;
#   mix_idx      = (r / N_NCLASSES) % N_MIX;
#   seed_idx     = (r / N_NCLASSES / N_MIX) % N_SEED;
#   density_idx  = r / N_NCLASSES / N_MIX / N_SEED.
# Array task t owns the CHUNK consecutive indices r = t * CHUNK + j for
# j in [0, CHUNK) (each launched as a background job) -- since CHUNK
# exactly equals N_NCLASSES * N_MIX, every task's 4 replicates share one
# (density, seed) pair and sweep all 4 (mix, n_classes) combinations,
# with no chunk straddling a seed or density boundary.
#
# Benchmarked the notebook's core SSWM loop single-threaded (non-cluster
# hardware) at ~70,000-83,000 generations/sec, so one 500M-generation
# replicate takes ~100-120 minutes -- comfortably inside the 4-hour job
# time limit below even allowing for slower cluster CPUs.
# i starts at 1 (not 0) to exclude density=0 (the degenerate "0 edges
# retained" case -- see above), so this keeps the same 1/99 spacing a
# full 100-point [0, 1] linspace would have, just dropping that one
# endpoint.
DENSITIES=($(awk 'BEGIN { for (i = 1; i < 100; i++) printf "%.6f ", i / 99 }'))
L1_SCALES=(0.5 0.9 0.5 0.99)
L2_SCALES=(0.5 0.1 0.0 0.01)
N_CLASSES_VALUES=(3)
SEEDS=(1 2 3 4 5 6 7 8 9 10)
N_DENSITY=${#DENSITIES[@]}
N_MIX=${#L1_SCALES[@]}
N_NCLASSES=${#N_CLASSES_VALUES[@]}
N_SEED=${#SEEDS[@]}
N_TASKS=$((N_DENSITY * N_MIX * N_NCLASSES * N_SEED))
CHUNK=$((N_NCLASSES * N_MIX))
N_ARRAY_TASKS=$(((N_TASKS + CHUNK - 1) / CHUNK))
NUM_EPOCH=138889
echo "N_DENSITY=${N_DENSITY} DENSITIES=${DENSITIES[*]}"
echo "N_MIX=${N_MIX} L1_SCALES=${L1_SCALES[*]} L2_SCALES=${L2_SCALES[*]}"
echo "N_NCLASSES=${N_NCLASSES} N_CLASSES_VALUES=${N_CLASSES_VALUES[*]}"
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
L1_SCALES=(${L1_SCALES[*]})
L2_SCALES=(${L2_SCALES[*]})
N_CLASSES_VALUES=(${N_CLASSES_VALUES[*]})
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

# Run one (density, seed, l1/l2 mix, n_classes) trial on CPU. Each
# replicate runs in its own working dir \${JOBDIR}/r<gid> and the notebook
# writes one timeseries csv plus G/B snapshot npz stores to that dir's
# dd_trial_outputs/, self-describing via keyname (every run option as a
# key=value segment) with a uuid replicate identifier stamped on every
# timeseries row.
run_replicate() {
    local gid="\$1"

    local nclasses_idx=\$((gid % ${N_NCLASSES}))
    local rem1=\$((gid / ${N_NCLASSES}))
    local mix_idx=\$((rem1 % ${N_MIX}))
    local rem2=\$((rem1 / ${N_MIX}))
    local seed_idx=\$((rem2 % ${N_SEED}))
    local density_idx=\$((rem2 / ${N_SEED}))

    local density="\${DENSITIES[\${density_idx}]}"
    local seed="\${SEEDS[\${seed_idx}]}"
    local l1="\${L1_SCALES[\${mix_idx}]}"
    local l2="\${L2_SCALES[\${mix_idx}]}"
    local nclasses="\${N_CLASSES_VALUES[\${nclasses_idx}]}"
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
    echo "  [gid=\${gid}] density=\${density} seed=\${seed} l1=\${l1} l2=\${l2} nclasses=\${nclasses} repdir=\${repdir}"
    python3.10 -m marimo export ipynb \
        --include-outputs --sort topological -f \
        "\${nbdir}/${NOTEBOOK_NAME}.py" \
        -o "\${repdir}/${NOTEBOOK_NAME}.ipynb" \
        -- \
        --seed "\${seed}" \
        --density "\${density}" \
        --l1-scale "\${l1}" \
        --l2-scale "\${l2}" \
        --n-classes "\${nclasses}" \
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
