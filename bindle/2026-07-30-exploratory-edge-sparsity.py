import marimo

__generated_with = "0.23.2"
app = marimo.App(width="full")


@app.cell
def import_std():
    import builtins
    import contextlib
    import os
    import sys
    import threading
    import time
    import uuid
    import zipfile

    return builtins, contextlib, os, sys, threading, time, uuid, zipfile


@app.cell
def import_pkg():
    from keyname import keyname as kn
    import marimo as mo
    from numba import njit
    import numpy as np
    import pandas as pd
    from scipy.stats import qmc
    from watermark import watermark

    return kn, mo, njit, np, pd, qmc, watermark


@app.cell(hide_code=True)
def do_watermark(mo, watermark):
    mo.md(
        f"""
    ```Text
    {watermark(
        current_date=True,
        iso8601=True,
        machine=True,
        updated=True,
        python=True,
        iversions=True,
        globals_=globals(),
    )}
    ```
    """
    )
    return


@app.cell(hide_code=True)
def delimit_intro(mo):
    mo.md(
        """
    # Single-trial elastic-net GRN run: edge-sparsity model-size sweep, no blips

    One trial = one (density, seed, n_classes, L1 scale, L2 scale,
    num_epoch) combination. Adapted from the `v`-sweep/blip-sweep family
    of notebooks in this project (see e.g.
    `bindle/2026-07-23-exploratory.py`), but with two deliberate
    simplifications for this sweep:

    - **No blips.** Every training block presents one of the actual
      training patterns (round-robin cycling through the `n_classes`-item
      training set); there's no blip-target substitution or
      `blip_freq`/`blip_mode` axis.
    - **No `v`/visible-gene masking (and so no `zero_init` axis either).**
      All 16 genes are always visible; instead, "model size" is
      controlled by zeroing out a random subset of the GRN's 256 possible
      regulatory edges (`--density`; see "Edge sparsity" below) -- kept
      fixed for the whole replicate.

    Also introduces an `--n-classes` axis (3 vs. 5 canonical training
    phenotypes; see "Training set" below), crossed with an
    L1-regularization on/off axis via `--l1-scale`/`--l2-scale`.

    Self-contained: every model definition (Kouvaris et al. 2017 GRN
    core, target patterns, elastic-net SSWM) is inlined below rather than
    imported from the surrounding project, so this file has no dependency
    on any other file in this repository -- only installable packages
    (marimo, numpy, pandas, scipy, numba, keyname, watermark).

    CLI-parameterizable, no interactive widgets: every value below reads
    its default from `mo.cli_args()`, so running `marimo run
    2026-07-30-exploratory-edge-sparsity.py -- --seed 11 --density 0.5
    --n-classes 5 --l1-scale 0.0 --l2-scale 1.0 --num-epoch 50000` sets
    all of them, and the trial runs immediately (no button to click).
    """
    )
    return


@app.cell(hide_code=True)
def delimit_configure_trial(mo):
    mo.md(
        """
    ## Configure trial
    """
    )
    return


@app.cell
def configure_trial(mo):
    _args = mo.cli_args()

    def _get(name, default, cast):
        v = _args.get(name)
        return cast(v) if v is not None else default

    seed = _get("seed", 1, int)
    density = _get("density", 1.0, float)
    l1_scale = _get("l1-scale", 0.995, float)
    l2_scale = _get("l2-scale", 0.005, float)
    n_classes = _get("n-classes", 3, int)
    num_epoch = _get("num-epoch", 100, int)
    return density, l1_scale, l2_scale, n_classes, num_epoch, seed


@app.cell
def show_config(density, l1_scale, l2_scale, n_classes, num_epoch, pd, seed):
    config_df = pd.DataFrame(
        [
            {
                "seed": seed,
                "density": density,
                "n_classes": n_classes,
                "l1_scale": l1_scale,
                "l2_scale": l2_scale,
                "num_epoch": num_epoch,
            }
        ]
    )
    config_df
    return


@app.cell(hide_code=True)
def delimit_grn_core(mo):
    mo.md(
        """
    ## Core GRN model (Kouvaris et al. 2017), inlined from grn.py

    Uses the plain 16-gene model (no hidden-gene extension, no
    visible-gene masking -- all 16 genes are always visible; see "Edge
    sparsity" below for how model size is instead controlled).
    """
    )
    return


@app.cell
def grn_core(njit, np):
    N = 16
    T = 10
    TAU1 = 1.0
    TAU2 = 0.2
    ALPHA = 0.5

    @njit(fastmath=True)
    def develop(G, B):
        n = G.shape[0]
        p = G.copy()
        inter = np.empty(n)
        for _ in range(T):
            np.dot(B, p, inter)
            for i in range(n):
                p[i] = p[i] + TAU1 * np.tanh(ALPHA * inter[i]) - TAU2 * p[i]
        return p

    @njit(fastmath=True)
    def benefit(Pa, S):
        Pa_norm = Pa / (TAU1 / TAU2)
        dot = 0.0
        for i in range(Pa.shape[0]):
            dot += Pa_norm[i] * S[i]
        return 0.5 * (1.0 + dot / Pa.shape[0])

    @njit(fastmath=True)
    def l1_cost(B):
        n2 = B.shape[0] * B.shape[1]
        total = 0.0
        for i in range(B.shape[0]):
            for j in range(B.shape[1]):
                total += abs(B[i, j])
        return total / n2

    @njit(fastmath=True)
    def l2_cost(B):
        n2 = B.shape[0] * B.shape[1]
        total = 0.0
        for i in range(B.shape[0]):
            for j in range(B.shape[1]):
                total += B[i, j] * B[i, j]
        return total / n2

    return ALPHA, N, T, TAU1, TAU2, benefit, develop, l1_cost, l2_cost


@app.cell(hide_code=True)
def delimit_targets(mo):
    mo.md(
        """
    ## Target phenotypes, inlined from targets.py

    `CLASS_8` is the same 8 canonical phenotypes used throughout this
    project's `v`-sweep notebooks -- indexed by `(m1, m2, m3)` in
    `{A,B}^3` (lexicographic order, `itertools.product("AB", repeat=3)`),
    with `m4` fixed to `A`.
    """
    )
    return


@app.cell
def targets_definitions(N, np):
    import itertools

    MOD_A = [
        np.array([-1, 1, -1, 1], dtype=np.float64),
        np.array([-1, -1, 1, 1], dtype=np.float64),
        np.array([-1, 1, 1, -1], dtype=np.float64),
        np.array([-1, -1, -1, -1], dtype=np.float64),
    ]
    MOD_B = [-m for m in MOD_A]

    def _pattern(states):
        mods = [
            MOD_A[i] if s == "A" else MOD_B[i] for i, s in enumerate(states)
        ]
        return np.concatenate(mods)

    CLASS_8 = np.stack(
        [
            _pattern([m1, m2, m3, "A"])
            for m1, m2, m3 in itertools.product("AB", repeat=3)
        ]
    )
    assert CLASS_8.shape == (8, N)

    # sanity: idx 0/3/6 are this project's original 3-class training set
    # S1/S2/S3 (literal transcription of Kouvaris et al. 2017 Appendix
    # Eq. 2).
    _S1_lit = np.array(
        [-1, 1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1, -1, -1],
        dtype=np.float64,
    )
    _S2_lit = np.array(
        [-1, 1, -1, 1, 1, 1, -1, -1, 1, -1, -1, 1, -1, -1, -1, -1],
        dtype=np.float64,
    )
    _S3_lit = np.array(
        [1, -1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1],
        dtype=np.float64,
    )
    assert np.array_equal(CLASS_8[0], _S1_lit)
    assert np.array_equal(CLASS_8[3], _S2_lit)
    assert np.array_equal(CLASS_8[6], _S3_lit)

    return CLASS_8, MOD_A


@app.cell(hide_code=True)
def delimit_generalisation(mo):
    mo.md(
        """
    ## Generalisation measurement, inlined from generalisation.py
    """
    )
    return


@app.cell
def generalisation_core(MOD_A, np):
    _CANONICAL_MOD4 = MOD_A[3]

    def fold_to_canonical(Pa_batch):
        signs = np.sign(Pa_batch)
        mod4_match = np.all(signs[:, 12:16] == _CANONICAL_MOD4, axis=1)
        return np.where(mod4_match[:, None], Pa_batch, -Pa_batch)

    def chi_squared(counts, M):
        k = counts.shape[0]
        freq = counts / M
        expected = 1.0 / k
        return float(np.sum((freq - expected) ** 2 / expected))

    return chi_squared, fold_to_canonical


@app.cell(hide_code=True)
def delimit_training_set(mo):
    mo.md(
        """
    ## Training set: 3-class vs. 5-class

    `--n-classes` selects which subset of the 8 canonical `CLASS_8`
    phenotypes is used as the training set (cycled round-robin across
    blocks -- see "Training schedule" below):

    | idx | label  | m1 m2 m3 | m1==m2 | canonical | in 3-class | in 5-class |
    |----:|--------|----------|:------:|:---------:|:----------:|:----------:|
    |   0 | class1 | A A A    |   yes  |    S1     |    yes     |    yes     |
    |   1 | class2 | A A B    |   yes  |    --     |    no      |    yes     |
    |   3 | class4 | A B B    |   no   |    S2     |    yes     |    yes     |
    |   6 | class7 | B B A    |   yes  |    S3     |    yes     |    yes     |
    |   7 | class8 | B B B    |   yes  |    --     |    no      |    yes     |

    3-class (`--n-classes 3`) is this project's original training set,
    `{S1, S2, S3}` = `CLASS_8[[0, 3, 6]]`. 5-class (`--n-classes 5`) adds
    `class2` and `class8`, `CLASS_8[[0, 1, 3, 6, 7]]` -- the two
    additional `m1==m2` phenotypes bracketing `S1`/`S3` in `CLASS_8`'s
    index order (`S2`/`class4` is the sole `m1!=m2` member of both
    training sets).
    """
    )
    return


@app.cell
def build_training_set(CLASS_8, n_classes, np):
    TRAIN_IDX_3 = [0, 3, 6]
    TRAIN_IDX_5 = [0, 1, 3, 6, 7]
    assert n_classes in (3, 5), "n_classes must be 3 or 5"
    train_idx = TRAIN_IDX_3 if n_classes == 3 else TRAIN_IDX_5
    training_set = CLASS_8[np.asarray(train_idx)]
    train_class_idx_str = ",".join(str(_i) for _i in train_idx)
    return train_class_idx_str, train_idx, training_set


@app.cell
def show_training_set(n_classes, pd, train_class_idx_str, training_set):
    training_set_df = pd.DataFrame(
        {
            "n_classes": n_classes,
            "train_class_idx": train_class_idx_str,
            "n_training_patterns": training_set.shape[0],
        },
        index=[0],
    )
    training_set_df
    return


@app.cell(hide_code=True)
def delimit_phenotype_measurement(mo):
    mo.md(
        """
    ## Phenotype sampling + classification, inlined from grn_output_masked.py

    Unlike the `v`-sweep notebooks' `classify_by_phenotype_output_masked`,
    there's no "bitflip" bucket here (that existed only to track blip
    targets, which this notebook doesn't use) -- phenotypes are
    classified as one of the 8 canonical `CLASS_8` classes, or "other".
    """
    )
    return


@app.cell
def phenotype_measurement(
    CLASS_8, N, chi_squared, develop, fold_to_canonical, np, qmc
):
    def sample_G(M, seed, n):
        m = max(1, (M - 1).bit_length())
        sampler = qmc.Sobol(d=n, scramble=True, seed=seed)
        unit_cube = sampler.random_base2(m)[:M]
        return 2.0 * unit_cube - 1.0

    def develop_batch(G_batch, B):
        Pa = np.empty_like(G_batch)
        for k in range(G_batch.shape[0]):
            Pa[k] = develop(G_batch[k], B)
        return Pa

    def classify_by_phenotype(Pa_batch):
        Pa_folded = fold_to_canonical(Pa_batch)
        k = CLASS_8.shape[0]
        M = Pa_batch.shape[0]
        signs = np.sign(Pa_folded)
        dots = signs @ CLASS_8.T
        match = dots == N
        assigned = np.full(M, -1, dtype=np.int64)
        still_open = np.ones(M, dtype=bool)
        for col in range(k):
            take = still_open & match[:, col]
            assigned[take] = col
            still_open &= ~take
        assigned[still_open] = k
        counts = np.zeros(k + 1, dtype=np.int64)
        for col in range(k + 1):
            counts[col] = int((assigned == col).sum())

        # chi-squared evenness of the phenotype distribution *within* the
        # "other" bucket (still_open: samples matching none of the 8
        # canonical classes) -- bit-pack each such sample's sign pattern
        # into an integer id, tally occurrences of each distinct pattern
        # actually observed, and score how evenly those occurrences are
        # spread (0 = perfectly even across whatever distinct patterns
        # were observed). NaN when "other" is empty -- nothing to measure.
        other_signs = signs[still_open]
        if other_signs.shape[0] > 0:
            bits = (other_signs > 0).astype(np.int64)
            pattern_ids = bits @ (
                1 << np.arange(bits.shape[1], dtype=np.int64)
            )
            _, pattern_counts = np.unique(pattern_ids, return_counts=True)
            other_chi2 = chi_squared(pattern_counts, other_signs.shape[0])
            n_other_classes = int(pattern_counts.shape[0])
        else:
            other_chi2 = float("nan")
            n_other_classes = 0

        return counts, other_chi2, n_other_classes

    return classify_by_phenotype, develop_batch, sample_G


@app.cell(hide_code=True)
def delimit_edge_mask(mo):
    mo.md(
        """
    ## Edge sparsity ("model size") sweep

    Replaces the earlier `v` (visible-gene count) double-descent axis:
    model size here is controlled by how many of the GRN's `N*N = 256`
    possible regulatory edges (`B` matrix entries, `N=16`) are
    permanently zeroed out, rather than by masking whole genes.
    `--density` in `[0, 1]` sets the target *fraction of edges retained*;
    `n_zero_edges = round((1 - density) * 256)` of the 256 possible edges
    are zeroed. Which specific edges are zeroed is drawn uniformly at
    random, seeded by `--seed` (so it varies per replicate) via a fixed
    random permutation of the 256 edge positions, of which the first
    `n_zero_edges` are zeroed -- for a fixed seed this also means the
    zeroed set only grows (is nested) as density decreases, though
    nothing downstream relies on that property.

    The same `edge_mask` is used for the entire replicate:
    `mutate_edge_masked` (below) re-applies it to `B` after every
    mutation step, so a zeroed edge's weight is pinned at exactly 0 for
    the whole run (rather than just masked at evaluation time) -- which
    also means zeroed edges pay no L1/L2 regularization cost, since
    `l1_cost`/`l2_cost` are computed directly on `B`.
    """
    )
    return


@app.cell
def build_edge_mask(N, density, np, seed):
    n_zero_edges = int(round((1.0 - density) * N * N))
    n_zero_edges = min(max(n_zero_edges, 0), N * N)
    _rng = np.random.default_rng(seed)
    _edge_perm = _rng.permutation(N * N)
    _zeroed_flat = _edge_perm[:n_zero_edges]
    edge_mask = np.ones((N, N), dtype=np.float64)
    edge_mask.reshape(-1)[_zeroed_flat] = 0.0
    return edge_mask, n_zero_edges


@app.cell
def show_edge_mask(N, edge_mask, n_zero_edges, pd):
    edge_mask_df = pd.DataFrame(
        [
            {
                "n_edges_total": N * N,
                "n_zero_edges": n_zero_edges,
                "n_kept_edges": int(edge_mask.sum()),
                "density_actual": float(edge_mask.mean()),
            }
        ]
    )
    edge_mask_df
    return


@app.cell(hide_code=True)
def delimit_edge_masked_model(mo):
    mo.md(
        """
    ## Elastic-net SSWM evolution over edge-masked B, inlined from grn_output_masked.py
    """
    )
    return


@app.cell
def edge_masked_model(
    N,
    benefit,
    builtins,
    chi_squared,
    classify_by_phenotype,
    develop,
    develop_batch,
    l1_cost,
    l2_cost,
    njit,
    np,
    sample_G,
):
    # marimo shadows the builtin `print` within each cell with its own
    # output-capturing wrapper, which numba's nopython mode can't type --
    # capture the real builtins.print here so the njit progress heartbeat
    # below can reference a supported callable via closure.
    _print = builtins.print

    @njit(fastmath=True)
    def fitness_edge_masked_elastic(G, B, S, lam1, lam2, w1, w2):
        Pa = develop(G, B)
        b = benefit(Pa, S)
        c = w1 * lam1 * l1_cost(B) + w2 * lam2 * l2_cost(B)
        return b - c

    @njit
    def mutate_edge_masked(G, B, edge_mask):
        n = G.shape[0]
        Gp = G.copy()
        i = np.random.randint(0, n)
        mu1 = np.random.uniform(-0.1, 0.1)
        gi = Gp[i] + mu1
        if gi < -1.0:
            gi = -1.0
        elif gi > 1.0:
            gi = 1.0
        Gp[i] = gi

        if np.random.random() < (1.0 / 15.0):
            bound = 1.0 / (150.0 * n * n)
            Bp = B + np.random.uniform(-bound, bound, size=(n, n))
            # pin zeroed edges (edge_mask entries == 0) at exactly 0 on
            # every mutation step, not just at evaluation time -- so a
            # zeroed edge's weight never drifts and never accrues L1/L2
            # cost.
            Bp = Bp * edge_mask
        else:
            Bp = B.copy()
        return Gp, Bp

    # nogil=True releases the GIL for the duration of this call (pure
    # nopython numeric code, no Python objects touched), so a plain Python
    # watcher thread in the caller can concurrently poll progress_counts
    # and drain newly-written G_snap/B_snap/B_trace rows to disk as they
    # appear -- this is what makes output progressively durable against a
    # SLURM job timeout, without restructuring this loop into resumable
    # chunks. G_snap/B_snap/B_trace are caller-allocated and filled
    # in-place; progress_counts (shape (2,), int64) publishes how many
    # snapshot/timeseries rows are safe to read -- the count is only
    # bumped *after* the corresponding row is fully written, so a watcher
    # thread that only reads indices below the last-seen count never
    # observes a partial row. numba's print() internally reacquires the
    # GIL for the duration of the call (see numba.cpython.printimpl), so
    # it's safe to call here despite nogil=True.
    @njit(nogil=True)
    def run_sswm_edge_masked_scheduled_traced_elastic(
        G0,
        B0,
        training_set,
        K,
        schedule,
        lam1,
        lam2,
        w1,
        w2,
        edge_mask,
        seed,
        snapshot_blocks,
        timeseries_blocks,
        G_snap,
        B_snap,
        B_trace,
        progress_counts,
    ):
        np.random.seed(seed)
        G = G0.copy()
        B = B0.copy()
        n_blocks = schedule.shape[0]
        total_gens = n_blocks * K

        # snapshot_blocks/timeseries_blocks are sorted-ascending, unique
        # block indices in [0, n_blocks] -- independent point sets, walked
        # via monotonic pointers rather than a fixed stride, so recording
        # cadence can be arbitrarily (non-uniformly) spaced.
        n_snap_pts = snapshot_blocks.shape[0]
        n_ts_pts = timeseries_blocks.shape[0]
        snap_idx = 0
        ts_idx = 0
        snap_ptr = 0
        ts_ptr = 0

        # Additional heartbeat directly from inside the loop, independent
        # of the caller's watcher thread -- numba's nopython mode doesn't
        # support time.time() (only plain print() with comma-separated
        # args), so approximate a ~20s cadence using the benchmarked
        # single-threaded throughput of this loop (~83,000 gens/sec on
        # non-cluster hardware) converted to a generation-count interval.
        # Actual cadence scales with real hardware speed -- this is a
        # "job is still alive" signal, not a precise timer.
        LOG_EVERY_GENS = 1_660_000

        if snap_ptr < n_snap_pts and snapshot_blocks[snap_ptr] == 0:
            G_snap[snap_idx] = G
            B_snap[snap_idx] = B
            snap_idx += 1
            snap_ptr += 1
            progress_counts[0] = snap_idx
        if ts_ptr < n_ts_pts and timeseries_blocks[ts_ptr] == 0:
            B_trace[ts_idx] = B
            ts_idx += 1
            ts_ptr += 1
            progress_counts[1] = ts_idx

        for gen in range(total_gens):
            block = gen // K
            t_idx = schedule[block]
            S = training_set[t_idx]

            f = fitness_edge_masked_elastic(G, B, S, lam1, lam2, w1, w2)
            Gp, Bp = mutate_edge_masked(G, B, edge_mask)
            fp = fitness_edge_masked_elastic(Gp, Bp, S, lam1, lam2, w1, w2)
            if fp > f:
                G = Gp
                B = Bp

            if (gen + 1) % LOG_EVERY_GENS == 0:
                _print(
                    "njit-heartbeat",
                    "seed",
                    seed,
                    "gen",
                    gen + 1,
                    "/",
                    total_gens,
                    "pct",
                    (gen + 1) * 100 // total_gens,
                )

            if (gen + 1) % K == 0:
                completed_block = (gen + 1) // K
                if (
                    snap_ptr < n_snap_pts
                    and completed_block == snapshot_blocks[snap_ptr]
                ):
                    G_snap[snap_idx] = G
                    B_snap[snap_idx] = B
                    snap_idx += 1
                    snap_ptr += 1
                    progress_counts[0] = snap_idx
                if (
                    ts_ptr < n_ts_pts
                    and completed_block == timeseries_blocks[ts_ptr]
                ):
                    B_trace[ts_idx] = B
                    ts_idx += 1
                    ts_ptr += 1
                    progress_counts[1] = ts_idx

        return G, B

    def compute_errors_edge_masked(B, seed, M, train_idx):
        G_batch = sample_G(M, seed, n=N)
        Pa_batch = develop_batch(G_batch, B)
        (
            class_counts,
            other_chi2,
            n_other_classes,
        ) = classify_by_phenotype(Pa_batch)
        train_counts = class_counts[train_idx]
        return (
            chi_squared(train_counts, M),
            chi_squared(class_counts[:8], M),
            class_counts,
            other_chi2,
            n_other_classes,
        )

    return (
        compute_errors_edge_masked,
        fitness_edge_masked_elastic,
        mutate_edge_masked,
        run_sswm_edge_masked_scheduled_traced_elastic,
    )


@app.cell(hide_code=True)
def delimit_model_constants(mo):
    mo.md(
        """
    ## Model constants
    """
    )
    return


@app.cell
def model_constants(os):
    LAM1 = 0.22
    LAM2 = 38.0
    TOTAL_BLOCKS = 3600
    # sampling-over-time density, 20-fold lower than the v-sweep notebooks'
    # 100/1_000 (fewer recorded points per replicate, since this sweep has
    # far more replicates -- 3960 vs. the low hundreds in prior sweeps --
    # and per-replicate output size scales directly with these targets).
    N_SNAPSHOT_TARGET = 5
    N_TIMESERIES_TARGET = 50
    OUTPUT_DIR = "dd_trial_outputs"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return (
        LAM1,
        LAM2,
        N_SNAPSHOT_TARGET,
        N_TIMESERIES_TARGET,
        OUTPUT_DIR,
        TOTAL_BLOCKS,
    )


@app.cell(hide_code=True)
def delimit_schedule(mo):
    mo.md(
        """
    ## Training schedule (no blips)

    No blip environments here (contrast the blip-sweep notebooks in this
    project) -- every block presents one of the `n_classes` actual
    training patterns, cycled round-robin (`block_idx % n_classes`) so
    each pattern gets exactly `TOTAL_BLOCKS / n_classes` blocks.
    `TOTAL_BLOCKS` (3600, fixed above) is divisible by both supported
    `n_classes` values (3 and 5), so the split is always exactly even.
    """
    )
    return


@app.cell
def build_round_robin_schedule(TOTAL_BLOCKS, n_classes, np):
    assert TOTAL_BLOCKS % n_classes == 0
    schedule = np.arange(TOTAL_BLOCKS, dtype=np.int64) % n_classes
    return (schedule,)


@app.cell(hide_code=True)
def delimit_timepoints(mo):
    mo.md(
        """
    ## Timepoint sampling (snapshots + timeseries)

    Snapshots (full G/B matrices, persisted to `.npz`) and timeseries rows
    (train/test chi2 + walltime, persisted to `.pqt`) are sampled
    independently over the block domain `[0, TOTAL_BLOCKS]`: each set is
    the union of an evenly-spaced and a log-spaced sample at its target
    count, plus the immediate successor of every sampled point (so
    consecutive-block deltas are always available). A domain smaller than
    the requested count degrades gracefully to the full domain rather than
    erroring or duplicating points past what exists.
    """
    )
    return


@app.cell
def sample_timepoints_fn(np):
    def sample_timepoints(domain_max, n_target):
        # Sorted unique block indices in [0, domain_max]: union of an
        # evenly-spaced and a log-spaced sample (each capped at
        # min(n_target, domain_max + 1) points -- a domain smaller than
        # n_target degrades gracefully to (a subset of) the full domain
        # instead of erroring or padding with out-of-range duplicates),
        # plus the immediate successor of every sampled point.
        domain_max = int(domain_max)
        n = max(1, min(int(n_target), domain_max + 1))
        even = np.linspace(0, domain_max, n).round().astype(np.int64)
        if domain_max >= 1 and n >= 2:
            log = (
                (np.geomspace(1, domain_max + 1, n) - 1)
                .round()
                .astype(np.int64)
            )
        else:
            log = np.zeros(n, dtype=np.int64)
        union = np.unique(
            np.clip(
                np.concatenate([even, log, [0, domain_max]]), 0, domain_max
            )
        )
        with_next = np.unique(
            np.clip(np.concatenate([union, union + 1]), 0, domain_max)
        )
        return with_next.astype(np.int64)

    return (sample_timepoints,)


@app.cell
def build_timepoints(
    N_SNAPSHOT_TARGET,
    N_TIMESERIES_TARGET,
    TOTAL_BLOCKS,
    sample_timepoints,
):
    SNAPSHOT_BLOCKS = sample_timepoints(TOTAL_BLOCKS, N_SNAPSHOT_TARGET)
    TIMESERIES_BLOCKS = sample_timepoints(TOTAL_BLOCKS, N_TIMESERIES_TARGET)
    return SNAPSHOT_BLOCKS, TIMESERIES_BLOCKS


@app.cell
def show_timepoint_counts(SNAPSHOT_BLOCKS, TIMESERIES_BLOCKS, pd):
    timepoint_counts_df = pd.DataFrame(
        [
            {
                "n_snapshot_blocks": len(SNAPSHOT_BLOCKS),
                "n_timeseries_blocks": len(TIMESERIES_BLOCKS),
            }
        ]
    )
    timepoint_counts_df
    return


@app.cell(hide_code=True)
def delimit_progressive_io(mo):
    mo.md(
        """
    ## Progressive output helpers

    G/B snapshots are appended to `.npz` files one array at a time (rather
    than written once via `np.savez_compressed` at the end) using the same
    zip-of-`.npy`-members format `np.savez` produces internally, just
    opened in append mode per write. Timeseries rows are appended to a
    `.csv` (rather than buffered into a DataFrame and written once as
    `.pqt`) since CSV supports genuine line-at-a-time appends. Both let a
    replicate's output survive a SLURM job timeout with only whatever was
    written between the last append and the kill lost, instead of losing
    the entire run.
    """
    )
    return


@app.cell
def progressive_io_helpers(np, pd, zipfile):
    from numpy.lib import format as npy_format

    def append_npz_array(path, key, arr, first):
        mode = "w" if first else "a"
        with zipfile.ZipFile(
            path, mode=mode, compression=zipfile.ZIP_DEFLATED
        ) as zf:
            with zf.open(f"{key}.npy", "w", force_zip64=True) as f:
                npy_format.write_array(f, np.asarray(arr))

    def append_csv_row(path, row, columns, first):
        pd.DataFrame([row], columns=columns).to_csv(
            path, mode="w" if first else "a", header=first, index=False
        )

    return append_csv_row, append_npz_array


@app.cell(hide_code=True)
def delimit_run_trial(mo):
    mo.md(
        """
    ## Run trial
    """
    )
    return


@app.cell
def run_trial(
    LAM1,
    LAM2,
    N,
    OUTPUT_DIR,
    SNAPSHOT_BLOCKS,
    TIMESERIES_BLOCKS,
    append_csv_row,
    append_npz_array,
    compute_errors_edge_masked,
    contextlib,
    density,
    edge_mask,
    kn,
    l1_cost,
    l1_scale,
    l2_cost,
    l2_scale,
    n_classes,
    n_zero_edges,
    np,
    num_epoch,
    pd,
    run_sswm_edge_masked_scheduled_traced_elastic,
    schedule,
    seed,
    sys,
    threading,
    time,
    train_class_idx_str,
    train_idx,
    training_set,
    uuid,
):
    _t0 = time.time()
    _w1 = l1_scale
    _w2 = l2_scale
    _seed = seed
    _K = num_epoch
    replicate_uid = str(uuid.uuid4())

    _G0 = np.zeros(N)
    _B0 = np.zeros((N, N))

    # M for the (dense, ~thousands of points) per-timepoint trace calls --
    # kept small relative to FINAL_M below since it's paid many times over.
    _TRACE_M = 2000

    _n_snap = SNAPSHOT_BLOCKS.shape[0]
    _n_ts = TIMESERIES_BLOCKS.shape[0]
    _snapshot_gens = np.asarray(SNAPSHOT_BLOCKS, dtype=np.int64) * _K
    trace_gens = np.asarray(TIMESERIES_BLOCKS, dtype=np.int64) * _K

    # G_snap/B_snap/B_trace are allocated here (rather than inside the
    # njit call) and filled in-place by it, so the watcher thread below
    # can read newly-completed rows out of the same arrays concurrently.
    _G_snap = np.empty((_n_snap, N))
    _B_snap = np.empty((_n_snap, N, N))
    _B_trace = np.empty((_n_ts, N, N))
    # progress_counts[0]/[1] publish how many snapshot/timeseries rows the
    # njit call has fully written -- see the nogil comment on the njit
    # function itself for the publish-count-last safety argument.
    _progress_counts = np.zeros(2, dtype=np.int64)

    # --- output paths, self-describing via keyname.pack (every run option
    # as a key=value segment). Timeseries rows are written progressively
    # to CSV as each row becomes available during the run (CSV supports
    # genuine line-at-a-time appends, unlike parquet), so a SLURM job
    # timeout only loses whatever was written since the last append
    # instead of the entire run's output; G/B snapshots are likewise
    # appended to .npz files one array at a time. Per-replicate output
    # stays CSV -- the downstream collation step converts to parquet
    # itself (joinem infers CSV input / parquet output from file
    # extensions) once all replicates' timeseries are joined into one
    # frame, rather than every replicate separately producing its own
    # parquet file only to be re-read and re-written at collation time.
    _run_params = {
        "density": density,
        "nzeroedges": n_zero_edges,
        "nclasses": n_classes,
        "seed": _seed,
        "l1scale": _w1,
        "l2scale": _w2,
        "numepoch": _K,
        "replicate": replicate_uid,
    }
    timeseries_path = f"{OUTPUT_DIR}/{kn.pack({**_run_params, 'ext': '.csv'})}"
    G_snapshots_path = (
        f"{OUTPUT_DIR}/{kn.pack({**_run_params, 'what': 'G', 'ext': '.npz'})}"
    )
    B_snapshots_path = (
        f"{OUTPUT_DIR}/{kn.pack({**_run_params, 'what': 'B', 'ext': '.npz'})}"
    )

    # Per-class fraction columns (share of _TRACE_M samples landing exactly
    # on each phenotype): test1_frac..test8_frac are the 8 CLASS_8 classes
    # in order -- which of these 8 are actually "training" classes depends
    # on n_classes/train_class_idx (see build_training_set above), so
    # unlike the v-sweep notebooks there's no separate, fixed-width
    # train{1..3}_frac duplication here (it would have a different width
    # for n_classes=3 vs. 5, breaking schema uniformity at collation time).
    _trace_columns = (
        [
            "epoch",
            "generation",
            "walltime_sec",
            "pure_train_chi2",
            "test_chi2",
        ]
        + [f"test{_j + 1}_frac" for _j in range(8)]
        + [
            "other_frac",
            "other_chi2",
            "other_n_classes",
            "l1_loss",
            "l2_loss",
            "regularization_loss",
            "density",
            "n_zero_edges",
            "n_classes",
            "seed",
            "l1_scale",
            "l2_scale",
            "num_epoch",
            "train_class_idx",
            "replicate_uid",
        ]
    )

    def _trace_row(_i, _t_evo_start):
        # L1/L2/regularization loss depend only on B (not on sampled
        # genotypes). l1_loss/l2_loss are the raw, unweighted per-entry
        # -mean costs -- recorded regardless of how l1_scale/l2_scale are
        # configured for this replicate -- while regularization_loss is
        # the actual weighted penalty subtracted from benefit in the
        # fitness function driving evolution.
        (_ptr, _te, _cc, _oc, _noc,) = compute_errors_edge_masked(
            _B_trace[_i], _seed + 2000, _TRACE_M, train_idx
        )
        _l1 = l1_cost(_B_trace[_i])
        _l2 = l2_cost(_B_trace[_i])
        _row = {
            "epoch": _i,
            "generation": int(trace_gens[_i]),
            # each row's timestamp is measured live as the watcher thread
            # observes it (rather than interpolated after the fact),
            # since the njit call runs concurrently (nogil) with this
            # thread instead of as one opaque, unobservable block.
            "walltime_sec": time.time() - _t_evo_start,
            "pure_train_chi2": float(_ptr),
            "test_chi2": float(_te),
        }
        for _j in range(8):
            _row[f"test{_j + 1}_frac"] = float(_cc[_j]) / _TRACE_M
        _row["other_frac"] = float(_cc[8]) / _TRACE_M
        _row["other_chi2"] = float(_oc)
        _row["other_n_classes"] = int(_noc)
        _row["l1_loss"] = float(_l1)
        _row["l2_loss"] = float(_l2)
        _row["regularization_loss"] = float(
            _w1 * LAM1 * _l1 + _w2 * LAM2 * _l2
        )
        _row["density"] = density
        _row["n_zero_edges"] = n_zero_edges
        _row["n_classes"] = n_classes
        _row["seed"] = _seed
        _row["l1_scale"] = _w1
        _row["l2_scale"] = _w2
        _row["num_epoch"] = _K
        _row["train_class_idx"] = train_class_idx_str
        _row["replicate_uid"] = replicate_uid
        return _row

    _trace_rows = []
    _stop_event = threading.Event()
    _watcher_exc = [None]
    _log_start = time.time()
    # Shared, mutable drain progress -- a dict (rather than locals closed
    # over by _watcher alone) so both the watcher thread's periodic polls
    # and the guaranteed final drain call after it exits (see below) can
    # resume from the same cursor.
    _drain_state = {
        "last_snap": 0,
        "last_ts": 0,
        "first_g": True,
        "first_b": True,
        "first_csv": True,
    }

    def _drain(_t_evo_start, _label):
        # Reads progress_counts (published by the nogil njit call running
        # concurrently on the main thread) and, if it has advanced,
        # writes the newly-available snapshot/timeseries rows straight to
        # disk. Every call logs the attempt; a second line logs the save
        # only when one actually happens, so "checked, nothing new yet"
        # and "wrote N rows" are distinguishable in the log.
        _snap_n = int(_progress_counts[0])
        _ts_n = int(_progress_counts[1])
        print(
            "poll-attempt",
            replicate_uid,
            "label",
            _label,
            "snap",
            _snap_n,
            "/",
            _n_snap,
            "ts",
            _ts_n,
            "/",
            _n_ts,
            "elapsed_sec",
            int(time.time() - _log_start),
        )
        if (
            _snap_n <= _drain_state["last_snap"]
            and _ts_n <= _drain_state["last_ts"]
        ):
            return
        # deliberate pause between observing that new rows are ready and
        # actually reading + writing them, as an extra margin of safety
        # on top of the publish-count-last ordering guarantee.
        time.sleep(1.0)
        _wrote_snap = 0
        while _drain_state["last_snap"] < _snap_n:
            _i = _drain_state["last_snap"]
            _key = f"gen{int(_snapshot_gens[_i]):012d}"
            append_npz_array(
                G_snapshots_path, _key, _G_snap[_i], _drain_state["first_g"]
            )
            append_npz_array(
                B_snapshots_path, _key, _B_snap[_i], _drain_state["first_b"]
            )
            _drain_state["first_g"] = False
            _drain_state["first_b"] = False
            _drain_state["last_snap"] += 1
            _wrote_snap += 1
        _wrote_ts = 0
        while _drain_state["last_ts"] < _ts_n:
            _i = _drain_state["last_ts"]
            _row = _trace_row(_i, _t_evo_start)
            _trace_rows.append(_row)
            append_csv_row(
                timeseries_path,
                _row,
                _trace_columns,
                _drain_state["first_csv"],
            )
            _drain_state["first_csv"] = False
            _drain_state["last_ts"] += 1
            _wrote_ts += 1
        print(
            "saved",
            replicate_uid,
            "label",
            _label,
            "wrote_snap",
            _wrote_snap,
            "wrote_ts",
            _wrote_ts,
        )

    def _watcher(_t_evo_start):
        # Checks for new data every ~60s -- marimo's per-cell print
        # capture would otherwise swallow this thread's logging, so the
        # caller redirects stdout to sys.__stdout__ for the duration of
        # this thread's lifetime.
        try:
            while True:
                _stopped = _stop_event.wait(60.0)
                _drain(_t_evo_start, "watcher")
                if _stopped:
                    break
        except Exception as _e:  # noqa: BLE001 -- re-raised after join
            _watcher_exc[0] = _e

    # marimo captures each cell's stdout internally (buffering it into the
    # exported notebook rather than passing it through to the real
    # terminal), so the print()-based logging above wouldn't otherwise be
    # visible while a long SLURM job is running. Redirecting to
    # sys.__stdout__ (the original stream, saved by Python at process
    # startup, before marimo's capture takes over) sidesteps that.
    _t_evo_start = time.time()
    with contextlib.redirect_stdout(sys.__stdout__):
        _thread = threading.Thread(
            target=_watcher, args=(_t_evo_start,), daemon=True
        )
        _thread.start()
        try:
            _G, _B = run_sswm_edge_masked_scheduled_traced_elastic(
                _G0,
                _B0,
                training_set,
                _K,
                schedule,
                LAM1,
                LAM2,
                _w1,
                _w2,
                edge_mask,
                _seed,
                SNAPSHOT_BLOCKS,
                TIMESERIES_BLOCKS,
                _G_snap,
                _B_snap,
                _B_trace,
                _progress_counts,
            )
        finally:
            _stop_event.set()
            _thread.join()
        if _watcher_exc[0] is not None:
            raise _watcher_exc[0]
        # The watcher's own final iteration (triggered by _stop_event
        # above) already drains everything, but calling _drain() once
        # more here -- synchronously, on the main thread, independent of
        # the watcher thread's internal loop timing -- guarantees the
        # complete run's data is on disk before this cell returns rather
        # than relying solely on that thread's internal control flow.
        # By this point _run_sswm has already returned, so there's
        # nothing left to become available and this is a cheap no-op.
        _drain(_t_evo_start, "final")

    FINAL_M = 100_000
    (
        pure_train_chi2,
        test_chi2,
        final_class_counts,
        final_other_chi2,
        final_n_other_classes,
    ) = compute_errors_edge_masked(_B, _seed + 1000, FINAL_M, train_idx)
    assert final_class_counts.sum() == FINAL_M
    final_l1_loss = l1_cost(_B)
    final_l2_loss = l2_cost(_B)
    final_reg_loss = _w1 * LAM1 * final_l1_loss + _w2 * LAM2 * final_l2_loss

    elapsed_sec = time.time() - _t0

    trace_df = pd.DataFrame(_trace_rows, columns=_trace_columns)
    trace_df = trace_df.astype(
        {
            "epoch": np.uint16,
            "generation": np.uint32,
            "walltime_sec": np.float32,
            "pure_train_chi2": np.float32,
            "test_chi2": np.float32,
            **{f"test{_j + 1}_frac": np.float32 for _j in range(8)},
            "other_frac": np.float32,
            "other_chi2": np.float32,
            "other_n_classes": np.uint32,
            "l1_loss": np.float32,
            "l2_loss": np.float32,
            "regularization_loss": np.float32,
            "density": np.float32,
            "n_zero_edges": np.uint16,
            "n_classes": np.uint8,
            "seed": np.uint32,
            "l1_scale": np.float32,
            "l2_scale": np.float32,
            "num_epoch": np.uint32,
        }
    )
    trace_df["train_class_idx"] = pd.Categorical(trace_df["train_class_idx"])
    trace_df["replicate_uid"] = pd.Categorical(trace_df["replicate_uid"])

    _final_test_fracs = {
        f"test{_j + 1}_frac": float(final_class_counts[_j] / FINAL_M)
        for _j in range(8)
    }

    result = {
        "density": density,
        "n_zero_edges": n_zero_edges,
        "n_classes": n_classes,
        "seed": _seed,
        "num_epoch": _K,
        "l1_scale": _w1,
        "l2_scale": _w2,
        "train_class_idx": train_class_idx_str,
        "pure_train_chi2": pure_train_chi2,
        "test_chi2": test_chi2,
        **_final_test_fracs,
        "other_frac": float(final_class_counts[8] / FINAL_M),
        "other_chi2": final_other_chi2,
        "other_n_classes": final_n_other_classes,
        "l1_loss": final_l1_loss,
        "l2_loss": final_l2_loss,
        "regularization_loss": final_reg_loss,
        "elapsed_sec": elapsed_sec,
        "replicate_uid": replicate_uid,
        "timeseries_path": timeseries_path,
        "G_snapshots_path": G_snapshots_path,
        "B_snapshots_path": B_snapshots_path,
    }
    return result, trace_df


@app.cell(hide_code=True)
def delimit_show_result(mo):
    mo.md(
        """
    ## Result
    """
    )
    return


@app.cell
def show_result(pd, result):
    result_df = pd.DataFrame([result])
    result_df
    return


@app.cell
def show_timeseries_peek(pd, trace_df):
    pd.concat([trace_df.head(), trace_df.tail()])
    return


@app.cell
def show_output_files(os, pd, result):
    _paths = [
        result["timeseries_path"],
        result["G_snapshots_path"],
        result["B_snapshots_path"],
    ]
    output_files_df = pd.DataFrame(
        {
            "file": _paths,
            "size_kb": [round(os.path.getsize(_p) / 1024, 1) for _p in _paths],
        }
    )
    output_files_df
    return


if __name__ == "__main__":
    app.run()
