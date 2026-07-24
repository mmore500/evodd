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
    # Single-trial elastic-net GRN run (self-contained, parameterizable)

    One trial = one (v, L1 scale, L2 scale, seed, zero-init, blip freq,
    num_epoch) combination from `run_output_masked_k100k_dd_heatmap.py`'s
    v1..v20 double-descent batch sweep, pulled out so a single condition can
    be inspected/tuned without launching the full batch job. Self-contained:
    every model definition (Kouvaris et al. 2017 GRN core, target patterns,
    masked development, elastic-net SSWM) is inlined below rather than
    imported from the surrounding project, so this file has no dependency on
    any other file in this repository -- only installable packages
    (marimo, numpy, pandas, scipy, numba, keyname, watermark).

    Uses the unified 20-gene model (16 standard visible genes + 4 extra
    hidden genes) -- v1..v16 unmask the standard genes in a fixed
    interleaved order, v17..v20 progressively unmask the 4 extra genes.

    CLI-parameterizable, no interactive widgets: every value below reads its
    default from `mo.cli_args()`, so running `marimo run dd_single_trial.py
    -- --seed 21 --v 12 --zero-init true --l1-scale 0.9 --l2-scale 0.1
    --blip-freq 0.5 --num-epoch 50000` sets all of them, and the trial runs
    immediately (no button to click).
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

    seed = _get("seed", 20, int)
    v_label = _get("v", 16, int)
    zero_init = _get(
        "zero-init", False, lambda s: str(s).lower() in ("1", "true", "yes")
    )
    l1_scale = _get("l1-scale", 0.995, float)
    l2_scale = _get("l2-scale", 0.005, float)
    blip_freq = _get("blip-freq", 0.66, float)
    num_epoch = _get("num-epoch", 100, int)
    # schedule_mode controls how ties are broken in the environment-order
    # schedule (see weighted_interleave_fn below): "none" (deterministic,
    # default), "local" (randomize among equally-due environments), or
    # "global" (fully randomize the whole schedule's order).
    schedule_mode = _get("schedule-mode", "none", lambda s: str(s).lower())
    return (
        blip_freq,
        l1_scale,
        l2_scale,
        num_epoch,
        schedule_mode,
        seed,
        v_label,
        zero_init,
    )


@app.cell
def show_config(
    blip_freq,
    l1_scale,
    l2_scale,
    num_epoch,
    pd,
    schedule_mode,
    seed,
    v_label,
    zero_init,
):
    config_df = pd.DataFrame(
        [
            {
                "v": v_label,
                "seed": seed,
                "zero_init": zero_init,
                "l1_scale": l1_scale,
                "l2_scale": l2_scale,
                "blip_freq": blip_freq,
                "num_epoch": num_epoch,
                "schedule_mode": schedule_mode,
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

    @njit
    def mutate(G, B):
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
        else:
            Bp = B.copy()
        return Gp, Bp

    return N, benefit, develop, l1_cost, l2_cost, mutate


@app.cell(hide_code=True)
def delimit_targets(mo):
    mo.md(
        """
    ## Target phenotypes, inlined from targets.py
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

    S1 = _pattern(["A", "A", "A", "A"])
    S2 = _pattern(["A", "B", "B", "A"])
    S3 = _pattern(["B", "B", "A", "A"])
    TRAINING_SET = np.stack([S1, S2, S3])

    # sanity: verify against literal transcription of S1 Appendix Eq. 2
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
    assert np.array_equal(S1, _S1_lit)
    assert np.array_equal(S2, _S2_lit)
    assert np.array_equal(S3, _S3_lit)

    CLASS_8 = np.stack(
        [
            _pattern([m1, m2, m3, "A"])
            for m1, m2, m3 in itertools.product("AB", repeat=3)
        ]
    )
    assert CLASS_8.shape == (8, N)
    for _s in TRAINING_SET:
        assert any(np.array_equal(_s, _c) for _c in CLASS_8)
    return CLASS_8, MOD_A, TRAINING_SET


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

    def classify_exact_counts(Pa_batch, candidates):
        signs = np.sign(Pa_batch)
        dots = signs @ candidates.T
        n = candidates.shape[1]
        return (dots == n).sum(axis=0)

    def chi_squared(counts, M):
        k = counts.shape[0]
        freq = counts / M
        expected = 1.0 / k
        return float(np.sum((freq - expected) ** 2 / expected))

    return chi_squared, classify_exact_counts, fold_to_canonical


@app.cell(hide_code=True)
def delimit_masked_model(mo):
    mo.md(
        """
    ## Masked development + zero-init generalization, inlined from grn_output_masked.py
    """
    )
    return


@app.cell
def masked_model_core(
    CLASS_8, N, chi_squared, develop, fold_to_canonical, njit, np, qmc
):
    def make_visible_mask(v, n, perm):
        mask = np.zeros(n, dtype=np.bool_)
        mask[perm[:v]] = True
        return mask

    @njit(fastmath=True)
    def develop_output_masked(G, B, visible_mask):
        n = G.shape[0]
        B_eff = B.copy()
        for i in range(n):
            if not visible_mask[i]:
                for j in range(n):
                    B_eff[j, i] = 0.0
        return develop(G, B_eff)

    @njit(fastmath=True)
    def develop_output_masked_zero_masked(G, B, visible_mask):
        n = G.shape[0]
        G_eff = G.copy()
        for i in range(n):
            if not visible_mask[i]:
                G_eff[i] = 0.0
        return develop_output_masked(G_eff, B, visible_mask)

    def sample_G(M, seed, n):
        m = max(1, (M - 1).bit_length())
        sampler = qmc.Sobol(d=n, scramble=True, seed=seed)
        unit_cube = sampler.random_base2(m)[:M]
        return 2.0 * unit_cube - 1.0

    def develop_batch_output_masked(G_batch, B, visible_mask):
        Pa = np.empty_like(G_batch)
        for k in range(G_batch.shape[0]):
            Pa[k] = develop_output_masked(G_batch[k], B, visible_mask)
        return Pa

    def classify_by_phenotype_output_masked(Pa_batch):
        Pa_folded = fold_to_canonical(Pa_batch)
        k = CLASS_8.shape[0]
        M = Pa_batch.shape[0]
        signs = np.sign(Pa_folded)
        dots = signs @ CLASS_8.T
        match = dots == N
        train_cols = [0, 3, 6]
        other_cols = [1, 2, 4, 5, 7]
        assigned = np.full(M, -1, dtype=np.int64)
        still_open = np.ones(M, dtype=bool)
        for col in train_cols + other_cols:
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

    return (
        classify_by_phenotype_output_masked,
        develop_batch_output_masked,
        develop_output_masked,
        develop_output_masked_zero_masked,
        make_visible_mask,
        sample_G,
    )


@app.cell
def masked_model_ext_elastic(
    BLIP_SET,
    benefit,
    builtins,
    chi_squared,
    classify_by_phenotype_output_masked,
    classify_exact_counts,
    develop_batch_output_masked,
    develop_output_masked,
    fold_to_canonical,
    l1_cost,
    l2_cost,
    mutate,
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
    def fitness_output_masked_ext_elastic(
        G, B, S, lam1, lam2, w1, w2, visible_mask, n_score
    ):
        Pa = develop_output_masked(G, B, visible_mask)
        b = benefit(Pa[:n_score], S)
        c = w1 * lam1 * l1_cost(B) + w2 * lam2 * l2_cost(B)
        return b - c

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
    def run_sswm_output_masked_scheduled_traced_ext_elastic(
        G0,
        B0,
        training_set,
        K,
        schedule,
        lam1,
        lam2,
        w1,
        w2,
        visible_mask,
        seed,
        snapshot_blocks,
        timeseries_blocks,
        n_score,
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

            f = fitness_output_masked_ext_elastic(
                G, B, S, lam1, lam2, w1, w2, visible_mask, n_score
            )
            Gp, Bp = mutate(G, B)
            fp = fitness_output_masked_ext_elastic(
                Gp, Bp, S, lam1, lam2, w1, w2, visible_mask, n_score
            )
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

    def compute_errors_output_masked_ext(
        B, visible_mask, seed, n_score, M=100_000
    ):
        n_total = B.shape[0]
        G_batch = sample_G(M, seed, n=n_total)
        Pa_batch = develop_batch_output_masked(G_batch, B, visible_mask)
        Pa_scored = Pa_batch[:, :n_score]
        Pa_folded = fold_to_canonical(Pa_scored)
        # class_counts is length 9: CLASS_8 classes 1..8 (indices 0..7,
        # order-preserving -- train_cols [0, 3, 6] are S1/S2/S3) plus
        # "other" (index 8, matches none of the 8 canonical classes).
        (
            class_counts,
            other_chi2,
            n_other_classes,
        ) = classify_by_phenotype_output_masked(Pa_scored)
        train_counts = class_counts[[0, 3, 6]]
        blip_counts = classify_exact_counts(Pa_folded, BLIP_SET)
        return (
            chi_squared(train_counts, M),
            chi_squared(class_counts[:8], M),
            chi_squared(blip_counts, M),
            class_counts,
            blip_counts,
            other_chi2,
            n_other_classes,
        )

    return (
        compute_errors_output_masked_ext,
        run_sswm_output_masked_scheduled_traced_ext_elastic,
    )


@app.cell
def masked_model_zero_masked_elastic(
    BLIP_SET,
    benefit,
    builtins,
    chi_squared,
    classify_by_phenotype_output_masked,
    classify_exact_counts,
    develop_batch_output_masked,
    develop_output_masked_zero_masked,
    fold_to_canonical,
    l1_cost,
    l2_cost,
    mutate,
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
    def fitness_output_masked_zero_masked_elastic(
        G, B, S, lam1, lam2, w1, w2, visible_mask, n_score
    ):
        Pa = develop_output_masked_zero_masked(G, B, visible_mask)
        b = benefit(Pa[:n_score], S)
        c = w1 * lam1 * l1_cost(B) + w2 * lam2 * l2_cost(B)
        return b - c

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
    def run_sswm_output_masked_scheduled_traced_zero_masked_elastic(
        G0,
        B0,
        training_set,
        K,
        schedule,
        lam1,
        lam2,
        w1,
        w2,
        visible_mask,
        seed,
        snapshot_blocks,
        timeseries_blocks,
        n_score,
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

            f = fitness_output_masked_zero_masked_elastic(
                G, B, S, lam1, lam2, w1, w2, visible_mask, n_score
            )
            Gp, Bp = mutate(G, B)
            fp = fitness_output_masked_zero_masked_elastic(
                Gp, Bp, S, lam1, lam2, w1, w2, visible_mask, n_score
            )
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

    def compute_errors_output_masked_zero_masked(
        B, visible_mask, seed, n_score, M=100_000
    ):
        n_total = B.shape[0]
        G_batch = sample_G(M, seed, n=n_total)
        G_batch = G_batch.copy()
        G_batch[:, ~visible_mask] = 0.0
        Pa_batch = develop_batch_output_masked(G_batch, B, visible_mask)
        Pa_scored = Pa_batch[:, :n_score]
        Pa_folded = fold_to_canonical(Pa_scored)
        # class_counts is length 9: CLASS_8 classes 1..8 (indices 0..7,
        # order-preserving -- train_cols [0, 3, 6] are S1/S2/S3) plus
        # "other" (index 8, matches none of the 8 canonical classes).
        (
            class_counts,
            other_chi2,
            n_other_classes,
        ) = classify_by_phenotype_output_masked(Pa_scored)
        train_counts = class_counts[[0, 3, 6]]
        blip_counts = classify_exact_counts(Pa_folded, BLIP_SET)
        return (
            chi_squared(train_counts, M),
            chi_squared(class_counts[:8], M),
            chi_squared(blip_counts, M),
            class_counts,
            blip_counts,
            other_chi2,
            n_other_classes,
        )

    return (
        compute_errors_output_masked_zero_masked,
        run_sswm_output_masked_scheduled_traced_zero_masked_elastic,
    )


@app.cell(hide_code=True)
def delimit_model_constants(mo):
    mo.md(
        """
    ## Model constants and blip schedule
    """
    )
    return


@app.cell
def weighted_interleave_fn(np):
    def weighted_interleave(counts, schedule_mode="none", rng=None):
        # Greedy fair-queueing schedule: at each step, whichever
        # environment is furthest behind its target proportion
        # (counts[i] / total) goes next. schedule_mode controls how ties
        # for "furthest behind" are broken -- ties are common here since
        # e.g. all 3 true patterns share one count and all 3 blip patterns
        # share another, so at any given step several environments are
        # often equally due:
        #   "none"   -- deterministic: always break toward the lowest
        #               environment index (original behavior).
        #   "local"  -- draw uniformly at random (via rng) among *all*
        #               environments currently tied for furthest-behind,
        #               instead of always the lowest index. This
        #               randomizes presentation order within each natural
        #               batch of equally-due environments while leaving
        #               the overall per-environment counts/frequencies
        #               exactly as requested.
        #   "global" -- build the "none"-order schedule, then apply one
        #               full random permutation (via rng) across the
        #               entire sequence, discarding all local structure.
        total = sum(counts)
        seq = np.empty(total, dtype=np.int64)
        appeared = [0] * len(counts)
        for step in range(total):
            deficits = np.asarray(
                [
                    (counts[i] / total) * (step + 1) - appeared[i]
                    for i in range(len(counts))
                ]
            )
            if schedule_mode == "local":
                eligible = np.flatnonzero(
                    np.isclose(deficits, deficits.max(), atol=1e-9)
                )
                i = int(rng.choice(eligible))
            else:
                i = int(np.argmax(deficits))
            seq[step] = i
            appeared[i] += 1
        if schedule_mode == "global":
            rng.shuffle(seq)
        return seq

    return (weighted_interleave,)


@app.cell
def model_constants(np, os):
    LAM1 = 0.22
    LAM2 = 38.0
    N_SCORE = 16
    N_TOTAL = 20
    TOTAL_BLOCKS = 3600
    N_SNAPSHOT_TARGET = 100
    N_TIMESERIES_TARGET = 10_000
    OUTPUT_DIR = "dd_trial_outputs"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # standard 16-gene interleaved order, then the 4 extra hidden genes appended --
    # v<=16 unmasks exactly this prefix, v17..v20 progressively adds the extras.
    INTERLEAVED_PERM20 = np.array(
        [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 16, 17, 18, 19],
        dtype=np.int64,
    )
    return (
        INTERLEAVED_PERM20,
        LAM1,
        LAM2,
        N_SCORE,
        N_SNAPSHOT_TARGET,
        N_TIMESERIES_TARGET,
        N_TOTAL,
        OUTPUT_DIR,
        TOTAL_BLOCKS,
    )


@app.cell
def build_schedule(
    TOTAL_BLOCKS,
    TRAINING_SET,
    blip_freq,
    np,
    schedule_mode,
    seed,
    weighted_interleave,
):
    def make_blips():
        S1b = TRAINING_SET[0].copy()
        S1b[0] *= -1
        S2b = TRAINING_SET[1].copy()
        S2b[4] *= -1
        S3b = TRAINING_SET[2].copy()
        S3b[8] *= -1
        return S1b, S2b, S3b

    S1b, S2b, S3b = make_blips()
    training_set = np.vstack([TRAINING_SET.astype(np.float64), S1b, S2b, S3b])
    BLIP_SET = np.stack([S1b, S2b, S3b])

    # blip_freq = blip:true pattern-count ratio (0.66 reproduces this project's
    # established "blip66" condition: [723,723,723,477,477,477]). 3 equal true
    # counts + 3 equal blip counts, rounded to integers, remainder folded into
    # the first true count so the total still hits TOTAL_BLOCKS exactly.
    _f = blip_freq
    _true_ct = round(TOTAL_BLOCKS / (3 * (1 + _f)))
    _blip_ct = round(_true_ct * _f)
    blip_counts = [_true_ct, _true_ct, _true_ct, _blip_ct, _blip_ct, _blip_ct]
    blip_counts[0] += TOTAL_BLOCKS - sum(blip_counts)
    assert sum(blip_counts) == TOTAL_BLOCKS

    assert schedule_mode in ("none", "local", "global")
    _rng = np.random.default_rng(seed)
    schedule = weighted_interleave(
        blip_counts, schedule_mode=schedule_mode, rng=_rng
    )
    assert schedule.shape[0] == TOTAL_BLOCKS
    return BLIP_SET, S1b, S2b, S3b, blip_counts, schedule, training_set


@app.cell
def show_blip_counts(blip_counts, pd):
    blip_counts_df = pd.DataFrame(
        [
            {
                "s1_true": blip_counts[0],
                "s2_true": blip_counts[1],
                "s3_true": blip_counts[2],
                "s1_blip": blip_counts[3],
                "s2_blip": blip_counts[4],
                "s3_blip": blip_counts[5],
            }
        ]
    )
    blip_counts_df
    return


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
    INTERLEAVED_PERM20,
    LAM1,
    LAM2,
    N_SCORE,
    N_TOTAL,
    OUTPUT_DIR,
    SNAPSHOT_BLOCKS,
    TIMESERIES_BLOCKS,
    append_csv_row,
    append_npz_array,
    blip_freq,
    compute_errors_output_masked_ext,
    compute_errors_output_masked_zero_masked,
    contextlib,
    kn,
    l1_cost,
    l1_scale,
    l2_cost,
    l2_scale,
    make_visible_mask,
    np,
    num_epoch,
    pd,
    run_sswm_output_masked_scheduled_traced_ext_elastic,
    run_sswm_output_masked_scheduled_traced_zero_masked_elastic,
    schedule,
    schedule_mode,
    seed,
    sys,
    threading,
    time,
    training_set,
    uuid,
    v_label,
    zero_init,
):
    _t0 = time.time()
    _w1 = l1_scale
    _w2 = l2_scale
    _seed = seed
    _K = num_epoch
    _v = v_label
    replicate_uid = str(uuid.uuid4())

    _mask = make_visible_mask(_v, n=N_TOTAL, perm=INTERLEAVED_PERM20)
    _G0 = np.zeros(N_TOTAL)
    _B0 = np.zeros((N_TOTAL, N_TOTAL))

    if zero_init:
        _run_sswm = run_sswm_output_masked_scheduled_traced_zero_masked_elastic

        def _errors_fn(B_, seed_, M_):
            return compute_errors_output_masked_zero_masked(
                B_, _mask, seed=seed_, n_score=N_SCORE, M=M_
            )

    else:
        _run_sswm = run_sswm_output_masked_scheduled_traced_ext_elastic

        def _errors_fn(B_, seed_, M_):
            return compute_errors_output_masked_ext(
                B_, _mask, seed=seed_, n_score=N_SCORE, M=M_
            )

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
    _G_snap = np.empty((_n_snap, N_TOTAL))
    _B_snap = np.empty((_n_snap, N_TOTAL, N_TOTAL))
    _B_trace = np.empty((_n_ts, N_TOTAL, N_TOTAL))
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
    # appended to .npz files one array at a time. Once the run completes,
    # trace_df is *also* written out as parquet (compact, and what the
    # collation step downstream expects) -- see near the end of this
    # cell, after trace_df is fully assembled.
    _run_params = {
        "v": _v,
        "seed": _seed,
        "zeroinit": zero_init,
        "l1scale": _w1,
        "l2scale": _w2,
        "blipfreq": blip_freq,
        "numepoch": _K,
        "schedulemode": schedule_mode,
        "replicate": replicate_uid,
    }
    timeseries_csv_path = (
        f"{OUTPUT_DIR}/{kn.pack({**_run_params, 'ext': '.csv'})}"
    )
    timeseries_pqt_path = (
        f"{OUTPUT_DIR}/{kn.pack({**_run_params, 'ext': '.pqt'})}"
    )
    G_snapshots_path = (
        f"{OUTPUT_DIR}/{kn.pack({**_run_params, 'what': 'G', 'ext': '.npz'})}"
    )
    B_snapshots_path = (
        f"{OUTPUT_DIR}/{kn.pack({**_run_params, 'what': 'B', 'ext': '.npz'})}"
    )

    # Per-class fraction columns (share of _TRACE_M samples landing exactly
    # on each phenotype): test1_frac..test8_frac are the 8 CLASS_8 classes
    # in order; train1_frac..train3_frac are the 3 of those 8 that are
    # also unblipped training patterns (CLASS_8 indices 0, 3, 6 -- S1, S2,
    # S3), duplicated under their own names for self-documenting clarity
    # even though train{i}_frac == test{1,4,7}_frac by construction.
    _trace_columns = (
        [
            "epoch",
            "generation",
            "walltime_sec",
            "pure_train_chi2",
            "test_chi2",
            "blip_train_chi2",
        ]
        + [f"test{_j + 1}_frac" for _j in range(8)]
        + [f"train{_j + 1}_frac" for _j in range(3)]
        + [f"s{_j + 1}_blip_match_frac" for _j in range(3)]
        + [
            "other_frac",
            "other_chi2",
            "other_n_classes",
            "l1_loss",
            "l2_loss",
            "regularization_loss",
            "v",
            "seed",
            "zero_init",
            "l1_scale",
            "l2_scale",
            "blip_freq",
            "num_epoch",
            "schedule_mode",
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
        _ptr, _te, _btr, _cc, _bc, _oc, _noc = _errors_fn(
            _B_trace[_i], _seed + 2000, _TRACE_M
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
            "blip_train_chi2": float(_btr),
        }
        for _j in range(8):
            _row[f"test{_j + 1}_frac"] = float(_cc[_j]) / _TRACE_M
        for _j, _tc in enumerate([0, 3, 6]):
            _row[f"train{_j + 1}_frac"] = float(_cc[_tc]) / _TRACE_M
        for _j in range(3):
            _row[f"s{_j + 1}_blip_match_frac"] = float(_bc[_j]) / _TRACE_M
        _row["other_frac"] = float(_cc[8]) / _TRACE_M
        _row["other_chi2"] = float(_oc)
        _row["other_n_classes"] = int(_noc)
        _row["l1_loss"] = float(_l1)
        _row["l2_loss"] = float(_l2)
        _row["regularization_loss"] = float(
            _w1 * LAM1 * _l1 + _w2 * LAM2 * _l2
        )
        _row["v"] = _v
        _row["seed"] = _seed
        _row["zero_init"] = zero_init
        _row["l1_scale"] = _w1
        _row["l2_scale"] = _w2
        _row["blip_freq"] = blip_freq
        _row["num_epoch"] = _K
        _row["schedule_mode"] = schedule_mode
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
                timeseries_csv_path,
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
            _G, _B = _run_sswm(
                _G0,
                _B0,
                training_set,
                _K,
                schedule,
                LAM1,
                LAM2,
                _w1,
                _w2,
                _mask,
                _seed,
                SNAPSHOT_BLOCKS,
                TIMESERIES_BLOCKS,
                N_SCORE,
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
        blip_train_chi2,
        final_class_counts,
        final_blip_counts,
        final_other_chi2,
        final_n_other_classes,
    ) = _errors_fn(_B, _seed + 1000, FINAL_M)
    assert final_class_counts.sum() == FINAL_M
    final_l1_loss = l1_cost(_B)
    final_l2_loss = l2_cost(_B)
    final_reg_loss = _w1 * LAM1 * final_l1_loss + _w2 * LAM2 * final_l2_loss
    s1_blip_match, s2_blip_match, s3_blip_match = (
        int(_c) for _c in final_blip_counts
    )

    elapsed_sec = time.time() - _t0

    trace_df = pd.DataFrame(_trace_rows, columns=_trace_columns)
    trace_df = trace_df.astype(
        {
            "epoch": np.uint16,
            "generation": np.uint32,
            "walltime_sec": np.float32,
            "pure_train_chi2": np.float32,
            "test_chi2": np.float32,
            "blip_train_chi2": np.float32,
            **{f"test{_j + 1}_frac": np.float32 for _j in range(8)},
            **{f"train{_j + 1}_frac": np.float32 for _j in range(3)},
            **{f"s{_j + 1}_blip_match_frac": np.float32 for _j in range(3)},
            "other_frac": np.float32,
            "other_chi2": np.float32,
            "other_n_classes": np.uint32,
            "l1_loss": np.float32,
            "l2_loss": np.float32,
            "regularization_loss": np.float32,
            "v": np.uint8,
            "seed": np.uint32,
            "zero_init": np.bool_,
            "l1_scale": np.float32,
            "l2_scale": np.float32,
            "blip_freq": np.float32,
            "num_epoch": np.uint32,
        }
    )
    trace_df["schedule_mode"] = pd.Categorical(trace_df["schedule_mode"])
    trace_df["replicate_uid"] = pd.Categorical(trace_df["replicate_uid"])

    # written once the run is complete (unlike the progressive CSV above,
    # parquet doesn't support cheap row-at-a-time appends) -- a compact
    # final artifact, and what the downstream collation step expects.
    trace_df.to_parquet(timeseries_pqt_path, compression="zstd", index=False)

    _final_test_fracs = {
        f"test{_j + 1}_frac": float(final_class_counts[_j] / FINAL_M)
        for _j in range(8)
    }
    _final_train_fracs = {
        f"train{_j + 1}_frac": float(final_class_counts[_tc] / FINAL_M)
        for _j, _tc in enumerate([0, 3, 6])
    }

    result = {
        "v": _v,
        "seed": _seed,
        "num_epoch": _K,
        "zero_init": zero_init,
        "l1_scale": _w1,
        "l2_scale": _w2,
        "blip_freq": blip_freq,
        "schedule_mode": schedule_mode,
        "pure_train_chi2": pure_train_chi2,
        "test_chi2": test_chi2,
        "blip_train_chi2": blip_train_chi2,
        **_final_test_fracs,
        **_final_train_fracs,
        "other_frac": float(final_class_counts[8] / FINAL_M),
        "other_chi2": final_other_chi2,
        "other_n_classes": final_n_other_classes,
        "l1_loss": final_l1_loss,
        "l2_loss": final_l2_loss,
        "regularization_loss": final_reg_loss,
        "s1_blip_match_frac": s1_blip_match / FINAL_M,
        "s2_blip_match_frac": s2_blip_match / FINAL_M,
        "s3_blip_match_frac": s3_blip_match / FINAL_M,
        "elapsed_sec": elapsed_sec,
        "replicate_uid": replicate_uid,
        "timeseries_csv_path": timeseries_csv_path,
        "timeseries_pqt_path": timeseries_pqt_path,
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
        result["timeseries_csv_path"],
        result["timeseries_pqt_path"],
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
