import marimo

__generated_with = "0.23.2"
app = marimo.App(width="full")


@app.cell
def import_std():
    import builtins
    import contextlib
    import os
    import sys
    import time
    import uuid

    return builtins, contextlib, os, sys, time, uuid


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
    mo.md("""
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
    """)
    return


@app.cell(hide_code=True)
def delimit_configure_trial(mo):
    mo.md("""
    ## Configure trial
    """)
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
    mo.md("""
    ## Core GRN model (Kouvaris et al. 2017), inlined from grn.py
    """)
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
    mo.md("""
    ## Target phenotypes, inlined from targets.py
    """)
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
    mo.md("""
    ## Generalisation measurement, inlined from generalisation.py
    """)
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
    mo.md("""
    ## Masked development + zero-init generalization, inlined from grn_output_masked.py
    """)
    return


@app.cell
def masked_model_core(
    CLASS_8,
    N,
    chi_squared,
    develop,
    fold_to_canonical,
    njit,
    np,
    qmc,
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

    @njit
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
        replicate_uid,
    ):
        np.random.seed(seed)
        n = G0.shape[0]
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
        G_snap = np.empty((n_snap_pts, n))
        B_snap = np.empty((n_snap_pts, n, n))
        # B_trace holds a B copy at every timeseries point for post-hoc
        # error computation -- G isn't needed there (compute_errors draws
        # fresh genotype samples rather than reusing evolved G).
        B_trace = np.empty((n_ts_pts, n, n))
        snap_idx = 0
        ts_idx = 0
        snap_ptr = 0
        ts_ptr = 0

        # Progress heartbeat: numba's nopython mode doesn't support
        # time.time() (only plain print() with comma-separated args), so
        # rather than pull in objmode just for a log line, approximate a
        # ~20s cadence using the benchmarked single-threaded throughput of
        # this loop (~83,000 gens/sec on non-cluster hardware) converted
        # to a generation-count interval. Actual cadence scales with real
        # hardware speed -- this is a "job is still alive" heartbeat, not
        # a precise timer. The caller redirects stdout to a real log file
        # for the duration of this call, since marimo's per-cell print
        # capture would otherwise swallow this output (buffered into the
        # exported notebook instead of being visible while the job runs).
        LOG_EVERY_GENS = 1_660_000

        if snap_ptr < n_snap_pts and snapshot_blocks[snap_ptr] == 0:
            G_snap[snap_idx] = G
            B_snap[snap_idx] = B
            snap_idx += 1
            snap_ptr += 1
        if ts_ptr < n_ts_pts and timeseries_blocks[ts_ptr] == 0:
            B_trace[ts_idx] = B
            ts_idx += 1
            ts_ptr += 1

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
                    "progress",
                    replicate_uid,
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
                if (
                    ts_ptr < n_ts_pts
                    and completed_block == timeseries_blocks[ts_ptr]
                ):
                    B_trace[ts_idx] = B
                    ts_idx += 1
                    ts_ptr += 1

        return G, B, G_snap, B_snap, B_trace

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

    @njit
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
        replicate_uid,
    ):
        np.random.seed(seed)
        n = G0.shape[0]
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
        G_snap = np.empty((n_snap_pts, n))
        B_snap = np.empty((n_snap_pts, n, n))
        # B_trace holds a B copy at every timeseries point for post-hoc
        # error computation -- G isn't needed there (compute_errors draws
        # fresh genotype samples rather than reusing evolved G).
        B_trace = np.empty((n_ts_pts, n, n))
        snap_idx = 0
        ts_idx = 0
        snap_ptr = 0
        ts_ptr = 0

        # Progress heartbeat: numba's nopython mode doesn't support
        # time.time() (only plain print() with comma-separated args), so
        # rather than pull in objmode just for a log line, approximate a
        # ~20s cadence using the benchmarked single-threaded throughput of
        # this loop (~83,000 gens/sec on non-cluster hardware) converted
        # to a generation-count interval. Actual cadence scales with real
        # hardware speed -- this is a "job is still alive" heartbeat, not
        # a precise timer. The caller redirects stdout to a real log file
        # for the duration of this call, since marimo's per-cell print
        # capture would otherwise swallow this output (buffered into the
        # exported notebook instead of being visible while the job runs).
        LOG_EVERY_GENS = 1_660_000

        if snap_ptr < n_snap_pts and snapshot_blocks[snap_ptr] == 0:
            G_snap[snap_idx] = G
            B_snap[snap_idx] = B
            snap_idx += 1
            snap_ptr += 1
        if ts_ptr < n_ts_pts and timeseries_blocks[ts_ptr] == 0:
            B_trace[ts_idx] = B
            ts_idx += 1
            ts_ptr += 1

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
                    "progress",
                    replicate_uid,
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
                if (
                    ts_ptr < n_ts_pts
                    and completed_block == timeseries_blocks[ts_ptr]
                ):
                    B_trace[ts_idx] = B
                    ts_idx += 1
                    ts_ptr += 1

        return G, B, G_snap, B_snap, B_trace

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
    mo.md("""
    ## Model constants and blip schedule
    """)
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
    return BLIP_SET, blip_counts, schedule, training_set


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
    mo.md("""
    ## Timepoint sampling (snapshots + timeseries)

    Snapshots (full G/B matrices, persisted to `.npz`) and timeseries rows
    (train/test chi2 + walltime, persisted to `.pqt`) are sampled
    independently over the block domain `[0, TOTAL_BLOCKS]`: each set is
    the union of an evenly-spaced and a log-spaced sample at its target
    count, plus the immediate successor of every sampled point (so
    consecutive-block deltas are always available). A domain smaller than
    the requested count degrades gracefully to the full domain rather than
    erroring or duplicating points past what exists.
    """)
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
def delimit_run_trial(mo):
    mo.md("""
    ## Run trial
    """)
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

    # marimo captures each cell's stdout internally (buffering it into the
    # exported notebook rather than passing it through to the real
    # terminal), so the print()-based progress heartbeats below wouldn't
    # otherwise be visible while a long SLURM job is running. Redirecting
    # to sys.__stdout__ (the original stream, saved by Python at process
    # startup, before marimo's capture takes over) sidesteps that.
    _t_evo_start = time.time()
    with contextlib.redirect_stdout(sys.__stdout__):
        if zero_init:
            (
                _G,
                _B,
                _G_snap,
                _B_snap,
                _B_trace,
            ) = run_sswm_output_masked_scheduled_traced_zero_masked_elastic(
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
                replicate_uid,
            )

            def _errors_fn(B_, seed_, M_):
                return compute_errors_output_masked_zero_masked(
                    B_, _mask, seed=seed_, n_score=N_SCORE, M=M_
                )

        else:
            (
                _G,
                _B,
                _G_snap,
                _B_snap,
                _B_trace,
            ) = run_sswm_output_masked_scheduled_traced_ext_elastic(
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
                replicate_uid,
            )

            def _errors_fn(B_, seed_, M_):
                return compute_errors_output_masked_ext(
                    B_, _mask, seed=seed_, n_score=N_SCORE, M=M_
                )

        # the SSWM loop is a single compiled call, so per-timepoint
        # timestamps aren't directly observable -- _sswm_wall is the true
        # measured duration of the whole call, and each row's walltime
        # below is interpolated proportional to how far through the
        # generation count that row sits (a reasonable estimate since
        # SSWM's cost is ~constant per generation).
        _sswm_wall = time.time() - _t_evo_start

        # M for the (dense, ~thousands of points) per-timepoint trace
        # calls -- kept small relative to FINAL_M below since it's paid
        # many times over.
        _TRACE_M = 2000

        _n_ts = _B_trace.shape[0]
        trace_gens = [int(_b) * _K for _b in TIMESERIES_BLOCKS]
        _total_gens = trace_gens[-1] if trace_gens[-1] > 0 else 1
        trace_walltime = [_sswm_wall * (g / _total_gens) for g in trace_gens]
        trace_pure_train, trace_test, trace_blip_train, trace_other_chi2 = (
            [],
            [],
            [],
            [],
        )
        trace_class_counts, trace_blip_counts, trace_n_other = [], [], []
        # L1/L2/regularization loss depend only on B (not on sampled
        # genotypes), so they're computed directly here rather than
        # threaded through _errors_fn. l1_loss/l2_loss are the raw,
        # unweighted per-entry-mean costs -- recorded regardless of how
        # l1_scale/l2_scale are configured for this replicate -- while
        # regularization_loss is the actual weighted penalty subtracted
        # from benefit in the fitness function driving evolution
        # (w1 * LAM1 * l1_cost(B) + w2 * LAM2 * l2_cost(B)).
        trace_l1_loss, trace_l2_loss, trace_reg_loss = [], [], []
        _trace_log_start = time.time()
        _trace_last_log = _trace_log_start
        for _i in range(_n_ts):
            _ptr, _te, _btr, _cc, _bc, _oc, _noc = _errors_fn(
                _B_trace[_i], _seed + 2000, _TRACE_M
            )
            trace_pure_train.append(_ptr)
            trace_test.append(_te)
            trace_blip_train.append(_btr)
            trace_class_counts.append(_cc)
            trace_blip_counts.append(_bc)
            trace_other_chi2.append(_oc)
            trace_n_other.append(_noc)
            _l1 = l1_cost(_B_trace[_i])
            _l2 = l2_cost(_B_trace[_i])
            trace_l1_loss.append(_l1)
            trace_l2_loss.append(_l2)
            trace_reg_loss.append(_w1 * LAM1 * _l1 + _w2 * LAM2 * _l2)
            _trace_now = time.time()
            if _trace_now - _trace_last_log >= 20:
                print(
                    "progress",
                    replicate_uid,
                    "seed",
                    _seed,
                    "trace",
                    _i + 1,
                    "/",
                    _n_ts,
                    "elapsed_sec",
                    int(_trace_now - _trace_log_start),
                )
                _trace_last_log = _trace_now
    # class_counts: (n_rows, 9) -- CLASS_8 classes 1..8 then "other".
    # blip_counts: (n_rows, 3) -- S1b, S2b, S3b exact-match counts.
    _class_counts_arr = np.asarray(trace_class_counts)
    _blip_counts_arr = np.asarray(trace_blip_counts)

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

    # --- write the time series as a parquet file, filename self-describing
    # via keyname.pack (every run option as a key=value segment, ending in
    # ext=.pqt). Numeric option columns use the smallest dtype that safely
    # covers their range -- pandas Categorical on a numeric/bool column gets
    # silently unwrapped by to_parquet/read_parquet, but parquet's own
    # RLE_DICTIONARY encoding compresses these low-cardinality columns
    # automatically regardless, so the on-disk size benefit doesn't depend
    # on the pandas dtype tag. replicate_uid is a genuine string, so
    # Categorical is used there and round-trips correctly.
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

    _n_rows = len(trace_gens)

    def _bcast(value, dtype):
        return np.full(_n_rows, value, dtype=dtype)

    # Per-class fraction columns (share of _TRACE_M samples landing exactly
    # on each phenotype): test1_frac..test8_frac are the 8 CLASS_8 classes
    # in order; train1_frac..train3_frac are the 3 of those 8 that are
    # also unblipped training patterns (CLASS_8 indices 0, 3, 6 -- S1, S2,
    # S3), duplicated under their own names for self-documenting clarity
    # even though train{i}_frac == test{1,4,7}_frac by construction.
    _test_frac_cols = {
        f"test{_j + 1}_frac": (_class_counts_arr[:, _j] / _TRACE_M).astype(
            np.float32
        )
        for _j in range(8)
    }
    _train_frac_cols = {
        f"train{_j + 1}_frac": (_class_counts_arr[:, _tc] / _TRACE_M).astype(
            np.float32
        )
        for _j, _tc in enumerate([0, 3, 6])
    }
    _blip_frac_cols = {
        f"s{_j + 1}_blip_match_frac": (
            _blip_counts_arr[:, _j] / _TRACE_M
        ).astype(np.float32)
        for _j in range(3)
    }

    trace_df = pd.DataFrame(
        {
            "epoch": np.arange(_n_rows, dtype=np.uint16),
            "generation": np.asarray(trace_gens, dtype=np.uint32),
            "walltime_sec": np.asarray(trace_walltime, dtype=np.float32),
            "pure_train_chi2": np.asarray(trace_pure_train, dtype=np.float32),
            "test_chi2": np.asarray(trace_test, dtype=np.float32),
            "blip_train_chi2": np.asarray(trace_blip_train, dtype=np.float32),
            **_test_frac_cols,
            **_train_frac_cols,
            **_blip_frac_cols,
            "other_frac": (_class_counts_arr[:, 8] / _TRACE_M).astype(
                np.float32
            ),
            "other_chi2": np.asarray(trace_other_chi2, dtype=np.float32),
            "other_n_classes": np.asarray(trace_n_other, dtype=np.uint32),
            "l1_loss": np.asarray(trace_l1_loss, dtype=np.float32),
            "l2_loss": np.asarray(trace_l2_loss, dtype=np.float32),
            "regularization_loss": np.asarray(
                trace_reg_loss, dtype=np.float32
            ),
            "v": _bcast(_v, np.uint8),
            "seed": _bcast(_seed, np.uint32),
            "zero_init": _bcast(zero_init, np.bool_),
            "l1_scale": _bcast(_w1, np.float32),
            "l2_scale": _bcast(_w2, np.float32),
            "blip_freq": _bcast(blip_freq, np.float32),
            "num_epoch": _bcast(_K, np.uint32),
            "schedule_mode": pd.Categorical([schedule_mode] * _n_rows),
            "replicate_uid": pd.Categorical([replicate_uid] * _n_rows),
        }
    )
    timeseries_path = f"{OUTPUT_DIR}/{kn.pack({**_run_params, 'ext': '.pqt'})}"
    trace_df.to_parquet(timeseries_path, compression="zstd", index=False)

    # --- write G_snap and B_snap to separate npz key-value stores, each
    # keyed by generation (zero-padded so keys sort lexicographically in
    # the same order as chronologically). Snapshot points are the sparser
    # SNAPSHOT_BLOCKS set, distinct from the denser TIMESERIES_BLOCKS set
    # trace_gens is built from above.
    _snapshot_gens = [int(_b) * _K for _b in SNAPSHOT_BLOCKS]
    _G_store = {
        f"gen{_g:012d}": _G_snap[_i] for _i, _g in enumerate(_snapshot_gens)
    }
    _B_store = {
        f"gen{_g:012d}": _B_snap[_i] for _i, _g in enumerate(_snapshot_gens)
    }
    G_snapshots_path = (
        f"{OUTPUT_DIR}/{kn.pack({**_run_params, 'what': 'G', 'ext': '.npz'})}"
    )
    B_snapshots_path = (
        f"{OUTPUT_DIR}/{kn.pack({**_run_params, 'what': 'B', 'ext': '.npz'})}"
    )
    np.savez_compressed(G_snapshots_path, **_G_store)
    np.savez_compressed(B_snapshots_path, **_B_store)

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
        "timeseries_path": timeseries_path,
        "G_snapshots_path": G_snapshots_path,
        "B_snapshots_path": B_snapshots_path,
    }
    return result, trace_df


@app.cell(hide_code=True)
def delimit_show_result(mo):
    mo.md("""
    ## Result
    """)
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
