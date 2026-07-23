import marimo

__generated_with = "0.23.2"
app = marimo.App(width="full")


@app.cell
def import_std():
    import os
    import time
    import uuid

    return os, time, uuid


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
    # shuffle_mode controls how ties are broken in the environment-order
    # schedule (see weighted_interleave_fn below): "none" (deterministic,
    # default), "local" (randomize among equally-due environments), or
    # "global" (fully randomize the whole schedule's order).
    shuffle_mode = _get("shuffle-mode", "none", lambda s: str(s).lower())
    return (
        blip_freq,
        l1_scale,
        l2_scale,
        num_epoch,
        seed,
        shuffle_mode,
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
    seed,
    shuffle_mode,
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
                "shuffle_mode": shuffle_mode,
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
def masked_model_core(CLASS_8, N, develop, fold_to_canonical, njit, np, qmc):
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

    def classify_priority_groups_output_masked(Pa_batch):
        Pa_folded = fold_to_canonical(Pa_batch)
        signs = np.sign(Pa_folded)
        dots = signs @ CLASS_8.T
        match = dots == N
        train_cols = [0, 3, 6]
        other_cols = [1, 2, 4, 5, 7]
        is_training = match[:, train_cols].any(axis=1)
        is_other_test = match[:, other_cols].any(axis=1) & ~is_training
        is_other = ~is_training & ~is_other_test
        return (
            int(is_training.sum()),
            int(is_other_test.sum()),
            int(is_other.sum()),
        )

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
        return counts

    return (
        classify_by_phenotype_output_masked,
        classify_priority_groups_output_masked,
        develop_batch_output_masked,
        develop_output_masked,
        develop_output_masked_zero_masked,
        make_visible_mask,
        sample_G,
    )


@app.cell
def masked_model_ext_elastic(
    CLASS_8,
    TRAINING_SET,
    benefit,
    chi_squared,
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
        Pa_folded = fold_to_canonical(Pa_batch[:, :n_score])
        train_counts = classify_exact_counts(Pa_folded, TRAINING_SET)
        test_counts = classify_exact_counts(Pa_folded, CLASS_8)
        return (
            chi_squared(train_counts, M),
            chi_squared(test_counts, M),
            test_counts,
        )

    return (
        compute_errors_output_masked_ext,
        run_sswm_output_masked_scheduled_traced_ext_elastic,
    )


@app.cell
def masked_model_zero_masked_elastic(
    CLASS_8,
    TRAINING_SET,
    benefit,
    chi_squared,
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
        Pa_folded = fold_to_canonical(Pa_batch[:, :n_score])
        train_counts = classify_exact_counts(Pa_folded, TRAINING_SET)
        test_counts = classify_exact_counts(Pa_folded, CLASS_8)
        return (
            chi_squared(train_counts, M),
            chi_squared(test_counts, M),
            test_counts,
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
    def weighted_interleave(counts, shuffle_mode="none", rng=None):
        # Greedy fair-queueing schedule: at each step, whichever
        # environment is furthest behind its target proportion
        # (counts[i] / total) goes next. shuffle_mode controls how ties
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
            if shuffle_mode == "local":
                eligible = np.flatnonzero(
                    np.isclose(deficits, deficits.max(), atol=1e-9)
                )
                i = int(rng.choice(eligible))
            else:
                i = int(np.argmax(deficits))
            seq[step] = i
            appeared[i] += 1
        if shuffle_mode == "global":
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
    seed,
    shuffle_mode,
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

    assert shuffle_mode in ("none", "local", "global")
    _rng = np.random.default_rng(seed)
    schedule = weighted_interleave(
        blip_counts, shuffle_mode=shuffle_mode, rng=_rng
    )
    assert schedule.shape[0] == TOTAL_BLOCKS
    return S1b, S2b, S3b, blip_counts, schedule, training_set


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
    S1b,
    S2b,
    S3b,
    SNAPSHOT_BLOCKS,
    TIMESERIES_BLOCKS,
    blip_freq,
    classify_by_phenotype_output_masked,
    classify_exact_counts,
    classify_priority_groups_output_masked,
    compute_errors_output_masked_ext,
    compute_errors_output_masked_zero_masked,
    develop_batch_output_masked,
    fold_to_canonical,
    kn,
    l1_scale,
    l2_scale,
    make_visible_mask,
    np,
    num_epoch,
    pd,
    run_sswm_output_masked_scheduled_traced_ext_elastic,
    run_sswm_output_masked_scheduled_traced_zero_masked_elastic,
    sample_G,
    schedule,
    seed,
    shuffle_mode,
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

    _t_evo_start = time.time()
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
        )

        def _errors_fn(B_, seed_, M_):
            return compute_errors_output_masked_ext(
                B_, _mask, seed=seed_, n_score=N_SCORE, M=M_
            )

    # the SSWM loop is a single compiled call, so per-timepoint timestamps
    # aren't directly observable -- _sswm_wall is the true measured duration
    # of the whole call, and each row's walltime below is interpolated
    # proportional to how far through the generation count that row sits
    # (a reasonable estimate since SSWM's cost is ~constant per generation).
    _sswm_wall = time.time() - _t_evo_start

    _n_ts = _B_trace.shape[0]
    trace_gens = [int(_b) * _K for _b in TIMESERIES_BLOCKS]
    _total_gens = trace_gens[-1] if trace_gens[-1] > 0 else 1
    trace_walltime = [_sswm_wall * (g / _total_gens) for g in trace_gens]
    trace_train, trace_test = [], []
    for _i in range(_n_ts):
        _tr, _te, _ = _errors_fn(_B_trace[_i], _seed + 2000, 2000)
        trace_train.append(_tr)
        trace_test.append(_te)

    FINAL_M = 100_000
    train_chi2, test_chi2, _ = _errors_fn(_B, _seed + 1000, FINAL_M)

    _G_batch = sample_G(FINAL_M, seed=_seed + 1000, n=N_TOTAL)
    if zero_init:
        _G_batch = _G_batch.copy()
        _G_batch[:, ~_mask] = 0.0
    _Pa_batch_full = develop_batch_output_masked(_G_batch, _B, _mask)
    _Pa_batch = _Pa_batch_full[:, :N_SCORE]
    _Pa_folded = fold_to_canonical(_Pa_batch)
    s1_blip_match = int(
        classify_exact_counts(_Pa_folded, S1b.reshape(1, -1))[0]
    )
    s2_blip_match = int(
        classify_exact_counts(_Pa_folded, S2b.reshape(1, -1))[0]
    )
    s3_blip_match = int(
        classify_exact_counts(_Pa_folded, S3b.reshape(1, -1))[0]
    )
    (
        training_ct,
        other_test_ct,
        other_ct,
    ) = classify_priority_groups_output_masked(_Pa_batch)
    assert training_ct + other_test_ct + other_ct == FINAL_M
    phenotype_counts = classify_by_phenotype_output_masked(_Pa_batch)
    assert phenotype_counts.sum() == FINAL_M

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
        "shufflemode": shuffle_mode,
        "replicate": replicate_uid,
    }

    _n_rows = len(trace_gens)

    def _bcast(value, dtype):
        return np.full(_n_rows, value, dtype=dtype)

    trace_df = pd.DataFrame(
        {
            "epoch": np.arange(_n_rows, dtype=np.uint16),
            "generation": np.asarray(trace_gens, dtype=np.uint32),
            "walltime_sec": np.asarray(trace_walltime, dtype=np.float32),
            "train_chi2": np.asarray(trace_train, dtype=np.float32),
            "test_chi2": np.asarray(trace_test, dtype=np.float32),
            "v": _bcast(_v, np.uint8),
            "seed": _bcast(_seed, np.uint32),
            "zero_init": _bcast(zero_init, np.bool_),
            "l1_scale": _bcast(_w1, np.float32),
            "l2_scale": _bcast(_w2, np.float32),
            "blip_freq": _bcast(blip_freq, np.float32),
            "num_epoch": _bcast(_K, np.uint32),
            "shuffle_mode": pd.Categorical([shuffle_mode] * _n_rows),
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

    result = {
        "v": _v,
        "seed": _seed,
        "num_epoch": _K,
        "zero_init": zero_init,
        "l1_scale": _w1,
        "l2_scale": _w2,
        "blip_freq": blip_freq,
        "shuffle_mode": shuffle_mode,
        "train_chi2": train_chi2,
        "test_chi2": test_chi2,
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
