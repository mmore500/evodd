import marimo

__generated_with = "0.23.2"
app = marimo.App(width="full")


@app.cell
def import_std():
    import colorsys
    import pathlib
    import tempfile
    import urllib.request

    return colorsys, pathlib, tempfile, urllib


@app.cell
def import_pkg():
    import marimo as mo
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from teeplot import teeplot as tp
    from watermark import watermark

    return (
        GridSpec,
        GridSpecFromSubplotSpec,
        mo,
        np,
        pd,
        plt,
        sns,
        tp,
        watermark,
    )


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
    # Exploratory sweep and noblip: double-descent analysis

    Downloads the collated timeseries parquet files for the
    `2026-07-23-exploratory-sweep` and `2026-07-23-exploratory-noblip` SLURM
    batch jobs (see `slurm/2026-07-23/`) from OSF, then -- for each swept
    condition (blip frequency, L1/L2 regularization mix, zero-init, and
    environment schedule mode) -- renders a compound plot summarizing
    training/testing dynamics, phenotype composition, and double descent
    across model size `v` and training time.
    """
    )
    return


@app.cell(hide_code=True)
def delimit_fetch_data(mo):
    mo.md(
        """
    ## Fetch data
    """
    )
    return


@app.cell
def osf_slugs():
    # https://osf.io/7n63x -- 2026-07-23-exploratory-sweep collated timeseries
    #   (blip_freq swept across {0.66, 0.63, 0.6, 0.5})
    # https://osf.io/xn6mk -- 2026-07-23-exploratory-noblip collated timeseries
    #   (blip_freq fixed at 0)
    OSF_SLUGS = {
        "exploratory-sweep": "7n63x",
        "exploratory-noblip": "xn6mk",
    }
    return (OSF_SLUGS,)


@app.cell
def fetch_osf_fn(pathlib, tempfile, urllib):
    def fetch_osf(slug: str) -> "pathlib.Path":
        """Download a file from OSF by its slug, caching it under the system
        temp directory, and return the path to the cached file.

        Tries a plain stdlib download first; if that fails for any reason
        (auth wall, transient network error, redirect handling, ...) falls
        back to the `osf_fetch` utility from this project's `pylib` (ported
        from the same-named utility added to
        github.com/mmore500/paperproject.git), which is more robust
        (session handling via `requests`, on-disk caching)."""
        cache_path = pathlib.Path(tempfile.gettempdir()) / slug
        url = f"https://osf.io/{slug}/download"
        try:
            if not cache_path.exists():
                print(f"downloading {url} -> {cache_path}")
                request = urllib.request.Request(
                    url, headers={"User-Agent": "evodd-bindle/1.0"}
                )
                with urllib.request.urlopen(request, timeout=180) as resp:
                    cache_path.write_bytes(resp.read())
            else:
                print(f"reusing cached {cache_path}")
        except Exception as e:
            print(
                f"direct download of {url} failed ({e!r}), falling back to pylib.osf_fetch"
            )
            from pylib import osf_fetch

            cache_path = osf_fetch(slug)
        print(f"size: {cache_path.stat().st_size} bytes")
        return cache_path

    return (fetch_osf,)


@app.cell
def fetch_data(OSF_SLUGS, fetch_osf):
    osf_paths = {name: fetch_osf(slug) for name, slug in OSF_SLUGS.items()}
    return (osf_paths,)


@app.cell(hide_code=True)
def delimit_load_data(mo):
    mo.md(
        """
    ## Load data
    """
    )
    return


@app.cell
def load_data(osf_paths, pd):
    df = pd.concat(
        [
            pd.read_parquet(path).assign(dataset=name)
            for name, path in osf_paths.items()
        ],
        ignore_index=True,
    )
    df["dataset"] = pd.Categorical(df["dataset"])
    return (df,)


@app.cell
def peek_data(df, pd):
    pd.concat([df.head(), df.tail()])
    return


@app.cell
def describe_data(df):
    df.describe()
    return


@app.cell(hide_code=True)
def delimit_prep(mo):
    mo.md(
        """
    ## Prepare conditions

    A "condition" is one unique combination of the swept, non-model-size
    knobs (`blip_freq`, `l1_scale`, `l2_scale`, `zero_init`,
    `schedule_mode`) within a dataset -- everything except model size `v`
    (the double-descent x-axis) and `seed` (the replicate axis).
    """
    )
    return


@app.cell
def prep_conditions(df, sns):
    CONDITION_COLS = [
        "dataset",
        "blip_freq",
        "l1_scale",
        "l2_scale",
        "zero_init",
        "schedule_mode",
    ]

    conditions = (
        df[CONDITION_COLS].drop_duplicates().sort_values(CONDITION_COLS)
    )

    # categorical (qualitative) color map for v, shared across every plot
    # below that encodes v via color.
    v_values_all = sorted(df["v"].unique())
    v_palette = dict(
        zip(
            v_values_all,
            sns.color_palette("tab20", n_colors=len(v_values_all)),
        )
    )
    return CONDITION_COLS, conditions, v_palette


@app.cell
def show_conditions(conditions, pd):
    pd.concat([conditions.head(), conditions.tail()])
    return


@app.cell(hide_code=True)
def delimit_plot_helpers(mo):
    mo.md(
        """
    ## Plotting helpers

    Phenotype classes `test1_frac`..`test8_frac` are the 8 canonical
    `CLASS_8` phenotypes; `train1_frac`..`train3_frac` duplicate
    `test1_frac`/`test4_frac`/`test7_frac` by construction (see
    `bindle/2026-07-23-exploratory.py`'s `_trace_row`), so the phenotype
    stackplot below stacks `test1_frac`..`test8_frac` (not the `train*`
    columns, which would double-count) and simply recolors the 3
    training-overlap slices distinctly from the other 5 test-only slices.
    Blip-pattern matches (`s1_blip_match_frac`..`s3_blip_match_frac`) are a
    subset of `other_frac`, so they're broken out of it rather than stacked
    on top.
    """
    )
    return


@app.cell
def plot_helpers(colorsys):
    # CLASS_8 indices 0,3,6 (1-indexed: test1,test4,test7) are the pure
    # training patterns S1,S2,S3.
    TRAIN_OVERLAP_TEST_IDX = [1, 4, 7]
    TEST_ONLY_IDX = [2, 3, 5, 6, 8]

    def dull(hex_color, sat_scale=0.35, light_boost=0.25):
        """Desaturate + lighten a bright hex color for the "testing" (as
        opposed to bright "training") stackplot palette."""
        r, g, b = (int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5))
        hue, lightness, sat = colorsys.rgb_to_hls(r, g, b)
        lightness = min(1.0, lightness + light_boost)
        sat = sat * sat_scale
        return colorsys.hls_to_rgb(hue, lightness, sat)

    BRIGHT_TRAIN_COLORS = ["#e6194B", "#3cb44b", "#4363d8"]
    DULL_TEST_COLORS = [dull(c) for c in BRIGHT_TRAIN_COLORS] + [
        dull("#f58231"),
        dull("#911eb4"),
    ]
    GRAY_BLIP_COLORS = ["#404040", "#808080", "#bfbfbf"]
    OTHER_COLOR = "#ffffff"
    return (
        BRIGHT_TRAIN_COLORS,
        DULL_TEST_COLORS,
        GRAY_BLIP_COLORS,
        OTHER_COLOR,
        TEST_ONLY_IDX,
        TRAIN_OVERLAP_TEST_IDX,
    )


@app.cell
def compound_plot_fn(
    BRIGHT_TRAIN_COLORS,
    DULL_TEST_COLORS,
    GRAY_BLIP_COLORS,
    GridSpec,
    GridSpecFromSubplotSpec,
    OTHER_COLOR,
    TEST_ONLY_IDX,
    TRAIN_OVERLAP_TEST_IDX,
    plt,
    sns,
):
    def make_compound_plot(df_cond, v_palette):
        # every generation-axis (log-scaled) plot below starts at 10^6,
        # not 1 -- drop earlier rows up front so they don't skew
        # autoscaled y-limits or inflate rendered path complexity for
        # data that's off the visible window anyway.
        GEN_LOG_FLOOR = 1e6
        df_cond = df_cond[df_cond["generation"] >= GEN_LOG_FLOOR]

        v_values = sorted(df_cond["v"].unique())
        n_v = len(v_values)

        fig = plt.figure(figsize=(max(10, 2.0 * n_v), 24))
        gs = GridSpec(
            5,
            1,
            figure=fig,
            height_ratios=[3, 2.4, 2.6, 2.4, 3.2],
            hspace=0.55,
        )

        # --- row 1: testing (solid) / ACTUAL training (dashed) loss over
        # generations, faceted by v, one line per replicate.
        row1_gs = GridSpecFromSubplotSpec(
            1, n_v, subplot_spec=gs[0], wspace=0.08
        )
        axes1 = [fig.add_subplot(row1_gs[0, 0])]
        for i in range(1, n_v):
            axes1.append(fig.add_subplot(row1_gs[0, i], sharey=axes1[0]))
        seed_values = sorted(df_cond["seed"].unique())
        seed_palette = dict(
            zip(
                seed_values,
                sns.color_palette("tab10", n_colors=len(seed_values)),
            )
        )
        for ax, v in zip(axes1, v_values):
            sub = df_cond[df_cond["v"] == v]
            for seed, g in sub.groupby("seed", observed=True):
                g = g.sort_values("generation")
                color = seed_palette[seed]
                ax.plot(
                    g["generation"],
                    g["test_chi2"],
                    color=color,
                    ls="-",
                    lw=1.2,
                )
                ax.plot(
                    g["generation"],
                    g["pure_train_chi2"],
                    color=color,
                    ls="--",
                    lw=1.2,
                )
            ax.set_xscale("log")
            ax.set_xlim(left=GEN_LOG_FLOOR)
            ax.set_ylim(bottom=0)
            ax.set_title(f"v={v}", fontsize=9, color=v_palette[v])
            ax.set_xlabel("generation", fontsize=7)
            ax.tick_params(labelsize=7)
            if ax is not axes1[0]:
                plt.setp(ax.get_yticklabels(), visible=False)
        axes1[0].set_ylabel("chi$^2$ error")
        legend_handles = [
            plt.Line2D([], [], color="black", ls="-", label="testing"),
            plt.Line2D(
                [], [], color="black", ls="--", label="training (actual)"
            ),
        ] + [
            plt.Line2D(
                [], [], color=seed_palette[s], label=f"replicate seed={s}"
            )
            for s in seed_values
        ]
        axes1[-1].legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            fontsize=7,
            frameon=False,
        )

        # --- row 2: median +/- [0th, 100th] percentile band across
        # replicates, training error | testing error, hue=v.
        row2_gs = GridSpecFromSubplotSpec(
            1, 2, subplot_spec=gs[1], wspace=0.15
        )
        ax_train = fig.add_subplot(row2_gs[0, 0])
        ax_test = fig.add_subplot(row2_gs[0, 1])
        for metric_col, ax, label in (
            ("pure_train_chi2", ax_train, "training error (actual)"),
            ("test_chi2", ax_test, "testing error"),
        ):
            agg = (
                df_cond.groupby(["v", "generation"], observed=True)[metric_col]
                .agg(median="median", lo="min", hi="max")
                .reset_index()
            )
            for v, g in agg.groupby("v", observed=True):
                g = g.sort_values("generation")
                color = v_palette[v]
                ax.plot(
                    g["generation"],
                    g["median"],
                    color=color,
                    lw=1.5,
                    label=f"v={v}",
                )
                ax.fill_between(
                    g["generation"],
                    g["lo"],
                    g["hi"],
                    color=color,
                    alpha=0.2,
                    lw=0,
                )
            ax.set_xscale("log")
            ax.set_xlim(left=GEN_LOG_FLOOR)
            ax.set_ylim(bottom=0)
            ax.set_xlabel("generation")
            ax.set_title(label, fontsize=10)
        ax_train.set_ylabel("chi$^2$ error")
        ax_test.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            fontsize=7,
            frameon=False,
            title="v",
        )

        # --- row 3: stackplot of testing phenotype distributions across v,
        # at each replicate's own final recorded generation. Bright =
        # training classes (overlap with test1/test4/test7), dull =
        # test-only classes, grayscale = blip matches, white = remaining
        # "other". Per-(v, seed) last-row (rather than a single shared max
        # generation across the whole condition) since replicates can be
        # truncated at different generations by a SLURM timeout (see
        # bindle/2026-07-23-exploratory.py's progressive save-out).
        ax3 = fig.add_subplot(gs[2])
        final = (
            df_cond.sort_values("generation")
            .groupby(["v", "seed"], observed=True)
            .tail(1)
        )
        frac_cols = [f"test{i}_frac" for i in range(1, 9)]
        blip_cols = [f"s{i}_blip_match_frac" for i in range(1, 4)]
        med = (
            final.groupby("v", observed=True)[
                frac_cols + blip_cols + ["other_frac"]
            ]
            .median()
            .reindex(v_values)
        )
        blip_sum = med[blip_cols].sum(axis=1)
        # blip matches are a subset of "other" (they don't coincide with
        # any of the 8 canonical CLASS_8 phenotypes) -- subtract them back
        # out so the stack doesn't double-count; clip guards against
        # floating-point/rare-coincidence edge cases.
        other_only = (med["other_frac"] - blip_sum).clip(lower=0)

        train_layers = [med[f"test{i}_frac"] for i in TRAIN_OVERLAP_TEST_IDX]
        test_only_layers = [med[f"test{i}_frac"] for i in TEST_ONLY_IDX]
        blip_layers = [med[c] for c in blip_cols]

        stack = train_layers + test_only_layers + blip_layers + [other_only]
        colors = (
            BRIGHT_TRAIN_COLORS
            + DULL_TEST_COLORS
            + GRAY_BLIP_COLORS
            + [OTHER_COLOR]
        )
        labels = (
            [f"train (test{i})" for i in TRAIN_OVERLAP_TEST_IDX]
            + [f"test-only (test{i})" for i in TEST_ONLY_IDX]
            + [f"blip s{i}" for i in range(1, 4)]
            + ["other"]
        )
        polys = ax3.stackplot(v_values, *stack, colors=colors, labels=labels)
        # "other" is white -- outline it so it's visible against the
        # figure background.
        polys[-1].set_edgecolor("black")
        polys[-1].set_linewidth(0.6)
        ax3.set_xlim(min(v_values), max(v_values))
        ax3.set_ylim(0, 1)
        ax3.set_xlabel("v (visible genes)")
        ax3.set_ylabel("phenotype fraction")
        ax3.set_title(
            "final-generation testing phenotype distribution across v",
            fontsize=10,
        )
        ax3.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            fontsize=7,
            frameon=False,
            ncol=1,
        )

        # --- row 4: final training (actual) / testing loss vs model size
        # v -- reuses `final` (each replicate's own last recorded row,
        # computed above for row 3) so truncated replicates are handled
        # the same way here; median with a shaded [0th, 100th] percentile
        # band across replicates per v, testing solid / training (actual)
        # dashed to match row 1's convention.
        ax_final = fig.add_subplot(gs[3])
        final_loss = (
            final.groupby("v", observed=True)[["pure_train_chi2", "test_chi2"]]
            .agg(["median", "min", "max"])
            .reindex(v_values)
        )
        for metric_col, color, ls, label in (
            ("test_chi2", "#1f77b4", "-", "testing"),
            ("pure_train_chi2", "#d62728", "--", "training (actual)"),
        ):
            ax_final.plot(
                v_values,
                final_loss[(metric_col, "median")],
                color=color,
                ls=ls,
                lw=1.8,
                label=label,
            )
            ax_final.fill_between(
                v_values,
                final_loss[(metric_col, "min")],
                final_loss[(metric_col, "max")],
                color=color,
                alpha=0.2,
                lw=0,
            )
        ax_final.set_xlim(min(v_values), max(v_values))
        ax_final.set_ylim(bottom=0)
        ax_final.set_xlabel("v (visible genes)")
        ax_final.set_ylabel("chi$^2$ error")
        ax_final.set_title(
            "final training (actual) / testing loss vs model size",
            fontsize=10,
        )
        ax_final.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            fontsize=7,
            frameon=False,
        )

        # --- row 5: double descent heatmap -- training time (y) x model
        # size v (x), colored by median testing error. Colormap matches
        # the one used for the double descent heatmaps in Nakkiran et al.
        # 2019 ("Deep Double Descent", arXiv:1912.02292, Figure 2) --
        # matplotlib's viridis (dark purple = low error, yellow = high
        # error).
        ax4 = fig.add_subplot(gs[4])
        grid = (
            df_cond.groupby(["generation", "v"], observed=True)["test_chi2"]
            .median()
            .unstack("v")
            .reindex(columns=v_values)
            .sort_index()
        )
        mesh = ax4.pcolormesh(
            grid.columns.to_numpy(dtype=float),
            grid.index.to_numpy(dtype=float),
            grid.to_numpy(),
            cmap="viridis",
            shading="nearest",
        )
        ax4.set_yscale("log")
        ax4.set_ylim(bottom=GEN_LOG_FLOOR)
        ax4.set_xlabel("v (visible genes)")
        ax4.set_ylabel("generation (training time)")
        ax4.set_title("double descent: median testing error", fontsize=10)
        cbar = fig.colorbar(mesh, ax=ax4, pad=0.02)
        cbar.set_label("median test chi$^2$ error")

        return fig

    return (make_compound_plot,)


@app.cell(hide_code=True)
def delimit_render(mo):
    mo.md(
        """
    ## Compound plots by condition
    """
    )
    return


@app.cell
def render_compound_plots(
    CONDITION_COLS,
    conditions,
    df,
    make_compound_plot,
    mo,
    np,
    pathlib,
    plt,
    tp,
    v_palette,
):
    for _, _cond in conditions.iterrows():
        _mask = np.logical_and.reduce(
            [df[c] == _cond[c] for c in CONDITION_COLS]
        )
        _df_cond = df[_mask]
        _n_replicates = _df_cond["seed"].nunique()

        # text spacer between compound plots (which are themselves
        # deliberately left untitled) identifying which condition follows.
        mo.output.append(
            mo.md(
                f"""
    ---
    **dataset**=`{_cond['dataset']}`
    &nbsp;**blip_freq**=`{_cond['blip_freq']}`
    &nbsp;**l1_scale**=`{_cond['l1_scale']}`
    &nbsp;**l2_scale**=`{_cond['l2_scale']}`
    &nbsp;**zero_init**=`{_cond['zero_init']}`
    &nbsp;**schedule_mode**=`{_cond['schedule_mode']}`
    &nbsp;**n_replicates(seeds)/v**=`{_n_replicates}`
    """
            )
        )

        with tp.teed(
            make_compound_plot,
            _df_cond,
            v_palette,
            teeplot_outattrs={
                "dataset": str(_cond["dataset"]),
                "blipfreq": str(_cond["blip_freq"]),
                "l1scale": str(_cond["l1_scale"]),
                "l2scale": str(_cond["l2_scale"]),
                "zeroinit": str(_cond["zero_init"]),
                "schedulemode": str(_cond["schedule_mode"]),
            },
            teeplot_subdir=pathlib.Path(__file__).stem,
            teeplot_show=False,
        ) as _fig:
            pass

        # mo.output.append renders the figure into the cell's output
        # immediately, so it's safe (and, across ~dozens of conditions,
        # necessary to avoid unbounded memory growth) to close it right
        # after.
        mo.output.append(_fig)
        plt.close(_fig)
    return


if __name__ == "__main__":
    app.run()
