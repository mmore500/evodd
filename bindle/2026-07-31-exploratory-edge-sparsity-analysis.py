import marimo

__generated_with = "0.23.2"
app = marimo.App(width="full")


@app.cell
def import_std():
    import pathlib
    import tempfile
    import urllib.request

    return pathlib, tempfile, urllib


@app.cell
def import_pkg():
    from matplotlib.colors import Normalize
    from matplotlib.gridspec import GridSpec
    import marimo as mo
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, ScalarFormatter
    import numpy as np
    import pandas as pd
    from teeplot import teeplot as tp
    from watermark import watermark

    return (
        GridSpec,
        MaxNLocator,
        Normalize,
        ScalarFormatter,
        mo,
        np,
        pd,
        plt,
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
    # Exploratory edge sparsity (2026-07-30): loss vs. n and over time

    Downloads the collated timeseries parquet file for the
    `2026-07-30-exploratory-edge-sparsity` SLURM batch job (see
    `slurm/2026-07-30/`) from OSF, then renders three 2x2-panel figures
    -- rows = L1 condition (with L1 / without L1), columns = training-set
    size (3-class / 5-class) -- covering all 4 swept (l1_scale, l2_scale,
    n_classes) conditions in one glance apiece:

    1. Final testing / training (actual) chi^2 error vs. `n_zero_edges`
       (how many of the GRN's 256 possible regulatory edges were zeroed),
       median with a shaded [min, max] band across the 10 replicate
       seeds at each n.
    2. Testing chi^2 error over evolutionary time (generation), one line
       per swept `n_zero_edges` value, colored by a continuous colormap.
    3. Training (actual) chi^2 error over evolutionary time, same layout
       as (2).

    Every chi^2 y-axis here is symlog -- linear below 1.0, log above --
    matching this project's usual convention (`CHI2_LINTHRESH = 1.0`).

    Unlike the source notebook's own `v`-sweep-family predecessors (e.g.
    `bindle/2026-07-29-exploratory-reseed-bitflip-blip50-l1-9966.py`),
    there's no per-condition loop here: with only 4 conditions total
    (crossing 2 L1 settings x 2 training-set sizes) and `n_zero_edges`
    itself the swept model-size axis (no blips, no `v`, no `zero_init`,
    no `schedule_mode` -- see `bindle/2026-07-30-exploratory-edge-sparsity.py`),
    all 4 conditions fit as subplots in a single figure per plot type,
    so each of the 3 figures is produced exactly once.
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
    # https://osf.io/dwhuq -- 2026-07-30-exploratory-edge-sparsity collated
    #   timeseries (density/n_zero_edges, l1_scale/l2_scale, and n_classes
    #   swept; no blips, no v/visible-gene masking, no zero_init)
    OSF_SLUGS = {
        "exploratory-edge-sparsity": "dwhuq",
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

    A "condition" here is one unique (`l1_scale`, `l2_scale`, `n_classes`)
    combination -- everything swept except `n_zero_edges`/`density` (the
    x-axis in plot 1, the color axis in plots 2/3) and `seed` (the
    replicate axis). `L1_CONDITIONS` orders the two (`l1_scale`,
    `l2_scale`) pairs "with L1" (`l1_scale=1.0`, pure L1) first, "without
    L1" (`l1_scale=l2_scale=0.0`, no regularization at all -- not a swap
    to pure L2) second, matching `slurm/2026-07-30/2026-07-30-exploratory-edge-sparsity.sh`'s
    `L1_SCALES`/`L2_SCALES`.
    """
    )
    return


@app.cell
def prep_conditions(df):
    CONDITION_COLS = ["l1_scale", "l2_scale", "n_classes"]
    conditions = (
        df[CONDITION_COLS].drop_duplicates().sort_values(CONDITION_COLS)
    )
    assert (
        len(conditions) == 4
    ), f"expected 4 conditions, got {len(conditions)}"

    L1_CONDITIONS = [
        (l1s, l2s, "with L1" if l1s > 0 else "without L1")
        for l1s, l2s in sorted(
            conditions[["l1_scale", "l2_scale"]]
            .drop_duplicates()
            .itertuples(index=False, name=None),
            reverse=True,
        )
    ]
    N_CLASSES_VALUES = sorted(conditions["n_classes"].unique())
    assert len(L1_CONDITIONS) == 2, L1_CONDITIONS
    assert len(N_CLASSES_VALUES) == 2, N_CLASSES_VALUES
    return CONDITION_COLS, L1_CONDITIONS, N_CLASSES_VALUES, conditions


@app.cell
def show_conditions(conditions, pd):
    pd.concat([conditions.head(), conditions.tail()])
    return


@app.cell(hide_code=True)
def delimit_final_loss_plot(mo):
    mo.md(
        """
    ## Plot 1: final test/train loss vs. n_zero_edges
    """
    )
    return


@app.cell
def final_loss_plot_fn(GridSpec, MaxNLocator, ScalarFormatter, plt):
    def _use_readable_symlog_ticks(axis):
        # matplotlib's default symlog locator/formatter collapses to a
        # single offset-notation label (instead of normal tick labels)
        # whenever the visible range never actually crosses linthresh --
        # MaxNLocator + ScalarFormatter reads fine in that case *and*
        # when the range does cross into the log region.
        axis.set_major_locator(MaxNLocator(nbins=6))
        axis.set_major_formatter(ScalarFormatter())

    def make_final_loss_plot(df, l1_conditions, n_classes_values):
        CHI2_LINTHRESH = 1.0

        # each replicate's own last recorded row -- not a single shared
        # max generation across the whole sweep -- since replicates can
        # be truncated at different generations by a SLURM timeout (see
        # bindle/2026-07-30-exploratory-edge-sparsity.py's progressive
        # save-out).
        final = (
            df.sort_values("generation")
            .groupby(
                [
                    "l1_scale",
                    "l2_scale",
                    "n_classes",
                    "n_zero_edges",
                    "seed",
                ],
                observed=True,
            )
            .tail(1)
        )

        fig = plt.figure(figsize=(11, 8), dpi=80)
        gs = GridSpec(
            len(l1_conditions),
            len(n_classes_values),
            figure=fig,
            hspace=0.35,
            wspace=0.25,
        )

        ax0 = None
        for row, (l1s, l2s, l1_label) in enumerate(l1_conditions):
            for col, ncls in enumerate(n_classes_values):
                if ax0 is None:
                    ax = fig.add_subplot(gs[row, col])
                    ax0 = ax
                else:
                    ax = fig.add_subplot(gs[row, col], sharex=ax0, sharey=ax0)

                sub = final[
                    (final["l1_scale"] == l1s)
                    & (final["l2_scale"] == l2s)
                    & (final["n_classes"] == ncls)
                ]
                n_values = sorted(sub["n_zero_edges"].unique())
                agg = (
                    sub.groupby("n_zero_edges", observed=True)[
                        ["pure_train_chi2", "test_chi2"]
                    ]
                    .agg(["median", "min", "max"])
                    .reindex(n_values)
                )
                for metric_col, color, ls, label in (
                    ("test_chi2", "#1f77b4", "-", "testing"),
                    (
                        "pure_train_chi2",
                        "#d62728",
                        "--",
                        "training (actual)",
                    ),
                ):
                    ax.plot(
                        n_values,
                        agg[(metric_col, "median")],
                        color=color,
                        ls=ls,
                        lw=1.6,
                        label=label,
                    )
                    ax.fill_between(
                        n_values,
                        agg[(metric_col, "min")],
                        agg[(metric_col, "max")],
                        color=color,
                        alpha=0.2,
                        lw=0,
                    )
                ax.set_yscale("symlog", linthresh=CHI2_LINTHRESH)
                ax.set_ylim(bottom=0)
                _use_readable_symlog_ticks(ax.yaxis)
                if n_values:
                    ax.set_xlim(min(n_values), max(n_values))
                ax.set_title(f"{l1_label}, n_classes={ncls}", fontsize=10)
                if row == len(l1_conditions) - 1:
                    ax.set_xlabel("n_zero_edges (of 256 possible)")
                if col == 0:
                    ax.set_ylabel("chi$^2$ error")
                if row == 0 and col == len(n_classes_values) - 1:
                    ax.legend(
                        loc="upper left",
                        bbox_to_anchor=(1.02, 1.0),
                        fontsize=8,
                        frameon=False,
                    )

        fig.suptitle(
            "final testing / training (actual) chi$^2$ error vs. n_zero_edges",
            fontsize=12,
        )
        return fig

    return (make_final_loss_plot,)


@app.cell
def render_final_loss_plot(
    L1_CONDITIONS,
    N_CLASSES_VALUES,
    df,
    make_final_loss_plot,
    mo,
    pathlib,
    plt,
    tp,
):
    with tp.teed(
        make_final_loss_plot,
        df,
        L1_CONDITIONS,
        N_CLASSES_VALUES,
        teeplot_outattrs={"dataset": "exploratory-edge-sparsity"},
        teeplot_subdir=pathlib.Path(__file__).stem,
        teeplot_show=False,
    ) as _fig:
        pass
    mo.output.append(_fig)
    plt.close(_fig)
    return


@app.cell(hide_code=True)
def delimit_over_time_plots(mo):
    mo.md(
        """
    ## Plots 2 & 3: loss over evolutionary time, colored by n_zero_edges

    Each panel overlays one median line per swept `n_zero_edges` value
    (99 lines/panel) against generation (log-scaled, starting at 10^6),
    colored by a continuous colormap (`n_zero_edges` -> viridis, shown
    via a shared colorbar) rather than a discrete per-line legend --
    with 99 distinct swept values, a discrete legend isn't practical
    (unlike this project's earlier `v`-sweep notebooks, which had at
    most 20 values). No min/max band is shown here (99 overlapping bands
    would be unreadable); see plot 1 above for replicate spread at each
    n. Y-axis is symlog (linear below chi^2=1, log above).
    """
    )
    return


@app.cell
def over_time_plot_fn(GridSpec, MaxNLocator, Normalize, ScalarFormatter, plt):
    def _use_readable_symlog_ticks(axis):
        axis.set_major_locator(MaxNLocator(nbins=6))
        axis.set_major_formatter(ScalarFormatter())

    def make_over_time_plot(
        df, l1_conditions, n_classes_values, metric_col, metric_label
    ):
        GEN_LOG_FLOOR = 1e6
        CHI2_LINTHRESH = 1.0
        df = df[df["generation"] >= GEN_LOG_FLOOR]

        n_min = df["n_zero_edges"].min()
        n_max = df["n_zero_edges"].max()
        norm = Normalize(vmin=n_min, vmax=n_max)
        cmap = plt.get_cmap("viridis")

        fig = plt.figure(figsize=(12, 8), dpi=80)
        gs = GridSpec(
            len(l1_conditions),
            len(n_classes_values),
            figure=fig,
            hspace=0.35,
            wspace=0.25,
        )

        ax0 = None
        axes = []
        for row, (l1s, l2s, l1_label) in enumerate(l1_conditions):
            for col, ncls in enumerate(n_classes_values):
                if ax0 is None:
                    ax = fig.add_subplot(gs[row, col])
                    ax0 = ax
                else:
                    ax = fig.add_subplot(gs[row, col], sharex=ax0, sharey=ax0)
                axes.append(ax)

                sub = df[
                    (df["l1_scale"] == l1s)
                    & (df["l2_scale"] == l2s)
                    & (df["n_classes"] == ncls)
                ]
                agg = (
                    sub.groupby(["n_zero_edges", "generation"], observed=True)[
                        metric_col
                    ]
                    .median()
                    .reset_index()
                )
                for n_val, g in agg.groupby("n_zero_edges", observed=True):
                    g = g.sort_values("generation")
                    ax.plot(
                        g["generation"],
                        g[metric_col],
                        color=cmap(norm(n_val)),
                        lw=0.8,
                        alpha=0.85,
                    )

                ax.set_xscale("log")
                ax.set_xlim(left=GEN_LOG_FLOOR)
                ax.set_yscale("symlog", linthresh=CHI2_LINTHRESH)
                ax.set_ylim(bottom=0)
                _use_readable_symlog_ticks(ax.yaxis)
                ax.set_title(f"{l1_label}, n_classes={ncls}", fontsize=10)
                if row == len(l1_conditions) - 1:
                    ax.set_xlabel("generation")
                if col == 0:
                    ax.set_ylabel("chi$^2$ error")

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, pad=0.02, aspect=40)
        cbar.set_label("n_zero_edges (of 256 possible)")

        fig.suptitle(
            f"{metric_label} chi$^2$ error over evolutionary time",
            fontsize=12,
        )
        return fig

    return (make_over_time_plot,)


@app.cell
def render_test_loss_over_time(
    L1_CONDITIONS,
    N_CLASSES_VALUES,
    df,
    make_over_time_plot,
    mo,
    pathlib,
    plt,
    tp,
):
    with tp.teed(
        make_over_time_plot,
        df,
        L1_CONDITIONS,
        N_CLASSES_VALUES,
        "test_chi2",
        "testing",
        teeplot_outattrs={
            "dataset": "exploratory-edge-sparsity",
            "metric": "test",
        },
        teeplot_subdir=pathlib.Path(__file__).stem,
        teeplot_show=False,
    ) as _fig:
        pass
    mo.output.append(_fig)
    plt.close(_fig)
    return


@app.cell
def render_train_loss_over_time(
    L1_CONDITIONS,
    N_CLASSES_VALUES,
    df,
    make_over_time_plot,
    mo,
    pathlib,
    plt,
    tp,
):
    with tp.teed(
        make_over_time_plot,
        df,
        L1_CONDITIONS,
        N_CLASSES_VALUES,
        "pure_train_chi2",
        "training (actual)",
        teeplot_outattrs={
            "dataset": "exploratory-edge-sparsity",
            "metric": "train",
        },
        teeplot_subdir=pathlib.Path(__file__).stem,
        teeplot_show=False,
    ) as _fig:
        pass
    mo.output.append(_fig)
    plt.close(_fig)
    return


if __name__ == "__main__":
    app.run()
