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
    import marimo as mo
    from matplotlib.colors import SymLogNorm
    import matplotlib.pyplot as plt
    import pandas as pd
    from teeplot import teeplot as tp
    from watermark import watermark

    return SymLogNorm, mo, pd, plt, tp, watermark


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
    # Edge sparsity L1 x L2 grid (2026-08-01): loss vs. regularization strength

    Downloads the collated timeseries parquet file for the
    `2026-08-01-edge-sparsity-grid` SLURM batch job (see
    `slurm/2026-08-01/2026-08-01-edge-sparsity-grid.sh`) from OSF, then
    renders three 5-panel heatmap grids -- one panel per swept
    `n_zero_edges` value (`0, 100, 125, 150, 175`, chosen to straddle the
    sparsity-driven phenotype-composition collapse observed around
    `n_zero_edges` ~95-110 in
    `bindle/2026-07-31-exploratory-edge-sparsity-analysis.py`) -- each
    panel a `l1_scale` (x, log) x `l2_scale` (y, log) heatmap, median
    across the 3 replicate seeds:

    1. Final testing chi^2 error.
    2. Final training (actual) chi^2 error.
    3. Final "classified" fraction (`1 - other_frac`, i.e. the share of
       the population landing on any of the 8 canonical `CLASS_8`
       phenotypes rather than "other") -- this is the metric that most
       directly showed the sparsity collapse in ad-hoc exploration of
       this dataset (and of `osf.io/6vrxc`/`osf.io/ykp74`, similar
       independent-L1/L2 grids not yet backed by a committed slurm
       script), so it's included here as its own panel rather than only
       inferred from the chi^2 plots.

    Unlike this project's other edge-sparsity analysis notebooks, L1 and
    L2 here are swept INDEPENDENTLY (`l1_scale` in 9 values from 0.0001
    to 1.0, `l2_scale` in 7 values from 0.00001 to 0.01, both in
    half-decade steps -- NOT constrained to `l1_scale + l2_scale = 1`
    like every other sweep in this project) -- so a heatmap over the
    (l1_scale, l2_scale) plane is the natural layout, rather than the
    line/stackplot-vs-n_zero_edges or 2x2-condition-panel layouts used
    elsewhere. `n_classes=3` and `doubled_class=-1` (uniform
    presentation) are both fixed for the whole dataset.
    """)
    return


@app.cell(hide_code=True)
def delimit_fetch_data(mo):
    mo.md("""
    ## Fetch data
    """)
    return


@app.cell
def osf_slugs():
    # https://osf.io/z3jbe -- 2026-08-01-edge-sparsity-grid collated
    #   timeseries (l1_scale x l2_scale independently swept; n_classes=3,
    #   doubled_class=-1 fixed; n_zero_edges in {0, 100, 125, 150, 175})
    OSF_SLUGS = {
        "edge-sparsity-grid": "z3jbe",
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
    mo.md("""
    ## Load data
    """)
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
    mo.md("""
    ## Verify the fixed condition, prepare the L1 x L2 grid

    `n_classes` and `doubled_class` should each be a single fixed value
    throughout this dataset, matching
    `slurm/2026-08-01/2026-08-01-edge-sparsity-grid.sh`; `n_zero_edges`,
    `l1_scale`, and `l2_scale` are the 3 swept axes.
    """)
    return


@app.cell
def verify_fixed_condition(df):
    assert df["n_classes"].nunique() == 1 and df["n_classes"].iloc[0] == 3
    assert (
        df["doubled_class"].nunique() == 1
        and df["doubled_class"].iloc[0] == -1
    )
    n_zero_values = sorted(df["n_zero_edges"].unique())
    l1_values = sorted(df["l1_scale"].unique())
    l2_values = sorted(df["l2_scale"].unique())
    return l1_values, l2_values, n_zero_values


@app.cell
def show_condition_summary(df, l1_values, l2_values, n_zero_values, pd):
    condition_summary_df = pd.DataFrame(
        [
            {
                "n_classes": df["n_classes"].iloc[0],
                "doubled_class": df["doubled_class"].iloc[0],
                "n_zero_edges_values": str(n_zero_values),
                "n_l1_values": len(l1_values),
                "n_l2_values": len(l2_values),
                "n_seeds": df["seed"].nunique(),
                "n_replicates": df["replicate_uid"].nunique(),
            }
        ]
    )
    condition_summary_df
    return


@app.cell(hide_code=True)
def delimit_heatmap_helpers(mo):
    mo.md("""
    ## Heatmap grid helper

    Each figure below is a 1-row-per-`n_zero_edges` (well, laid out as a
    single row of panels, one per value) grid of `l1_scale` (x, log) x
    `l2_scale` (y, log) heatmaps, colored by a per-cell statistic
    computed from each replicate's own FINAL recorded generation
    (median across the 3 seeds at each grid cell) -- not a single shared
    max generation across the whole sweep, since replicates can be
    truncated at different generations by a SLURM timeout (see
    `bindle/2026-07-30-exploratory-edge-sparsity.py`'s progressive
    save-out; in practice every replicate in this dataset reaches the
    full 500,000,400-generation budget). The chi^2 colormaps use the
    same symlog normalization (linear below 1.0, log above) and viridis
    colormap as this project's other double-descent heatmaps; the
    classified-fraction colormap is plain linear over `[0, 1]`.
    """)
    return


@app.cell
def heatmap_grid_fn(SymLogNorm, plt):
    def make_l1l2_heatmap_grid(
        df, n_zero_values, l1_values, l2_values, metric_col, metric_label, kind
    ):
        final = (
            df.sort_values("generation")
            .groupby(
                ["n_zero_edges", "l1_scale", "l2_scale", "seed"], observed=True
            )
            .tail(1)
        )
        if metric_col == "classified_frac":
            final = final.assign(classified_frac=1.0 - final["other_frac"])

        agg = (
            final.groupby(
                ["n_zero_edges", "l1_scale", "l2_scale"], observed=True
            )[metric_col]
            .median()
            .reset_index()
        )

        if kind == "chi2":
            CHI2_LINTHRESH = 1.0
            norm = SymLogNorm(linthresh=CHI2_LINTHRESH, vmin=0)
            cmap = "viridis"
        else:
            norm = None
            cmap = "viridis"

        n_panels = len(n_zero_values)
        fig, axes = plt.subplots(
            1, n_panels, figsize=(4.2 * n_panels, 4.5), dpi=90, sharey=True
        )
        if n_panels == 1:
            axes = [axes]

        # l1_values/l2_values are log-spaced but NOT evenly so on a linear
        # scale -- pcolormesh's default cell-edge computation uses linear
        # midpoints between the given coordinates, so feeding it these
        # values directly with a log-scaled axis blows the outermost
        # cells' edges out to near-zero (rendering as e.g. 1e-19 on a log
        # axis instead of the actual smallest swept value). Plotting on
        # plain integer index positions instead -- then relabeling ticks
        # with the actual values -- sidesteps that entirely and is the
        # more natural representation anyway, since l1_scale/l2_scale are
        # a small set of discrete swept values, not a continuous field.
        mesh = None
        for ax, n_val in zip(axes, n_zero_values):
            sub = agg[agg["n_zero_edges"] == n_val]
            grid = sub.pivot(
                index="l2_scale", columns="l1_scale", values=metric_col
            ).reindex(index=l2_values, columns=l1_values)
            mesh = ax.pcolormesh(
                grid.to_numpy(),
                cmap=cmap,
                norm=norm,
                vmin=(0.0 if kind != "chi2" else None),
                vmax=(1.0 if kind != "chi2" else None),
                shading="flat",
            )
            ax.set_xticks(
                [i + 0.5 for i in range(len(l1_values))],
                [f"{v:g}" for v in l1_values],
                rotation=90,
                fontsize=7,
            )
            ax.set_yticks(
                [i + 0.5 for i in range(len(l2_values))],
                [f"{v:g}" for v in l2_values],
                fontsize=7,
            )
            ax.set_xlabel("l1_scale")
            ax.set_title(f"n_zero_edges={n_val}", fontsize=10)
        axes[0].set_ylabel("l2_scale")

        cbar = fig.colorbar(mesh, ax=axes, pad=0.02, aspect=30)
        cbar.set_label(f"median final {metric_label}")
        fig.suptitle(
            f"edge-sparsity L1xL2 grid: final {metric_label} "
            "vs. l1_scale/l2_scale",
            fontsize=13,
        )
        return fig

    return (make_l1l2_heatmap_grid,)


@app.cell(hide_code=True)
def delimit_test_loss_heatmap(mo):
    mo.md("""
    ## Plot 1: final testing loss heatmap
    """)
    return


@app.cell
def render_test_loss_heatmap(
    df,
    l1_values,
    l2_values,
    make_l1l2_heatmap_grid,
    mo,
    n_zero_values,
    pathlib,
    plt,
    tp,
):
    with tp.teed(
        make_l1l2_heatmap_grid,
        df,
        n_zero_values,
        l1_values,
        l2_values,
        "test_chi2",
        "testing chi$^2$ error",
        "chi2",
        teeplot_outattrs={"dataset": "edge-sparsity-grid", "metric": "test"},
        teeplot_subdir=pathlib.Path(__file__).stem,
        teeplot_show=False,
    ) as _fig:
        pass
    mo.output.append(_fig)
    plt.close(_fig)
    return


@app.cell(hide_code=True)
def delimit_train_loss_heatmap(mo):
    mo.md("""
    ## Plot 2: final training (actual) loss heatmap
    """)
    return


@app.cell
def render_train_loss_heatmap(
    df,
    l1_values,
    l2_values,
    make_l1l2_heatmap_grid,
    mo,
    n_zero_values,
    pathlib,
    plt,
    tp,
):
    with tp.teed(
        make_l1l2_heatmap_grid,
        df,
        n_zero_values,
        l1_values,
        l2_values,
        "pure_train_chi2",
        "training (actual) chi$^2$ error",
        "chi2",
        teeplot_outattrs={"dataset": "edge-sparsity-grid", "metric": "train"},
        teeplot_subdir=pathlib.Path(__file__).stem,
        teeplot_show=False,
    ) as _fig:
        pass
    mo.output.append(_fig)
    plt.close(_fig)
    return


@app.cell(hide_code=True)
def delimit_classified_heatmap(mo):
    mo.md("""
    ## Plot 3: final classified fraction heatmap (1 - other_frac)
    """)
    return


@app.cell
def render_classified_heatmap(
    df,
    l1_values,
    l2_values,
    make_l1l2_heatmap_grid,
    mo,
    n_zero_values,
    pathlib,
    plt,
    tp,
):
    with tp.teed(
        make_l1l2_heatmap_grid,
        df,
        n_zero_values,
        l1_values,
        l2_values,
        "classified_frac",
        "classified fraction (1 - other_frac)",
        "fraction",
        teeplot_outattrs={
            "dataset": "edge-sparsity-grid",
            "metric": "classified",
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
