#!/usr/bin/env python3
"""Combine each teeplots/ subdirectory's PDF members into one PDF.

For every immediate subdirectory of the given teeplots root (default:
"teeplots"), collects that subdirectory's *.pdf files (sorted for
determinism), stamps each page with its source file's path in a header
band above the original content, and writes the concatenated result to
"<root>/<subdirectory-name>.pdf" (a sibling of the subdirectory itself).
"""
import io
import pathlib
import subprocess
import sys
import tempfile

import matplotlib

matplotlib.use("pdf")
from PIL import Image  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from pypdf import (  # noqa: E402
    PageObject,
    PdfReader,
    PdfWriter,
    Transformation,
)

HEADER_HEIGHT_PT = 20.0
HEADER_FONT_SIZE = 7
SIZE_BUDGET_BYTES = 40_000_000
RASTER_FALLBACK_DPI = 100


def _make_header_page(text: str, width_pt: float) -> PageObject:
    fig = plt.figure(figsize=(width_pt / 72.0, HEADER_HEIGHT_PT / 72.0))
    fig.text(
        0.01,
        0.5,
        text,
        fontsize=HEADER_FONT_SIZE,
        family="monospace",
        va="center",
        ha="left",
    )
    buf = io.BytesIO()
    fig.savefig(buf, format="pdf")
    plt.close(fig)
    buf.seek(0)
    return PdfReader(buf).pages[0]


def _stamped_page(source_page: PageObject, label: str) -> PageObject:
    # grows the page by HEADER_HEIGHT_PT and merges the original content
    # in at the bottom (unchanged) and a label in the new band at the
    # top, so the two never overlap regardless of source page size.
    width = float(source_page.mediabox.width)
    height = float(source_page.mediabox.height)
    stamped = PageObject.create_blank_page(
        width=width, height=height + HEADER_HEIGHT_PT
    )
    stamped.merge_page(source_page)
    stamped.merge_transformed_page(
        _make_header_page(label, width),
        Transformation().translate(tx=0, ty=height),
    )
    return stamped


def _gs_compress(path: pathlib.Path) -> None:
    compressed = path.with_name(path.stem + ".compressed.pdf")
    subprocess.run(
        [
            "gs",
            "-sDEVICE=pdfwrite",
            "-dCompatibilityLevel=1.4",
            "-dPDFSETTINGS=/screen",
            "-dNOPAUSE",
            "-dBATCH",
            "-dQUIET",
            f"-sOutputFile={compressed}",
            str(path),
        ],
        check=True,
    )
    compressed.replace(path)


def _rasterize(path: pathlib.Path, dpi: int) -> None:
    # last-resort fallback when compression alone isn't enough: forces a
    # hard byte-size ceiling (unlike vector content, whose size scales
    # with plot complexity) by flattening every page to a raster image.
    with tempfile.TemporaryDirectory() as tmp_dir:
        prefix = pathlib.Path(tmp_dir) / "page"
        subprocess.run(
            [
                "gs",
                "-sDEVICE=png16m",
                f"-r{dpi}",
                "-dNOPAUSE",
                "-dBATCH",
                "-dQUIET",
                f"-o{prefix}-%04d.png",
                str(path),
            ],
            check=True,
        )
        images = [
            Image.open(png).convert("RGB")
            for png in sorted(pathlib.Path(tmp_dir).glob("page-*.png"))
        ]
        images[0].save(path, save_all=True, append_images=images[1:])


def combine_subdir(subdir: pathlib.Path, out_path: pathlib.Path) -> None:
    members = sorted(subdir.glob("*.pdf"))
    if not members:
        return
    writer = PdfWriter()
    for member in members:
        reader = PdfReader(str(member))
        for page in reader.pages:
            writer.add_page(_stamped_page(page, str(member)))
    with open(out_path, "wb") as f:
        writer.write(f)

    _gs_compress(out_path)
    if out_path.stat().st_size > SIZE_BUDGET_BYTES:
        _rasterize(out_path, RASTER_FALLBACK_DPI)

    size_mb = out_path.stat().st_size / 1e6
    print(f"{out_path} <- {len(members)} member(s), {size_mb:.1f} MB")


def main(teeplots_root: pathlib.Path) -> None:
    if not teeplots_root.is_dir():
        print(f"{teeplots_root} does not exist, nothing to combine")
        return
    for subdir in sorted(teeplots_root.iterdir()):
        if subdir.is_dir():
            combine_subdir(subdir, teeplots_root / f"{subdir.name}.pdf")


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "teeplots"))
