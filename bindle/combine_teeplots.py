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
import sys

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402
from pypdf import (  # noqa: E402
    PageObject,
    PdfReader,
    PdfWriter,
    Transformation,
)

HEADER_HEIGHT_PT = 20.0
HEADER_FONT_SIZE = 7


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


def combine_subdir(subdir: pathlib.Path, out_path: pathlib.Path) -> None:
    members = sorted(subdir.glob("*.pdf"))
    if not members:
        return
    writer = PdfWriter()
    for member in members:
        reader = PdfReader(str(member))
        for page in reader.pages:
            writer.add_page(_stamped_page(page, str(member)))
    print(f"{out_path} <- {len(members)} member(s)")
    with open(out_path, "wb") as f:
        writer.write(f)


def main(teeplots_root: pathlib.Path) -> None:
    if not teeplots_root.is_dir():
        print(f"{teeplots_root} does not exist, nothing to combine")
        return
    for subdir in sorted(teeplots_root.iterdir()):
        if subdir.is_dir():
            combine_subdir(subdir, teeplots_root / f"{subdir.name}.pdf")


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "teeplots"))
