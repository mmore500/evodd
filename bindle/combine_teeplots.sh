#!/bin/bash

################################################################################
echo
echo "running combine_teeplots.sh"
echo "---------------------------------------------"
################################################################################

# fail on error
set -e

################################################################################
echo
echo "other initialization"
echo "--------------------"
################################################################################

# adapted from https://stackoverflow.com/a/24114056
script_dir="$(dirname -- "$BASH_SOURCE")"
echo "script_dir ${script_dir}"

################################################################################
echo
echo "combine each teeplots subdirectory into one pdf"
echo "-------------------------------------------------"
################################################################################

python3 - "${script_dir}/teeplots" << 'PYEOF'
import io
import pathlib
import subprocess
import sys
import tempfile

import matplotlib

matplotlib.use("pdf")
from PIL import Image  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from pypdf import PageObject, PdfReader, PdfWriter, Transformation  # noqa: E402

HEADER_HEIGHT_PT = 20.0
HEADER_FONT_SIZE = 7
SIZE_BUDGET_BYTES = 40_000_000
RASTER_FALLBACK_DPI = 100


def make_header_page(text, width_pt):
    fig = plt.figure(figsize=(width_pt / 72.0, HEADER_HEIGHT_PT / 72.0))
    fig.text(
        0.01, 0.5, text, fontsize=HEADER_FONT_SIZE,
        family="monospace", va="center", ha="left",
    )
    buf = io.BytesIO()
    fig.savefig(buf, format="pdf")
    plt.close(fig)
    buf.seek(0)
    return PdfReader(buf).pages[0]


def stamped_page(source_page, label):
    width = float(source_page.mediabox.width)
    height = float(source_page.mediabox.height)
    stamped = PageObject.create_blank_page(
        width=width, height=height + HEADER_HEIGHT_PT
    )
    stamped.merge_page(source_page)
    stamped.merge_transformed_page(
        make_header_page(label, width),
        Transformation().translate(tx=0, ty=height),
    )
    return stamped


def gs_compress(path):
    compressed = path.with_name(path.stem + ".compressed.pdf")
    subprocess.run(
        [
            "gs", "-sDEVICE=pdfwrite", "-dCompatibilityLevel=1.4",
            "-dPDFSETTINGS=/screen", "-dNOPAUSE", "-dBATCH", "-dQUIET",
            f"-sOutputFile={compressed}", str(path),
        ],
        check=True,
    )
    compressed.replace(path)


def rasterize(path, dpi):
    with tempfile.TemporaryDirectory() as tmp_dir:
        prefix = pathlib.Path(tmp_dir) / "page"
        subprocess.run(
            [
                "gs", "-sDEVICE=png16m", f"-r{dpi}", "-dNOPAUSE",
                "-dBATCH", "-dQUIET", f"-o{prefix}-%04d.png", str(path),
            ],
            check=True,
        )
        images = [
            Image.open(png).convert("RGB")
            for png in sorted(pathlib.Path(tmp_dir).glob("page-*.png"))
        ]
        images[0].save(path, save_all=True, append_images=images[1:])


def combine_subdir(subdir, out_path):
    members = sorted(subdir.glob("*.pdf"))
    if not members:
        return
    writer = PdfWriter()
    for member in members:
        reader = PdfReader(str(member))
        for page in reader.pages:
            writer.add_page(stamped_page(page, str(member)))
    with open(out_path, "wb") as f:
        writer.write(f)

    gs_compress(out_path)
    if out_path.stat().st_size > SIZE_BUDGET_BYTES:
        rasterize(out_path, RASTER_FALLBACK_DPI)

    size_mb = out_path.stat().st_size / 1e6
    print(f"{out_path} <- {len(members)} member(s), {size_mb:.1f} MB")


teeplots_root = pathlib.Path(sys.argv[1])
if teeplots_root.is_dir():
    for subdir in sorted(teeplots_root.iterdir()):
        if subdir.is_dir():
            combine_subdir(subdir, teeplots_root / f"{subdir.name}.pdf")
PYEOF

################################################################################
echo
echo "recurse to subdirectories"
echo "-------------------------"
################################################################################

shopt -s nullglob

for script in "${script_dir}/"*/combine_teeplots.sh; do
  "${script}"
done

shopt -u nullglob
