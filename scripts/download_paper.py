#!/usr/bin/env python3
"""Download a research paper PDF and convert it to markdown with marker-pdf."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import unquote, urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
PAPERS_DIR = REPO_ROOT / "doc" / "papers"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a research paper and convert it to markdown with marker-pdf."
    )
    parser.add_argument("url", type=str, help="URL of the PDF to download.")
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Paper title used for the output filename (spaces become underscores).",
    )
    return parser.parse_args()


def title_to_filename(title: str) -> str:
    """Convert a paper title into a safe markdown filename stem."""
    name = title.strip().replace(" ", "_")
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", name)
    name = re.sub(r"_+", "_", name).strip("._")
    if not name:
        raise ValueError("Title resolves to an empty filename.")
    return name


def filename_from_url(url: str) -> str:
    path = unquote(urlparse(url).path)
    stem = Path(path).stem
    return stem or "paper"


def download_pdf(url: str, dest_dir: Path) -> Path:
    """Download a PDF from ``url`` into ``dest_dir`` and return its path."""
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "unitree_rl_lab/download_paper"},
    )
    try:
        with urllib.request.urlopen(request) as response:
            content_type = (response.headers.get_content_type() or "").lower()
            disposition = response.headers.get("Content-Disposition", "")
            data = response.read()
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to download PDF from {url}: {exc}") from exc

    filename = None
    match = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', disposition, re.I)
    if match:
        filename = Path(unquote(match.group(1).strip())).name

    if not filename:
        filename = filename_from_url(url)
        if not filename.lower().endswith(".pdf"):
            filename = f"{filename}.pdf"

    if content_type and "pdf" not in content_type and not data.startswith(b"%PDF"):
        raise RuntimeError(
            f"Downloaded content does not look like a PDF (Content-Type: {content_type or 'unknown'})."
        )

    pdf_path = dest_dir / filename
    pdf_path.write_bytes(data)
    return pdf_path


def run_marker(pdf_path: Path, output_dir: Path) -> Path:
    """Convert ``pdf_path`` with marker_single and return the generated markdown path."""
    marker_bin = shutil.which("marker_single")
    if marker_bin is None:
        raise RuntimeError(
            "marker_single not found on PATH. Install it with: pip install marker-pdf"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        marker_bin,
        str(pdf_path),
        "--output_dir",
        str(output_dir),
        "--output_format",
        "markdown",
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"marker_single failed with exit code {result.returncode}.")

    md_files = sorted(output_dir.rglob("*.md"))
    if not md_files:
        raise FileNotFoundError(f"No markdown file produced under {output_dir}.")
    return md_files[0]


def cleanup_marker_artifacts(marker_output_dir: Path, keep_md: Path) -> None:
    """Delete marker-extracted images and metadata; keep only the chosen markdown file."""
    for path in marker_output_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.resolve() == keep_md.resolve():
            continue
        path.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="download_paper_") as tmp:
        tmp_dir = Path(tmp)
        download_dir = tmp_dir / "download"
        marker_dir = tmp_dir / "marker"
        download_dir.mkdir()

        print(f"Downloading: {args.url}")
        pdf_path = download_pdf(args.url, download_dir)
        print(f"Saved PDF to temporary path: {pdf_path}")

        title = args.title or pdf_path.stem
        out_name = title_to_filename(title)
        out_path = PAPERS_DIR / f"{out_name}.md"

        md_path = run_marker(pdf_path, marker_dir)
        cleanup_marker_artifacts(marker_dir, md_path)

        out_path.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Wrote markdown: {out_path}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - surface a clean CLI error
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
