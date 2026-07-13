# Unitree RL Lab — Agent Guide

## Research Papers

Parsed markdown versions of reference papers are stored in `doc/papers/`.
They can be consulted for background on algorithms, architectures, and methods used in this project.
Each file ends with a `## Notes` section containing the original title and URL.

To see what papers are available, list `doc/papers/`.

### How to add a paper

```bash
python3 scripts/download_paper.py <pdf_url> --title "Paper Title"
```

The markdown is saved to `doc/papers/<Paper_Title>.md`.
