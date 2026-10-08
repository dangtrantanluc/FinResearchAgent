"""Build notebooks/train_growth_model.ipynb from its .py source.

The notebook must run on Kaggle from a single uploaded file, so the
`finresearch` package and the reference table are embedded as %%writefile
cells. Edit the .py sources, then re-run this script.

    python scripts/make_notebook.py
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks/train_growth_model.py"
TARGET = ROOT / "notebooks/train_growth_model.ipynb"
EMBED = ["finresearch/__init__.py", "finresearch/extract.py", "finresearch/build.py", "finresearch/panel.py", "reference/companies.csv"]


def cells_from_percent(text: str):
    """Split a `# %%` script into (kind, source) pairs."""
    kind, lines = None, []
    for line in text.split("\n"):
        if line.startswith("# %%"):
            if kind:
                yield kind, "\n".join(lines).strip("\n")
            kind, lines = ("markdown" if "[markdown]" in line else "code"), []
        elif kind == "markdown":
            lines.append(line[2:] if line.startswith("# ") else line.lstrip("#"))
        elif kind:
            lines.append(line)
    if kind:
        yield kind, "\n".join(lines).strip("\n")


def main():
    cells = list(cells_from_percent(SOURCE.read_text(encoding="utf-8")))
    nb = nbf.v4.new_notebook()
    nb.cells.append(nbf.v4.new_markdown_cell(cells[0][1]))
    nb.cells.append(nbf.v4.new_markdown_cell(
        "## 0. Mã nguồn xử lý dữ liệu\n\n"
        "Các ô dưới đây ghi package `finresearch` và bảng tham chiếu mã chứng khoán ra đĩa, "
        "để notebook chạy được mà không cần clone repo. Chúng được sinh tự động từ repo bằng "
        "`scripts/make_notebook.py`; muốn sửa thì sửa trong repo rồi sinh lại."
    ))
    nb.cells.append(nbf.v4.new_code_cell('import os\n\nfor d in ("finresearch", "reference"):\n    os.makedirs(d, exist_ok=True)'))
    for rel in EMBED:
        body = (ROOT / rel).read_text(encoding="utf-8")
        nb.cells.append(nbf.v4.new_code_cell(f"%%writefile {rel}\n{body}"))
    for kind, src in cells[1:]:
        nb.cells.append(nbf.v4.new_markdown_cell(src) if kind == "markdown" else nbf.v4.new_code_cell(src))
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python"}
    nbf.write(nb, TARGET)
    print(f"wrote {TARGET.relative_to(ROOT)}: {len(nb.cells)} cells, {TARGET.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
