"""
Run the qis reference runner and the AI-written scripts and compare their outputs.

Each script is run unchanged as a subprocess in its own subfolder next to this file, so the
nav.csv files do not overwrite each other. The stdout and stderr of each script are saved in its
subfolder. The eight lines each script prints are tabulated side by side, and each AI nav is
compared with nav_reference.csv on their common dates.

Place this file next to qis_blog_examples.py and ai_1.py to ai_4.py (a missing script is
skipped) and run it in an environment with qis[data]. It writes run_info.txt,
ai_comparison_table.csv and nav_comparison.csv next to itself. It computes no statistics of its
own: every number is one a script printed, or a difference of two navs.
"""

# packages
import platform
import subprocess
import sys
import pandas as pd
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Dict, List, Optional

HERE: Path = Path(__file__).resolve().parent
SCRIPTS: Dict[str, str] = {
    "reference": "qis_blog_examples.py",
    "ai_1": "ai_1.py",
    "ai_2": "ai_2.py",
    "ai_3": "ai_3.py",
    "ai_4": "ai_4.py",
}
NAV_FILES: Dict[str, str] = {
    "reference": "nav_reference.csv",
    "ai_1": "nav.csv",
    "ai_2": "nav.csv",
    "ai_3": "nav.csv",
    "ai_4": "nav.csv",
}
LABELS: List[str] = [
    "first_date",
    "last_date",
    "terminal_nav",
    "total_return",
    "annual_return",
    "volatility",
    "sharpe",
    "max_drawdown",
]
PACKAGES: List[str] = ["qis", "yfinance", "pandas", "numpy", "numba", "matplotlib"]


def write_run_info() -> None:
    """record when and with which interpreter and package versions the scripts were run."""
    lines = [
        f"run started: {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"python: {sys.version.split()[0]} at {sys.executable}",
        f"platform: {platform.platform()}",
    ]
    for package in PACKAGES:
        try:
            lines.append(f"{package}: {metadata.version(package)}")
        except metadata.PackageNotFoundError:
            lines.append(f"{package}: not installed")
    (HERE / "run_info.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


def run_script(name: str) -> Dict[str, str]:
    """run one script in its own subfolder and return the eight printed lines as label -> text."""
    workdir = HERE / f"output_{name}"
    workdir.mkdir(exist_ok=True)
    print(f"running {SCRIPTS[name]} ...", flush=True)
    result = subprocess.run(
        [sys.executable, str(HERE / SCRIPTS[name])],
        cwd=workdir,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    (workdir / "stdout.txt").write_text(result.stdout, encoding="utf-8")
    (workdir / "stderr.txt").write_text(result.stderr, encoding="utf-8")
    printed = {"ran": "yes" if result.returncode == 0 else f"no (exit {result.returncode})"}
    if result.returncode != 0:
        print(
            f"{name} failed; last lines of stderr:\n"
            + "\n".join(result.stderr.strip().splitlines()[-6:]),
            flush=True,
        )
    for line in result.stdout.splitlines():
        label, _, value = line.partition(":")
        if label.strip() in LABELS:
            printed[label.strip()] = value.strip()  # the last occurrence wins
    return printed


def load_nav(name: str) -> Optional[pd.Series]:
    """read the nav a script saved, or None when the file is missing."""
    path = HERE / f"output_{name}" / NAV_FILES[name]
    if not path.exists():
        return None
    frame = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    return frame["nav"].rename(name)


if __name__ == "__main__":
    write_run_info()
    names = [name for name, file_name in SCRIPTS.items() if (HERE / file_name).exists()]
    table = pd.DataFrame({name: run_script(name=name) for name in names}).reindex(["ran"] + LABELS)
    navs = {name: load_nav(name=name) for name in names}
    reference = navs.get("reference")
    if reference is not None:
        for name, nav in navs.items():
            if nav is None or name == "reference":
                continue
            common = reference.index.intersection(nav.index)
            difference = (nav.loc[common] - reference.loc[common]).abs()
            table.loc["common_days", name] = str(len(common))
            table.loc["max_abs_nav_difference", name] = f"{difference.max():.2e}"
        pd.concat([nav for nav in navs.values() if nav is not None], axis=1).to_csv(
            HERE / "nav_comparison.csv"
        )
    table.to_csv(HERE / "ai_comparison_table.csv")
    print(table.fillna("").to_string(), flush=True)
