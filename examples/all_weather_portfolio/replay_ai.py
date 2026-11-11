"""Replay the original AI accounting using archived inputs instead of live market-data calls."""

import argparse
from contextlib import chdir, redirect_stderr, redirect_stdout
from pathlib import Path
import runpy
import subprocess
import sys
from unittest.mock import patch

import pandas as pd

import chatgpt_high
import deepseek

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "ai"


def levels(name):
    return pd.read_csv(DATA / name, index_col=0, parse_dates=True, float_precision="round_trip")


def run_in(folder, operation):
    folder.mkdir(parents=True, exist_ok=True)
    with (
        chdir(folder),
        open("stdout.txt", "w", encoding="utf-8") as out,
        open("stderr.txt", "w", encoding="utf-8") as err,
        redirect_stdout(out),
        redirect_stderr(err),
    ):
        operation()


def replay_three(target):
    for case, policy in (
        ("claude_opus_5_5_high", "cash"),
        ("claude_renormalize", "renormalize"),
    ):
        work = target / case
        work.mkdir(parents=True, exist_ok=True)
        with (
            (work / "stdout.txt").open("w", encoding="utf-8") as out,
            (work / "stderr.txt").open("w", encoding="utf-8") as err,
        ):
            subprocess.run(
                [
                    sys.executable,
                    str(HERE / "claude_opus_5_5_high.py"),
                    "--cache-dir",
                    str(DATA / "claude"),
                    "--outdir",
                    "results",
                    "--missing",
                    policy,
                ],
                cwd=work,
                stdout=out,
                stderr=err,
                check=True,
            )

    yahoo = levels("deepseek_adjusted_close.csv")
    with (
        patch.object(
            deepseek,
            "download_adj_close",
            side_effect=lambda ticker: yahoo[ticker].dropna(),
        ),
        patch.object(
            deepseek,
            "download_yield",
            return_value=levels("irx_percent.csv").iloc[:, 0].dropna(),
        ),
    ):
        run_in(target / "deepseek", deepseek.main)

    saved = {
        "close": levels("bullion.csv").iloc[:, 0],
        "DGS3MO": levels("dgs3mo.csv").iloc[:, 0],
    }
    with (
        patch.object(
            chatgpt_high,
            "fetch_etfs",
            return_value=levels("chatgpt_adjusted_close.csv"),
        ),
        patch.object(
            chatgpt_high,
            "fetch_csv_series",
            side_effect=lambda url, preferred: saved[preferred].copy(),
        ),
    ):
        run_in(target / "chatgpt_high", chatgpt_high.main)


def replay_four(target):
    import yfinance as yf

    # All four earlier prompts request adjusted ETF closes on exchange sessions.
    # The NAV comparison from the original date is also archived alongside the scripts.
    source = HERE / "experiments" / "four_prompts"
    panel = levels("claude/prices.csv")

    def download(tickers, start, end, **kwargs):
        sample = panel.loc[
            (panel.index >= pd.Timestamp(start)) & (panel.index < pd.Timestamp(end)),
            list(tickers),
        ]
        return pd.concat({"Close": sample}, axis=1)

    with patch.object(yf, "download", side_effect=download):
        for name in ("qis_blog_examples", "ai_1", "ai_2", "ai_3", "ai_4"):
            run_in(
                target / name,
                lambda: runpy.run_path(str(source / f"{name}.py"), run_name="__main__"),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    target = args.output_dir.resolve()
    replay_three(target / "three_models")
    replay_four(target / "four_prompts")
    print(f"AI outputs written to {target}")
