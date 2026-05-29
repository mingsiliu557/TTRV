#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
STEP_RE = re.compile(r"step:(\d+)\s+-\s+(.*)")
METRIC_RE = re.compile(r"([A-Za-z0-9_./@-]+):(-?\d+(?:\.\d+)?(?:e[-+]?\d+)?)")
AT_RE = re.compile(r"@(\d+)")


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text).replace("\r", "\n")


def parse_step_metrics(log_file: Path) -> list[tuple[int, dict[str, float]]]:
    metrics_by_step = []
    for line in strip_ansi(log_file.read_text(errors="ignore")).splitlines():
        match = STEP_RE.search(line)
        if not match:
            continue
        step = int(match.group(1))
        metrics = {name: float(value) for name, value in METRIC_RE.findall(match.group(2))}
        if metrics:
            metrics_by_step.append((step, metrics))
    return metrics_by_step


def select_metric(metrics: dict[str, float], prefix: str) -> float | None:
    candidates = [(name, value) for name, value in metrics.items() if name.startswith(prefix)]
    if not candidates:
        return None

    def rank(item):
        name, _ = item
        match = AT_RE.search(name)
        return int(match.group(1)) if match else -1

    return max(candidates, key=rank)[1]


def format_value(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def load_baseline(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text())
    return data.get("summary", data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--baseline-json")
    parser.add_argument("--title", default="ModelNet40 Free-Form TTRV Results")
    parser.add_argument("--command", default="")
    args = parser.parse_args()

    log_file = Path(args.log_file)
    output_file = Path(args.output_file)
    baseline = load_baseline(Path(args.baseline_json) if args.baseline_json else None)
    steps = parse_step_metrics(log_file)
    validation_steps = [
        (step, metrics)
        for step, metrics in steps
        if any(name.startswith("val-core/modelnet40_freeform/acc/mean@") for name in metrics)
    ]
    if not validation_steps:
        raise RuntimeError(f"No validation metrics found in {log_file}")

    first_step, first_metrics = validation_steps[0]
    final_step, final_metrics = validation_steps[-1]
    first_acc = select_metric(first_metrics, "val-core/modelnet40_freeform/acc/mean@")
    final_acc = select_metric(final_metrics, "val-core/modelnet40_freeform/acc/mean@")

    rows = [
        (
            f"step={first_step}",
            first_acc,
            select_metric(first_metrics, "val-core/modelnet40_freeform/acc/best@"),
            select_metric(first_metrics, "val-core/modelnet40_freeform/acc/maj@"),
            select_metric(first_metrics, "val-aux/modelnet40_freeform/unknown/mean@"),
        ),
        (
            f"step={final_step}",
            final_acc,
            select_metric(final_metrics, "val-core/modelnet40_freeform/acc/best@"),
            select_metric(final_metrics, "val-core/modelnet40_freeform/acc/maj@"),
            select_metric(final_metrics, "val-aux/modelnet40_freeform/unknown/mean@"),
        ),
    ]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w") as f:
        f.write(f"# {args.title}\n\n")
        if baseline:
            f.write("## Standalone Greedy Baseline\n\n")
            f.write(f"- N: {baseline.get('n', 'n/a')}\n")
            f.write(f"- Accuracy: {format_value(baseline.get('accuracy'))}\n")
            f.write(f"- Unknown rate: {format_value(baseline.get('unknown_rate'))}\n\n")
        f.write("## verl TTRV Validation\n\n")
        f.write("| split | acc_mean | acc_best | acc_majority | unknown_mean |\n")
        f.write("| --- | ---: | ---: | ---: | ---: |\n")
        for label, acc, best, majority, unknown in rows:
            f.write(
                f"| {label} | {format_value(acc)} | {format_value(best)} | "
                f"{format_value(majority)} | {format_value(unknown)} |\n"
            )
        if first_acc is not None and final_acc is not None:
            f.write(f"\n- Final - step0 acc delta: {final_acc - first_acc:+.4f}\n")
        f.write(f"- Log file: `{log_file}`\n")
        if args.command:
            f.write(f"- Command: `{args.command}`\n")


if __name__ == "__main__":
    main()
