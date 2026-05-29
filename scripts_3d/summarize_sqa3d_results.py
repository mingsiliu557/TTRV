#!/usr/bin/env python3
import argparse
import re
from pathlib import Path


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
STEP_RE = re.compile(r"step:(\d+)\s+-\s+(.*)")
METRIC_RE = re.compile(r"([A-Za-z0-9_./@-]+):(-?\d+(?:\.\d+)?(?:e[-+]?\d+)?)")
PRETTY_METRIC_RE = re.compile(
    r"'([^']+)':\s+(?:np\.float64\()?(-?\d+(?:\.\d+)?(?:e[-+]?\d+)?)(?:\))?"
)
ANSWER_TYPES = ("what", "is", "how", "can", "which", "other")
VIEWS = ("baseline", "sample-level", "dataset-level")


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text).replace("\r", "\n")


def parse_step_metrics(log_file: Path) -> list[tuple[int, dict[str, float]]]:
    lines = strip_ansi(log_file.read_text(errors="ignore")).splitlines()
    metric_blocks = []
    current_block = None
    current_start = 0
    for line_idx, line in enumerate(lines):
        values = {
            name: float(value)
            for name, value in PRETTY_METRIC_RE.findall(line)
            if name.startswith("val-")
        }
        if values:
            if current_block is None:
                current_block = {}
                current_start = line_idx
            current_block.update(values)
        if current_block is not None and line.rstrip().endswith('}")'):
            metric_blocks.append((current_start, line_idx, current_block))
            current_block = None
    if current_block is not None:
        metric_blocks.append((current_start, len(lines) - 1, current_block))

    metrics_by_step = []
    for line_idx, line in enumerate(lines):
        match = STEP_RE.search(line)
        if not match:
            continue
        step = int(match.group(1))
        metrics = {name: float(value) for name, value in METRIC_RE.findall(match.group(2))}
        if "val-sqa3d/sqa3d/em/overall/baseline" in metrics:
            nearest_metrics = None
            nearest_distance = 10**9
            for block_start, block_end, block_metrics in metric_blocks:
                if "val-sqa3d/sqa3d/em/overall/baseline" not in block_metrics:
                    continue
                distance = 0
                if line_idx < block_start:
                    distance = block_start - line_idx
                elif line_idx > block_end:
                    distance = line_idx - block_end
                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_metrics = block_metrics
            if nearest_metrics is not None and nearest_distance <= 250:
                metrics.update(nearest_metrics)
        if metrics:
            metrics_by_step.append((step, metrics))
    return metrics_by_step


def metric(metrics: dict[str, float], name: str) -> str:
    value = metrics.get(name)
    return "n/a" if value is None else f"{value:.4f}"


def max_metric(steps: list[tuple[int, dict[str, float]]], name: str) -> str:
    values = [metrics[name] for _, metrics in steps if name in metrics]
    return "n/a" if not values else f"{max(values):.4f}"


def last_metric(steps: list[tuple[int, dict[str, float]]], name: str) -> str:
    for _, metrics in reversed(steps):
        if name in metrics:
            return f"{metrics[name]:.4f}"
    return "n/a"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--title", default="SQA3D TTRV Results")
    args = parser.parse_args()

    log_file = Path(args.log_file)
    output_file = Path(args.output_file)
    steps = parse_step_metrics(log_file)
    val_steps = [
        (step, metrics)
        for step, metrics in steps
        if "val-sqa3d/sqa3d/em/overall/baseline" in metrics
    ]
    if not val_steps:
        raise RuntimeError(f"No SQA3D validation metrics found in {log_file}")

    first_step, first_metrics = val_steps[0]
    final_step, final_metrics = val_steps[-1]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        f.write(f"# {args.title}\n\n")
        f.write("## Stepwise Official SQA3D EM\n\n")
        f.write("| step | baseline | sample-level | dataset-level |\n")
        f.write("| ---: | ---: | ---: | ---: |\n")
        for step, metrics in val_steps:
            row = [metric(metrics, f"val-sqa3d/sqa3d/em/overall/{view}") for view in VIEWS]
            f.write(f"| {step} | {' | '.join(row)} |\n")
        f.write("\n")

        f.write("## Stepwise Sanity Metrics\n\n")
        f.write("| step | official_em | strict_em | closed_em | unknown | closed_unknown | blank |\n")
        f.write("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for step, metrics in val_steps:
            row = [
                metric(metrics, "val-sqa3d/sqa3d/em/overall/baseline"),
                metric(metrics, "val-sqa3d/sqa3d/strict_em/overall/baseline"),
                metric(metrics, "val-sqa3d/sqa3d/closed_em/overall/baseline"),
                metric(metrics, "val-sqa3d/sqa3d/unknown/overall/baseline"),
                metric(metrics, "val-sqa3d/sqa3d/closed_unknown/overall/baseline"),
                metric(metrics, "val-sqa3d/sqa3d/blank/overall/baseline"),
            ]
            f.write(f"| {step} | {' | '.join(row)} |\n")
        f.write("\n")

        for title, metric_name in (
            ("Official SQA3D EM", "em"),
            ("Strict Exact EM", "strict_em"),
            ("Closed-Set Diagnostic EM", "closed_em"),
        ):
            f.write(f"## {title}\n\n")
            f.write("| split | baseline | sample-level | dataset-level |\n")
            f.write("| --- | ---: | ---: | ---: |\n")
            for label, step, metrics in (
                ("initial", first_step, first_metrics),
                ("final", final_step, final_metrics),
            ):
                row = [metric(metrics, f"val-sqa3d/sqa3d/{metric_name}/overall/{view}") for view in VIEWS]
                f.write(f"| {label} (step={step}) | {' | '.join(row)} |\n")
            f.write("\n")

        f.write("## Final Official Per-Type EM\n\n")
        f.write("| question_type | baseline | sample-level | dataset-level |\n")
        f.write("| --- | ---: | ---: | ---: |\n")
        for answer_type in ANSWER_TYPES:
            row = [metric(final_metrics, f"val-sqa3d/sqa3d/em/{answer_type}/{view}") for view in VIEWS]
            f.write(f"| {answer_type} | {' | '.join(row)} |\n")

        f.write("\n## Answer-Space Unknown Rate\n\n")
        f.write("| split | baseline | sample-level | dataset-level |\n")
        f.write("| --- | ---: | ---: | ---: |\n")
        for label, step, metrics in (
            ("initial", first_step, first_metrics),
            ("final", final_step, final_metrics),
        ):
            row = [metric(metrics, f"val-sqa3d/sqa3d/unknown/overall/{view}") for view in VIEWS]
            f.write(f"| {label} (step={step}) | {' | '.join(row)} |\n")

        f.write("\n## Closed-Set Unknown Rate\n\n")
        f.write("| split | baseline | sample-level | dataset-level |\n")
        f.write("| --- | ---: | ---: | ---: |\n")
        for label, step, metrics in (
            ("initial", first_step, first_metrics),
            ("final", final_step, final_metrics),
        ):
            row = [metric(metrics, f"val-sqa3d/sqa3d/closed_unknown/overall/{view}") for view in VIEWS]
            f.write(f"| {label} (step={step}) | {' | '.join(row)} |\n")

        f.write("\n## Blank Rate\n\n")
        f.write("| split | baseline | sample-level | dataset-level |\n")
        f.write("| --- | ---: | ---: | ---: |\n")
        for label, step, metrics in (
            ("initial", first_step, first_metrics),
            ("final", final_step, final_metrics),
        ):
            row = [metric(metrics, f"val-sqa3d/sqa3d/blank/overall/{view}") for view in VIEWS]
            f.write(f"| {label} (step={step}) | {' | '.join(row)} |\n")

        f.write("\n## Runtime Diagnostics\n\n")
        f.write("| metric | value |\n")
        f.write("| --- | ---: |\n")
        runtime_metrics = (
            ("max_memory_allocated_gb", "perf/max_memory_allocated_gb", "max"),
            ("max_memory_reserved_gb", "perf/max_memory_reserved_gb", "max"),
            ("max_total_num_tokens", "perf/total_num_tokens", "max"),
            ("max_prompt_length_mean", "prompt_length/mean", "max"),
            ("max_response_length_mean", "response_length/mean", "max"),
            ("final_actor_grad_norm", "actor/grad_norm", "last"),
            ("final_actor_lr", "actor/lr", "last"),
            ("final_actor_lr_x1e6", "actor/lr_x1e6", "last"),
            ("final_actor_pg_loss", "actor/pg_loss", "last"),
            ("final_param_checksum_delta", "actor/param_checksum_delta", "last"),
            ("final_param_checksum_delta_abs_scaled", "actor/param_checksum_delta_abs_scaled", "last"),
            ("final_param_sample_delta_abs_mean", "actor/param_sample_delta_abs_mean", "last"),
            ("final_param_sample_delta_abs_max", "actor/param_sample_delta_abs_max", "last"),
            ("final_param_sample_delta_abs_sum", "actor/param_sample_delta_abs_sum", "last"),
            ("final_param_sample_count", "actor/param_sample_count", "last"),
            ("final_sqa3d_freq_mean", "train/sqa3d/freq_mean", "last"),
            ("final_sqa3d_freq_max", "train/sqa3d/freq_max", "last"),
            ("final_sqa3d_unique_pred_ratio", "train/sqa3d/unique_pred_ratio", "last"),
            ("final_sqa3d_unknown_rate", "train/sqa3d/unknown_rate", "last"),
            ("final_sqa3d_blank_rate", "train/sqa3d/blank_rate", "last"),
            ("final_sqa3d_entropy_mean", "train/sqa3d/entropy_mean", "last"),
            ("final_sqa3d_raw_entropy_mean", "train/sqa3d/raw_entropy_mean", "last"),
            ("final_sqa3d_pass@votes", "train/sqa3d/pass@votes", "last"),
            ("final_sqa3d_em_mean", "train/sqa3d/em_mean", "last"),
        )
        for label, name, reducer in runtime_metrics:
            value = max_metric(steps, name) if reducer == "max" else last_metric(steps, name)
            f.write(f"| {label} | {value} |\n")
        f.write(f"\n- Log file: `{log_file}`\n")


if __name__ == "__main__":
    main()
