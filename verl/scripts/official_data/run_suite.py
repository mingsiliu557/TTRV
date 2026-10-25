"""Submit/run sequential gates inside an existing allocation; never calls Slurm.

Each subprocess has a separate lifetime. A cgroup-wide memory guard stops only this
suite's child process tree. Missing inputs or correctness failures block training.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
LIMIT = 4 * 1024 ** 3
RESERVE = 512 * 1024 ** 2


def verify_method_artifacts(directory, method):
    """A successful process exit alone is not evidence that PROVE ran."""
    if method == "base":
        return
    count = 0
    with open(Path(directory) / "train_rollouts.jsonl") as stream:
        for line in stream:
            row = json.loads(line)
            expected_style = "density_cluster_answer_entropy" if method == "prove" else "frequency_entropy"
            if row["reward_style"] != expected_style or len(row["samples"]) != 32:
                raise ValueError("Unexpected training reward/rollout budget")
            if sum(bool(sample["used_for_update"]) for sample in row["samples"]) != 16:
                raise ValueError("Unexpected update-sample budget")
            if not all(math.isfinite(sample["reward"]) for sample in row["samples"]):
                raise ValueError("Nonfinite rollout rewards")
            if method == "prove" and not all(sample.get("evidence_query") for sample in row["samples"]):
                raise ValueError("PROVE evidence-query branch did not produce per-rollout queries")
            if method == "prove" and "density_temperature" not in row:
                raise ValueError("PROVE density diagnostics missing")
            count += 1
    if count == 0:
        raise ValueError("Training emitted no rollout records")
    if method == "prove":
        with open(Path(directory) / "train.log") as stream:
            if not any("reward/visual_dep_raw_mean" in line for line in stream):
                raise ValueError("PROVE visual reward branch has no execution evidence")


def memory_cgroups():
    """Return all readable enclosing memory controllers, including parent limits."""
    candidates = []
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        _, controllers, relative = line.split(":", 2)
        if not controllers:
            base, current, maximum = Path("/sys/fs/cgroup"), "memory.current", "memory.max"
        elif "memory" in controllers.split(","):
            base, current, maximum = Path("/sys/fs/cgroup/memory"), "memory.usage_in_bytes", "memory.limit_in_bytes"
        else:
            continue
        path = base / relative.lstrip("/")
        while path == base or base in path.parents:
            if (path / current).is_file() and (path / maximum).is_file():
                value = (path / maximum).read_text().strip()
                candidates.append((path / current, int(value) if value != "max" else None))
            if path == base:
                break
            path = path.parent
    if not candidates:
        raise RuntimeError("Cannot inspect job cgroup memory; refusing unguarded GPU training")
    return candidates


def memory_sample(groups):
    readings = [(str(path), int(path.read_text()), maximum) for path, maximum in groups]
    # Apply the user cap to the WHOLE job, not just this task_special subgroup.
    # All explicit parent limits are still enforced independently.
    path, usage, maximum = next((r for r in readings if Path(r[0]).parent.name.startswith("job_")), readings[0])
    effective = min(LIMIT, maximum) if maximum is not None else LIMIT
    violations = [(path, usage, effective)] if usage > effective - RESERVE else []
    violations += [(p, u, m) for p, u, m in readings if m is not None and m < 2**60 and u > m - RESERVE]
    return readings, violations


def stop_owned_processes(process):
    # Ray workers can start their own sessions. Snapshot descendants before
    # terminating the launcher, then signal only this owned tree, never ray stop.
    import psutil
    try:
        children = psutil.Process(process.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        children = []
    for child in reversed(children):
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    _, alive = psutil.wait_procs(children, timeout=3)
    for child in alive:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source-root", required=True, help="Local official TSVs, named AI2D_TEST.tsv and MME.tsv")
    parser.add_argument("--data-root", required=True, help="Fresh generated-data parent, outside git")
    parser.add_argument("--output", required=True, help="Fresh experiment directory, outside git")
    parser.add_argument("--model", required=True, help="Local InternVL3-2B snapshot")
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    output, data_root = Path(args.output).resolve(), Path(args.data_root).resolve()
    if ROOT.parent == output or ROOT.parent in output.parents or ROOT.parent == data_root or ROOT.parent in data_root.parents:
        raise ValueError("Experiment data/results must be outside the source worktree")
    output.mkdir(parents=True, exist_ok=False)
    state = dict(status="starting", stages=[], submitted_at=datetime.now(timezone.utc).isoformat(), blockers=[])

    def save():
        (output / "run_status.json").write_text(json.dumps(state, indent=2) + "\n")

    save()
    env = dict(os.environ)
    # Private paths only enter runtime artifacts, never committed configs.
    env.update(PYTHONPATH=str(ROOT), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", USE_COT="0",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
               TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1", PYTHONNOUSERSITE="1",
               HF_HOME=str(output / "hf_cache"), HF_MODULES_CACHE=str(output / "hf_modules"),
               XDG_CACHE_HOME=str(output / "cache"), RAY_DEDUP_LOGS="0")
    devices = [s.strip() for s in env.get("CUDA_VISIBLE_DEVICES", "").split(",") if s.strip()]
    try:
        if len(devices) < 2:
            raise RuntimeError("Existing allocation must expose at least two GPUs; no Slurm request will be made")
        devices = devices[:2]
        env["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
        state["cuda_visible_devices"] = devices
        state["source_revision"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        state["source_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
        tracked = subprocess.check_output(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=ROOT).decode().split("\0")
        code_hashes = {}
        for name in tracked:
            path = ROOT / name
            if path.is_file() and path.suffix in {".py", ".sh", ".json", ".yaml", ".yml"}:
                code_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        (output / "source_code_hashes.json").write_text(json.dumps(code_hashes, indent=2))
        groups = memory_cgroups()
        state["memory_cgroups"] = [(str(path), cap) for path, cap in groups]
        if not Path(args.model, "config.json").is_file():
            raise FileNotFoundError("Local model snapshot missing")
        missing = [str(Path(args.source_root) / f"{dataset}.tsv") for dataset in ("AI2D_TEST", "MME")
                   if not (Path(args.source_root) / f"{dataset}.tsv").is_file()]
        if missing:
            state["blockers"].append(dict(type="missing_official_tsv", paths=missing))

        def gpu_idle():
            usage = subprocess.check_output(["nvidia-smi", "-i", ",".join(devices), "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
            if any(int(value.strip()) > 1024 for value in usage.splitlines()):
                raise RuntimeError("Selected GPUs already occupied; no existing process will be stopped")

        def run(name, command, additions=None):
            state["status"], state["current_stage"] = "running", name
            stage = dict(name=name, command=command, started_at=datetime.now(timezone.utc).isoformat())
            state["stages"].append(stage)
            save()
            readings, violations = memory_sample(groups)
            if violations:
                raise RuntimeError(f"Insufficient cgroup memory headroom before {name}: {violations}")
            child_env = dict(env, **(additions or {}))
            with open(output / f"{name}.log", "x") as log, open(output / f"{name}.memory.jsonl", "x") as memory_log:
                process = subprocess.Popen(command, cwd=ROOT, env=child_env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                stage["pid"] = process.pid
                save()
                try:
                    while process.poll() is None:
                        readings, violations = memory_sample(groups)
                        memory_log.write(json.dumps(dict(time=time.time(), readings=readings)) + "\n")
                        memory_log.flush()
                        if violations:
                            raise RuntimeError(f"Cgroup memory reserve breached: {violations}")
                        time.sleep(0.5)
                except BaseException:
                    stop_owned_processes(process)
                    raise
                stage["returncode"] = process.returncode
                stage["finished_at"] = datetime.now(timezone.utc).isoformat()
                save()
                if process.returncode:
                    raise RuntimeError(f"Gate failed: {name} (exit {process.returncode}); see {name}.log")

        gpu_idle()
        run("data_scorer_tests", [sys.executable, "-m", "pytest", "tests/official_data/test_data_and_score.py", "-q", "-p", "no:cacheprovider"])
        run("single_gpu_loader", [sys.executable, "scripts/smoke_gpu_checkpoint.py", "--model-path", args.model, "--dtype", "fp32"],
            {"CUDA_VISIBLE_DEVICES": devices[0]})
        run("gpu_contract", [sys.executable, "-m", "pytest", "tests/official_data/test_gpu_contract.py", "-q", "-p", "no:cacheprovider"],
            {"CUDA_VISIBLE_DEVICES": devices[0]})
        if missing:
            raise FileNotFoundError("Official TSV inputs missing. No downloads or legacy-data substitution performed.")
        for dataset in ("AI2D_TEST", "MME"):
            run(f"prepare_{dataset}", [sys.executable, "-m", "scripts.official_data.prepare", "--dataset", dataset,
                "--tsv", str(Path(args.source_root) / f"{dataset}.tsv"), "--output", str(data_root / dataset)])
            run(f"contract_{dataset}", [sys.executable, "-m", "scripts.official_data.check_contract", "--data-dir", str(data_root / dataset),
                "--model", args.model, "--output", str(output / f"input_{dataset}"), "--forward"], {"CUDA_VISIBLE_DEVICES": devices[0]})
        methods = json.loads((ROOT / "configs/official_data/methods.json").read_text())
        for phase in (["smoke"] if args.smoke_only else ["smoke", "full"]):
            for dataset in ("AI2D_TEST", "MME"):
                for method in ("base", "ttrv", "prove"):
                    gpu_idle()
                    name = f"{phase}_{dataset}_{method}"
                    profile = dict(methods["common"], **methods[dataset])
                    profile.update(methods[method])
                    profile.update(TASK=dataset, METHOD=method, BACKBONE_PATH=args.model, PYTHON_BIN=sys.executable,
                                   DATA_LOCAL_DIR=str(data_root), OFFICIAL_RUN_DIR=str(output / name),
                                   OFFICIAL_SMOKE="1" if phase == "smoke" else "0", OFFICIAL_STEPS="1" if phase == "smoke" else "20")
                    (output / f"{name}.profile.json").write_text(json.dumps(profile, indent=2))
                    run(name, ["bash", "examples/ttrv/official_ai2d_mme.sh"], profile)
                    verify_method_artifacts(output / name, method)
                    command = [sys.executable, "-m", "scripts.official_eval.score_predictions", "--data-dir", str(data_root / dataset),
                        "--predictions", str(output / name / "predictions.raw.jsonl"), "--output", str(output / name / "scoring"),
                        "--method", method, "--run-id", name]
                    if phase == "smoke":
                        command.append("--smoke")
                    run(f"score_{name}", command)
        state["status"] = "complete"
    except BaseException as error:
        state["status"] = "blocked"
        state["blockers"].append(dict(type=type(error).__name__, message=str(error)))
        save()
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
