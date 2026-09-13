"""Run reproducible MPGNN training tasks with isolated logs and concurrency."""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


EXPECTED_PYTHON = Path(
    os.environ.get("MPGNN_PYTHON", sys.executable)
).resolve()
EXPECTED_PROTOCOL = "r2_pdf_locked_v1"


def _load_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("version") != 1:
        raise ValueError("Only training-suite version 1 is supported")
    if config.get("protocol_id") != EXPECTED_PROTOCOL:
        raise ValueError(
            f"Formal training requires protocol_id={EXPECTED_PROTOCOL}; "
            f"got {config.get('protocol_id')!r}"
        )
    return config, path


def _cli_name(key):
    return "--" + key


def _cli_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _task_paths(output_root, task_name, seed):
    task_root = Path(output_root) / task_name / f"seed_{seed}"
    return task_root, task_root / "stdout.log", task_root / "launcher_status.json"


def _build_command(config, task, seed, output_root, assigned_gpu=None):
    python = Path(config.get("python", EXPECTED_PYTHON)).resolve()
    if config.get("require_zhihao", False) and python != EXPECTED_PYTHON.resolve():
        raise ValueError(f"Training must use configured Python: {EXPECTED_PYTHON}")
    workspace = Path(config["workspace"]).resolve()
    task_root, _, _ = _task_paths(output_root, task["name"], seed)
    save_root = Path(output_root) / "checkpoints"
    arguments = dict(config.get("common_train_args", {}))
    arguments.update(task.get("train_args", {}))
    if assigned_gpu is not None:
        arguments["gpu_id"] = int(assigned_gpu)
    arguments.update({
        "seed": seed,
        "trial": seed,
        "dataset": task["dataset"],
        "budget": task["budget"],
        "graph_path": task["graph_path"],
        "data_saving_dir": str(save_root.resolve()),
        "log_path": str((Path(output_root) / "garage").resolve()),
        "torch_n_threads": config.get("torch_threads_per_task", 2),
    })
    if task.get("community_data_config"):
        arguments["community_data_config"] = task["community_data_config"]
    elif task.get("data_config"):
        arguments["data_config"] = task["data_config"]
    else:
        arguments["data_path1"] = task["data_path"]
    command = [str(python), str(workspace / "main.py")]
    for key, value in arguments.items():
        if value is None:
            continue
        command.extend([_cli_name(key), _cli_value(value)])
    return command, workspace, task_root, arguments


def _run_one(config, task, seed, output_root, resume, assigned_gpu=None):
    command, workspace, task_root, arguments = _build_command(
        config, task, seed, output_root, assigned_gpu=assigned_gpu
    )
    task_root.mkdir(parents=True, exist_ok=True)
    _, log_path, status_path = _task_paths(output_root, task["name"], seed)
    if resume and status_path.exists():
        previous = json.loads(status_path.read_text(encoding="utf-8"))
        if previous.get("status") == "completed":
            return previous

    status = {
        "task": task["name"],
        "training_seed": seed,
        "command": command,
        "arguments": arguments,
        "python": command[0],
        "assigned_gpu": assigned_gpu,
        "status": "running",
        "started_at_unix": time.time(),
        "log_path": str(log_path.resolve()),
    }
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    env = os.environ.copy()
    threads = str(config.get("torch_threads_per_task", 2))
    env.update({
        "OMP_NUM_THREADS": threads,
        "MKL_NUM_THREADS": threads,
        "OPENBLAS_NUM_THREADS": threads,
        "NUMEXPR_NUM_THREADS": threads,
        "PYTHONUNBUFFERED": "1",
    })
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write("COMMAND: " + " ".join(command) + "\n")
        log_file.flush()
        process = subprocess.Popen(
            command,
            cwd=str(workspace),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        status["child_pid"] = process.pid
        status_path.write_text(
            json.dumps(status, indent=2) + "\n", encoding="utf-8"
        )
        return_code = process.wait()
    status["finished_at_unix"] = time.time()
    status["elapsed_seconds"] = status["finished_at_unix"] - status["started_at_unix"]
    status["return_code"] = return_code
    status["status"] = "completed" if return_code == 0 else "failed"
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    return status


def _run_job_group(config, jobs, output_root, resume, assigned_gpu):
    results = []
    for task, seed in jobs:
        result = _run_one(
            config,
            task,
            seed,
            output_root,
            resume,
            assigned_gpu=assigned_gpu,
        )
        results.append(result)
        print(
            f"{task['name']} seed={seed}: {result['status']} gpu={assigned_gpu}",
            flush=True,
        )
    return results


def _preflight(config):
    python = Path(config.get("python", EXPECTED_PYTHON)).resolve()
    if config.get("require_zhihao", False) and python != EXPECTED_PYTHON.resolve():
        raise ValueError(f"Training must use configured Python: {EXPECTED_PYTHON}")
    code = (
        "import json,sys,torch,igraph,leidenalg,garage; "
        "print(json.dumps({'python':sys.executable,'torch':torch.__version__,"
        "'cuda_available':torch.cuda.is_available(),'leiden':True,'garage':True}))"
    )
    result = subprocess.run(
        [str(python), "-c", code], capture_output=True, text=True, check=True
    )
    lines = [line for line in result.stdout.splitlines() if line.startswith("{")]
    if not lines:
        raise RuntimeError("Unable to parse zhihao environment preflight output")
    return json.loads(lines[-1])


def run_suite(args):
    config, config_path = _load_config(args.config)
    from formal_protocol_gate import audit as audit_protocol
    protocol_report = audit_protocol(config_path)
    if protocol_report["status"] != "pass":
        raise RuntimeError(
            "Formal protocol gate failed: " + "; ".join(protocol_report["errors"][:8])
        )
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    environment = _preflight(config)
    cpu_fallback = bool(config.get("cpu_fallback", True))
    if not environment["cuda_available"] and not cpu_fallback and max(
        (task.get("train_args", {}).get("gpu_id", config.get("common_train_args", {}).get("gpu_id", -1))
         for task in config.get("tasks", [])),
        default=-1,
    ) >= 0:
        raise RuntimeError("CUDA is unavailable; set gpu_id=-1 for every task")
    (output_root / "suite_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    gpu_ids = args.gpu_ids
    if gpu_ids is None:
        gpu_ids = config.get("gpu_ids")
    if isinstance(gpu_ids, str):
        gpu_ids = [int(value) for value in gpu_ids.split(",") if value.strip()]
    if not environment["cuda_available"] and cpu_fallback:
        # Assigned GPU ids are omitted; main.py already selects CPU when CUDA
        # is unavailable, allowing the same suite to resume after driver repair.
        gpu_ids = []
    jobs = [
        (task, int(seed))
        for task in config.get("tasks", [])
        for seed in config.get("training_seeds", [])
    ]
    if args.task:
        jobs = [job for job in jobs if job[0]["name"] in set(args.task)]
    if args.seed:
        jobs = [job for job in jobs if job[1] in set(args.seed)]
    for exclusion in args.exclude_job or []:
        task_name, seed_text = exclusion.rsplit(":", 1)
        excluded_seed = int(seed_text)
        jobs = [
            job for job in jobs
            if not (job[0]["name"] == task_name and job[1] == excluded_seed)
        ]
    if not jobs:
        raise ValueError("No training jobs match the requested filters")
    max_parallel = args.max_parallel or int(config.get("max_parallel", 1))
    if gpu_ids:
        max_parallel = min(max_parallel, len(gpu_ids))
    worker_devices = gpu_ids if gpu_ids else [None] * max_parallel
    job_groups = [[] for _ in worker_devices]
    for index, job in enumerate(jobs):
        job_groups[index % len(job_groups)].append(job)
    results = []
    with ThreadPoolExecutor(max_workers=len(worker_devices)) as executor:
        futures = [
            executor.submit(
                _run_job_group,
                config,
                job_group,
                output_root,
                args.resume,
                assigned_gpu,
            )
            for job_group, assigned_gpu in zip(job_groups, worker_devices)
            if job_group
        ]
        for future in as_completed(futures):
            results.extend(future.result())
    summary = {
        "config": str(config_path),
        "python": str(EXPECTED_PYTHON),
        "environment": environment,
        "job_count": len(results),
        "completed": sum(item["status"] == "completed" for item in results),
        "failed": sum(item["status"] == "failed" for item in results),
        "jobs": sorted(results, key=lambda item: (item["task"], item["training_seed"])),
    }
    summary_path = output_root / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved training summary to {summary_path}")
    return 1 if summary["failed"] else 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-parallel", type=int)
    parser.add_argument("--task", action="append")
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument(
        "--exclude-job",
        action="append",
        help="exclude one task/seed pair using TASK_NAME:SEED",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--gpu-ids",
        help="comma-separated GPU ids; each concurrent task gets one GPU",
    )
    return parser


if __name__ == "__main__":
    sys.exit(run_suite(build_parser().parse_args()))
