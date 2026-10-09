"""Scene-local artifacts for one fixed-scene experiment."""
import json
import os
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


@contextmanager
def capture_console(path):
    """Tee both OS output descriptors, including simulator output, to run.log."""
    sys.stdout.flush()
    sys.stderr.flush()
    originals = [os.dup(fd) for fd in (1, 2)]
    reader, writer = os.pipe()
    with open(path, "ab", buffering=0) as log:
        def drain():
            while True:
                chunk = os.read(reader, 65536)
                if not chunk:
                    break
                log.write(chunk)
                # stdout/stderr share a chronological log and console stream.
                remaining = chunk
                while remaining:
                    remaining = remaining[os.write(originals[0], remaining):]
        thread = threading.Thread(target=drain, daemon=True)
        thread.start()
        try:
            os.dup2(writer, 1)
            os.dup2(writer, 2)
            os.close(writer)
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            for fd, original in zip((1, 2), originals):
                os.dup2(original, fd)
            thread.join()
            os.close(reader)
            for original in originals:
                os.close(original)


class RunOutput:
    def __init__(self, cfg, trial):
        scene_cfg = cfg["unisis"]
        if not scene_cfg.get("scene_path"):
            raise ValueError("scene_source=unisis requires unisis.scene_path.")
        scene = Path(scene_cfg["scene_path"]).expanduser().resolve()
        if scene.is_dir():
            scene = scene / "scene.yaml"
        self.scene = scene
        self.seed = int(scene_cfg["seed"]) + trial
        run_id = scene_cfg.get("run_id")
        if run_id:
            if run_id in {".", ".."} or Path(run_id).name != run_id or "\\" in run_id:
                raise ValueError("unisis.run_id must be a single directory name.")
            if int(cfg["trials"]) > 1:
                run_id += f"_t{trial + 1:03d}"
        else:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_s{self.seed}"
        root = Path(scene_cfg.get("output_dir") or scene.parent / "gensim").expanduser().resolve()
        self.path = root / run_id
        self.path.mkdir(parents=True, exist_ok=False)
        for folder in ("llm", "code", "trajectory", "artifacts"):
            (self.path / folder).mkdir()
        self.phase = "initialization"
        self.started = time.monotonic()
        self.calls = []
        self.result = {"status": "running", "native_success": None, "success": None,
                       "failure_stage": None, "failure_reason": None, "attempts": []}
        self.meta = {"run_id": run_id, "robot": "franka", "method": "gensim", "scene_source": "unisis",
                     "scene": os.path.relpath(scene, self.path), "scene_sha256": None,
                     "started_at": datetime.now(timezone.utc).isoformat(), "seed": self.seed,
                     "trial": trial + 1, "trials": int(cfg["trials"]), "mode": cfg["mode"],
                     "end_effector": scene_cfg["end_effector"],
                     "task_code_candidate_num": int(cfg["task_code_candidate_num"]),
                     "load_memory": bool(cfg["load_memory"]), "use_template": bool(cfg["use_template"]),
                     "max_env_run_cnt": int(cfg["max_env_run_cnt"]),
                     "save_data": bool(cfg["save_data"]), "save_video": bool(cfg["record"]["save_video"])}
        try:
            self.meta["source_commit"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
                stderr=subprocess.DEVNULL, text=True).strip()
            self.meta["source_dirty"] = bool(subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=Path(__file__).resolve().parents[1], text=True).strip())
        except (OSError, subprocess.CalledProcessError):
            self.meta["source_commit"] = None
        write_json(self.path / "run_meta.json", self.meta)
        write_json(self.path / "result.json", self.result)
        write_json(self.path / "llm/index.json", self.calls)

    def begin_call(self, request):
        name = f"turn_{len(self.calls) + 1:04d}_{self.phase}.json"
        entry = {"file": name, "phase": self.phase, "status": "running"}
        self.calls.append(entry)
        write_json(self.path / "llm/index.json", self.calls)
        record = {"phase": self.phase, "request": request, "status": "running"}
        write_json(self.path / "llm" / name, record)
        return entry, record

    def end_call(self, call, **values):
        entry, record = call
        record.update(values)
        entry["status"] = record["status"]
        write_json(self.path / "llm" / entry["file"], record)
        write_json(self.path / "llm/index.json", self.calls)

    def fail(self, error):
        self.result.update(status="error", native_success=False,
                           failure_stage=self.phase, failure_reason=str(error))
        write_json(self.path / "result.json", self.result)

    def finish(self):
        if self.result["status"] == "running":
            self.result["status"] = "completed"
        self.result["elapsed_seconds"] = time.monotonic() - self.started
        self.result["llm_call_count"] = len(self.calls)
        self.result["trajectory_count"] = len(list((self.path / "trajectory/action").glob("*.pkl")))
        write_json(self.path / "run_meta.json", self.meta)
        write_json(self.path / "result.json", self.result)
