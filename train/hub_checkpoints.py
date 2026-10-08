"""Synchronous, atomic private Hub checkpoint commits with optimizer and RNG recovery."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download, CommitOperationDelete
from transformers import TrainerCallback

# These settings must match when continuing an interrupted optimizer trajectory.
RESUME_KEYS = ("base_identity", "base_revision", "dataset_sha256", "eval_dataset_sha256",
               "method", "lora_rank", "lora_alpha", "max_seq_length", "batch_size",
               "gradient_accumulation_steps", "learning_rate", "num_epochs", "seed", "decide", "max_steps",
               "warmup_steps", "lr_scheduler_type", "weight_decay", "max_grad_norm", "dtype",
               "use_gradient_checkpointing", "prompt_source_sha256", "chat_template_sha256", "single_token_labels")


def private_repo(repo_id, api=None):
    api = api or HfApi()
    api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
    if not api.repo_info(repo_id, repo_type="model").private:
        raise ValueError(f"Refusing to write to public model repository {repo_id}")
    return api


def validate_resume(path, cfg):
    path = Path(path)
    saved = json.loads((path / "run_config.json").read_text())
    changed = [key for key in RESUME_KEYS if saved.get(key) != cfg.get(key)]
    if not cfg.get("base_identity") and saved.get("base_model") != cfg.get("base_model"):
        changed.append("base_model")
    if changed:
        raise ValueError(f"Resume config changed: {changed}; use a new output directory")
    for name in ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth",
                 "adapter_model.safetensors", "adapter_config.json"):
        if not (path / name).is_file():
            raise ValueError(f"Incomplete checkpoint: missing {name}")
    return str(path)


def resolve_resume(cfg, checkpoint_dir):
    requested = cfg.get("resume", "auto")
    if requested == "none":
        return None
    if requested not in ("auto", "hub"):
        return validate_resume(requested, cfg)
    root = Path(checkpoint_dir)
    candidates = sorted(root.glob("checkpoint-*/complete.json"),
                        key=lambda p: int(p.parent.name.split("-")[-1]))
    if candidates and requested == "auto":
        return validate_resume(candidates[-1].parent, cfg)
    repo = cfg.get("checkpoint_repo")
    if not repo:
        if requested == "hub":
            raise ValueError("--resume hub requires a checkpoint repository")
        return None
    api = private_repo(repo)
    revision = api.repo_info(repo).sha
    markers = [p for p in api.list_repo_files(repo, revision=revision) if p.startswith("checkpoints/checkpoint-")
               and p.endswith("/complete.json")]
    if not markers:
        return None
    latest = max(markers, key=lambda p: int(p.split("/")[1].split("-")[-1]))
    prefix = latest.rsplit("/", 1)[0]
    snapshot = Path(snapshot_download(repo, revision=revision, allow_patterns=[prefix + "/*"]))
    dest = root / prefix.split("/")[-1]
    shutil.copytree(snapshot / prefix, dest, dirs_exist_ok=True)
    return validate_resume(dest, cfg)


class HubCheckpoint(TrainerCallback):
    def __init__(self, cfg):
        self.cfg = cfg
        self.api = private_repo(cfg["checkpoint_repo"]) if cfg.get("checkpoint_repo") else None

    def on_save(self, args, state, control, **kwargs):
        folder = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        (folder / "run_config.json").write_text(json.dumps(self.cfg, indent=2) + "\n")
        validate_resume(folder, self.cfg)
        (folder / "complete.json").write_text(json.dumps({"step": state.global_step}) + "\n")
        if self.api:
            private_repo(self.cfg["checkpoint_repo"], self.api)
            # upload_folder is one commit; a remote complete marker never precedes its files.
            self.api.upload_folder(repo_id=self.cfg["checkpoint_repo"], folder_path=folder,
                                   path_in_repo=f"checkpoints/{folder.name}",
                                   commit_message=f"Training checkpoint {state.global_step}", ignore_patterns=["README.md"])

            files = self.api.list_repo_files(self.cfg["checkpoint_repo"])
            complete = sorted({path.rsplit("/", 1)[0] for path in files
                               if path.startswith("checkpoints/checkpoint-") and path.endswith("/complete.json")},
                              key=lambda path: int(path.split("-")[-1]))
            keep = int(self.cfg.get("save_total_limit", 2))
            old = complete[:-keep] if keep > 0 else []
            deletions = [CommitOperationDelete(path_in_repo=path) for path in files
                         if any(path.startswith(prefix + "/") for prefix in old)]
            if deletions:
                self.api.create_commit(repo_id=self.cfg["checkpoint_repo"], operations=deletions,
                                       commit_message=f"Keep latest {keep} training checkpoints")
