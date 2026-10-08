"""One command stages both pinned chat models and runs train, merge, calibrate, publish."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

BASES = {
    "9b": ("Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a"),
    "27b": ("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"),
}
HERE = Path(__file__).resolve().parent


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path):
    """One record per newline. Not str.splitlines(): JSON strings may hold U+2028/U+0085, which it also splits on."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def validate_data(train, calib):
    from jebadiah_prompt import wire_keys
    from ainode_prompt_verbatim import translate_one, criteria_pairs, MAX_CRITERIA
    from train_jebadiah import label_index, make_target
    report, families, ids = {}, [], []
    for name, path in (("train", train), ("calib", calib)):
        records = read_jsonl(path)
        if not records:
            raise ValueError(f"Empty {name} split")
        fam, row_ids, types, count = set(), set(), set(), 0
        for record in records:
            if not record.get("family_id") or not record.get("id"):
                raise ValueError("Every record must have id and family_id")
            if record["id"] in row_ids:
                raise ValueError(f"Duplicate record id in {name}")
            row_ids.add(record["id"])
            fam.add(record["family_id"])
            if not record["questions"]:
                raise ValueError("Record has no questions")
            for qid, question in record["questions"].items():
                if "label" in question or "target" in question:
                    raise ValueError("Supervision must not occur inside a question")
                if question.get("type") == "choice" and len(question.get("criteria", {})) > MAX_CRITERIA:
                    criteria_pairs(qid, question["criteria"], "choice")
                    if not isinstance(question.get("instructions"), str) or not question["instructions"].strip():
                        raise ValueError("Choice instructions must be non-empty text")
                else:
                    translate_one(qid, question)
                label = record["label"][qid]
                if question["type"] == "noul" and not isinstance(label, bool):
                    raise ValueError("Noul gold must be a JSON boolean")
                if question["type"] == "score" and (not isinstance(label, int) or isinstance(label, bool)):
                    raise ValueError("Score gold must be an integer level index")
                keys = wire_keys(question)
                label_index(question, record["label"][qid], keys)
                target = make_target(question, record["label"][qid], keys,
                                     record.get("target", {}).get(qid), "ordinal", 0.2)
                if any(not (0 <= x <= 1) for x in target) or abs(sum(target) - 1) > 1e-5:
                    raise ValueError("Invalid target distribution")
                types.add(question["type"])
                count += 1
        if types != {"choice", "noul", "score"}:
            raise ValueError(f"{name} must cover choice, noul and score")
        families.append(fam)
        ids.append(row_ids)
        report[name] = {"sha256": sha256(path), "rows": len(records), "questions": count}
    if families[0] & families[1] or ids[0] & ids[1]:
        raise ValueError("Training/calibration families or record IDs overlap")
    return report


def run(script, *args):
    subprocess.run([sys.executable, str(HERE / script), *map(str, args)], check=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(os.environ.get("JEB_ROOT", "/workspace/jeb-v21")))
    p.add_argument("--sizes", nargs="+", choices=BASES, default=["9b", "27b"])
    p.add_argument("--dataset-id", default="frontier-infra/jebadiah-data-v2-1")
    p.add_argument("--dataset-revision", default="main")
    p.add_argument("--train-file", default="train.jsonl")
    p.add_argument("--calib-file", default="calib.jsonl")
    p.add_argument("--manifest-file", default="manifest.json")
    p.add_argument("--data-dir", type=Path, help="local split files, used for offline smoke tests")
    p.add_argument("--base", type=Path, help="local tiny-model snapshot, single size only")
    p.add_argument("--model-repo", help="private model repository, use {size} for both sizes")
    p.add_argument("--checkpoint-repo", help="private checkpoint repository, use {size} for both sizes")
    p.add_argument("--no-upload", action="store_true", help="local verification only")
    p.add_argument("--setup-only", action="store_true")
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--alpha", type=int, default=128)
    p.add_argument("--max-seq-length", type=int, default=4096)
    p.add_argument("--epochs", type=float, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--microbatch", type=int, default=1)
    p.add_argument("--accumulation", type=int, default=8)
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True,
                   help="recompute activations to save memory; disable only after measuring peak memory")
    p.add_argument("--group-by-length", action="store_true",
                   help="shuffle length-sorted windows to reduce batch padding")
    p.add_argument("--pad-to-multiple-of", type=int,
                   help="round padded batch lengths to this token multiple to reduce kernel tuning shapes")
    p.add_argument("--checkpoint-min-tokens", type=int,
                   help="checkpoint batches at or above this padded length; retain short-batch activations")
    p.add_argument("--backbone-autocast", action="store_true",
                   help="execute backbone LoRA projections in bf16 while retaining the fp32 candidate head")
    p.add_argument("--eval-steps", type=int, default=0, help="held-out accuracy/NLL interval during training")
    p.add_argument("--calib-eval-limit", type=int, help="maximum held-out questions per training evaluation")
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--resume", default="auto", help="auto, hub, none or checkpoint path")
    p.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cuda")
    return p


def main():
    from huggingface_hub import HfApi, snapshot_download
    from hub_checkpoints import private_repo
    args = parser().parse_args()
    if args.base and len(args.sizes) != 1:
        raise ValueError("A local base requires one size")
    if not args.no_upload and not args.setup_only:
        if not args.model_repo or not args.checkpoint_repo:
            raise ValueError("Private model and checkpoint repositories are required")
        if len(args.sizes) > 1 and any("{size}" not in r for r in (args.model_repo, args.checkpoint_repo)):
            raise ValueError("Both repository flags need {size} when training both sizes")
    args.root = args.root.resolve()
    args.root.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    run("test_prompt_identity.py")
    if args.data_dir:
        data = args.data_dir.resolve()
        data_revision = "local"
    else:
        info = api.repo_info(args.dataset_id, repo_type="dataset", revision=args.dataset_revision)
        if not info.private:
            raise ValueError("Expected a private training dataset")
        data_revision = info.sha
        data = Path(snapshot_download(args.dataset_id, repo_type="dataset", revision=data_revision,
                                     allow_patterns=[args.train_file, args.calib_file, args.manifest_file]))
    train, calib = data / args.train_file, data / args.calib_file
    data_report = validate_data(train, calib)
    manifest = json.loads((data / args.manifest_file).read_text())
    for split, filename in (("train", args.train_file), ("calib", args.calib_file)):
        entry = manifest["files"][filename]
        for key in ("sha256", "rows", "questions"):
            if entry[key] != data_report[split][key]:
                raise ValueError(f"Manifest mismatch: {filename} {key}")
    (args.root / "data_provenance.json").write_text(json.dumps({"dataset": args.dataset_id,
        "revision": data_revision, "splits": data_report}, indent=2) + "\n")

    if args.device == "cuda" and not args.setup_only:
        run("cuda_preflight.py")
    # Download both checkpoints before entering training, with immutable resolved revisions.
    staged = {}
    previous = args.root / "models.json"
    pinned = json.loads(previous.read_text()) if previous.exists() else {}
    for size in args.sizes:
        repo, revision = BASES[size]
        if args.base:
            path, resolved = str(args.base.resolve()), "local"
        else:
            resolved = pinned.get(size, {}).get("revision") or api.model_info(repo, revision=revision).sha
            if not resolved.startswith(revision):
                raise ValueError(f"Resolved revision does not match pin for {repo}")
            path = snapshot_download(repo, revision=resolved, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "LICENSE*"])
        staged[size] = {"repo": repo, "revision": resolved, "path": path}
    previous.write_text(json.dumps(staged, indent=2) + "\n")
    if args.setup_only:
        return
    for size in args.sizes:
        spec = staged[size]
        out = args.root / "runs" / size
        out.mkdir(parents=True, exist_ok=True)
        model_repo = args.model_repo.format(size=size) if args.model_repo and not args.no_upload else None
        checkpoint_repo = args.checkpoint_repo.format(size=size) if args.checkpoint_repo and not args.no_upload else None
        if model_repo:
            private_repo(model_repo)
            private_repo(checkpoint_repo)
        cfg = {"base_model": spec["path"], "base_revision": None, "base_identity": spec["repo"] + "@" + spec["revision"], "method": "lora",
               "dataset_path": str(train), "eval_dataset_path": str(calib), "output_dir": str(out),
               "dataset_sha256": data_report["train"]["sha256"], "eval_dataset_sha256": data_report["calib"]["sha256"],
               "lora_rank": args.rank, "lora_alpha": args.alpha, "max_seq_length": args.max_seq_length,
               "num_epochs": args.epochs, "learning_rate": args.lr, "batch_size": args.microbatch,
               "gradient_accumulation_steps": args.accumulation, "use_gradient_checkpointing": args.gradient_checkpointing,
               "warmup_steps": 30 if args.max_steps < 0 else 0, "seed": 17, "save_steps": args.save_steps,
               "eval_steps": args.eval_steps,
               "checkpoint_repo": checkpoint_repo, "resume": args.resume, "device": args.device,
               "dtype": "bfloat16" if args.device == "cuda" else "float32", "logging_steps": 10,
               "decide": {"target_modules": "all-linear", "lora_dropout": 0.05, "score_targets": "ordinal",
                          "score_ordinal_adjacent": 0.2, "shuffle_choice_options": True}}
        if args.eval_steps < 0 or (args.calib_eval_limit is not None and args.calib_eval_limit < 1):
            raise ValueError("Evaluation interval must be nonnegative and limit positive")
        if args.calib_eval_limit is not None:
            cfg["decide"]["calib_eval_limit"] = args.calib_eval_limit
        if args.group_by_length:
            cfg["group_by_length"] = True
        if args.pad_to_multiple_of is not None:
            if args.pad_to_multiple_of < 1:
                raise ValueError("--pad-to-multiple-of must be positive")
            cfg["pad_to_multiple_of"] = args.pad_to_multiple_of
        if args.checkpoint_min_tokens is not None:
            if args.checkpoint_min_tokens < 1 or not args.gradient_checkpointing:
                raise ValueError("--checkpoint-min-tokens requires a positive threshold and gradient checkpointing")
            cfg["checkpoint_min_tokens"] = args.checkpoint_min_tokens
        if args.backbone_autocast:
            cfg["backbone_autocast"] = True
        if not args.base:
            pinned_contract = json.loads((HERE.parent / "results" / "runs" / f"{size}-chat-v1" / "adapter" / "prompt_contract.json").read_text())
            for key in ("prompt_source_sha256", "chat_template_sha256", "single_token_labels"):
                cfg[key] = pinned_contract[key]
        config_path = out / "config.json"
        config_path.write_text(json.dumps(cfg, indent=2) + "\n")
        run("train_jebadiah.py", "--config", config_path, "--max-steps", args.max_steps)
        merged = out / "merged"
        run("merge_export.py", "--base", spec["path"], "--adapter", out / "adapter", "--output", merged)
        run("verify_merge.py", "--base", spec["path"], "--adapter", out / "adapter", "--merged", merged,
            "--calib", calib, "--max-tokens", args.max_seq_length, "--device", args.device)
        run("fit_temperature.py", "--base", merged, "--output-dir", merged, "--calib", calib,
            "--max-tokens", args.max_seq_length, "--batch-size", 1, "--target", "train", "--score-fit", "hard",
            "--score-targets", "ordinal", "--device", args.device, "--dtype", cfg["dtype"])
        provenance = {"base_model": spec["repo"], "base_revision": spec["revision"],
                      "dataset": args.dataset_id, "dataset_revision": data_revision,
                      "effective_batch_size": args.microbatch * args.accumulation, "config": cfg}
        # No local paths or checkpoint routing in the published provenance.
        provenance["config"] = {k: v for k, v in cfg.items() if k not in
                                ("base_model", "dataset_path", "eval_dataset_path", "output_dir", "checkpoint_repo", "resume")}
        (merged / "training_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        card_base = f"base_model: {spec['repo']}\n" if not args.base else ""
        (merged / "README.md").write_text("---\n" + card_base + "library_name: transformers\n---\n\n"
            + (f"# Jebadiah {size} v2.1\n\n" if not args.base else "# Tiny trainer smoke model\n\n")
            + "Merged bf16 weights for the fp32 candidate-label decision objective. Load the published scripts/ "
            + "inference payload and temperatures.json to preserve the training prompt and calibrated probabilities. "
            + "See training_provenance.json, prompt_contract.json, and merge_verification.json for the run evidence.\n")
        if not args.no_upload:
            uploader = private_repo(model_repo)
            uploader.upload_folder(repo_id=model_repo, folder_path=merged, commit_message="Merged v2.1 decision model")
            if not uploader.repo_info(model_repo).private:
                raise ValueError("Model repository privacy changed during upload")
        print(f"V21_DONE size={size} local={merged} private_repo={model_repo}", flush=True)


if __name__ == "__main__":
    main()
