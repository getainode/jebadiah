"""Behavioral checks for candidate precision, export completeness and restart safety."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from safetensors.torch import load_file, save_file

from hub_checkpoints import private_repo, validate_resume
from jebadiah_model import option_logits
from merge_export import merge
from train_jebadiah import make_target, DecideTrainer, NonFiniteLoss
from v21_pipeline import validate_data, parser, read_jsonl, split_manifest_entry


@pytest.mark.parametrize("autocast", [False, True])
def test_candidate_head_remains_fp32(autocast):
    hidden = torch.tensor([[[0.998, 1.02], [1.11, 0.13]]], dtype=torch.bfloat16, requires_grad=True)
    head = torch.nn.Linear(2, 4, bias=False, dtype=torch.bfloat16)
    head.weight.data.copy_(torch.tensor([[1., .2], [1.01, .19], [0., 1.], [2., 0.]]))
    core = SimpleNamespace(model=lambda **kwargs: SimpleNamespace(last_hidden_state=hidden), lm_head=head)
    ids, mask, candidates = torch.tensor([[1, 2]]), torch.tensor([[1, 1]]), torch.tensor([[0, 1, -1]])
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        logits = option_logits(core, ids, mask, candidates)
    assert logits.dtype == torch.float32
    torch.testing.assert_close(logits[0, :2], head.weight[:2].float() @ hidden[0, 1].float())
    assert torch.isneginf(logits[0, 2])
    logits[0, :2].sum().backward()
    assert torch.isfinite(hidden.grad).all()


def test_ordinal_and_soft_choice_targets_unchanged():
    q = {"type": "score"}
    target = make_target(q, 1, ["0", "1", "2"], None, "ordinal", .2)
    assert target == pytest.approx([1/7, 5/7, 1/7])
    assert make_target({"type": "choice"}, "x", ["y", "x"], {"x": .8, "y": .2}, "ordinal", .2) == [.2, .8]


def test_public_repository_refused_before_upload():
    api = Mock()
    api.repo_info.return_value.private = False
    with pytest.raises(ValueError, match="public"):
        private_repo("org/model", api)
    api.upload_folder.assert_not_called()


def write_merge_inputs(root, layout="model.language_model.layers.0.proj.weight"):
    base, adapter = root / "base", root / "adapter"
    base.mkdir(); adapter.mkdir()
    save_file({layout: torch.ones(2, 3), "vision.weight": torch.ones(1)}, base / "model.safetensors")
    (base / "config.json").write_text('{"model_type":"qwen3_5"}')
    (adapter / "adapter_config.json").write_text('{"r":1,"lora_alpha":2}')
    save_file({"base_model.model.model.layers.0.proj.lora_A.weight": torch.tensor([[1., 2., 3.]]),
               "base_model.model.model.layers.0.proj.lora_B.weight": torch.tensor([[1.], [2.]])}, adapter / "adapter_model.safetensors")
    (adapter / "prompt_contract.json").write_text('{}')
    return base, adapter


def test_merge_matches_every_pair_and_preserves_multimodal_layout(tmp_path):
    base, adapter = write_merge_inputs(tmp_path)
    out = tmp_path / "merged"
    report = merge(base, adapter, out)
    assert report["merged_pairs"] == 1
    weights = load_file(out / "model.safetensors")
    assert weights["model.language_model.layers.0.proj.weight"].dtype == torch.bfloat16
    torch.testing.assert_close(weights["model.language_model.layers.0.proj.weight"], torch.tensor([[3., 5., 7.], [5., 9., 13.]], dtype=torch.bfloat16))
    assert (out / "config.json").read_bytes() == (base / "config.json").read_bytes()
    assert sorted(p.name for p in (out / "scripts").iterdir()) == ["ainode_prompt_verbatim.py", "jebadiah_model.py", "jebadiah_prompt.py"]


def test_merge_refuses_unmatched_adapter(tmp_path):
    base, adapter = write_merge_inputs(tmp_path, "wrong.weight")
    with pytest.raises(ValueError, match="Unmatched"):
        merge(base, adapter, tmp_path / "merged")


def test_resume_rejects_recipe_changes_and_incomplete_checkpoints(tmp_path):
    cfg = {"base_identity": "Qwen/pinned", "lora_rank": 64, "dataset_sha256": "abc"}
    (tmp_path / "run_config.json").write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="Incomplete"):
        validate_resume(tmp_path, cfg)
    with pytest.raises(ValueError, match="lora_rank"):
        validate_resume(tmp_path, {**cfg, "lora_rank": 16})


def test_v21_defaults():
    args = parser().parse_args([])
    assert args.sizes == ["9b", "27b"]
    assert (args.rank, args.alpha, args.max_seq_length, args.microbatch * args.accumulation) == (64, 128, 4096, 8)
    assert args.gradient_checkpointing and not args.group_by_length
    tuned = parser().parse_args(["--no-gradient-checkpointing", "--group-by-length"])
    assert not tuned.gradient_checkpointing and tuned.group_by_length


def test_length_grouping_reduces_padding_without_dropping_questions():
    from transformers import TrainingArguments
    from train_jebadiah import DecideDataset
    # Distinct lengths alternate: a random batch spends most tokens on padding.
    dataset = DecideDataset([])
    dataset.items = [None] * 400
    dataset.lengths = [32, 128, 512, 2048] * 100
    trainer = object.__new__(DecideTrainer)
    trainer.train_dataset = dataset
    trainer.args = TrainingArguments(output_dir="unused", per_device_train_batch_size=4,
        gradient_accumulation_steps=2, train_sampling_strategy="group_by_length", use_cpu=True,
        report_to=[])
    torch.manual_seed(17)
    indices = list(trainer._get_train_sampler())
    assert sorted(indices) == list(range(len(dataset)))
    def padded(order):
        return sum(4 * max(dataset.lengths[j] for j in order[i:i+4]) for i in range(0, len(order), 4))
    torch.manual_seed(17)
    assert padded(indices) < padded(torch.randperm(len(dataset)).tolist()) / 2


def test_length_grouping_recipe_change_refuses_resume(tmp_path):
    cfg = {"base_identity": "tiny", "group_by_length": True}
    (tmp_path / "run_config.json").write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="group_by_length"):
        validate_resume(tmp_path, {"base_identity": "tiny"})


def test_kernel_selection_detects_reference_fallback():
    from cuda_preflight import selected_kernel
    from functools import wraps
    def fallback():
        pass
    def wrapper(implementation):
        @wraps(fallback)
        def wrapped(*args, **kwargs):
            return implementation(*args, **kwargs)
        return wrapped
    assert selected_kernel(wrapper(fallback)) == {"module": __name__, "name": "fallback"}


def test_rounded_padding_preserves_tokens_targets_and_fp32_logits(tmp_path):
    from tiny_smoke import prepare
    from jebadiah_model import load_tokenizer, load_base
    from jebadiah_prompt import Renderer
    from train_jebadiah import DecideDataset, DecideCollator, read_jsonl
    prepare(tmp_path)
    tok = load_tokenizer(str(tmp_path / "base"))
    renderer = Renderer(tok, 4096)
    dataset = DecideDataset(read_jsonl(tmp_path / "data/train.jsonl"))
    rows = [dataset[i] for i in range(3)]
    plain = DecideCollator(tok, renderer, False, score_targets="ordinal")(rows)
    rounded = DecideCollator(tok, renderer, False, score_targets="ordinal", pad_to_multiple_of=64)(rows)
    assert rounded["input_ids"].shape[1] % 64 == 0
    for i in range(len(rows)):
        n = int(plain["attention_mask"][i].sum())
        torch.testing.assert_close(plain["input_ids"][i, :n], rounded["input_ids"][i, :n])
        assert not rounded["attention_mask"][i, n:].any()
    for key in ("cand_ids", "label_idx", "target"):
        torch.testing.assert_close(plain[key], rounded[key], atol=0, rtol=0)
    model = load_base(str(tmp_path / "base"), device="cpu", dtype=torch.float32).eval()
    with torch.no_grad():
        a = option_logits(model, plain["input_ids"], plain["attention_mask"], plain["cand_ids"])
        b = option_logits(model, rounded["input_ids"], rounded["attention_mask"], rounded["cand_ids"])
    assert a.dtype == b.dtype == torch.float32
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)


def test_adaptive_checkpointing_preserves_loss_gradients_and_dropout(tmp_path):
    import copy
    from transformers import TrainingArguments, Qwen3_5ForCausalLM
    from peft import get_peft_model, LoraConfig
    from cuda_preflight import tiny_config
    torch.manual_seed(17)
    config = tiny_config()
    config.num_hidden_layers = 1
    config.layer_types = ["full_attention"]
    initial = get_peft_model(Qwen3_5ForCausalLM(config), LoraConfig(r=2, lora_alpha=4,
        lora_dropout=.05, target_modules="all-linear", task_type="CAUSAL_LM"))
    models = [copy.deepcopy(initial) for _ in range(2)]
    trainers = []
    for i, model in enumerate(models):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        trainers.append(DecideTrainer(model=model, args=TrainingArguments(output_dir=str(tmp_path / str(i)),
            use_cpu=True, report_to=[]), checkpoint_min_tokens=4 if i == 1 else None))
    for length in (3, 8, 3):
        inputs = {"input_ids": torch.arange(length).repeat(2, 1),
                  "attention_mask": torch.ones(2, length, dtype=torch.long),
                  "cand_ids": torch.tensor([[30, 31], [30, 31]]),
                  "target": torch.tensor([[1., 0.], [0., 1.]])}
        losses = []
        for trainer in trainers:
            trainer.model.train()
            trainer.model.zero_grad(set_to_none=True)
            torch.manual_seed(123)
            loss = trainer.compute_loss(trainer.model, inputs)
            loss.backward()
            losses.append(loss.detach())
        torch.testing.assert_close(*losses, atol=0, rtol=0)
        assert all(m.gradient_checkpointing == (length >= 4) for m in trainers[1].checkpoint_modules)
        for (_, a), (_, b) in zip(models[0].named_parameters(), models[1].named_parameters()):
            if a.requires_grad:
                torch.testing.assert_close(a.grad, b.grad, atol=1e-6, rtol=1e-5)


def test_family_leakage_rejected(tmp_path):
    rows = [{"id": "row", "family_id": "family", "state": "s", "questions": {
        "c": {"type": "choice", "instructions": "Pick", "criteria": {"a": None, "b": None}},
        "n": {"type": "noul", "instructions": "Yes?"},
        "s": {"type": "score", "instructions": "Rate", "criteria": ["low", "high"]}},
        "label": {"c": "a", "n": True, "s": 1}}]
    paths = [tmp_path / name for name in ("train.jsonl", "calib.jsonl")]
    for path in paths:
        path.write_text(json.dumps(rows[0]) + "\n")
    with pytest.raises(ValueError, match="overlap"):
        validate_data(*paths)


@pytest.mark.parametrize("grouped", [False, True])
def test_effective_batch_and_optimizer_resume_match(tmp_path, grouped):
    import copy
    from transformers import TrainingArguments, TrainerCallback, Qwen3_5ForCausalLM
    from peft import get_peft_model, LoraConfig
    from cuda_preflight import tiny_config
    from hub_checkpoints import HubCheckpoint, resolve_resume
    torch.manual_seed(17)
    config = tiny_config()
    config.num_hidden_layers = 1
    config.layer_types = ["full_attention"]
    initial = get_peft_model(Qwen3_5ForCausalLM(config), LoraConfig(r=2, lora_alpha=4, lora_dropout=0,
                            target_modules="all-linear", task_type="CAUSAL_LM"))
    class Rows(list):
        lengths = [3] * 4
    dataset = Rows([{"input_ids": torch.tensor([1, 2, i+3]), "attention_mask": torch.ones(3, dtype=torch.long),
                "cand_ids": torch.tensor([30, 31]), "target": torch.tensor([float(i % 2), float(1-i % 2)])} for i in range(4)]
    )
    def collate(rows):
        return {key: torch.stack([row[key] for row in rows]) for key in rows[0]}
    recipe = {"base_identity": "tiny", "lora_rank": 2, "dataset_sha256": "fixed"}
    class Stop(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == 1:
                control.should_training_stop = True
    def train(name, micro, accum, interrupted=False, resumed=False):
        folder = tmp_path / name
        args = TrainingArguments(output_dir=str(folder), max_steps=2, per_device_train_batch_size=micro,
            gradient_accumulation_steps=accum, learning_rate=.001, lr_scheduler_type="constant",
            save_strategy="steps", save_steps=1, use_cpu=True, remove_unused_columns=False,
            report_to=[], disable_tqdm=True, seed=17,
            train_sampling_strategy="group_by_length" if grouped else "random")
        model = copy.deepcopy(initial)
        callbacks = [HubCheckpoint(recipe)] + ([Stop()] if interrupted else [])
        trainer = DecideTrainer(model=model, args=args, train_dataset=dataset, data_collator=collate, callbacks=callbacks)
        trainer.model_accepts_loss_kwargs = False
        checkpoint = resolve_resume(recipe, folder) if resumed else None
        trainer.train(resume_from_checkpoint=checkpoint)
        return {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
    full = train("full", 4, 1)
    accumulated = train("accumulated", 1, 4)
    train("resumed", 1, 4, interrupted=True)
    resumed = train("resumed", 1, 4, resumed=True)
    for name in full:
        torch.testing.assert_close(full[name], accumulated[name], atol=2e-6, rtol=1e-5)
        torch.testing.assert_close(accumulated[name], resumed[name], atol=0, rtol=0)


def test_unsupported_method_rejected_before_model_download(tmp_path, monkeypatch):
    from train_jebadiah import main
    config = tmp_path / "config.json"
    config.write_text('{"method":"full"}')
    monkeypatch.setattr("sys.argv", ["trainer", "--config", str(config)])
    with pytest.raises(ValueError, match="Only method=lora"):
        main()


def test_sequence_budget_preserves_question_and_rejects_oversized_rubric():
    import string
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    from jebadiah_prompt import Renderer
    labels = list(string.ascii_uppercase) + [a+b for a in string.ascii_uppercase for b in string.ascii_uppercase]
    vocab = {word: i for i, word in enumerate(["[UNK]", "[EOS]", *labels])}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", eos_token="[EOS]")
    tok.chat_template = "{% for m in messages %}{{m['content'] + '\\n'}}{% endfor %}{{'assistant:\\n'}}"
    renderer = Renderer(tok, max_tokens=64)
    question = {"type": "choice", "instructions": "Pick the first option", "criteria": {"x": "Yes", "y": "No"}}
    rendered = renderer.render("state " * 1000, question)
    assert rendered.truncated
    assert len(tok.encode(rendered.prompt, add_special_tokens=False)) <= 64
    assert "QUESTION: Pick the first option" in rendered.prompt
    with pytest.raises(ValueError, match="sequence length"):
        renderer.render("s", {**question, "instructions": "long " * 1000})


def test_setup_downloads_both_pinned_chat_checkpoints(tmp_path, monkeypatch):
    import huggingface_hub
    import v21_pipeline
    from tiny_smoke import prepare
    prepare(tmp_path / "tiny")
    calls = []
    class API:
        def model_info(self, repo, revision):
            return SimpleNamespace(sha=revision)
    def download(repo, revision, **kwargs):
        calls.append((repo, revision))
        return str(tmp_path / "tiny/base")
    monkeypatch.setattr(huggingface_hub, "HfApi", API)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    monkeypatch.setattr(v21_pipeline, "run", lambda *args: None)
    monkeypatch.setattr("sys.argv", ["pipeline", "--root", str(tmp_path / "run"), "--data-dir", str(tmp_path / "tiny/data"), "--setup-only"])
    v21_pipeline.main()
    assert calls == [("Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a"),
                     ("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0")]



def test_temperature_fit_stays_in_bounds_at_extreme_optima():
    from fit_temperature import fit_one
    logits = torch.tensor([[10., 0.]])
    mask = torch.tensor([[True, True]])
    cold, _, _ = fit_one(logits, torch.tensor([[1., 0.]]), mask)
    hot, _, _ = fit_one(logits, torch.tensor([[.5, .5]]), mask)
    assert .05 <= cold <= 20
    assert .05 <= hot <= 20
    assert hot == pytest.approx(20, abs=1e-4)
    with pytest.raises(ValueError, match="finite"):
        fit_one(torch.tensor([[float("nan"), 0.]]), torch.tensor([[1., 0.]]), mask)


def test_read_jsonl_keeps_unicode_line_separators(tmp_path):
    path = tmp_path / "rows.jsonl"
    rows = [{"state": "a\u2028b\u0085c"}, {"state": "plain"}]
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    assert read_jsonl(path) == rows


@pytest.mark.parametrize("filename,manifest_file,key", [
    ("train.jsonl", "metadata/manifest.json", "train.jsonl"),
    ("a1/train.jsonl", "a1/manifest.json", "train.jsonl"),
    ("a1/train.jsonl", "metadata/manifest.json", "a1/train.jsonl"),
])
def test_manifest_paths_preserve_direct_keys_and_support_relative_splits(filename, manifest_file, key):
    entry = {"sha256": "test-hash", "rows": 2, "questions": 3}
    assert split_manifest_entry({"files": {key: entry}}, filename, manifest_file) == entry
