"""CPU verification of the finite source and fail-closed A3 composition gate."""

from collections import Counter
import copy
import itertools
import json
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import item47_evidence_worlds as E
import item47_rung1_mix as M

LICENSES = Path(__file__).with_name("manifests") / "item47-source-licenses.json"


class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.splits = E.generate()

    def test_complete_source_determinism_license_and_counts(self):
        self.assertEqual(self.splits, E.generate())
        E.audit(self.splits)
        self.assertEqual(
            {s: len(rs) for s, rs in self.splits.items()},
            {"train": 6000, "diagnostic": 300},
        )
        policy = json.loads(LICENSES.read_text())
        source = policy["sources"][E.SOURCE]
        E.validate_source(source, policy["source_exclusions"])
        self.assertEqual(source["generator_sha256"], E.sha(Path(E.__file__)))
        self.assertEqual(
            source["license_sha256"],
            E.sha(Path(__file__).resolve().parents[1] / "LICENSE"),
        )
        self.assertEqual(
            source["complete_split_sha256"],
            {s: E.rows_hash(rs) for s, rs in self.splits.items()},
        )
        self.assertEqual(source["external_inputs"], [])
        self.assertIsNone(source["generator_model"])
        for rs in self.splits.values():
            c = E.counts(rs)
            self.assertEqual(c["types"]["choice"], c["types"]["noul"])
            self.assertEqual(len(set(c["evidence_status"].values())), 1)
            self.assertEqual(c["negated_claims"], len(rs) // 2)
        for ancestor in (
            "Wikipedia",
            "HoVer",
            "FEVER",
            "HotpotQA",
            "RAGTruth",
            "CLINC150",
        ):
            bad = copy.deepcopy(source)
            bad["ancestry"] = ["derived from " + ancestor]
            with self.assertRaisesRegex(ValueError, "Protected source"):
                E.validate_source(bad, policy["source_exclusions"])

    def test_independent_exhaustive_oracle_every_question(self):
        for rs in self.splits.values():
            for r in rs:
                self.assertEqual(
                    E.reference_oracle(r["world"]), r["evidence_status"], r["id"]
                )
                self.assertEqual(E.oracle(r["world"]), r["proof"])

    def test_minimal_flips_and_document_necessity(self):
        missing = Counter()
        for rs in self.splits.values():
            for r in rs:
                if (
                    r["evidence_status"] != "supported"
                    or r["questions"]["decision"]["type"] != "choice"
                ):
                    continue
                base = r["world"]
                basefacts = {f["fid"]: f for f in base["facts"]}
                ref = E.variant(base, "refuted")
                refs = {f["fid"]: f for f in ref["facts"]}
                self.assertEqual(
                    [k for k in basefacts if basefacts[k] != refs[k]], ["claim-status"]
                )
                self.assertEqual(E.oracle(ref)["status"], "refuted")
                unknown = E.variant(base, "insufficient")
                ufacts = {f["fid"]: f for f in unknown["facts"]}
                self.assertEqual(set(basefacts) - set(ufacts), {base["missing_fid"]})
                self.assertTrue(all(v == basefacts[k] for k, v in ufacts.items()))
                self.assertEqual(E.oracle(unknown)["status"], "insufficient")
                missing[
                    "join" if base["missing_fid"].startswith("join") else "terminal"
                ] += 1
                for doc in range(base["documents"]):
                    dropped = copy.deepcopy(base)
                    dropped["facts"] = [
                        f for f in base["facts"] if f["document"] != doc
                    ]
                    self.assertEqual(
                        E.oracle(dropped)["status"], "insufficient", (r["rule"], doc)
                    )
                    self.assertEqual(E.reference_oracle(dropped), "insufficient")
        self.assertGreater(missing["join"], 0)
        self.assertGreater(missing["terminal"], 0)

    def test_proof_entity_time_negation_and_order_binding(self):
        for rule in E.RULES:
            for number in (0, 1):
                world = E.world_for(rule, number)
                for status in E.STATUSES:
                    w = E.variant(world, status)
                    expected = E.oracle(w)
                    random.Random(17).shuffle(w["facts"])
                    self.assertEqual(E.oracle(w), expected)
                wrongtime = copy.deepcopy(world)
                wrongtime["query"]["period"] = "2099-12"
                self.assertEqual(E.oracle(wrongtime)["status"], "insufficient")
                wrongentity = copy.deepcopy(world)
                wrongentity["query"]["anchor"] = "UNRECORDED"
                self.assertEqual(E.oracle(wrongentity)["status"], "insufficient")
                bad = copy.deepcopy(world)
                f = next(f for f in bad["facts"] if f["fid"] == "claim-status")
                second = copy.deepcopy(f)
                second["fid"] = "conflicting"
                second["object"] = "unrecorded-status"
                bad["facts"].append(second)
                for oracle in (E.oracle, E.reference_oracle):
                    with self.assertRaisesRegex(ValueError, "Inconsistent"):
                        oracle(bad)
                # True means proved, rather than merely not explicitly refuted.
                unknown = E.variant(world, "insufficient")
                self.assertEqual(E.oracle(unknown)["status"], "insufficient")

    def test_no_gold_in_rendered_payload_and_rule_holdout(self):
        for rs in self.splits.values():
            for r in rs:
                self.assertEqual(
                    set(r["state"]), {"scope", "documents", "reference_query", "claim"}
                )
                self.assertEqual(len(r["state"]["documents"]), r["world"]["documents"])
                self.assertTrue(all(d["text"] for d in r["state"]["documents"]))
        self.assertFalse(set(E.TRAIN_RULES) & set(E.DIAGNOSTIC_RULES))
        train = self.splits["train"]
        diag = self.splits["diagnostic"]
        for key in ("id", "family_id", "group_id"):
            self.assertFalse({r[key] for r in train} & {r[key] for r in diag})
        self.assertFalse(
            {E.canonical(r["state"]) for r in train}
            & {E.canonical(r["state"]) for r in diag}
        )
        for mutation, message in (
            (
                lambda s: s["train"][0]["label"].update(decision="refuted"),
                "Wrong label",
            ),
            (
                lambda s: s["train"][0]["state"].update(claim="changed"),
                "Rendered state",
            ),
            (
                lambda s: s["train"][0]["proof"].update(status="refuted"),
                "Wrong executable proof",
            ),
            (lambda s: s["diagnostic"].append(s["train"][0]), "Rule-family leakage"),
        ):
            bad = copy.deepcopy(self.splits)
            mutation(bad)
            with self.assertRaisesRegex(ValueError, message):
                E.audit(bad)

    def test_wrong_index_location_fails_before_output_or_load(self):
        with tempfile.TemporaryDirectory(
            prefix="item47-test-", dir=os.environ.get("TMPDIR")
        ) as d:
            out = Path(d) / "out"
            with patch.object(E, "scan") as scanner:
                with self.assertRaisesRegex(ValueError, "Studio"):
                    E.build(out, Path(d), LICENSES, Path(d) / "protected.pkl")
                scanner.assert_not_called()
            self.assertFalse(out.exists())

    def test_any_overlap_rejects_whole_source_and_removes_index_symlink(self):
        with tempfile.TemporaryDirectory(
            prefix="item47-test-", dir=os.environ.get("TMPDIR")
        ) as d:
            root = Path(d)
            index = root / "protected.pkl"
            index.write_bytes(b"test-only index stand-in")
            out = root / "out"
            actual_sha = E.sha

            def pinned_sha(path):
                return (
                    E.ITEM25_INDEX_SHA256 if Path(path) == index else actual_sha(path)
                )

            def one_hit(work):
                with (work / "candidates.jsonl").open() as stream:
                    scanned = [json.loads(line) for line in stream]
                self.assertEqual(len(scanned), 6300)
                self.assertEqual({r["family"] for r in scanned}, {E.SOURCE})
                E.write_json(
                    work / "scan-report.json",
                    {
                        "scanned_records": 6300,
                        "direct_hits": 1,
                        "removed_records": 6300,
                        "retained_records": 0,
                    },
                )

            with patch.object(E, "INDEX_PATH", str(index)), patch.object(
                E, "sha", side_effect=pinned_sha
            ), patch.object(E, "render_check", return_value={}), patch.object(
                E, "scan", side_effect=one_hit
            ):
                with self.assertRaisesRegex(ValueError, "Entire owned source rejected"):
                    E.build(out, root, LICENSES, index)
            self.assertEqual(
                json.loads((out / "overlap-scan-report.json").read_text())["status"],
                "rejected",
            )
            self.assertFalse((out / "source-scan/protected.pkl").is_symlink())
            self.assertFalse((out / "train.jsonl").exists())
            self.assertFalse((out / "manifest.json").exists())

    def test_source_histogram_aggregates_record_locators_only(self):
        rows = [
            {
                "source": f"owned-source rev=123 row={i}",
                "area": "language",
                "questions": {"q": {"type": "noul"}},
            }
            for i in (10, 20)
        ]
        self.assertEqual(
            M.histograms(rows)["source_questions"], {"owned-source rev=123": 2}
        )
        rows.append({**rows[0], "source": "owned-source rev=456 row=10"})
        self.assertEqual(
            M.histograms(rows)["source_questions"],
            {"owned-source rev=123": 2, "owned-source rev=456": 1},
        )

    def test_slot_match_cap_and_mixed_row_preservation(self):
        rows = []
        for i in range(100):
            rows.append(
                {
                    "id": f"a3:{i}",
                    "family_id": f"a3:{i}",
                    "source": "old-source",
                    "area": "language",
                    "state": f"original {i}",
                    "questions": {
                        "c": {
                            "type": "choice",
                            "instructions": "Choose",
                            "criteria": {"x": None, "y": None},
                        },
                        "b": {"type": "noul", "instructions": "True?"},
                        "s": {
                            "type": "score",
                            "instructions": "Rate",
                            "criteria": ["low", "high"],
                        },
                    },
                    "label": {"c": "x", "b": True, "s": 1},
                    "target": {"c": {"x": 1, "y": 0}, "b": {"true": 1, "false": 0}},
                }
            )
        before = copy.deepcopy(rows)
        result, receipt = M.replace_slots(rows, self.splits["train"])
        self.assertEqual(rows, before)
        self.assertEqual(receipt["replaced_slots"], 30)
        self.assertEqual(receipt["selected_types"], {"choice": 15, "noul": 15})
        self.assertEqual(sum(len(r["questions"]) for r in result), 300)
        self.assertEqual(receipt["before"]["area_type"], receipt["after"]["area_type"])
        for original in rows:
            retained = next(r for r in result if r["id"] == original["id"])
            self.assertEqual(retained["state"], original["state"])
            self.assertEqual(retained["questions"]["s"], original["questions"]["s"])
            for qid, q in retained["questions"].items():
                self.assertEqual(q, original["questions"][qid])
                self.assertEqual(retained["label"][qid], original["label"][qid])
            for field in ("questions", "label", "target"):
                self.assertFalse(
                    set(retained.get(field, {})) - set(retained["questions"])
                )
        with self.assertRaisesRegex(ValueError, "already exposed"):
            M.replace_slots([self.splits["train"][0]], self.splits["train"])
        with self.assertRaisesRegex(ValueError, "No matched"):
            M.replace_slots([{**rows[0], "area": "arts"}], self.splits["train"])

    def artifact_fixture(self, root, passed=True):
        base = root / "base"
        owned = root / "owned"
        base.mkdir()
        owned.mkdir()
        rows = [
            {
                "id": f"base:{i}",
                "family_id": f"base:{i}",
                "source": "old-source",
                "area": "language",
                "state": f"base state {i}",
                "questions": {
                    "c": {
                        "type": "choice",
                        "instructions": "Choose",
                        "criteria": {"x": None, "y": None},
                    },
                    "b": {"type": "noul", "instructions": "True?"},
                },
                "label": {"c": "x", "b": True},
            }
            for i in range(100)
        ]
        calib = [
            {
                "id": "calib",
                "family_id": "calib",
                "state": "calibration state",
                "questions": {"b": {"type": "noul", "instructions": "True?"}},
                "label": {"b": True},
            }
        ]
        E.write_rows(base / "train.jsonl", rows)
        E.write_rows(base / "calib.jsonl", calib)
        E.write_json(base / "manifest.json", {})
        for s, rs in self.splits.items():
            E.write_rows(owned / f"{s}.jsonl", rs)
        scan = {
            "status": "passed" if passed else "pending_studio_scan",
            "source_level": True,
            "protected_index_sha256": E.ITEM25_INDEX_SHA256,
            "scanned_records": 6300,
            "retained_records": 6300,
            "removed_records": 0,
            "removed_families": 0,
            "invalid_empty_state_records": 0,
            "direct_hits": 0,
            "scanner_sha256": E.sha(
                Path(E.__file__).with_name("item33_full_suite_scan.py")
            ),
            "candidates_sha256": E.rows_hash(E.scan_rows(self.splits)),
        }
        from jebadiah_prompt import PROMPT_SOURCE_SHA256

        contract = json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "results/runs/9b-chat-v1/adapter/prompt_contract.json"
            ).read_text()
        )
        manifest = {
            "generator_sha256": E.sha(Path(E.__file__)),
            "license_manifest_sha256": E.sha(LICENSES),
            "overlap_scan": scan,
            "render_validation": {
                "truncated": 0,
                "max_seq_length": 2048,
                "prompt_budget": 1984,
                "padding_reserve": 64,
                "checked_renders": 18900,
                "tokenizer_sha256": E.TOKENIZER_SHA256,
                "prompt_source_sha256": PROMPT_SOURCE_SHA256,
                "chat_template_sha256": contract["chat_template_sha256"],
                "max_prompt_tokens_by_type": {"choice": 1500, "noul": 1400},
            },
            "splits": {s: E.counts(rs) for s, rs in self.splits.items()},
            "files": {
                f"{s}.jsonl": {"sha256": E.sha(owned / f"{s}.jsonl")}
                for s in self.splits
            },
        }
        E.write_json(owned / "manifest.json", manifest)
        E.write_json(owned / "overlap-scan-report.json", scan)
        return base, owned, manifest

    def compose_fixture(self, base, owned, out):
        # Test-only synthetic scan attestations, never a real contamination pass.
        with patch.object(
            M, "A3_HASHES", {n: E.sha(base / n) for n in M.A3_HASHES}
        ), patch.object(
            M, "A3_MANIFEST_SHA256", E.sha(base / "manifest.json")
        ), patch.object(
            M, "TRAIN_QUESTIONS", 200
        ), patch.object(
            M, "CALIB_QUESTIONS", 1
        ):
            return M.compose(base, owned, out, LICENSES)

    def test_composition_success_original_calibration_and_holdout_bytes(self):
        with tempfile.TemporaryDirectory(
            prefix="item47-test-", dir=os.environ.get("TMPDIR")
        ) as d:
            root = Path(d)
            base, owned, _ = self.artifact_fixture(root)
            report = self.compose_fixture(base, owned, root / "out")
            self.assertEqual(report["questions"], 200)
            self.assertEqual(report["slots"]["replaced_slots"], 20)
            self.assertEqual(
                (base / "calib.jsonl").read_bytes(),
                (root / "out/calib.jsonl").read_bytes(),
            )
            self.assertEqual(
                (owned / "diagnostic.jsonl").read_bytes(),
                (root / "out/evidence-diagnostic.jsonl").read_bytes(),
            )
            self.assertFalse(report["training_launched"])
            with self.assertRaisesRegex(ValueError, "empty"):
                self.compose_fixture(base, owned, root / "out")

    def test_composition_pending_and_unbound_scan_fail_without_output(self):
        with tempfile.TemporaryDirectory(
            prefix="item47-test-", dir=os.environ.get("TMPDIR")
        ) as d:
            root = Path(d)
            base, owned, manifest = self.artifact_fixture(root, False)
            with self.assertRaisesRegex(ValueError, "Studio scan"):
                self.compose_fixture(base, owned, root / "out")
            self.assertFalse((root / "out").exists())
            manifest["overlap_scan"]["status"] = "passed"
            for field, badvalue in (
                ("candidates_sha256", "wrong"),
                ("scanner_sha256", "wrong"),
                ("protected_index_sha256", "wrong"),
                ("scanned_records", 6000),
                ("source_level", False),
                ("removed_records", 1),
                ("direct_hits", 1),
            ):
                bad = copy.deepcopy(manifest)
                bad["overlap_scan"][field] = badvalue
                E.write_json(owned / "manifest.json", bad)
                E.write_json(owned / "overlap-scan-report.json", bad["overlap_scan"])
                with self.assertRaisesRegex(ValueError, "unbound source scan"):
                    self.compose_fixture(base, owned, root / "out")
                self.assertFalse((root / "out").exists())

    def test_composition_file_render_license_and_base_pins(self):
        with tempfile.TemporaryDirectory(
            prefix="item47-test-", dir=os.environ.get("TMPDIR")
        ) as d:
            root = Path(d)
            base, owned, manifest = self.artifact_fixture(root)
            for field, value in (
                ("truncated", 1),
                ("max_seq_length", 4096),
                ("checked_renders", 18000),
                ("tokenizer_sha256", "wrong"),
                ("prompt_budget", 2048),
            ):
                bad = copy.deepcopy(manifest)
                bad["render_validation"][field] = value
                E.write_json(owned / "manifest.json", bad)
                with self.assertRaisesRegex(ValueError, "Render checks"):
                    self.compose_fixture(base, owned, root / "out")
            bad = copy.deepcopy(manifest)
            bad["license_manifest_sha256"] = "wrong"
            E.write_json(owned / "manifest.json", bad)
            with self.assertRaisesRegex(ValueError, "license manifest"):
                self.compose_fixture(base, owned, root / "out")
            E.write_json(owned / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "pinned A3"):
                M.compose(base, owned, root / "out", LICENSES)
            with (owned / "train.jsonl").open("a") as f:
                f.write("\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                self.compose_fixture(base, owned, root / "out")
            self.assertFalse((root / "out").exists())

    @unittest.skipUnless(
        os.environ.get("ITEM47_TOKENIZER"),
        "Pinned tokenizer required for live CPU collator",
    )
    def test_live_renderer_and_training_collator_gold_mapping(self):
        from transformers import AutoTokenizer
        from jebadiah_prompt import Renderer, wire_keys
        from train_jebadiah import DecideCollator, DecideDataset

        tok = AutoTokenizer.from_pretrained(
            os.environ["ITEM47_TOKENIZER"], local_files_only=True
        )
        renderer = Renderer(tok, max_tokens=1984)
        rows = [
            next(
                r
                for r in self.splits["train"]
                if r["questions"]["decision"]["type"] == kind
                and r["evidence_status"] == status
            )
            for kind, status in itertools.product(("choice", "noul"), E.STATUSES)
        ]
        collator = DecideCollator(
            tok, renderer, shuffle_choice=True, pad_to_multiple_of=64
        )
        saved = random.getstate()
        try:
            random.seed(17)
            rng = random.Random()
            rng.setstate(collator.rng.getstate())
            gold = []
            for r in rows:
                q = r["questions"]["decision"]
                keys = wire_keys(q)
                if q["type"] == "choice":
                    rng.shuffle(keys)
                label = r["label"]["decision"]
                wire = label if q["type"] == "choice" else "true" if label else "false"
                gold.append(keys.index(wire))
            batch = collator(list(DecideDataset(rows)))
        finally:
            random.setstate(saved)
        self.assertEqual(batch["label_idx"].tolist(), gold)
        self.assertEqual(batch["target"].argmax(dim=-1).tolist(), gold)
        self.assertLessEqual(batch["input_ids"].shape[1], 2048)
        self.assertEqual(collator.truncated, 0)
        self.assertTrue((batch["cand_ids"][3:, 2:] == -1).all())


if __name__ == "__main__":
    unittest.main()
