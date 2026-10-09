"""CPU regression tests for original-label conversion and fail-closed admission."""
import copy
import csv
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import item49_corr2cause as C
import item49_rung2_mix as M

CHAIN = ("Suppose there is a closed system of 3 variables, A, B and C. All the statistical relations "
         "among these 3 variables are as follows: A correlates with B. A correlates with C. "
         "B correlates with C. However, A and C are independent given B.")
COLLIDER = ("Suppose there is a closed system of 3 variables, A, B and C. All the statistical relations "
            "among these 3 variables are as follows: A correlates with B. B correlates with C. "
            "However, A is independent of C.")
TRIANGLE = ("Suppose there is a closed system of 3 variables, A, B and C. All the statistical relations "
            "among these 3 variables are as follows: A correlates with B. A correlates with C. B correlates with C.")


def source_row(label="1", premise=CHAIN, hypothesis="A affects C."):
    return {"input": f"Premise: {premise}\nHypothesis: {hypothesis}",
            "label": label, "num_variables": "3", "template": "non-parent ancestor"}


class ConverterTests(unittest.TestCase):
    def test_binary_labels_are_not_three_way_nli(self):
        for label in ("0", "1"):
            for kind in ("choice", "noul"):
                row = C.convert(source_row(label), 7, kind)
                self.assertEqual(row["state"], source_row(label)["input"])
                expected = ("valid" if label == "1" else "not_valid") if kind == "choice" else label == "1"
                self.assertEqual(row["label"]["decision"], expected)
                self.assertNotIn("neutral", row["questions"]["decision"].get("criteria", {}))
                self.assertEqual(row["provenance"]["original_split"], "train")
                C.audit({"train": [row]})

    def test_reject_invalid_label_and_malformed_input(self):
        for label in ("neutral", "2", "True", ""):
            with self.assertRaisesRegex(ValueError, "Invalid binary"):
                C.convert(source_row(label), 0, "noul")
        for text in ("", "Hypothesis: A.", "Premise: A.", "Premise: A.\nHypothesis: B.\nHypothesis: C."):
            with self.assertRaises(ValueError):
                C.split_input(text)

    def test_family_is_invariant_under_variable_renaming(self):
        import re
        renamed = re.sub(r"\b[A-C]\b", lambda m: {"A": "Z", "B": "Y", "C": "X"}[m[0]], CHAIN)
        self.assertEqual(C.graph_family(CHAIN), C.graph_family(renamed))
        # A path with a collider is a different MEC but the same conservative
        # skeleton family; a triangle must remain a different family.
        self.assertEqual(C.graph_family(CHAIN), C.graph_family(COLLIDER))
        self.assertNotEqual(C.graph_family(CHAIN), C.graph_family(TRIANGLE))
        self.assertEqual(C.graph_family(CHAIN), "corr2cause:skeleton:3:3")

    def test_all_isomorphic_skeletons_share_family(self):
        from itertools import permutations
        edges = ((0, 1), (1, 2), (2, 3), (1, 4))
        expected = C.skeleton_key(5, edges)
        for perm in permutations(range(5)):
            relabeled = tuple(sorted(tuple(sorted((perm[a], perm[b]))) for a, b in edges))
            self.assertEqual(C.skeleton_key(5, relabeled), expected)
        self.assertNotEqual(C.skeleton_key(5, ((0, 1), (1, 2), (2, 3), (3, 4))), expected)

    def test_unknown_premise_fails_closed(self):
        for premise in ("Unknown graph", CHAIN.replace("B correlates with C.", ""),
                        CHAIN.replace("B and C", "B and Z")):
            with self.assertRaises(ValueError):
                C.graph_family(premise)

    def test_all_a3_paired_text_formats_resolve_same_graph(self):
        for prefix, separator in (("text_A: ", "\ntext_B: "), ("A: ", "\nB: "),
                                  ("Passage A:\n", "\n\nPassage B:\n")):
            self.assertEqual(C.a3_premise(prefix + CHAIN + separator + "claim"), CHAIN)
        with self.assertRaisesRegex(ValueError, "Unrecognized A3"):
            C.a3_premise("unrecognized")

    def test_family_siblings_and_duplicate_states_cannot_cross_split(self):
        one = C.convert(source_row(), 0, "choice")
        two = C.convert(source_row(hypothesis="B affects C."), 1, "noul")
        with self.assertRaisesRegex(ValueError, "Graph-family leakage"):
            C.audit({"train": [one], "diagnostic": [two]})
        two = copy.deepcopy(one); two["id"] = "other"
        with self.assertRaisesRegex(ValueError, "Duplicate candidate"):
            C.audit({"train": [one, two]})

    def test_entire_source_ancestry_rejected(self):
        policy = json.loads(C.POLICY.read_text())
        for ancestor in ("CLadder", "SNLI", "HotpotQA", "GSM8K"):
            source = copy.deepcopy(policy["sources"][C.SOURCE]); source["ancestry"].append(ancestor)
            with self.assertRaisesRegex(ValueError, "Protected source ancestry"):
                C.validate_source(source, policy["source_exclusions"])
        source = copy.deepcopy(policy["sources"][C.SOURCE]); source["commercial_use_ok"] = False
        with self.assertRaisesRegex(ValueError, "Commercial permission"):
            C.validate_source(source, policy["source_exclusions"])

    def test_raw_scan_includes_every_file_and_split(self):
        with tempfile.TemporaryDirectory(prefix="item49-test-") as tmp:
            root = Path(tmp)
            for name in ("train.csv", "dev.csv", "test.csv", "perturbation.csv"):
                with (root / name).open("w") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(source_row())); writer.writeheader()
                    writer.writerow(source_row())
            (root / "paraphrase.json").write_text(json.dumps([{"premise": "original", "hypothesis": "claim"}]))
            policy = {"sources": {C.SOURCE: {"complete_files": {p.name: {} for p in root.iterdir()}}}}
            rows = list(C.raw_scan_rows(root, policy))
            self.assertEqual(len(rows), 5)
            self.assertEqual({r["family"] for r in rows}, {C.SOURCE})
            self.assertTrue(any("dev.csv" in r["id"] for r in rows))
            self.assertTrue(any("paraphrase.json" in r["id"] for r in rows))


class CompositionTests(unittest.TestCase):
    def donors(self, choices=10, nouls=10):
        rows = []
        for i in range(max(choices, nouls)):
            questions = {}; labels = {}; target = {}
            for kind, count in (("choice", choices), ("noul", nouls)):
                if i < count:
                    q = {"type": kind, "instructions": "original question"}
                    if kind == "choice": q["criteria"] = {"a": None, "b": None}
                    questions[kind] = q; labels[kind] = "a" if kind == "choice" else True
                    target[kind] = {"a": 1.0} if kind == "choice" else {"true": 1.0, "false": 0.0}
            rows.append({"id": f"base:{i}", "source": "original", "area": "knowledge", "family_id": f"base:{i}",
                         "state": f"original state {i}", "questions": questions, "label": labels, "target": target})
        return rows

    def incoming(self, n=100):
        return [C.convert(source_row(hypothesis=f"claim {i}."), i, ("choice", "noul")[i % 2]) for i in range(n)]

    def test_exact_matched_slot_replacement_preserves_remaining_questions(self):
        donors = self.donors()
        result, receipt = M.replace_slots(donors, self.incoming())
        self.assertEqual(receipt["replaced_slots"], 2)
        self.assertEqual(sum(len(r["questions"]) for r in result), 20)
        self.assertEqual(receipt["before"]["area_type"], receipt["after"]["area_type"])
        mapping = {(r["original_id"], r["question_id"]) for r in receipt["replacements"]}
        kept = {r["id"]: r for r in result if not M.is_upstream(r)}
        for old in donors:
            for qid in old["questions"]:
                if (old["id"], qid) in mapping:
                    if old["id"] in kept:
                        self.assertNotIn(qid, kept[old["id"]]["questions"])
                        self.assertNotIn(qid, kept[old["id"]]["label"])
                        self.assertNotIn(qid, kept[old["id"]]["target"])
                else:
                    for field in ("questions", "label", "target"):
                        self.assertEqual(kept[old["id"]][field][qid], old[field][qid])
                    self.assertEqual(kept[old["id"]]["state"], old["state"])
        self.assertEqual(M.replace_slots(donors, self.incoming()), (result, receipt))

    def test_inherited_exposure_counts_variants_even_when_renamed(self):
        donors = self.donors(50, 50)
        donors[0]["source"] = "renamed-variant"
        donors[0]["source_family"] = "Corr2Cause"
        result, receipt = M.replace_slots(donors, self.incoming())
        self.assertEqual(receipt["existing_upstream_questions"], 2)
        self.assertEqual(receipt["replaced_slots"], 8)
        self.assertLessEqual(receipt["upstream_questions"], 10)
        self.assertEqual(sum(len(r["questions"]) for r in result), 100)

    def test_fewer_donors_reduce_replacements(self):
        donors = self.donors(1, 0)
        # Add presentations outside the matching area/type to get cap headroom.
        extra = copy.deepcopy(donors[0]); extra["id"] = "unmatched"; extra["area"] = "language"
        extra["questions"] = {str(i): {"type": "noul", "instructions": "untouched"} for i in range(19)}
        extra["label"] = {str(i): True for i in range(19)}
        result, receipt = M.replace_slots(donors + [extra], self.incoming())
        self.assertEqual(receipt["replaced_slots"], 1)
        self.assertIn(extra, result)

    def test_no_source_cap_headroom_fails(self):
        donors = self.donors()
        donors[0]["source"] = "corr2cause"
        with self.assertRaisesRegex(ValueError, "cap headroom"):
            M.replace_slots(donors, self.incoming())

    def test_pending_scan_gate_does_not_read_index_or_write_composition(self):
        with tempfile.TemporaryDirectory(prefix="item49-test-") as tmp:
            root = Path(tmp); causal = root / "causal"; causal.mkdir()
            (causal / "manifest.json").write_text(json.dumps({"overlap_scan": {"status": "pending_studio_scan"}}))
            with patch.object(C, "validate_policy", side_effect=AssertionError("must not read source")):
                with self.assertRaisesRegex(ValueError, "Studio scan required"):
                    M.validate_artifacts(root / "raw", causal, C.POLICY)
            self.assertFalse((root / "composition").exists())

    def test_scan_rejects_non_studio_index_before_reading(self):
        with tempfile.TemporaryDirectory(prefix="item49-test-") as tmp:
            root = Path(tmp)
            with patch.object(C, "sha", side_effect=AssertionError("must not read index")):
                with self.assertRaisesRegex(ValueError, "exact Studio path"):
                    C.scan_phase([], root, root / "fake.pkl", "raw-source-scan")

    def test_upstream_overlap_stops_before_sampling_and_rendering(self):
        with tempfile.TemporaryDirectory(prefix="item49-test-") as tmp:
            root = Path(tmp)
            with (patch.object(C, "validate_policy", return_value={"sources": {C.SOURCE: {"complete_files": {}}}}),
                  patch.object(C, "scan_phase", side_effect=ValueError("Entire upstream source rejected")),
                  patch.object(C, "select", side_effect=AssertionError("must not sample")),
                  patch.object(C, "renderer", side_effect=AssertionError("must not render"))):
                with self.assertRaisesRegex(ValueError, "Entire upstream"):
                    C.build(root, root, root / "out", root, index=root / "fake.pkl")
            self.assertFalse((root / "out/train.jsonl").exists())

    def test_scanner_error_removes_temporary_index_symlink(self):
        with tempfile.TemporaryDirectory(prefix="item49-test-") as tmp:
            root = Path(tmp)
            with (patch.object(C, "check_index"), patch.object(C, "scan", side_effect=RuntimeError("scanner failed"))):
                with self.assertRaisesRegex(RuntimeError, "scanner failed"):
                    C.scan_phase([], root, root / "synthetic-index.pkl", "raw-source-scan")
            self.assertFalse((root / "raw-source-scan/protected.pkl").is_symlink())

    def test_one_hit_rejects_the_entire_source(self):
        with tempfile.TemporaryDirectory(prefix="item49-test-") as tmp:
            root = Path(tmp)
            def rejected(work):
                C.write_json(work / "scan-report.json", {"removed_records": 2, "direct_hits": 1,
                                                        "scanned_records": 2, "removed_families": 1, "retained_records": 0})
            rows = [{"id": str(i), "family": C.SOURCE, "state": "synthetic", "questions": {}} for i in range(2)]
            with patch.object(C, "check_index"), patch.object(C, "scan", side_effect=rejected):
                with self.assertRaisesRegex(ValueError, "no row salvage"):
                    C.scan_phase(rows, root, root / "synthetic-index.pkl", "raw-source-scan")
            self.assertEqual(json.loads((root / "overlap-scan-report.json").read_text())["status"], "rejected")
            self.assertFalse((root / "train.jsonl").exists())


@unittest.skipUnless(os.environ.get("ITEM49_RAW") and os.environ.get("ITEM49_CANDIDATES"), "requires pinned local CPU build")
class LiveArtifactTests(unittest.TestCase):
    def test_pinned_release_and_original_train_provenance(self):
        raw = Path(os.environ["ITEM49_RAW"]); output = Path(os.environ["ITEM49_CANDIDATES"])
        policy = C.validate_policy(raw)
        self.assertEqual(sum(f["records"] for f in policy["sources"][C.SOURCE]["complete_files"].values()), 1035117)
        splits = {s: C.read(output / f"{s}.jsonl") for s in ("train", "diagnostic")}
        C.audit(splits)
        self.assertEqual([len(splits[s]) for s in splits], [6000, 300])
        selected = {r["provenance"]["row"]: r for rows in splits.values() for r in rows}
        for i, row in enumerate(C.upstream_rows(raw / "train.csv")):
            if i in selected:
                candidate = selected[i]
                self.assertEqual(C.convert(row, i, candidate["questions"]["decision"]["type"]), candidate)
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest["render_validation"]["truncated"], 0)
        self.assertEqual(manifest["render_validation"]["checked_renders"], 9450)
        self.assertEqual(manifest["overlap_scan"]["status"], "pending_studio_scan")


class ScanReceiptTests(unittest.TestCase):
    """Synthetic receipts exercise gate logic, never claim Studio clearance."""
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="item49-test-receipts-")
        cls.root = Path(cls.tmp.name)
        cls.raw = cls.root / "raw"; cls.raw.mkdir()
        cls.causal = cls.root / "causal"; cls.causal.mkdir()
        cls.original = [source_row(str(i % 2), CHAIN if i < 6000 else TRIANGLE, f"claim {i}.") for i in range(6300)]
        with (cls.raw / "train.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(cls.original[0])); writer.writeheader(); writer.writerows(cls.original)
        cls.splits = {s: [C.convert(cls.original[i], i, ("choice", "noul")[i % 2]) for i in indices]
                      for s, indices in (("train", range(6000)), ("diagnostic", range(6000, 6300)))}
        for s, rows in cls.splits.items(): C.write_rows(cls.causal / f"{s}.jsonl", rows)
        cls.policy = json.loads(C.POLICY.read_text())
        cls.policy["sources"][C.SOURCE]["complete_files"] = {"train.csv": {"sha256": C.sha(cls.raw / "train.csv"), "records": 6300}}
        def phase(rows):
            return {"status": "passed", "candidates_sha256": M.stream_hash(rows),
                    "scanned_records": 6300, "retained_records": 6300, "removed_records": 0,
                    "direct_hits": 0, "invalid_empty_state_records": 0, "removed_families": 0,
                    "source_level": True, "protected_index_sha256": C.ITEM25_INDEX_SHA256,
                    "scanner_sha256": C.sha(Path(C.__file__).with_name("item33_full_suite_scan.py"))}
        cls.report = {"status": "passed", "source_level": True, "protected_index_sha256": C.ITEM25_INDEX_SHA256,
                      "raw_files_sha256": C.digest(json.dumps(cls.policy["sources"][C.SOURCE]["complete_files"], sort_keys=True)),
                      "converted_sha256": {s: C.sha(cls.causal / f"{s}.jsonl") for s in cls.splits}, "scanned_records": 12600,
                      "raw": phase(C.raw_scan_rows(cls.raw, cls.policy)), "converted": phase(C.converted_scan_rows(cls.splits))}
        from jebadiah_prompt import PROMPT_SOURCE_SHA256
        contract = json.loads((Path(C.__file__).resolve().parents[1] / "results/runs/9b-chat-v1/adapter/prompt_contract.json").read_text())
        cls.manifest = {"overlap_scan": cls.report, "generator_sha256": C.sha(Path(C.__file__)),
                        "license_manifest_sha256": C.sha(C.POLICY), "source_revision": C.REVISION,
                        "source_files": cls.policy["sources"][C.SOURCE]["complete_files"],
                        "files": {f"{s}.jsonl": {"sha256": C.sha(cls.causal / f"{s}.jsonl")} for s in cls.splits},
                        "splits": {s: C.counts(rows) for s, rows in cls.splits.items()},
                        "render_validation": {"truncated": 0, "max_seq_length": 2048, "prompt_budget": 1984,
                                              "padding_reserve": 64, "checked_renders": 9450,
                                              "tokenizer_sha256": C.TOKENIZER_SHA256,
                                              "prompt_source_sha256": PROMPT_SOURCE_SHA256,
                                              "chat_template_sha256": contract["chat_template_sha256"],
                                              "max_prompt_tokens_by_type": {"choice": 200, "noul": 180},
                                              "choice_orders": ["canonical", "reversed"], "noul_order": "true,false"}}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def write_receipt(self, manifest):
        C.write_json(self.causal / "manifest.json", manifest)
        C.write_json(self.causal / "overlap-scan-report.json", manifest["overlap_scan"])

    def test_full_source_and_converted_receipts_are_required_and_bound(self):
        self.write_receipt(self.manifest)
        with patch.object(C, "validate_policy", return_value=self.policy):
            M.validate_artifacts(self.raw, self.causal, C.POLICY)
            for phase in ("raw", "converted"):
                for key, wrong in (("candidates_sha256", "incorrect"), ("scanned_records", 6299),
                                   ("direct_hits", 1), ("removed_families", 1),
                                   ("scanner_sha256", "incorrect"), ("protected_index_sha256", "incorrect")):
                    with self.subTest(phase=phase, key=key):
                        manifest = copy.deepcopy(self.manifest); manifest["overlap_scan"][phase][key] = wrong
                        self.write_receipt(manifest)
                        with self.assertRaisesRegex(ValueError, "source scan"):
                            M.validate_artifacts(self.raw, self.causal, C.POLICY)
            for phase in ("raw", "converted"):
                manifest = copy.deepcopy(self.manifest); del manifest["overlap_scan"][phase]
                self.write_receipt(manifest)
                with self.assertRaisesRegex(ValueError, "source scan"):
                    M.validate_artifacts(self.raw, self.causal, C.POLICY)

    def test_rehashed_wrong_label_still_fails_original_train_verification(self):
        bad = copy.deepcopy(self.splits["train"])
        bad[0]["label"]["decision"] = "valid"
        C.write_rows(self.causal / "train.jsonl", bad)
        manifest = copy.deepcopy(self.manifest)
        manifest["files"]["train.jsonl"]["sha256"] = C.sha(self.causal / "train.jsonl")
        manifest["splits"]["train"] = C.counts(bad)
        self.write_receipt(manifest)
        try:
            with patch.object(C, "validate_policy", return_value=self.policy):
                with self.assertRaisesRegex(ValueError, "original train row"):
                    M.validate_artifacts(self.raw, self.causal, C.POLICY)
        finally:
            C.write_rows(self.causal / "train.jsonl", self.splits["train"])

    def test_synthetic_gated_composition_preserves_calibration_bytes(self):
        self.write_receipt(self.manifest)
        base = self.root / "base"; base.mkdir(exist_ok=True)
        donors = CompositionTests().donors()
        C.write_rows(base / "train.jsonl", donors)
        # Intentional whitespace tests byte preservation beyond parsed equality.
        calib = {"id": "calibration", "source": "original", "family_id": "calib",
                 "state": "held out", "questions": {"q": {"type": "noul", "instructions": "held out"}}, "label": {"q": True}}
        (base / "calib.jsonl").write_text(json.dumps(calib, separators=(", ", ": ")) + "\n")
        C.write_json(base / "manifest.json", {"synthetic_test": True})
        out = self.root / "composition"
        with (patch.object(C, "validate_policy", return_value=self.policy),
              patch.object(C, "existing_families", return_value=set()),
              patch.object(M, "A3_HASHES", {n: C.sha(base / n) for n in ("train.jsonl", "calib.jsonl")}),
              patch.object(M, "A3_MANIFEST_SHA256", C.sha(base / "manifest.json")),
              patch.object(M, "PRESENTATIONS", 20), patch.object(M, "CALIBRATION", 1)):
            report = M.compose(self.raw, base, self.causal, out)
        self.assertEqual((out / "calib.jsonl").read_bytes(), (base / "calib.jsonl").read_bytes())
        self.assertEqual(report["slots"]["replaced_slots"], 2)
        self.assertEqual(len(C.read(out / "causal-diagnostic.jsonl")), 300)
        self.assertEqual((out / "NOTICE-Corr2Cause.txt").read_text(), self.policy["sources"][C.SOURCE]["notice_text"])


if __name__ == "__main__":
    unittest.main()
