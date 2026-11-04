import ast
import copy
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, mock_open, patch


NOTEBOOK = Path(__file__).resolve().parents[1] / "HPLC_AI.ipynb"
CELLS = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]


def source(index):
    return "".join(CELLS[index]["source"])


def load_worker(masses, targets, intact):
    tree = ast.parse(source(5))
    worker = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "process_peak"
    )
    namespace = {
        "TOLERANCE_DA": 0.05,
        "tree_mzs": masses,
        "tree_targets": targets,
        "tree_is_intact": intact,
        "tree_names": [
            f"T({target}) " + ("[M+H]+" if anchor else "Frag")
            for target, anchor in zip(targets, intact)
        ],
        "tree_abunds": [100] * len(masses),
    }
    exec(compile(ast.Module(body=[worker], type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    return namespace["process_peak"]


def run_advice(records, enabled=False, client=None, key=""):
    tree = ast.parse(source(7))
    if enabled:
        for node in tree.body:
            if isinstance(node, ast.Assign) and node.targets[0].id == "ENABLE_GEMINI":
                node.value = ast.Constant(True)
    namespace = {
        "assigned_df": SimpleNamespace(empty=False, to_json=lambda **_: json.dumps(records)),
        "gemini_api_key": key,
        "gemini_advice": {"summary": "stale"},
    }
    with patch.dict("sys.modules", {"google": SimpleNamespace(genai=client)}):
        with patch("getpass.getpass", return_value=""), redirect_stdout(io.StringIO()):
            exec(compile(ast.fix_missing_locations(tree), str(NOTEBOOK), "exec"), namespace)
    return namespace


class AssignmentTests(unittest.TestCase):
    def task(self, spectrum):
        return {"Peak #": 1, "Retention Time (min)": "2.00", "Spectrum": spectrum}

    def test_intact_match_is_tentative_and_preserves_input(self):
        worker = load_worker([100.123456], ["1"], [True])
        task = self.task({100.123457: 100})
        original = copy.deepcopy(task)
        rows = worker(task)
        self.assertEqual(task, original)
        self.assertEqual(rows[0]["Assignment Status"], "MS1 CANDIDATE")
        self.assertEqual(rows[0]["Confidence"], "Tentative MS1 candidate")
        self.assertEqual(rows[0]["Calculated m/z"], 100.123456)
        self.assertAlmostEqual(rows[0]["Mass Difference"], 0.000001)
        self.assertNotIn("Spectrum", rows[0])

    def test_all_compatible_targets_remain_ambiguous(self):
        worker = load_worker([100, 100.01], ["1", "2"], [True, True])
        rows = worker(self.task({100: 100}))
        self.assertEqual({row["Candidate Target"] for row in rows}, {"1", "2"})
        self.assertEqual({row["Assignment Status"] for row in rows}, {"AMBIGUOUS"})
        self.assertTrue(all(row["Confidence"] == "Ambiguous MS1 candidates" for row in rows))

    def test_coeluting_targets_are_not_reduced_to_a_winner(self):
        worker = load_worker([100, 200], ["1", "2"], [True, True])
        rows = worker(self.task({100: 100, 200: 5}))
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["Assignment Status"] == "AMBIGUOUS" for row in rows))

    def test_multiple_ions_for_one_target_are_not_multiple_targets(self):
        worker = load_worker([100, 200], ["1", "1"], [True, False])
        rows = worker(self.task({100: 100, 200: 10}))
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["Assignment Status"] == "MS1 CANDIDATE" for row in rows))

    def test_fragment_only_evidence_is_insufficient(self):
        row = load_worker([100], ["1"], [False])(self.task({100: 10}))[0]
        self.assertEqual(row["Assignment Status"], "FRAGMENT MATCH ONLY")
        self.assertEqual(row["Confidence"], "Insufficient evidence")
        self.assertIn("not measured MS/MS", row["Assignment Rationale"])

    def test_unmatched_and_empty_signals(self):
        worker = load_worker([100], ["1"], [True])
        unknown = worker(self.task({300: 10}))[0]
        self.assertEqual(unknown["Assignment Status"], "UNASSIGNED")
        self.assertEqual(unknown["Identified Adduct"], "Unassigned")
        self.assertEqual(unknown["Confidence"], "Unknown")
        empty = worker(self.task({}))[0]
        self.assertEqual(empty["Assignment Status"], "SKIPPED")
        self.assertNotIn("Spectrum", empty)

    def test_tolerance_and_signed_error(self):
        worker = load_worker([100], ["1"], [True])
        self.assertEqual(worker(self.task({100.049: 1}))[0]["Assignment Status"], "MS1 CANDIDATE")
        self.assertEqual(worker(self.task({100.051: 1}))[0]["Assignment Status"], "UNASSIGNED")
        self.assertLess(worker(self.task({99.99: 1}))[0]["Mass Difference"], 0)
        self.assertEqual(worker(self.task({99.99: 1})), worker(self.task({99.99: 1})))

    def test_screening_has_no_gemini_or_key_dependency(self):
        self.assertNotIn("genai", source(5))
        self.assertNotIn("getpass", source(5))
        self.assertNotIn("generate_content", source(5))
        for index in (5, 6, 8):
            self.assertNotIn("'VERIFIED'", source(index))
            self.assertNotIn("Verification Status", source(index))


class AdvisoryTests(unittest.TestCase):
    advice = {"summary": "Tentative matches only", "limitations": ["MS1"], "suggested_checks": ["Blanks"]}

    def test_default_is_offline_and_clears_stale_advice(self):
        namespace = run_advice([])
        self.assertEqual(namespace["gemini_advice_status"], "DISABLED")
        self.assertIsNone(namespace["gemini_advice"])

    def test_strict_schema_rejects_malformed_or_assignment_fields(self):
        parse = run_advice([])["parse_gemini_advice"]
        self.assertEqual(parse(json.dumps(self.advice)), self.advice)
        for payload in [
            "VERIFIED", "[]", "null", "{}",
            json.dumps({**self.advice, "confidence": "confirmed"}),
            json.dumps({**self.advice, "summary": 1}),
            json.dumps({**self.advice, "limitations": "none"}),
            json.dumps({**self.advice, "suggested_checks": [1]}),
            json.dumps({**self.advice, "summary": "x" * 4001}),
            json.dumps({**self.advice, "limitations": ["x"] * 21}),
        ]:
            with self.subTest(payload=payload[:80]), self.assertRaises((ValueError, TypeError)):
                parse(payload)

    def test_structured_request_cannot_mutate_records(self):
        namespace = run_advice([])
        client = SimpleNamespace(models=Mock())
        client.models.generate_content.return_value.text = json.dumps(self.advice)
        records = [{"Assignment Status": "UNASSIGNED", "Confidence": "Unknown"}]
        original = copy.deepcopy(records)
        result = namespace["request_gemini_advice"](client, "chosen-model", records)
        self.assertEqual(result, self.advice)
        self.assertEqual(records, original)
        arguments = client.models.generate_content.call_args.kwargs
        self.assertEqual(arguments["model"], "chosen-model")
        self.assertEqual(json.loads(arguments["contents"]), original)
        self.assertEqual(arguments["config"]["response_mime_type"], "application/json")
        self.assertFalse(arguments["config"]["response_json_schema"]["additionalProperties"])

    def test_missing_sdk_or_blank_key_is_nonfatal(self):
        missing = run_advice([], enabled=True, key="not-a-real-key")
        self.assertEqual(missing["gemini_advice_status"], "UNAVAILABLE")
        self.assertIsNone(missing["gemini_advice"])
        blank = run_advice([], enabled=True, client=SimpleNamespace())
        self.assertEqual(blank["gemini_advice_status"], "SKIPPED")

    def test_successful_advice_stays_separate_from_assignments(self):
        client = Mock()
        client.__enter__ = Mock(return_value=client)
        client.__exit__ = Mock(return_value=False)
        client.models.generate_content.return_value.text = json.dumps(self.advice)
        records = [{"Assignment Status": "AMBIGUOUS", "Confidence": "Ambiguous MS1 candidates"}]
        namespace = run_advice(
            records, enabled=True, client=SimpleNamespace(Client=Mock(return_value=client)),
            key="not-a-real-key",
        )
        self.assertEqual(namespace["gemini_advice_status"], "ADVISORY ONLY")
        self.assertEqual(namespace["gemini_advice"], self.advice)
        self.assertEqual(json.loads(namespace["assigned_df"].to_json()), records)

    def test_service_failure_and_invalid_output_preserve_assignments(self):
        for result in [RuntimeError("service unavailable"), SimpleNamespace(text='{"confidence":"confirmed"}')]:
            with self.subTest(result=result):
                client = Mock()
                client.__enter__ = Mock(return_value=client)
                client.__exit__ = Mock(return_value=False)
                if isinstance(result, Exception):
                    client.models.generate_content.side_effect = result
                else:
                    client.models.generate_content.return_value = result
                records = [{"Assignment Status": "UNASSIGNED", "Confidence": "Unknown"}]
                namespace = run_advice(
                    records, enabled=True, client=SimpleNamespace(Client=Mock(return_value=client)),
                    key="not-a-real-key",
                )
                self.assertEqual(namespace["gemini_advice_status"], "UNAVAILABLE")
                self.assertIsNone(namespace["gemini_advice"])
                self.assertEqual(json.loads(namespace["assigned_df"].to_json()), records)

    def test_report_separates_and_escapes_advice(self):
        for advice in [None, {**self.advice, "summary": "<script>alert(1)</script>"}]:
            with self.subTest(advice=advice):
                namespace = {"gemini_advice": advice}
                display = SimpleNamespace(HTML=lambda text: text, display=lambda _: None)
                with patch.dict("sys.modules", {"IPython": SimpleNamespace(), "IPython.display": display}):
                    with patch("builtins.open", mock_open()), redirect_stdout(io.StringIO()):
                        exec(compile(source(8), str(NOTEBOOK), "exec"), namespace)
                report = namespace["html_content"]
                self.assertNotIn("<script>", report)
                if advice:
                    self.assertIn("&lt;script&gt;", report)
                    self.assertIn("Not Evidence or Confidence", report)
                else:
                    self.assertNotIn("Optional Gemini Advice", report)

    def test_changed_cells_compile(self):
        for index in (5, 6, 7, 8):
            with self.subTest(cell=index):
                compile(source(index), str(NOTEBOOK), "exec")


if __name__ == "__main__":
    unittest.main()
