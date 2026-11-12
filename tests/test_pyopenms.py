import ast
import html
import importlib.util
import json
import math
import re
import sqlite3
import tempfile
import threading
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


NOTEBOOK = Path(__file__).resolve().parents[1] / "pyOpenMS.ipynb"
CELLS = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
STEP3 = next("".join(cell["source"]) for cell in CELLS
             if cell["cell_type"] == "code" and
             "".join(cell["source"]).startswith("#@title Step 3"))
TREE = ast.parse(STEP3)


def load_functions(*names, namespace=None):
    namespace = namespace or {}
    namespace.update(json=json, html=html, math=math, re=re, sqlite3=sqlite3, uuid=uuid)
    for node in TREE.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in (
                "SPECTRAL_STATE", "BENCH_CONTAMINANTS", "NEUTRAL_LOSSES")
            for target in node.targets
        ):
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    functions = [node for node in TREE.body if isinstance(node, ast.FunctionDef)
                 and node.name in names]
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    return namespace


class NotebookStructureTests(unittest.TestCase):
    def test_each_code_cell_has_bulleted_explanation(self):
        for i, cell in enumerate(CELLS):
            if cell["cell_type"] == "code":
                self.assertGreater(i, 0)
                previous = CELLS[i - 1]
                self.assertEqual(previous["cell_type"], "markdown")
                self.assertIn("- ", "".join(previous["source"]))

    def test_minimal_tool_does_not_return_arrays(self):
        functions = {n.name: n for n in TREE.body if isinstance(n, ast.FunctionDef)}
        parameters = [arg.arg for arg in functions["update_lcms_plot"].args.args]
        self.assertEqual(parameters, ["rt_window", "extracted_ion_mz", "ppm_tolerance",
                                      "select_scan_rt", "overlay_formula_isotopes",
                                      "peak_annotations"])
        self.assertNotIn("raw_ms1_top_25", STEP3)


class OfflineChemistryTests(unittest.TestCase):
    def test_reaction_roles_and_deltas(self):
        ns = load_functions("parse_reaction_candidates", "reaction_mass_hypotheses",
                            namespace={"calculate_expected_ions": lambda smi: {
                                "exact_mass": {"SM": 100.0, "R": 20.0, "P": 110.0}[smi],
                                "positive": {"[M+H]+": {"SM": 101.0073, "R": 21.0073,
                                                       "P": 111.0073}[smi]},
                                "formula": "C2H5Br" if smi == "P" else "C2H6O",
                                "cLogP": 1.5, "TPSA": 20}})
        reaction = ns["parse_reaction_candidates"]("SM>R>P")
        self.assertEqual([x["role"] for x in reaction],
                         ["Starting Material", "Reagent", "Target Product"])
        self.assertEqual(ns["parse_reaction_candidates"]("SM, R, P"), reaction)
        shifts = ns["reaction_mass_hypotheses"](reaction)
        self.assertEqual(shifts[0]["name"], "Unreacted SM")
        self.assertAlmostEqual(shifts[0]["delta"], -10)
        self.assertAlmostEqual(next(x for x in shifts if x["name"] == "Des-Br")["delta"],
                               -77.9105)
        self.assertIsNone(next(x for x in shifts if x["name"] == "Hydrolysis")["TPSA"])

    def test_contaminants_losses_and_compact_context(self):
        ns = load_functions("_flag_artifacts", "annotate_neutral_losses",
                            "compact_peak_context")
        ions = [{"mz": m, "intensity": intensity} for m, intensity in
                [(279.093, 100), (323.119, 90), (200, 80), (181.99, 70)]]
        self.assertIn("TPPO", ns["_flag_artifacts"](ions))
        self.assertIn("PEG-400", ns["_flag_artifacts"](ions))
        self.assertIn("H2O", [x["loss"] for x in ns["annotate_neutral_losses"](ions)])
        ns["SPECTRAL_STATE"]["peaks"] = [{
            "rt_min": float(i), "ions": ions, "artifacts": ["TPPO"],
            "neutral_losses": [], "candidates": [{"formula": "C2H6O",
                                                   "cLogP": 0.1, "TPSA": 20}],
        } for i in range(6)]
        compact = ns["compact_peak_context"]()
        self.assertEqual(len(json.loads(compact)), 6)
        self.assertEqual(json.loads(compact)[0]["r"], [[0.1, 20]])
        self.assertLess(len(compact), 950)

    def test_disk_cache_reuses_success_and_not_failures(self):
        ns = load_functions("_cached_chemical_lookup")
        with tempfile.TemporaryDirectory() as folder:
            ns["CHEM_CACHE_PATH"] = str(Path(folder) / "lookups.sqlite")
            calls = []

            def fetch():
                calls.append(1)
                return {"pubchem_hits": [{"id": 12}]}

            first = ns["_cached_chemical_lookup"]("pubchem", {"formula": "CO"}, fetch)
            self.assertEqual(first, ns["_cached_chemical_lookup"]("pubchem", {"formula": "CO"},
                                                                  fetch))
            self.assertEqual(len(calls), 1)
            for _ in range(2):
                ns["_cached_chemical_lookup"]("chembl", {"formula": "CO"},
                                              lambda: calls.append(1) or {"error": "offline"})
            self.assertEqual(len(calls), 3)

    def test_external_lookup_runs_independent_requests_in_parallel(self):
        barrier = threading.Barrier(3)

        def lookup(*args):
            barrier.wait(timeout=2)
            return {"hits": []}

        ns = load_functions("search_chemical_databases", namespace={
            "ThreadPoolExecutor": ThreadPoolExecutor,
            "search_pubchem_by_mass_or_formula": lookup,
            "search_chembl_metabolites_and_isomers": lookup,
            "query_massbank_reference_spectrum": lookup,
        })
        self.assertEqual(set(ns["search_chemical_databases"]("C2H6O", "CCO", 47.049).keys()),
                         {"pubchem", "chembl", "massbank"})

    def test_uv_to_ms_delay(self):
        ns = load_functions("match_uv_apex_to_ms_scan")
        ns["SPECTRAL_STATE"]["scans"] = [(1.00,), (1.08,), (1.2,)]
        self.assertEqual(ns["match_uv_apex_to_ms_scan"](1.00), 1.08)
        ns["SPECTRAL_STATE"]["scans"] = []
        self.assertIsNone(ns["match_uv_apex_to_ms_scan"](1.00))

    def test_chat_prunes_tool_history_and_routes_complex_turns(self):
        types = SimpleNamespace(**{
            name: lambda **fields: SimpleNamespace(**fields)
            for name in ("Content", "Part", "GenerateContentConfig",
                         "ThinkingConfig", "AutomaticFunctionCallingConfig")
        })
        client = SimpleNamespace(chats=SimpleNamespace(create=Mock(return_value=object())))
        selection = SimpleNamespace(value="gemini-2.5-flash")
        ns = load_functions("init_chat_session", namespace={
            "client": client, "types": types, "model_dropdown": selection,
            "reaction_input": SimpleNamespace(value=""),
            "uv_ms_delay_input": SimpleNamespace(value=.08),
            "rdkit_tools": [], "build_system_instruction": lambda: "compact instruction",
            "chat_history_ui": [("user", "u" * 800), ("model", "tool summary"),
                                ("user", "question"), ("model", "answer"),
                                ("user", "recent question")],
        })
        ns["SPECTRAL_STATE"].update(data_id=id(None), reaction_text="")
        ns["init_chat_session"]()
        kwargs = client.chats.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "gemini-2.5-flash")
        self.assertEqual(len(kwargs["history"]), 4)
        self.assertEqual(kwargs["history"][0].parts[0].text, "tool summary")
        self.assertEqual(kwargs["config"].thinking_config.thinking_budget, 0)
        self.assertEqual(kwargs["config"].automatic_function_calling.maximum_remote_calls, 6)
        selection.value = "gemini-2.5-pro"
        ns["init_chat_session"]()
        self.assertEqual(client.chats.create.call_args.kwargs["config"].thinking_config.thinking_budget,
                         1024)


@unittest.skipUnless(all(importlib.util.find_spec(p) for p in
                         ("numpy", "scipy", "pandas", "rdkit", "plotly")),
                     "Notebook scientific dependencies are not installed")
class ScientificTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        import pandas as pd
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        from rdkit import Chem
        from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
        from scipy.integrate import trapezoid
        from scipy.signal import find_peaks, peak_widths

        cls.np, cls.pd = np, pd
        cls.ns = load_functions(
            "safe_mol_from_smiles", "calculate_expected_ions", "parse_reaction_candidates",
            "reaction_mass_hypotheses", "theoretical_isotope_envelope",
            "annotate_neutral_losses", "_centroid_scans", "_flag_artifacts",
            "prepare_spectral_state", "register_uv_trace", "match_uv_apex_to_ms_scan",
            "_xic", "update_lcms_plot", "compact_peak_context",
            namespace=dict(Chem=Chem, Crippen=Crippen, Descriptors=Descriptors,
                           rdMolDescriptors=rdMolDescriptors, np=np, pd=pd, go=go,
                           make_subplots=make_subplots, find_peaks=find_peaks,
                           peak_widths=peak_widths, trapezoid=trapezoid, display=None))

    def setUp(self):
        self.ns["SPECTRAL_STATE"].clear()
        self.ns.pop("od", None)
        self.ns.pop("hplc_df", None)
        self.ns.pop("reaction_input", None)

    def test_adducts_reaction_and_isotope_ratios(self):
        ions = self.ns["calculate_expected_ions"]("CCBr")
        self.assertAlmostEqual(ions["positive"]["[2M+Na]+"],
                               2 * ions["exact_mass"] + 22.9892, places=3)
        self.assertAlmostEqual(ions["negative"]["[M+Cl]-"],
                               ions["exact_mass"] + 34.9694, places=3)
        self.assertIn("error", self.ns["calculate_expected_ions"]("not a smiles"))
        reaction = self.ns["parse_reaction_candidates"]("CCO.O>>CCBr")
        self.assertEqual([c["role"] for c in reaction],
                         ["Starting Material", "Starting Material", "Target Product"])
        des_br = next(x for x in self.ns["reaction_mass_hypotheses"](reaction)
                      if x["name"] == "Des-Br")
        self.assertIsNotNone(des_br["cLogP"])
        self.assertEqual(des_br["smiles"], "CC")
        for formula, ratio in (("C2H5Cl", .32), ("C2H5Br", .97)):
            envelope = {x["label"]: x for x in
                        self.ns["theoretical_isotope_envelope"](formula)}
            self.assertAlmostEqual(envelope["M+2"]["relative_intensity"] /
                                   envelope["M"]["relative_intensity"], ratio, delta=.06)
            self.assertGreater(envelope["M+1"]["relative_intensity"], 0)
        self.assertIn("M+4", [x["label"] for x in
                              self.ns["theoretical_isotope_envelope"]("CCl2")])

    def test_baseline_pick_uv_and_plot_without_model_arrays(self):
        np = self.np
        class Spectrum:
            def __init__(self, rt, mzs, ints):
                self.rt, self.mzs, self.ints = rt, np.array(mzs), np.array(ints)

            def getMSLevel(self):
                return 1

            def getRT(self):
                return self.rt

            def get_peaks(self):
                return self.mzs, self.ints

        scans = []
        for i in range(40):
            peak = max(0, 1 - abs(i - 15) / 4) * 10000
            scans.append(Spectrum((1 + i * .1) * 60,
                                  [59.05123, 123.092, 200, 181.99, 250.5],
                                  [1000 + peak, peak / 2, peak / 3, peak / 4, 5000]))
        self.ns["od"] = scans
        self.ns["reaction_input"] = SimpleNamespace(value="CCO>>CCBr")
        state = self.ns["prepare_spectral_state"]()
        self.assertEqual(len(state["peaks"]), 1)
        self.assertEqual(state["peaks"][0]["artifacts"], ["DMAP"])
        self.assertLessEqual(len(state["peaks"][0]["ions"]), 10)
        self.assertNotIn(250.5, [i["mz"] for i in state["peaks"][0]["ions"]])
        self.assertIn(59.0512, [i["mz"] for i in state["peaks"][0]["ions"]])
        self.assertLess(len(self.ns["compact_peak_context"]()), 950)
        channels = self.ns["register_uv_trace"](
            self.pd.DataFrame({"Time": np.array([s.rt / 60 for s in scans]) - .08,
                               "254 nm": [max(0, 1 - abs(i - 15) / 4) for i in range(40)]}))
        self.assertEqual(channels, ["254 nm"])
        self.assertAlmostEqual(self.ns["match_uv_apex_to_ms_scan"](2.42), 2.5)
        result = self.ns["update_lcms_plot"](extracted_ion_mz=[59.05123],
            select_scan_rt=2.5, overlay_formula_isotopes="C2H5Br")
        self.assertEqual(result, "LCMS dashboard updated locally; the plot is shown in the notebook.")
        self.assertNotIn("59.05123", result)
        figure = state["figure"]
        self.assertEqual(len(figure._grid_ref), 3)
        self.assertTrue(any("%" in str(a.text) for a in figure.layout.annotations))
        self.assertTrue(any("Theoretical" in str(t.name) for t in figure.data))
        self.assertTrue(any("XIC" in str(t.name) for t in figure.data))
        state.pop("uv")
        self.assertIn("updated locally", self.ns["update_lcms_plot"]())
        self.assertTrue(any("UV/DAD not supplied" in str(a.text)
                            for a in state["figure"].layout.annotations))


if __name__ == "__main__":
    unittest.main()
