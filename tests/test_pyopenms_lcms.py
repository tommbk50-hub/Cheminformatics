"""Offline regression tests for the Step 3 LC-MS agent cell of ``pyOpenMS.ipynb``.

The notebook cannot be imported (it needs RDKit, pyOpenMS, Plotly, ipywidgets and the
Gemini SDK), so — exactly like ``tests/test_hplc_ai.py`` — each test extracts the
deterministic helpers from the notebook source with :mod:`ast` and executes only those
definitions inside a stub namespace.  Everything here is stdlib-only and fully offline:
no network calls, no temporary directories and no optional third-party packages.
"""

import ast
import functools
import hashlib
import json
import math
import os
import re
import sqlite3
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "pyOpenMS.ipynb"


def _load_notebook():
    with NOTEBOOK_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


NOTEBOOK = _load_notebook()


def code_cell_starting_with(prefix):
    """Returns the full source of the first code cell whose first line starts with ``prefix``."""
    for cell in NOTEBOOK["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if source.startswith(prefix):
            return source
    raise AssertionError(f"No code cell starting with {prefix!r} found in {NOTEBOOK_PATH}")


STEP3_SOURCE = code_cell_starting_with("#@title Step 3 (Complete Self-Contained Script)")
STEP3_TREE = ast.parse(STEP3_SOURCE)

BASE_NAMESPACE = {
    "ast": ast,
    "functools": functools,
    "hashlib": hashlib,
    "json": json,
    "math": math,
    "os": os,
    "re": re,
    "sqlite3": sqlite3,
    "threading": threading,
    "time": time,
    "_scipy_signal": None,
    "_markdown_lib": None,
}


def load_symbols(names, extra_globals=None):
    """Executes only the requested top-level definitions/assignments of the Step 3 cell."""
    wanted = []
    for node in STEP3_TREE.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            wanted.append(node)
        elif isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if any(target in names for target in targets):
                wanted.append(node)
    found = set()
    for node in wanted:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            found.add(node.name)
        else:
            found.update(t.id for t in node.targets if isinstance(t, ast.Name))
    missing = set(names) - found
    if missing:
        raise AssertionError(f"Step 3 cell is missing definitions: {sorted(missing)}")

    namespace = dict(BASE_NAMESPACE)
    namespace["__name__"] = "step3_under_test"
    namespace.update(extra_globals or {})
    module = ast.fix_missing_locations(ast.Module(body=wanted, type_ignores=[]))
    exec(compile(module, str(NOTEBOOK_PATH), "exec"), namespace)
    return namespace


def function_node(name):
    for node in STEP3_TREE.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name:
            return node
    raise AssertionError(f"{name} is not defined at top level of the Step 3 cell")


class NotebookStructureTests(unittest.TestCase):
    """The notebook must stay a valid, Colab-runnable document."""

    def test_every_code_cell_compiles(self):
        for index, cell in enumerate(NOTEBOOK["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            if source.lstrip().startswith("!") or "\n!" in source:
                continue  # shell-magic install cells are not valid Python
            with self.subTest(cell=index):
                compile(source, f"cell_{index}", "exec")

    def test_every_code_cell_has_a_bullet_point_markdown_cell_above_it(self):
        cells = NOTEBOOK["cells"]
        for index, cell in enumerate(cells):
            if cell["cell_type"] != "code":
                continue
            with self.subTest(cell=index):
                self.assertGreater(index, 0, "a code cell may not be the first cell")
                previous = cells[index - 1]
                self.assertEqual(previous["cell_type"], "markdown")
                text = "".join(previous["source"])
                bullets = [line for line in text.splitlines() if line.strip().startswith("- ")]
                self.assertGreaterEqual(len(bullets), 2, f"expected bullet points, got:\n{text}")

    def test_step3_preserves_the_self_contained_colab_workflow(self):
        for required in (
            "build_interactive_chromatogram_html",
            "generate_in_silico_fragments",
            "assign_peaks_and_replot_chromatogram",
            "disambiguate_isobaric_peak_isomers",
            "get_candidates_from_table",
            "_notebook_figure_registry",
            "widgets.VBox",
            "display(ui)",
        ):
            self.assertIn(required, STEP3_SOURCE)

    def test_vision_qc_cell_still_present(self):
        vision = code_cell_starting_with("#@title Step 4: Multimodal Vision QC")
        self.assertIn("_notebook_figure_registry", vision)
        self.assertIn("types.Part.from_bytes", vision)

    def test_update_lcms_plot_signature_matches_the_specification(self):
        node = function_node("update_lcms_plot")
        self.assertEqual(
            [arg.arg for arg in node.args.args],
            ["rt_window", "extracted_ion_mz", "ppm_tolerance", "select_scan_rt",
             "overlay_formula_isotopes", "peak_annotations"],
        )
        self.assertEqual(ast.unparse(node.returns), "str")
        self.assertEqual(ast.unparse(node.args.annotations[0] if False else node.args.args[0].annotation),
                         "list[float] | None")

    def test_new_tools_are_registered_with_gemini(self):
        tools_assignment = next(
            node for node in STEP3_TREE.body
            if isinstance(node, ast.Assign)
            and any(getattr(t, "id", "") == "rdkit_tools" for t in node.targets)
        )
        registered = {element.id for element in tools_assignment.value.elts
                      if isinstance(element, ast.Name)}
        for tool in (
            "update_lcms_plot",
            "get_distilled_peak_summary",
            "analyze_reaction_mixture",
            "annotate_peak_neutral_losses",
            "check_elution_order_consistency",
            "screen_bench_contaminants",
            "theoretical_isotope_envelope",
            "calculate_expected_ions",
            "lookup_compound_everywhere",
        ):
            with self.subTest(tool=tool):
                self.assertIn(tool, registered)


class MarkdownRenderingTests(unittest.TestCase):
    """Part 1 — Markdown is rendered as styled HTML, and model HTML is sanitized."""

    def setUp(self):
        self.ns = load_symbols([
            "MARKDOWN_CONTAINER_STYLE", "_MARKDOWN_STYLE_BLOCK", "_UNSAFE_TAG_RE",
            "_EVENT_ATTR_RE", "_JS_URL_RE", "sanitize_model_html",
            "_fallback_markdown_to_html", "markdown_to_styled_html",
        ])

    def test_container_style_uses_the_requested_typography(self):
        style = self.ns["MARKDOWN_CONTAINER_STYLE"]
        self.assertIn("line-height: 1.6", style)
        self.assertIn("font-size: 15px", style)
        self.assertIn("padding: 14px", style)

    def test_style_block_covers_tables_and_code_blocks(self):
        css = self.ns["_MARKDOWN_STYLE_BLOCK"]
        self.assertIn(".gemini-md table", css)
        self.assertIn("border-collapse: collapse", css)
        self.assertIn(".gemini-md pre", css)
        self.assertIn(".gemini-md code", css)

    def test_headings_lists_tables_and_code_are_converted(self):
        html = self.ns["markdown_to_styled_html"](
            "## Peak 3\n\n- **Product** confirmed\n- `m/z 418.1234`\n\n"
            "| Peak | m/z |\n| --- | --- |\n| 3 | 418.1 |\n\n```python\nx = 1\n```\n"
        )
        self.assertIn("<h2>", html)
        self.assertIn("<li>", html)
        self.assertIn("<strong>", html)
        self.assertIn("<code>", html)
        self.assertIn("<table>", html)
        self.assertIn("<pre>", html)
        self.assertIn('class="gemini-md"', html)
        self.assertIn(self.ns["MARKDOWN_CONTAINER_STYLE"], html)

    def test_model_authored_html_is_sanitized(self):
        dirty = (
            '<p onclick="steal()">hi</p><script>fetch("http://evil")</script>'
            '<iframe src="http://evil"></iframe><a href="javascript:alert(1)">x</a>'
        )
        clean = self.ns["sanitize_model_html"](dirty)
        self.assertNotIn("<script", clean.lower())
        self.assertNotIn("<iframe", clean.lower())
        self.assertNotIn("onclick", clean.lower())
        self.assertNotIn("javascript:", clean.lower())
        self.assertIn("hi", clean)


class AdductAndContaminantTests(unittest.TestCase):
    """Feature 1 — deterministic adduct/dimer matrices and bench-contaminant screening."""

    def setUp(self):
        self.ns = load_symbols([
            "PROTON_MASS", "POSITIVE_ADDUCTS", "NEGATIVE_ADDUCTS", "BENCH_CONTAMINANTS",
            "PEG_REPEAT_DA", "PEG_ADDUCT_OFFSETS", "expected_ion_table",
            "match_bench_contaminants", "detect_peg_series", "flag_contaminants",
        ])

    def test_expected_ion_table_contains_the_required_adducts(self):
        table = self.ns["expected_ion_table"](100.0)
        positive = {row["adduct"]: row["mz"] for row in table["positive_mode"]}
        negative = {row["adduct"]: row["mz"] for row in table["negative_mode"]}
        self.assertAlmostEqual(positive["[M+H]+"], 101.0073, places=3)
        self.assertAlmostEqual(positive["[M+Na]+"], 122.9892, places=3)
        self.assertAlmostEqual(positive["[M+K]+"], 138.9632, places=3)
        self.assertAlmostEqual(positive["[M+NH4]+"], 118.0338, places=3)
        self.assertAlmostEqual(positive["[2M+H]+"], 201.0073, places=3)
        self.assertAlmostEqual(positive["[M+2H]2+"], 51.0073, places=3)
        self.assertAlmostEqual(negative["[M-H]-"], 98.9927, places=3)
        self.assertAlmostEqual(negative["[M+HCOO]-"], 144.9986, places=3)
        self.assertAlmostEqual(negative["[2M-H]-"], 198.9927, places=3)

    def test_triphenylphosphine_oxide_and_amine_bases_are_flagged(self):
        hits = self.ns["match_bench_contaminants"]([(279.0938, 100.0), (130.1590, 12.0)])
        names = {hit["contaminant"] for hit in hits}
        self.assertIn("TPPO (triphenylphosphine oxide)", names)
        self.assertIn("DIPEA (Hunig's base)", names)
        self.assertEqual(hits[0]["rel_intensity_pct"], 100.0)
        self.assertLess(abs(hits[0]["delta_mDa"]), 20.0)

    def test_unrelated_masses_are_not_flagged(self):
        self.assertEqual(self.ns["match_bench_contaminants"]([512.3331, 613.1]), [])

    def test_peg_ladder_detection(self):
        ladder = [327.2014, 371.2276, 415.2538, 459.2800]
        series = self.ns["detect_peg_series"](ladder)
        self.assertEqual(len(series), 1)
        self.assertEqual(series[0]["length"], 4)
        self.assertIn("44.026", series[0]["series"])
        self.assertEqual(self.ns["detect_peg_series"]([327.2014, 371.2276]), [])

    def test_flag_contaminants_combines_both_screens(self):
        flags = self.ns["flag_contaminants"]([279.0938, 327.2014, 371.2276, 415.2538])
        self.assertTrue(flags["bench_artifacts"])
        self.assertTrue(flags["polymer_series"])


class IsotopeEnvelopeTests(unittest.TestCase):
    """Feature 3 — natural-abundance isotope envelopes (Cl 3:1, Br 1:1, S ~4.4%)."""

    def setUp(self):
        self.ns = load_symbols([
            "PROTON_MASS", "ELEMENT_ISOTOPES", "_FORMULA_TOKEN_RE", "POSITIVE_ADDUCTS",
            "NEGATIVE_ADDUCTS", "parse_molecular_formula", "_convolve_isotope_dists",
            "isotope_pattern_from_formula", "theoretical_isotope_envelope",
        ], extra_globals={"safe_mol_from_smiles": lambda s: (None, s, "RDKit unavailable")})

    def test_formula_parsing(self):
        self.assertEqual(self.ns["parse_molecular_formula"]("C10H15N3O5"),
                         {"C": 10, "H": 15, "N": 3, "O": 5})
        self.assertEqual(self.ns["parse_molecular_formula"]("[C9H8ClNO2]+"),
                         {"C": 9, "H": 8, "Cl": 1, "N": 1, "O": 2})
        self.assertEqual(self.ns["parse_molecular_formula"]("C6H4(CH3)2"),
                         {"C": 8, "H": 10})
        with self.assertRaises(ValueError):
            self.ns["parse_molecular_formula"]("Xx2")

    def test_monoisotopic_mz_is_accurate(self):
        peaks = self.ns["isotope_pattern_from_formula"]("C6H6")
        self.assertAlmostEqual(peaks[0]["mz"], 79.0542, places=3)
        self.assertEqual(peaks[0]["isotope"], "M")
        self.assertEqual(peaks[0]["rel_intensity"], 100.0)

    def test_chlorine_gives_a_three_to_one_m_plus_two(self):
        peaks = {p["isotope"]: p["rel_intensity"] for p in
                 self.ns["isotope_pattern_from_formula"]("C6H5Cl")}
        self.assertAlmostEqual(peaks["M+2"], 32.0, delta=2.0)

    def test_bromine_gives_a_near_one_to_one_doublet(self):
        peaks = {p["isotope"]: p["rel_intensity"] for p in
                 self.ns["isotope_pattern_from_formula"]("C6H5Br")}
        self.assertAlmostEqual(peaks["M+2"], 97.3, delta=3.0)

    def test_sulfur_m_plus_two_is_about_four_percent(self):
        peaks = {p["isotope"]: p["rel_intensity"] for p in
                 self.ns["isotope_pattern_from_formula"]("C2H6S")}
        self.assertAlmostEqual(peaks["M+2"], 4.5, delta=1.0)

    def test_carbon_count_drives_the_m_plus_one_intensity(self):
        peaks = {p["isotope"]: p["rel_intensity"] for p in
                 self.ns["isotope_pattern_from_formula"]("C20H42")}
        self.assertAlmostEqual(peaks["M+1"], 20 * 1.08, delta=1.5)

    def test_envelope_tool_reports_halogen_diagnostics(self):
        result = self.ns["theoretical_isotope_envelope"]("C9H8ClNO2")
        self.assertEqual(result["formula"], "C9H8ClNO2")
        self.assertTrue(any("Cl" in note for note in result["halogen_diagnostic"]))
        self.assertAlmostEqual(result["isotope_peaks"][0]["mz"], 198.0316, places=2)

    def test_dimer_adducts_are_rejected_for_isotope_overlays(self):
        self.assertIn("error", self.ns["theoretical_isotope_envelope"]("C6H6", adduct="[2M+H]+"))


class NeutralLossTests(unittest.TestCase):
    """Feature 5 — in-source fragmentation / neutral-loss annotation."""

    def setUp(self):
        self.ns = load_symbols(["NEUTRAL_LOSS_LIBRARY", "detect_neutral_losses"])

    def test_water_and_boc_losses_are_annotated(self):
        ions = [{"mz": 418.1234, "rel": 100.0},
                {"mz": 400.1128, "rel": 45.0},
                {"mz": 318.0710, "rel": 20.0}]
        losses = {a["neutral_loss"] for a in self.ns["detect_neutral_losses"](ions)}
        self.assertIn("H2O", losses)
        self.assertIn("Boc (-CO2 -isobutene)", losses)

    def test_tolerance_rejects_near_misses(self):
        ions = [(418.1234, 100.0), (400.0734, 40.0)]  # delta 18.05 Da — not water
        self.assertEqual(self.ns["detect_neutral_losses"](ions), [])

    def test_results_are_sorted_by_precursor_intensity(self):
        ions = [(300.0, 10.0), (281.9895, 5.0), (500.0, 100.0), (481.9895, 60.0)]
        annotations = self.ns["detect_neutral_losses"](ions)
        self.assertTrue(annotations)
        intensities = [a["precursor_rel_intensity"] for a in annotations]
        self.assertEqual(intensities, sorted(intensities, reverse=True))


class PolarityElutionOrderTests(unittest.TestCase):
    """Feature 6 — cLogP / TPSA versus retention-time consistency on reverse phase."""

    def setUp(self):
        self.ns = load_symbols(["_rank", "check_rt_polarity_consistency"])

    def test_monotonic_data_is_consistent(self):
        entries = [{"name": "A", "cLogP": -1.2, "rt": 1.0},
                   {"name": "B", "cLogP": 0.8, "rt": 2.5},
                   {"name": "C", "cLogP": 2.4, "rt": 4.0},
                   {"name": "D", "cLogP": 4.1, "rt": 6.2}]
        result = self.ns["check_rt_polarity_consistency"](entries)
        self.assertEqual(result["verdict"], "consistent")
        self.assertAlmostEqual(result["spearman_rho"], 1.0, places=3)
        self.assertEqual(result["outliers"], [])

    def test_reversed_order_is_flagged_with_outliers(self):
        entries = [{"name": "A", "cLogP": 4.1, "rt": 1.0},
                   {"name": "B", "cLogP": 2.4, "rt": 2.5},
                   {"name": "C", "cLogP": 0.8, "rt": 4.0},
                   {"name": "D", "cLogP": -1.2, "rt": 6.2}]
        result = self.ns["check_rt_polarity_consistency"](entries)
        self.assertEqual(result["verdict"], "inconsistent")
        self.assertAlmostEqual(result["spearman_rho"], -1.0, places=3)
        self.assertTrue(result["outliers"])

    def test_insufficient_data(self):
        result = self.ns["check_rt_polarity_consistency"]([{"name": "A", "cLogP": 1.0, "rt": 1.0}])
        self.assertEqual(result["verdict"], "insufficient_data")
        self.assertIsNone(result["spearman_rho"])


class ReactionMonitoringTests(unittest.TestCase):
    """Feature 2 — reaction parsing and deterministic failure-mode masses."""

    def setUp(self):
        self.ns = load_symbols([
            "_ROLE_ALIASES", "_ROLE_LABEL_RE", "REACTION_FAILURE_MODES",
            "REACTION_MARKER_COLORS", "_match_role_label", "parse_reaction_input",
            "reaction_failure_mode_masses",
        ])

    def test_reaction_smiles_with_two_starting_materials(self):
        parsed = self.ns["parse_reaction_input"]("CCO.ClC(=O)C>>CCOC(C)=O")
        self.assertEqual(parsed["input_type"], "reaction_smiles")
        self.assertEqual(len(parsed["starting_materials"]), 2)
        self.assertEqual(len(parsed["products"]), 1)
        self.assertEqual(parsed["products"][0]["smiles"], "CCOC(C)=O")

    def test_three_part_reaction_smiles_captures_reagents(self):
        parsed = self.ns["parse_reaction_input"]("CCO>CC(=O)Cl>CCOC(C)=O")
        self.assertEqual([r["smiles"] for r in parsed["reagents"]], ["CC(=O)Cl"])
        self.assertEqual([p["smiles"] for p in parsed["products"]], ["CCOC(C)=O"])

    def test_labelled_component_list(self):
        parsed = self.ns["parse_reaction_input"]("SM: CCO, reagent: ClC(=O)C, product: CCOC(C)=O")
        self.assertEqual(parsed["input_type"], "component_list")
        self.assertEqual(parsed["starting_materials"][0]["smiles"], "CCO")
        self.assertEqual(parsed["reagents"][0]["smiles"], "ClC(=O)C")
        self.assertEqual(parsed["products"][0]["smiles"], "CCOC(C)=O")

    def test_single_smiles_is_treated_as_the_product(self):
        parsed = self.ns["parse_reaction_input"]("CC(=O)Oc1ccccc1C(=O)O")
        self.assertEqual(parsed["input_type"], "single_smiles")
        self.assertEqual(len(parsed["products"]), 1)
        self.assertEqual(self.ns["parse_reaction_input"]("")["input_type"], "empty")

    def test_failure_modes_cover_hydrolysis_and_deprotection(self):
        rows = self.ns["reaction_failure_mode_masses"](350.1, label="Product", role="product")
        modes = {row["failure_mode"]: row for row in rows}
        self.assertIn("Hydrolysis (+H2O)", modes)
        self.assertIn("Boc cleavage (-Boc + H)", modes)
        self.assertIn("Protodeboronation (-B(OH)2 + H)", modes)
        self.assertIn("Des-chloro (-Cl + H)", modes)
        self.assertAlmostEqual(modes["Hydrolysis (+H2O)"]["neutral_mass"], 368.1106, places=3)
        self.assertAlmostEqual(modes["Hydrolysis (+H2O)"]["[M+H]+"], 369.1179, places=3)
        self.assertAlmostEqual(modes["Hydrolysis (+H2O)"]["[M-H]-"], 367.1033, places=3)
        for row in rows:
            self.assertIn(row["color"], self.ns["REACTION_MARKER_COLORS"].values())

    def test_impossible_negative_masses_are_skipped(self):
        rows = self.ns["reaction_failure_mode_masses"](90.0)
        self.assertTrue(all(row["neutral_mass"] > 20.0 for row in rows))
        self.assertNotIn("Boc cleavage (-Boc + H)", {row["failure_mode"] for row in rows})


class PeakDistillationTests(unittest.TestCase):
    """Features 4 and 7 — client-side peak picking, background subtraction, distillation."""

    def setUp(self):
        self.ns = load_symbols([
            "UV_MS_DELAY_MIN", "DEFAULT_TOP_CHROM_PEAKS", "DEFAULT_MIN_REL_PCT",
            "DEFAULT_TOP_IONS", "LLM_TOP_IONS", "BACKGROUND_OFFSET_MIN",
            "APEX_SCANS_TO_AVERAGE", "SPECTRAL_STATE", "_rt_scale", "map_uv_rt_to_ms_rt",
            "relative_area_percent", "_window_stats", "_find_peaks_pure", "_peak_bounds",
            "find_chromatographic_peaks", "average_scans", "subtract_background",
            "top_centroids", "extract_xic", "_scan_to_dict", "_nearest_scan_index",
            "get_distilled_peak_summary", "distill_peak_summary",
        ])

    @staticmethod
    def _gaussian_trace():
        times, signal = [], []
        for i in range(400):
            t = i * 0.05
            value = (100.0 * math.exp(-((t - 5.0) ** 2) / 0.2)
                     + 40.0 * math.exp(-((t - 12.0) ** 2) / 0.3) + 1.0)
            times.append(t)
            signal.append(value)
        return times, signal

    def test_uv_to_ms_delay_default_matches_the_specification(self):
        self.assertAlmostEqual(self.ns["UV_MS_DELAY_MIN"], 0.08, places=6)
        self.assertAlmostEqual(self.ns["map_uv_rt_to_ms_rt"](300.0, rt_unit="s"), 304.8, places=4)
        self.assertAlmostEqual(self.ns["map_uv_rt_to_ms_rt"](5.0, rt_unit="min"), 5.08, places=4)
        self.assertAlmostEqual(self.ns["map_uv_rt_to_ms_rt"](5.0, delay_min=0.0, rt_unit="min"), 5.0)

    def test_relative_area_percent_normalises_to_one_hundred(self):
        percentages = self.ns["relative_area_percent"]([50.0, 25.0, 25.0])
        self.assertEqual(percentages, [50.0, 25.0, 25.0])
        self.assertAlmostEqual(sum(self.ns["relative_area_percent"]([3.0, 7.0, 11.0])), 100.0, places=1)
        self.assertEqual(self.ns["relative_area_percent"]([0.0, 0.0]), [0.0, 0.0])

    def test_peak_picker_finds_both_gaussians_without_scipy(self):
        times, signal = self._gaussian_trace()
        peaks = self.ns["find_chromatographic_peaks"](times, signal)
        self.assertEqual(len(peaks), 2)
        self.assertAlmostEqual(peaks[0]["rt"], 5.0, delta=0.1)
        self.assertAlmostEqual(peaks[1]["rt"], 12.0, delta=0.1)
        self.assertEqual([p["peak_id"] for p in peaks], [1, 2])
        self.assertAlmostEqual(sum(p["area_pct"] for p in peaks), 100.0, places=1)
        self.assertGreater(peaks[0]["area_pct"], peaks[1]["area_pct"])
        self.assertLess(peaks[0]["rt_start"], peaks[0]["rt"])
        self.assertGreater(peaks[0]["rt_end"], peaks[0]["rt"])

    def test_peak_picker_keeps_only_the_top_n(self):
        times, signal = self._gaussian_trace()
        self.assertEqual(len(self.ns["find_chromatographic_peaks"](times, signal, top_n=1)), 1)
        self.assertEqual(self.ns["find_chromatographic_peaks"]([], []), [])

    def test_apex_scan_averaging_and_background_subtraction(self):
        averaged = self.ns["average_scans"]([{100.0: 10.0, 200.0: 4.0},
                                             {100.0: 20.0, 200.0: 6.0},
                                             {100.0: 30.0}])
        self.assertAlmostEqual(averaged[100.0], 20.0)
        self.assertAlmostEqual(averaged[200.0], 10.0 / 3.0)
        cleaned = self.ns["subtract_background"](averaged, {200.0: 50.0})
        self.assertIn(100.0, cleaned)
        self.assertNotIn(200.0, cleaned, "column-bleed ions must be removed")

    def test_top_centroids_applies_the_noise_floor_and_cap(self):
        spectrum = {418.1: 1000.0, 400.1: 250.0, 318.1: 15.0}
        centroids = self.ns["top_centroids"](spectrum)
        self.assertEqual([c["mz"] for c in centroids], [418.1, 400.1])
        self.assertEqual(centroids[0]["rel"], 100.0)
        self.assertEqual(centroids[1]["rel"], 25.0)
        self.assertEqual(len(self.ns["top_centroids"]({float(i): 1000.0 - i for i in range(50)},
                                                      min_rel_pct=0.0, top_n=10)), 10)
        self.assertEqual(self.ns["top_centroids"]({}), [])

    def test_extracted_ion_chromatogram_respects_the_ppm_window(self):
        scans = [
            {"rt": 1.0, "mz": [417.0, 418.1234, 500.0], "int": [5.0, 100.0, 9.0]},
            {"rt": 2.0, "mz": [418.1300, 500.0], "int": [70.0, 9.0]},
            {"rt": 3.0, "mz": [419.5000], "int": [80.0]},
        ]
        times, intensities = self.ns["extract_xic"](scans, 418.1234, ppm_tolerance=10.0)
        self.assertEqual(times, [1.0, 2.0, 3.0])
        self.assertEqual(intensities, [100.0, 0.0, 0.0], "15.8 ppm neighbour must be excluded")
        _times, wide = self.ns["extract_xic"](scans, 418.1234, ppm_tolerance=20.0)
        self.assertEqual(wide[1], 70.0)

    def test_scan_binning_and_nearest_scan_lookup(self):
        binned = self.ns["_scan_to_dict"]({"mz": [418.12341, 418.12344, 500.0],
                                           "int": [10.0, 5.0, 2.0]})
        self.assertAlmostEqual(binned[418.123], 15.0)
        self.assertAlmostEqual(binned[500.0], 2.0)
        self.assertEqual(self.ns["_nearest_scan_index"](2.4, [1.0, 2.0, 3.0]), 1)

    def test_distilled_summary_is_compact_and_capped(self):
        self.ns["SPECTRAL_STATE"]["distilled"] = {
            "peaks": [{"peak_id": 1, "rt": 5.0, "area_pct": 62.0,
                       "ions": [{"mz": 400.0 + i, "rel": 100.0 - i} for i in range(10)]}],
            "n_scans": 1200,
        }
        summary = self.ns["get_distilled_peak_summary"](max_ions_per_peak=3)
        payload = json.loads(summary)
        self.assertEqual(len(payload["peaks"][0]["ions"]), 3)
        self.assertNotIn(", ", summary, "payload must use compact JSON separators")
        self.assertLess(len(summary) / 4, 300, "distilled summary must stay under ~300 tokens")
        self.assertEqual(len(self.ns["SPECTRAL_STATE"]["distilled"]["peaks"][0]["ions"]), 10)


class DatabaseCacheTests(unittest.TestCase):
    """Feature 9 — disk-cached, trimmed external database lookups."""

    def setUp(self):
        self.ns = load_symbols([
            "PROTON_MASS", "CHEM_CACHE_TTL_SECONDS", "MAX_HITS_RETURNED_TO_LLM",
            "ChemLookupCache", "disk_cached", "relevance_from_ppm", "trim_database_hit",
        ], extra_globals={"CHEM_CACHE_DIR": ":memory:"})
        self.cache = self.ns["ChemLookupCache"](path=":memory:")

    def test_cache_roundtrip_and_statistics(self):
        key = self.ns["ChemLookupCache"].make_key("pubchem", {"formula": "C10H15N3O5"})
        self.assertIsNone(self.cache.get(key))
        self.cache.set(key, {"hits": [1, 2, 3]})
        self.assertEqual(self.cache.get(key), {"hits": [1, 2, 3]})
        stats = self.cache.stats()
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 1)

    def test_cache_keys_are_order_independent_but_value_sensitive(self):
        make_key = self.ns["ChemLookupCache"].make_key
        self.assertEqual(make_key("pubchem", {"a": 1, "b": 2}), make_key("pubchem", {"b": 2, "a": 1}))
        self.assertNotEqual(make_key("pubchem", {"a": 1}), make_key("chembl", {"a": 1}))

    def test_expired_entries_are_treated_as_misses(self):
        cache = self.ns["ChemLookupCache"](path=":memory:", ttl_seconds=0.001)
        cache.set("k", {"v": 1})
        time.sleep(0.01)
        self.assertIsNone(cache.get("k"))

    def test_disk_cached_decorator_avoids_repeat_network_calls(self):
        calls = {"n": 0}

        @self.ns["disk_cached"]("pubchem", cache=self.cache)
        def fetch(formula):
            calls["n"] += 1
            return {"records": [{"id": 1, "formula": formula}]}

        first = fetch("C10H15N3O5")
        second = fetch("C10H15N3O5")
        third = fetch("C6H6")
        self.assertEqual(calls["n"], 2)
        self.assertNotIn("cache", first)
        self.assertEqual(second["cache"], "hit")
        self.assertEqual(second["records"], first["records"])
        self.assertNotIn("cache", third)

    def test_errors_are_never_cached(self):
        calls = {"n": 0}

        @self.ns["disk_cached"]("chembl", cache=self.cache)
        def failing():
            calls["n"] += 1
            return {"error": "timeout"}

        failing()
        failing()
        self.assertEqual(calls["n"], 2)

    def test_database_hits_are_trimmed_to_six_fields(self):
        verbose = {
            "id": "CID 135398635", "iupac_name": "x" * 200, "name": "Acyclovir",
            "formula": "C8H11N5O3", "exact_mass": 225.0862, "smiles": "C" * 400,
            "synonyms": ["a"] * 50, "xlogp": -1.6, "inchi": "InChI=1S/...",
        }
        trimmed = self.ns["trim_database_hit"](verbose, observed_mh_mz=226.0935)
        self.assertEqual(set(trimmed), {"id", "iupac_name", "formula", "exact_mass",
                                        "canonical_smiles", "relevance_score"})
        self.assertLessEqual(len(trimmed["iupac_name"]), 90)
        self.assertLessEqual(len(trimmed["canonical_smiles"]), 160)
        self.assertGreater(trimmed["relevance_score"], 0.9)
        self.assertLess(len(json.dumps(trimmed)) / 4, 100, "each hit must stay under ~100 tokens")

    def test_relevance_score_decays_with_mass_error(self):
        relevance = self.ns["relevance_from_ppm"]
        self.assertEqual(relevance(226.0935, 0.0), 1.0)
        self.assertAlmostEqual(relevance(226.0935, 226.0935), 1.0, places=3)
        self.assertLess(relevance(226.0935, 226.1), relevance(226.0935, 226.0940))
        self.assertEqual(relevance(100.0, 200.0), 0.0)


class DistillationEndToEndTests(unittest.TestCase):
    """Feature 7 end-to-end: a synthetic LC-MS dataset must collapse to a tiny payload."""

    PRODUCT_MZ = 418.1234
    TPPO_MZ = 279.0930

    def setUp(self):
        self.ns = load_symbols([
            "UV_MS_DELAY_MIN", "DEFAULT_TOP_CHROM_PEAKS", "DEFAULT_MIN_REL_PCT",
            "DEFAULT_TOP_IONS", "LLM_TOP_IONS", "BACKGROUND_OFFSET_MIN",
            "APEX_SCANS_TO_AVERAGE", "SPECTRAL_STATE", "PROTON_MASS", "PEG_REPEAT_DA",
            "PEG_ADDUCT_OFFSETS", "BENCH_CONTAMINANTS", "NEUTRAL_LOSS_LIBRARY",
            "_rt_scale", "map_uv_rt_to_ms_rt", "relative_area_percent", "_window_stats",
            "_find_peaks_pure", "_peak_bounds", "find_chromatographic_peaks",
            "average_scans", "subtract_background", "top_centroids", "extract_xic",
            "_scan_to_dict", "_nearest_scan_index", "hydrate_spectral_state",
            "_candidate_polarity_rows", "distill_peak_summary", "get_distilled_peak_summary",
            "match_bench_contaminants", "detect_peg_series", "flag_contaminants",
            "detect_neutral_losses", "estimate_tokens",
        ])
        self._build_synthetic_run()

    def _build_synthetic_run(self):
        """600 MS1 scans (seconds) with a product peak, its 13C/water-loss ions and a
        constant TPPO background, plus a UV trace that elutes 0.08 min earlier."""
        scans, uv_times, uv_signal = [], [], []
        for i in range(600):
            rt = 60.0 + i * 0.5
            product = 1.0e6 * math.exp(-((rt - 300.0) ** 2) / 60.0)
            scans.append({
                "rt": rt,
                "mz": [self.PRODUCT_MZ, self.PRODUCT_MZ + 1.0034,
                       self.PRODUCT_MZ - 18.0106, self.TPPO_MZ, 150.0],
                "int": [product, product * 0.22, product * 0.30, 3.0e4, 2.0e4],
            })
            uv_times.append(rt - 4.8)
            uv_signal.append(product * 1.0e-3 + 5.0)

        state = self.ns["SPECTRAL_STATE"]
        state.update({
            "scans": scans,
            "times": [s["rt"] for s in scans],
            "tic": [sum(s["int"]) for s in scans],
            "bpc": [max(s["int"]) for s in scans],
            "rt_unit": "s",
            "source": "synthetic",
            "uv": {"times": uv_times, "signal": uv_signal, "label": "UV 254 nm"},
            "hydrated": True,
            "distilled": None,
        })
        state["chrom_peaks"] = self.ns["find_chromatographic_peaks"](uv_times, uv_signal)
        self.scans = scans

    def test_uv_peak_is_located(self):
        peaks = self.ns["SPECTRAL_STATE"]["chrom_peaks"]
        self.assertEqual(len(peaks), 1)
        self.assertAlmostEqual(peaks[0]["rt"], 295.2, delta=1.0)

    def test_distilled_payload_contains_the_right_chemistry(self):
        payload = self.ns["distill_peak_summary"]()
        self.assertEqual(payload["rt_unit"], "s")
        self.assertAlmostEqual(payload["uv_ms_delay_min"], 0.08)
        self.assertEqual(payload["detector"], "UV 254 nm")
        peak = payload["peaks"][0]
        self.assertAlmostEqual(peak["ions"][0][0], 418.123, places=2,
                               msg="the UV->MS delay must land on the product apex scan")
        self.assertEqual(peak["ions"][0][1], 100.0)
        self.assertTrue(any("H2O" in loss for loss in peak["losses"]))
        self.assertLessEqual(len(peak["ions"]), self.ns["DEFAULT_TOP_IONS"])

    def test_constant_background_ions_are_subtracted(self):
        payload = self.ns["distill_peak_summary"]()
        observed = [round(ion[0], 2) for ion in payload["peaks"][0]["ions"]]
        self.assertNotIn(round(self.TPPO_MZ, 2), observed,
                         "ions present in the pre-peak background must be removed")

    def test_token_reduction_exceeds_eighty_percent(self):
        summary = self.ns["get_distilled_peak_summary"](refresh=True)
        raw = json.dumps([{"rt": s["rt"], "mz": s["mz"], "int": s["int"]} for s in self.scans])
        raw_tokens = self.ns["estimate_tokens"](raw)
        distilled_tokens = self.ns["estimate_tokens"](summary)
        reduction = 1.0 - distilled_tokens / raw_tokens
        self.assertGreater(reduction, 0.80, f"only {reduction:.1%} reduction")
        self.assertLess(distilled_tokens, 300)


class RoutingAndContextTests(unittest.TestCase):
    """Feature 10 — tiered Flash/Pro routing and history pruning."""

    def setUp(self):
        self.ns = load_symbols([
            "FAST_MODEL_DEFAULT", "DEEP_MODEL_DEFAULT", "FAST_THINKING_BUDGET",
            "DEEP_THINKING_BUDGET", "TOOL_OUTPUT_MAX_CHARS", "PRUNE_AFTER_EXCHANGES",
            "DEEP_REASONING_TRIGGERS", "estimate_tokens", "route_model",
            "summarize_tool_payload", "prune_history_contents",
        ])

    def test_default_models(self):
        self.assertEqual(self.ns["FAST_MODEL_DEFAULT"], "gemini-2.5-flash")
        self.assertEqual(self.ns["DEEP_MODEL_DEFAULT"], "gemini-2.5-pro")
        self.assertEqual(self.ns["FAST_THINKING_BUDGET"], 0)
        self.assertGreater(self.ns["DEEP_THINKING_BUDGET"], 0)

    def test_routine_turns_stay_on_flash(self):
        for prompt in ("Replot peak 3 with an XIC at 418.12",
                       "Which peak matches the product?",
                       "Zoom into 4-6 minutes"):
            with self.subTest(prompt=prompt):
                route = self.ns["route_model"](prompt)
                self.assertEqual(route["model"], "gemini-2.5-flash")
                self.assertEqual(route["tier"], "fast")
                self.assertEqual(route["thinking_budget"], 0)

    def test_complex_questions_escalate_to_pro(self):
        for prompt in ("Elucidate the unknown structure at 6.2 min",
                       "Explain the fragmentation pathway",
                       "Why does the byproduct form? Think step by step"):
            with self.subTest(prompt=prompt):
                route = self.ns["route_model"](prompt)
                self.assertEqual(route["model"], "gemini-2.5-pro")
                self.assertEqual(route["tier"], "deep")
                self.assertGreater(route["thinking_budget"], 0)

    def test_modes_can_be_forced(self):
        self.assertEqual(self.ns["route_model"]("elucidate", mode="fast")["tier"], "fast")
        self.assertEqual(self.ns["route_model"]("hello", mode="deep")["tier"], "deep")

    def test_bulky_tool_payloads_are_summarised(self):
        payload = {"scans": [{"mz": 100.0 + i, "int": 5.0} for i in range(500)]}
        summary = self.ns["summarize_tool_payload"](payload)
        self.assertLessEqual(len(summary), self.ns["TOOL_OUTPUT_MAX_CHARS"] + 40)
        self.assertIn("pruned", summary)
        self.assertEqual(self.ns["summarize_tool_payload"]("short"), "short")

    def test_history_pruning_keeps_the_most_recent_tool_results(self):
        def tool_turn(size):
            response = SimpleNamespace(response={"blob": "x" * size})
            return SimpleNamespace(parts=[SimpleNamespace(function_response=response)])

        history = [
            SimpleNamespace(parts=[SimpleNamespace(function_response=None, text="hi")]),
            tool_turn(5000), tool_turn(5000), tool_turn(5000), tool_turn(5000),
        ]
        before = sum(len(str(turn)) for turn in history)
        pruned = self.ns["prune_history_contents"](history, keep_last_tool_turns=2)
        payloads = [turn.parts[0].function_response.response for turn in pruned[1:]]
        self.assertIn("pruned_summary", payloads[0])
        self.assertIn("pruned_summary", payloads[1])
        self.assertNotIn("pruned_summary", payloads[2])
        self.assertNotIn("pruned_summary", payloads[3])
        self.assertLess(sum(len(str(turn)) for turn in pruned), before - 8000)

    def test_token_estimate(self):
        self.assertEqual(self.ns["estimate_tokens"]("x" * 400), 100)
        self.assertEqual(self.ns["estimate_tokens"](""), 1)


if __name__ == "__main__":
    unittest.main()
