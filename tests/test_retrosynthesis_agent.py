import ast
from collections import Counter, defaultdict
import functools
import gc
import html
from html.parser import HTMLParser
import json
import logging
import math
import os
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
from typing import List, Literal, Optional
import unittest
from unittest.mock import Mock, patch
import urllib.parse

try:
    import duckdb
    import numpy as np
    import pandas as pd
    from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
    import requests
    from rdkit import Chem, DataStructs, RDConfig
    from rdkit.Chem import AllChem, Descriptors, FilterCatalog, FunctionalGroups, rdChemReactions, rdFMCS, rdMolDescriptors
    HAS_CHEMISTRY = True
except ImportError:
    HAS_CHEMISTRY = False

try:
    from ord_schema.proto import reaction_pb2
except ImportError:
    reaction_pb2 = None


NOTEBOOK = Path(__file__).resolve().parents[1] / "Retrosynthesis_Agent.ipynb"
CELLS = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
CELL_TWO = next("".join(c["source"]) for c in CELLS if c["source"][0].startswith("# @title 2. Chemistry Workbench"))
TREE = ast.parse(CELL_TWO)
SNAR = "CS(=O)(=O)c1ncccn1.Nc1ccccc1>>c1ccc(Nc2ncccn2)cc1"
ANHYDRIDE = "O=C1OC(=O)c2ccccc21.Nc1ccccc1>>O=C1N(c2ccccc2)C(=O)c2ccccc21"
PYRIMIDINE = "CN(C)C=CC(=O)c1ccccc1.NC(N)=N>>Nc1nccc(-c2ccccc2)n1"
QUINOLINE = "Nc1ccccc1.CC(=O)CC=O>>Cc1ccnc2ccccc12"


def load_cell_two():
    """Load definitions, not Colab, the model, DuckDB globals, or UI setup."""
    data_names = {
        "_SMARTSRX_URL", "_STEP_USPTO_TEMPLATE_REGISTRY", "_STEP_TEMPLATE_FIELDS", "_COMPETING_ELECTROPHILES",
        "ORD_DB_PATH", "_USPTO_FG_SOURCE", "_ELN_COLUMNS",
    }
    body = [
        n for n in TREE.body
        if (isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name != "_build_genai_config")
        or (isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in data_names)
    ]
    ns = dict(globals(), _HAS_ORD_SCHEMA=False, _CANON_SMILES_TO_NAME={}, GEMINI_STEP_CACHE={}, MAX_STEP_CACHE_SIZE=250,
              ACTIVE_GENAI_SETTINGS={"model": "test", "thinking_level": "low"}, _CHEM_SVG_CARDS=set())
    exec(compile(ast.Module(body=body, type_ignores=[]), str(NOTEBOOK), "exec"), ns)
    ns["_load_smartsrx_data"] = lambda: [
        {"specific_type": "TestSulfone", "smarts": "[#6]S(=O)(=O)[#6]"},
    ]
    return ns


def ingredient(name, role, smiles, eq=0.0, mw=0.0, molarity=0.0):
    return dict(name=name, role=role, smiles_or_formula=smiles, equivalents=eq, mw_g_mol=mw, molarity_m=molarity)


class NotebookIntegrationTests(unittest.TestCase):
    def test_existing_callbacks_and_later_cells_are_present(self):
        names = {node.args[0].value for node in ast.walk(TREE) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == "register_callback"}
        self.assertTrue({"chemforge.run_synthesis", "chemforge.reclassify_routes", "chemforge.sync_ord_shards",
                         "chemforge.sync_canvas", "chemforge.agent_chat"} <= names)
        self.assertIn("finder.target_smiles = canon_target", CELL_TWO)
        self.assertTrue(any(c["source"][0].startswith("# @title 6.") for c in CELLS))
        self.assertTrue(any(c["source"][0].startswith("# @title 7.") for c in CELLS))


@unittest.skipUnless(HAS_CHEMISTRY, "requires existing Cell 2 scientific dependencies")
class ReactionGateTests(unittest.TestCase):
    def setUp(self):
        self.ns = load_cell_two()
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.validator = self.ns["ORDRetrosynthesisValidator"](str(Path(temp.name) / "ord.duckdb"))

    def test_distinct_matching_can_reassign_ambiguous_molecules(self):
        compile_queries = self.ns["_compile_rxn_smarts_queries"]
        matches = self.ns["matches_reaction_smarts_mols"]
        queries = compile_queries("[#6].CO>>CC")
        product = [Chem.MolFromSmiles("CC")]
        self.assertFalse(matches([Chem.MolFromSmiles("CO"), Chem.MolFromSmiles("O")], product, queries))
        self.assertTrue(matches([Chem.MolFromSmiles("CO"), Chem.MolFromSmiles("CC")], product, queries))

    def test_catalog_counts_losses_even_when_one_group_remains(self):
        consumed, formed, present = self.ns["compute_rdkit_internal_delta_fg"]("OCCO>>CCO")
        self.assertGreater(self.ns["_build_rdkit_internal_fg_catalog"]().GetNumEntries(), 85)
        self.assertIn("RDKitFG_Alcohol", present)
        self.assertIn("RDKitFG_Alcohol", consumed)
        self.assertNotIn("RDKitFG_Alcohol", formed)
        self.assertIn("SMARTSRX_TestSulfone", self.ns["compute_rdkit_internal_delta_fg"](SNAR)[0])

    def test_harvests_metadata_and_code_zero_before_cleanup(self):
        retro = "[C:1][N:2]>>[C:1]Br.[N:2]"
        nodes = [
            SimpleNamespace(metadata={"template_code": 0}, reaction_smiles=lambda: "CNC>>CBr.CN"),
            SimpleNamespace(metadata={"retro_template": retro, "classification": "Amination", "template_code": 3},
                            reaction_smiles=lambda: "CCNC>>CCBr.CN"),
        ]
        routes = [{"reaction_tree": SimpleNamespace(reactions=lambda: iter(nodes))}]
        db = pd.DataFrame([{"retro_template": retro, "classification": "Alkylation"}])
        finder = SimpleNamespace(routes=routes, tree=object(), target_smiles="CNC",
                                 expansion_policy={"uspto": SimpleNamespace(templates=db)})
        steps = self.ns["_harvest_uspto_templates_from_finder"](finder)
        self.ns["get_process_ram_mb"] = lambda: 0.0
        self.ns["cleanup_aizynth_memory"](finder, trim_ipython_history=False)
        self.assertEqual(finder.routes, [])
        self.assertEqual(finder.target_smiles, "CNC")
        self.assertEqual(self.ns["_STEP_USPTO_TEMPLATE_REGISTRY"][steps[0]], {
            "retro_template": retro, "classification": "Alkylation", "template_code": "0",
        })
        tiers = self.ns["build_progressive_reaction_smarts"](steps[0])
        self.assertEqual([t["base_sim"] for t in tiers], [0.82, 0.68, 0.58])
        self.assertEqual(tiers[0]["smarts"], "[C:1]Br.[N:2]>>[C:1][N:2]")

    def test_nodes_with_identical_smiles_keep_their_own_templates(self):
        smi = "CC=O>>CCO"
        mapped = "[CH3:1][CH2:2][OH:3]>>[CH3:1][CH:2]=[O:3]"
        retro_a = "[CH2:2][OH:3]>>[CH:2]=[O:3]"
        retro_b = "[C:1][CH2:2][OH:3]>>[C:1][CH:2]=[O:3]"
        nodes = [
            SimpleNamespace(metadata={"template": retro_a, "classification": "Carbonyl reduction",
                                      "mapped_reaction_smiles": mapped}, reaction_smiles=lambda: "CCO>>CC=O"),
            SimpleNamespace(metadata={"template_code": 0, "retro_template": float("nan")},
                            reaction_smiles=lambda: "CCO>>CC=O"),
        ]
        trees = [SimpleNamespace(reactions=lambda node=node: iter([node]), child_reactions=lambda node: []) for node in nodes]
        policy = SimpleNamespace(template_column="custom_smarts",
                                 templates=pd.DataFrame([{"classification": "Aldehyde reduction", "custom_smarts": retro_b}]))
        finder = SimpleNamespace(routes=[{"reaction_tree": tree} for tree in trees], expansion_policy={"uspto": policy})
        node_templates = {}
        self.assertEqual(self.ns["_harvest_uspto_templates_from_finder"](finder, node_templates), [smi])
        steps = [self.ns["_flatten_reaction_tree"](tree, node_templates)[0] for tree in trees]
        self.assertEqual([step["retro_template"] for step in steps], [retro_a, retro_b])
        self.assertEqual(steps[0]["mapped_forward_smiles"], self.ns["_reverse_reaction"](mapped))
        self.insert("ord-reduction", smi)
        self.ns["ORD_VALIDATOR"] = self.validator
        lookup = self.ns["_classify_route_steps"](steps, use_gemini=False)
        results = [lookup[self.ns["_route_step_key"](step)] for step in steps]
        self.assertEqual([result["reaction_name"] for result in results], ["Carbonyl reduction", "Aldehyde reduction"])
        self.assertTrue(all(result["ord_validation"]["matched_smarts_label"].startswith("Tier 1:") for result in results))

    def insert(self, ord_id, rxn, query=None, yield_pct=75):
        product = self.ns["_parse_reaction_mols"](query or rxn)[1][0]
        with duckdb.connect(self.validator.db_path) as con:
            con.execute(
                "INSERT INTO ord_reactions (ord_id, rxn_smiles, prod_connectivity, yield_pct, fp_bytes, procedure_details)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                [ord_id, rxn, Chem.MolToInchiKey(product).split("-")[0], yield_pct,
                 self.ns["compute_folded_rxn_fp"](query or rxn), "  Verbatim ORD note.\nSecond line.  "],
            )
        self.validator._refresh_index()

    def test_same_product_and_identical_fingerprint_do_not_bypass_gate(self):
        self.insert("ord-good", SNAR)
        self.insert("ord-anhydride", ANHYDRIDE, query=SNAR, yield_pct=99)
        product = SNAR.split(">>")[1]
        self.insert("ord-spectator", f"{product}>>{product}", query=SNAR, yield_pct=100)
        difference_fp = Mock(wraps=self.ns["compute_rxn_center_diff_fp"])
        self.ns["compute_rxn_center_diff_fp"] = difference_fp
        report = self.validator.validate_step(SNAR)
        self.assertEqual(report["smarts_matched_count"], 1)
        self.assertEqual(report["exact_product_hits"], 1)
        self.assertAlmostEqual(report["max_rxn_similarity"], 0.45 * 0.68 + 0.55, places=3)
        self.assertIn("radius 1", report["matched_smarts_label"])
        self.assertEqual([p["ord_id"] for p in report["top_precedents"]], ["ord-good"])
        self.assertEqual(report["top_precedents"][0]["procedure_details"], "  Verbatim ORD note.\nSecond line.  ")
        self.assertEqual({call.args[0] for call in difference_fp.call_args_list}, {SNAR})
        self.assertEqual(self.validator.validate_step(f"{product}>>{product}")["top_precedents"], [])

    def test_quinoline_not_admitted_as_pyrimidine_and_delta_only_can_pass(self):
        self.insert("ord-quinoline", QUINOLINE, query=PYRIMIDINE, yield_pct=99)
        self.assertEqual(self.validator.validate_step(PYRIMIDINE)["smarts_matched_count"], 0)
        self.insert("ord-pyrimidine", PYRIMIDINE)
        self.ns["build_progressive_reaction_smarts"] = lambda *args: []
        report = self.validator.validate_step(PYRIMIDINE)
        self.assertEqual(report["smarts_matched_count"], 1)
        self.assertEqual(report["matched_smarts_label"], "RDKit ΔFG match")

    def test_competing_electrophile_veto_applies_even_to_smarts_matches(self):
        r, p = self.ns["_parse_reaction_mols"](ANHYDRIDE)
        delta = self.ns["compute_rdkit_internal_delta_fg"](ANHYDRIDE)
        gate = self.ns["_match_ord_gate"]
        tiers = self.ns["build_progressive_reaction_smarts"](ANHYDRIDE)
        self.assertIn("fr_anhydride", delta[0])
        self.assertIsNone(gate(r, p, delta, tiers, self.ns["compute_rdkit_internal_delta_fg"](SNAR)))
        self.assertIsNotNone(gate(r, p, delta, tiers, delta))

    def test_unchanged_motifs_do_not_make_a_mapped_smarts_match(self):
        query = "CC=O>>CCO"
        self.ns["_STEP_USPTO_TEMPLATE_REGISTRY"][query] = {"retro_template": "[CH2:1][OH:2]>>[CH:1]=[O:2]"}
        self.insert("ord-debromination", "O=Cc1ccc(CO)cc1Br>>O=Cc1ccc(CO)cc1", query=query, yield_pct=99)
        self.insert("ord-reduction", "O=CCc1ccccc1>>OCCc1ccccc1")
        report = self.validator.validate_step(query)
        self.assertEqual([p["ord_id"] for p in report["top_precedents"]], ["ord-reduction"])
        self.assertIn("Tier 1", report["matched_smarts_label"])

    def test_filtered_shard_can_be_rescanned_for_a_new_gate(self):
        target = "CC=O>>CCO"
        dataset = SimpleNamespace(reactions=[target])
        download = Mock(return_value="/tmp/ord-fixture.parquet")
        self.ns.update(_HAS_HF_HUB=True, _HAS_ORD_SCHEMA=True, hf_hub_download=download,
                       list_repo_files=lambda **kwargs: ["fixture.parquet"],
                       ord_datasets=SimpleNamespace(load_dataset=lambda _: dataset),
                       parse_ord_reaction_protobuf=lambda smi, admit: (
                           {"ord_id": "ord-reduction", "rxn_smiles": smi} if admit(smi) else None))
        sync = self.validator.sync_huggingface_parquet_shards
        self.assertEqual(sync(forward_smiles_list=[SNAR])["added"], 0)
        self.assertEqual(sync(forward_smiles_list=[target])["added"], 1)
        self.assertEqual(sync(forward_smiles_list=[target])["added"], 0)
        self.assertEqual(download.call_count, 2)
        self.assertEqual(self.validator.validate_step(target)["smarts_matched_count"], 1)

    @unittest.skipIf(reaction_pb2 is None, "requires existing ORD schema")
    def test_ord_procedure_and_measured_amounts_round_trip(self):
        self.ns.update(_HAS_ORD_SCHEMA=True, ord_message_helpers=SimpleNamespace(
            smiles_from_compound=lambda comp: comp.identifiers[0].value),
            _smiles_to_english_name=lambda name: name,
        )
        p = reaction_pb2
        reaction = p.Reaction(reaction_id="ord-measured")
        reaction.notes.procedure_details = "  Original ORD text\nwith whitespace.  "
        reactant = reaction.inputs["r"].components.add(reaction_role=p.ReactionRole.REACTANT, is_limiting=True)
        reactant.identifiers.add(type=p.CompoundIdentifier.SMILES, value="CC=O")
        reactant.amount.moles.units, reactant.amount.moles.value = p.Moles.MILLIMOLE, 2
        solvent = reaction.inputs["s"].components.add(reaction_role=p.ReactionRole.SOLVENT)
        solvent.identifiers.add(type=p.CompoundIdentifier.SMILES, value="O")
        solvent.amount.volume.units, solvent.amount.volume.value = p.Volume.MICROLITER, 10000
        outcome = reaction.outcomes.add()
        outcome.reaction_time.units, outcome.reaction_time.value = p.Time.MINUTE, 90
        product = outcome.products.add(is_desired_product=True)
        product.identifiers.add(type=p.CompoundIdentifier.SMILES, value="CCO")
        product.measurements.add(type=p.ProductMeasurement.YIELD).percentage.value = 72
        record = self.ns["parse_ord_reaction_protobuf"](reaction)
        df_bulk = pd.DataFrame([record])
        with duckdb.connect(self.validator.db_path) as con:
            con.execute("INSERT INTO ord_reactions BY NAME SELECT * FROM df_bulk")
        self.validator._refresh_index()
        self.ns["ORD_VALIDATOR"] = self.validator
        info = self.ns["rdkit_classify_step"]("CC=O>>CCO")
        hit = info["ord_validation"]["top_precedents"][0]
        self.assertEqual(hit["procedure_details"], reaction.notes.procedure_details)
        self.assertEqual(hit["duration_h"], 1.5)
        self.assertEqual(info["assumed_step_yield"], 0.72)
        self.assertEqual(info["ranked_conditions"][0]["citation_or_ord_id"], "ord-measured")
        self.assertEqual(info["reagent_stoichiometry"][-1]["molarity_m"], 0.2)


@unittest.skipUnless(HAS_CHEMISTRY, "requires existing Cell 2 scientific dependencies")
class StepConditionsAndElnTests(unittest.TestCase):
    def setUp(self):
        self.ns = load_cell_two()

    def test_structured_response_uses_exact_evidence_and_clamps_scores(self):
        smi = "CC=O>>CCO"
        report = dict(smarts_matched_count=1, mean_analog_yield_pct=97.0, top_precedents=[
            {"ord_id": "ord-allowed", "procedure_details": "  Verbatim.\n", "yield_pct": 97.0, "temp_c": 25},
        ])
        self.ns["ORD_VALIDATOR"] = SimpleNamespace(validate_step=lambda _: report, index_revision=1)
        base = self.ns["rdkit_classify_step"](smi)
        self.assertEqual(base["assumed_step_yield"], 0.95)
        item = dict(base, safety_green_score=6, typical_conditions="ignore unsourced conditions",
                    source_doi_or_patent="ord-allowed", assumed_step_yield=0.10)
        item["ranked_conditions"] = list(reversed(base["ranked_conditions"]))
        self.ns["get_gemini_client"] = lambda: object()
        gemini = Mock(return_value=(SimpleNamespace(text=json.dumps({"steps": [item]})), "test", "low"))
        self.ns["call_gemini_with_config"] = gemini
        result = self.ns["batch_classify_with_gemini"]([smi])[smi]
        self.assertEqual(result["safety_green_score"], 5)
        self.assertEqual(result["assumed_step_yield"], 0.95)
        self.assertEqual(result["source_doi_or_patent"], "ord-allowed")
        self.assertEqual([c["rank"] for c in result["ranked_conditions"]], [1, 2, 3])
        self.assertEqual([c["citation_or_ord_id"] for c in result["ranked_conditions"]],
                         ["ord-allowed", "USPTO/ΔFG Standard", "USPTO/ΔFG Standard"])
        self.assertIn("25 °C", result["typical_conditions"])
        prompt = gemini.call_args.kwargs["contents"].split("Evidence JSON:\n", 1)[1]
        evidence = json.loads(prompt)[0]
        self.assertEqual(evidence["verified_ord_precedents"][0]["procedure_details"], "  Verbatim.\n")
        product = next(row for row in evidence["predicted_ipc_masses"] if row["role"] == "Product")
        self.assertAlmostEqual(product["[M+H]+ m/z"], 47.0491, places=4)
        self.assertNotEqual(self.ns["batch_classify_with_gemini"]([smi], use_gemini=False)[smi]["safety_green_score"], 5)

    def test_unverified_citation_rejects_the_claims_not_just_the_identifier(self):
        smi = "CC=O>>CCO"
        report = {"smarts_matched_count": 1, "top_precedents": [{"ord_id": "ord-good", "procedure_details": "ORD text"}]}
        self.ns["ORD_VALIDATOR"] = SimpleNamespace(validate_step=lambda _: report, index_revision=1)
        self.ns["get_gemini_client"] = lambda: object()
        baseline = self.ns["rdkit_classify_step"](smi)
        proposed = dict(baseline, source_doi_or_patent="ord-unrelated", experimental_procedure="Unverified draft")
        claim = dict(baseline["ranked_conditions"][0], citation_or_ord_id="ord-unrelated", expected_yield_range="99% reported")
        proposed["ranked_conditions"] = [claim, *baseline["ranked_conditions"][1:]]
        response = SimpleNamespace(text=json.dumps({"steps": [proposed]}))
        self.ns["call_gemini_with_config"] = Mock(return_value=(response, "test", "low"))
        result = self.ns["batch_classify_with_gemini"]([smi])[smi]
        self.assertFalse(result["gemini_enriched"])
        self.assertEqual(result["source_doi_or_patent"], "")
        self.assertEqual(result["experimental_procedure"], "")
        self.assertEqual(result["ranked_conditions"][0], baseline["ranked_conditions"][0])

    def test_literature_cross_checks_doi_and_queries_real_terms(self):
        pubchem = Mock()
        pubchem.json.return_value = {"InformationList": {"Information": [{"PubMedID": [123456]}]}}
        europe = Mock()
        europe.json.return_value = {"resultList": {"result": [
            {"pmid": "123456", "doi": "10.1234/confirmed", "abstractText": "A relevant summary."},
            {"doi": "10.1234/mismatch"},
        ]}}
        accepted = Mock()
        accepted.json.return_value = {"message": {"DOI": "10.1234/confirmed", "title": ["Example"]}}
        wrong = Mock()
        wrong.json.return_value = {"message": {"DOI": "10.1234/different"}}
        with patch.object(requests, "post", return_value=pubchem) as post, patch.object(
            requests, "get", side_effect=[europe, accepted, wrong],
        ) as get:
            hits = self.ns["fetch_verified_literature_dois"]("CCO", "Amination OR Synthetic Transformation")
        self.assertEqual([w["doi"] for w in hits], ["10.1234/confirmed"])
        self.assertTrue(hits[0]["product_pubmed_xref"])
        self.assertIn("/xrefs/PubMedID/", post.call_args.args[0])
        query = get.call_args_list[0].kwargs["params"]["query"]
        self.assertIn('TITLE_ABS:"Amination"', query)
        self.assertIn("EXT_ID:123456", query)
        self.assertNotIn("Synthetic Transformation", query)
        self.assertEqual(get.call_args_list[1].args[0], "https://api.crossref.org/works/10.1234/confirmed")

    def test_convergent_eln_uses_each_step_yield_and_reagent_loadings(self):
        target = "CCOC(C)=O"
        steps = [
            dict(step_index=1, forward_smiles="CC=O>>CCO", assumed_step_yield=0.50),
            dict(step_index=2, forward_smiles="CC(=O)O>>CC(=O)Cl", assumed_step_yield=0.80),
            dict(step_index=3, forward_smiles=f"CCO.CC(=O)Cl>>{target}", assumed_step_yield=0.80,
                 reagent_stoichiometry=[
                     ingredient("Ethanol", "Reactant", "CCO", 1),
                     ingredient("Coupling partner", "Reactant", "CC(=O)Cl", 1.5),
                     ingredient("Catalyst", "Catalyst", "[Pd]", 0.05, 106.42),
                     ingredient("Ligand", "Ligand", "L", 0.10, 200),
                     ingredient("Base", "Base/Reagent", "K2CO3", 2.0, 138.21),
                     ingredient("Solvent", "Solvent", "O", molarity=0.20),
                 ]),
        ]
        route = {"steps": steps}
        scale_g = 8 * Descriptors.MolWt(Chem.MolFromSmiles(target)) / 1000
        plans = self.ns["_scale_route_eln"](route, target, scale_g)
        by_step = {p["step"]["step_index"]: p for p in plans}
        for idx, mmol in ((1, 20.0), (2, 18.75), (3, 10.0)):
            self.assertAlmostEqual(by_step[idx]["limiting_mmol"], mmol)
        self.ns["LATEST_RETRO_RESULTS"] = {"target_smiles": target, "routes": [route]}
        markdown = self.ns["generate_eln_markdown"](1, scale_g, step_filter=3)
        rendered = self.ns["generate_eln_html_table"](1, scale_g, step_filter=3)
        for output in (markdown, rendered):
            for value in ("Limiting Reactant", "Coupling Partner", "Ligand", "Base/Reagent", "Expected Product",
                          "0.05 (5 mol%)", "53.21", "50.000", "8.000"):
                self.assertIn(value, output)
            self.assertNotIn("18.750", output)

    def test_limiting_substitution_preserves_recorded_partner_ratio(self):
        ingredients = [
            ingredient("Methanol", "Reactant", "CO", 1),
            ingredient("Acetic acid", "Reactant", "CC(=O)O", 2),
            ingredient("Water", "Solvent", "O", molarity=0.1),
        ]
        rows = self.ns["_merge_step_stoichiometry"]("CC(=O)O.CCO>>CCOC(C)=O", ingredients)
        self.assertEqual([(r["smiles_or_formula"], r["equivalents"]) for r in rows[:2]], [("CCO", 1), ("CC(=O)O", 2)])
        self.assertEqual(rows[2]["molarity_m"], 0.1)

    def test_scaled_eln_retains_two_producers_of_the_same_intermediate(self):
        target = "CCOC(=O)CC=O"
        reactions = [
            ("CC=O>>CCO", 0.50, []),
            ("CCBr>>CCO", 0.80, []),
            ("CCO.O=CCl>>CCOC=O", 0.80, [1]),
            ("CCO.CC(=O)Cl>>CCOC(C)=O", 0.80, [2]),
            (f"CCOC=O.CCOC(C)=O>>{target}", 0.80, [3, 4]),
        ]
        steps = [dict(step_index=i, forward_smiles=smi, assumed_step_yield=y, precursor_steps=parents)
                 for i, (smi, y, parents) in enumerate(reactions, 1)]
        scale_g = 8 * Descriptors.MolWt(Chem.MolFromSmiles(target)) / 1000
        route = {"steps": steps}
        plans = self.ns["_scale_route_eln"](route, target, scale_g)
        charges = {p["step"]["step_index"]: p["limiting_mmol"] for p in plans}
        self.assertEqual(len(charges), 5)
        self.assertAlmostEqual(charges[1], 25)
        self.assertAlmostEqual(charges[2], 15.625)
        self.ns["LATEST_RETRO_RESULTS"] = {"target_smiles": target, "routes": [route]}
        overridden = self.ns["generate_eln_markdown"](1, scale_g, assumed_step_yield=0.5, step_filter=5)
        self.assertIn("50% assumed yield", overridden)
        self.assertIn("16.000", overridden)

    def test_zero_hit_panel_does_not_render_old_fingerprint_hits(self):
        render = self.ns["_render_ord_step_panel_html"]
        report = {"smarts_matched_count": 0, "top_precedents": [{"ord_id": "ord-stale"}],
                  "rdkit_delta_fg_signature": {"consumed_fgs": ["fr_sulfone"], "formed_fgs": ["fr_aniline"]}}
        panel = render(report)
        self.assertIn("0 ORD reactions in current shard matched this Reaction SMARTS / ΔFG gate", panel)
        self.assertIn("RDKit ΔFG Signature: fr_sulfone → fr_aniline", panel)
        self.assertNotIn("ord-stale", panel)
        del report["smarts_matched_count"]
        self.assertIn("Reclassify it to verify precedents", render(report))
        self.assertNotIn("ord-stale", render(report))

    def test_step_card_has_ranked_conditions_verbatim_evidence_analytics_and_copy(self):
        smi = "Brc1ccccc1.N>>Nc1ccccc1"
        report = {
            "smarts_matched_count": 1, "matched_smarts_label": "Tier 1: Exact USPTO Template SMARTS",
            "top_precedents": [{"ord_id": "ord-allowed", "procedure_details": "  Author's <b>literal note</b>\nNext line.",
                                "doi": "10.1234/unverified", "yield_pct": 72.0}],
        }
        self.ns["ORD_VALIDATOR"] = SimpleNamespace(validate_step=lambda _: report, get_index_size=lambda: 1)
        self.ns["_generate_strategy_header"] = lambda _: {}
        self.ns["get_process_ram_mb"] = lambda: 10
        self.ns["mol_or_rxn_to_svg_html"] = lambda *a, **k: "<svg></svg>"
        step = dict(self.ns["rdkit_classify_step"](smi), step_index=1, safety_green_score=9,
                    experimental_procedure="Analog's <em>draft</em>")
        route = {"steps": [step], "step_count": 1,
                 "starting_materials": {"precursors": [], "in_stock_count": 1, "total_leaves": 1}}
        res = {"target_smiles": "Nc1ccccc1", "target_scale_grams": 1.0, "routes": [route]}
        result = self.ns["_render_routes_html_from_results"](res)
        for expected in ("Safety:</b> 5/5", "SMARTS-Verified ORD Hits: 1", "Rank #1", "Rank #2", "Rank #3",
                         "LC-MS &amp; Workup", "hydrodehalogenation impurity", "94.0651", "ΔTPSA", "Expected Product",
                         "Copy Step ELN to Clipboard", "Author&#x27;s &lt;b&gt;literal note&lt;/b&gt;\nNext line."):
            self.assertIn(expected, result)
        self.assertNotIn("10.1234/unverified", result)

        class CopyButtonParser(HTMLParser):
            payload = None

            def handle_starttag(self, tag, attrs):
                attributes = dict(attrs)
                if tag == "button" and "data-eln" in attributes:
                    self.payload = attributes["data-eln"]

        parser = CopyButtonParser()
        parser.feed(result)
        self.assertIn("Limiting Reactant", parser.payload)
        self.assertIn("Analog's &lt;em&gt;draft&lt;/em&gt;", parser.payload)


if __name__ == "__main__":
    unittest.main()
