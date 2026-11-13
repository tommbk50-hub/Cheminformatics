import ast
import csv
import html
import importlib.util
import inspect
import io
import json
import os
import re
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


NOTEBOOK = Path(__file__).resolve().parents[1] / "AiZynthFinder_with_Gemini.ipynb"
NB_DATA = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
CELLS = NB_DATA["cells"]


def cell_source(index: int) -> str:
    return "".join(CELLS[index]["source"])


class _FakeAtom:
    def __init__(self, symbol: str, idx: int, atom_map: int = 0):
        self._symbol = symbol
        self._idx = idx
        self._map = atom_map
        self._atomic_num = {
            "C": 6, "N": 7, "O": 8, "F": 9, "P": 15, "S": 16, "Cl": 17,
            "Br": 35, "I": 53, "B": 5, "Si": 14, "Sn": 50, "Mg": 12, "Zn": 30,
        }.get(symbol, 6)

    def GetIdx(self):
        return self._idx

    def GetAtomMapNum(self):
        return self._map

    def SetAtomMapNum(self, val: int):
        self._map = int(val)

    def GetAtomicNum(self):
        return self._atomic_num

    def GetSymbol(self):
        return self._symbol


class _FakeMol:
    """Lightweight deterministic molecule stub for offline unit testing without RDKit."""

    def __init__(self, text: str, is_smarts: bool = False):
        self.text = text.strip()
        self.is_smarts = is_smarts
        # Parse mapped atoms like [CH3:1] alongside plain heavy atoms
        self._atoms = []
        idx = 0
        pos = 0
        while pos < len(self.text):
            if self.text[pos] == "[":
                end = self.text.find("]", pos)
                token = self.text[pos + 1:end] if end != -1 else self.text[pos + 1:]
                m = re.match(r"([A-Z][a-z]?).*?(?::(\d+))?$", token)
                sym = m.group(1) if m and m.group(1) else "C"
                amap = int(m.group(2)) if (m and m.group(2)) else 0
                if sym != "H":
                    self._atoms.append(_FakeAtom(sym, idx, amap))
                    idx += 1
                pos = (end + 1) if end != -1 else len(self.text)
            else:
                two = self.text[pos:pos + 2]
                one = self.text[pos]
                if two in ("Cl", "Br", "Si", "Sn", "Mg", "Zn"):
                    self._atoms.append(_FakeAtom(two, idx, 0))
                    idx += 1
                    pos += 2
                elif one in "BCNOFPSIcnops":
                    sym = one.upper()
                    self._atoms.append(_FakeAtom(sym, idx, 0))
                    idx += 1
                    pos += 1
                else:
                    pos += 1

    def GetAtoms(self):
        return self._atoms

    def GetNumAtoms(self):
        return len(self._atoms)

    def GetNumHeavyAtoms(self):
        return len(self._atoms)

    def _matches_tag(self, pat_text: str) -> bool:
        s = self.text
        checks = {
            "[#5](O)O": "B(O)O" in s,
            "[#5]1O[#6][#6]O1": "B1OC" in s,
            "[#50]": "Sn" in s,
            "[#12,#30]": ("Mg" in s or "Zn" in s),
            "c[Cl,Br,I]": bool(re.search(r"c[12]?(?:ccc)?[ClBrI]|Brc1|Ic1|Clc1|c\(Br\)|c\(Cl\)|c\(I\)", s)),
            "cOS(=O)(=O)C(F)(F)F": "OS(=O)(=O)C(F)(F)F" in s,
            "[CX4][Cl,Br,I]": bool(re.search(r"C[ClBrI]|CCBr|CCCl", s)),
            "[CX3](=O)[OX2H1]": ("C(=O)O" in s and not re.search(r"C\(=O\)O[Cc]", s)),
            "[#6][CX3](=O)[OX2][#6]": bool(re.search(r"C\(=O\)O[Cc]", s)),
            "[#6][CX3](=O)[OX2]Cc1ccccc1": "C(=O)OCc1ccccc1" in s,
            "[CX3](=O)[Cl,Br,F]": ("C(=O)Cl" in s or "C(=O)Br" in s),
            "[CX3](=O)O[CX3](=O)": "C(=O)OC(=O)" in s,
            "[#6][CX3](=O)[NX3]": ("C(=O)N" in s and "OC(C)(C)C" not in s and "OCc1ccccc1" not in s),
            "[NX3;H2,H1;!$(NC=O);!$(NS=O)]": bool(re.search(r"(?:^|[^=])N(?![C=]|S|\(|\[|\+)", s) or "NCC" in s or "Nc1" in s),
            "[NX2]=[NX2+]=[NX1-]": "N=[N+]=[N-]" in s,
            "[NX1-]=[NX2+]=[NX2]": "[N-]=[N+]=N" in s,
            "c1ncnn1": "c1ncnn1" in s,
            "c1nnn[nH]1": "c1nnn[nH]1" in s,
            "c1ncsc1": "c1ncsc1" in s,
            "[CX3H1](=O)[#6]": "CH=O" in s or "C=O" in s,
            "[CX3](=O)([#6])[#6]": "CC(=O)c" in s or "CC(=O)C" in s,
            "[CX4][OX2H1]": bool(re.search(r"C[CH2]*O(?:$|\.)", s) or "NCCO" in s),
            "c[OX2H1]": "Oc1" in s or "c(O)" in s,
            "[#6][SX2H1]": "CCS" in s or "Sc1" in s,
            "[NX3]C(=O)OC(C)(C)C": "NC(=O)OC(C)(C)C" in s,
            "[NX3]C(=O)OCc1ccccc1": "NC(=O)OCc1ccccc1" in s,
            "[#8][#14]([#6])([#6])[#6]": "O[Si]" in s,
        }
        if pat_text in checks:
            return checks[pat_text]
        return pat_text in s

    def HasSubstructMatch(self, pat: "_FakeMol") -> bool:
        if pat is None:
            return False
        return self._matches_tag(pat.text)

    def GetSubstructMatches(self, pat: "_FakeMol"):
        if not self.HasSubstructMatch(pat):
            return ()
        if pat.text == "c[Cl,Br,I]":
            count = len(re.findall(r"Br|Cl|I", self.text))
            return tuple((i,) for i in range(max(1, count)))
        return ((0,),)

    def GetSubstructMatch(self, pat: "_FakeMol"):
        if pat is None:
            return ()
        if pat.text == self.text:
            return tuple(range(len(self._atoms)))
        if pat.text in self.text or self.HasSubstructMatch(pat):
            n = min(len(pat._atoms) or 1, len(self._atoms))
            return tuple(range(n))
        return ()


def _make_fake_chem():
    def MolFromSmiles(smi):
        s = (smi or "").strip()
        if not s or "INVALID" in s or s in ("bad_smiles", "header"):
            return None
        return _FakeMol(s, is_smarts=False)

    def MolFromSmarts(sma):
        s = (sma or "").strip()
        if not s:
            return None
        return _FakeMol(s, is_smarts=True)

    def Mol(mol_obj):
        clone = _FakeMol(mol_obj.text)
        clone._atoms = [_FakeAtom(a.GetSymbol(), a.GetIdx(), a.GetAtomMapNum()) for a in mol_obj.GetAtoms()]
        return clone

    def MolToSmiles(mol_obj):
        cleaned = re.sub(r"\[([A-Z][a-z]?)[H\d]*:\d+\]", r"\1", mol_obj.text)
        return cleaned

    def MolToInchiKey(mol_obj):
        canon = MolToSmiles(mol_obj)
        tag = re.sub(r"[^A-Za-z0-9]", "", canon).upper().ljust(14, "X")[:14]
        return f"{tag}-UHFFFAOYSA-N"

    def FindMolChiralCenters(mol_obj, includeUnassigned=True):
        centers = []
        for idx, m in enumerate(re.finditer(r"\[C@@?H\]", mol_obj.text)):
            centers.append((idx, "R" if "@@" in m.group(0) else "S"))
        if includeUnassigned and "CHIRAL_UNASSIGNED" in mol_obj.text:
            centers.append((0, "?"))
        return centers

    def MolFragmentToSmiles(mol_obj, atomsToUse):
        return "c1ncsc1" if "c1ncsc1" in mol_obj.text else mol_obj.text[:4]

    return SimpleNamespace(
        MolFromSmiles=MolFromSmiles,
        MolFromSmarts=MolFromSmarts,
        Mol=Mol,
        MolToSmiles=MolToSmiles,
        MolToInchiKey=MolToInchiKey,
        FindMolChiralCenters=FindMolChiralCenters,
        MolFragmentToSmiles=MolFragmentToSmiles,
    )


class _FakeBaseModel:
    """Minimal Pydantic-like BaseModel stub supporting class field defaults and model_dump()."""

    def __init__(self, **kwargs):
        annotations = {}
        for cls in reversed(self.__class__.__mro__):
            annotations.update(getattr(cls, "__annotations__", {}))
        for field_name in annotations:
            if field_name in kwargs:
                setattr(self, field_name, kwargs[field_name])
            elif hasattr(self.__class__, field_name):
                setattr(self, field_name, getattr(self.__class__, field_name))

    def model_dump(self):
        return dict(self.__dict__)


def _load_cell2_namespace(client=None):
    tree = ast.parse(cell_source(2))
    body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    ns = {
        "os": os,
        "json": json,
        "logging": __import__("logging"),
        "BaseModel": _FakeBaseModel,
        "Chem": _make_fake_chem(),
        "genai": SimpleNamespace(Client=lambda **_: client),
        "types": SimpleNamespace(),
    }
    if client is not None:
        ns["_GEMINI_CLIENT"] = client
    exec(compile(ast.Module(body=body, type_ignores=[]), str(NOTEBOOK), "exec"), ns)
    return ns


class NotebookStructureAndGuardrailTests(unittest.TestCase):
    def test_valid_notebook_json_and_metadata(self):
        self.assertEqual(NB_DATA["nbformat"], 4)
        self.assertEqual(len(CELLS), 8)
        expected_titles = {
            1: "# @title 1. Install Dependencies & Download Models",
            2: "# @title 2. Core Definitions: SMARTS Library, Reaction Classifiers & Gemini Helpers",
            3: "# @title 3. Initialize AiZynthFinder Engine",
            4: "# @title 4. Interactive Target Molecule Sketcher (JSME) + Multi-Identifier Resolver",
            5: "# @title 5. Configure Search & Route Ranking Settings",
            6: "# @title 5B. Launch Retrosynthesis Search",
            7: "# @title Chat with Gemini about the retrosynthesis routes",
        }
        for idx, title in expected_titles.items():
            self.assertEqual(CELLS[idx]["cell_type"], "code")
            if idx not in (5, 6):
                self.assertEqual(CELLS[idx]["metadata"].get("cellView"), "form")
            self.assertTrue(cell_source(idx).startswith(title), f"Cell {idx} missing title")

    def test_all_python_cells_pass_ast_parse(self):
        for idx in range(1, 8):
            raw = cell_source(idx)
            cleaned = "\n".join(
                re.sub(r"^(\s*)!.*", r"\1pass", line)
                for line in raw.splitlines()
            )
            ast.parse(cleaned, filename=f"cell_{idx}.py")

    def test_cell1_and_clean_literals_guardrails(self):
        self.assertIn("paretoset", cell_source(1))
        self.assertIn("aizynthfinder", cell_source(1))
        for idx in range(1, 8):
            src = cell_source(idx)
            self.assertNotIn("[cite:", src)
            if idx != 7:
                self.assertNotIn("tommbk50-hub/Cheminformatics#", src)

    def test_cell4_resolver_functions_preserved_and_ui_drawer_present(self):
        src4 = cell_source(4)
        tree4 = ast.parse(src4)
        fns = {n.name: n for n in tree4.body if isinstance(n, ast.FunctionDef)}
        self.assertIn("_fetch_pubchem_smiles", fns)
        self.assertIn("_resolve_chemical_identifier", fns)
        self.assertEqual([a.arg for a in fns["_fetch_pubchem_smiles"].args.args], ["namespace", "identifier"])
        self.assertEqual([a.arg for a in fns["_resolve_chemical_identifier"].args.args], ["query_str"])
        self.assertIn("<details", src4)
        self.assertIn("frozen-atoms-input", src4)
        self.assertIn("custom-stock-input", src4)
        self.assertIn("📦 Load Lab Stock", src4)

    def test_cell6_css_preserved_and_quick_chips_added(self):
        src6 = cell_source(7)
        self.assertIn(".retro-chat-box {", src6)
        self.assertIn(".retro-quick-btns {", src6)
        self.assertIn("⚡ Re-rank for Process Scale-Up", src6)
        self.assertIn("Generate 5.0g Stoichiometry", src6)
        self.assertIn("Route #1", src6)

    def test_search_only_runs_from_launch_cell(self):
        tree5 = ast.parse(cell_source(5))
        self.assertFalse(any(
            isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
            and isinstance(n.value.func, ast.Name) and n.value.func.id == "run_retrosynthesis_pipeline"
            for n in tree5.body
        ))
        self.assertIn("run_retrosynthesis_pipeline(", cell_source(6))
        self.assertEqual(len([
            n for n in ast.parse(cell_source(7)).body
            if isinstance(n, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "AGENT_PYTHON_TOOLS" for t in n.targets
            )
        ]), 1)


class Pass1ChemistryCriticsTests(unittest.TestCase):
    def test_step_classification_9_field_defaults(self):
        ns = _load_cell2_namespace()
        step = ns["StepClassification"](
            forward_smiles="A >> B",
            reaction_name="Suzuki-Miyaura Cross-Coupling",
            typical_conditions="Pd(PPh3)4, K2CO3",
        ).model_dump()
        self.assertEqual(
            set(step.keys()),
            {
                "forward_smiles",
                "reaction_name",
                "typical_conditions",
                "selectivity_warning",
                "protecting_group_note",
                "safety_green_score",
                "greener_solvent_swap",
                "is_chiral_transformation",
                "stereochem_note",
            },
        )
        self.assertEqual(step["selectivity_warning"], "None")
        self.assertEqual(step["protecting_group_note"], "None")
        self.assertEqual(step["safety_green_score"], 4)
        self.assertEqual(step["greener_solvent_swap"], "None")
        self.assertFalse(step["is_chiral_transformation"])
        self.assertEqual(step["stereochem_note"], "Achiral / Racemic")

    def test_competing_nucleophiles_and_protecting_group_choreography(self):
        ns = _load_cell2_namespace()
        info = ns["rdkit_classify_step"]("NCCO.CC(=O)Cl >> CC(=O)NCCO")
        self.assertIsInstance(info, dict)
        self.assertIn("Acylation", info["reaction_name"])
        self.assertNotEqual(info["selectivity_warning"], "None")
        self.assertIn("Competing nucleophiles", info["selectivity_warning"])
        self.assertIn("Boc", info["protecting_group_note"])
        self.assertIn("EtOAc", info["greener_solvent_swap"])

    def test_safety_green_scoring_stannane_azide_and_green_deprotection(self):
        ns = _load_cell2_namespace()
        stille = ns["rdkit_classify_step"]("c1ccc(Sn(CCCC)(CCCC)CCCC)cc1.Brc1ccccc1 >> c1ccc(-c2ccccc2)cc1")
        self.assertEqual(stille["reaction_name"], "Stille Cross-Coupling")
        self.assertEqual(stille["safety_green_score"], 1)
        self.assertIn("organotin", stille["greener_solvent_swap"])

        click = ns["rdkit_classify_step"]("CCN=[N+]=[N-].C#Cc1ccccc1 >> c1ncnn1")
        self.assertLessEqual(click["safety_green_score"], 2)
        self.assertIn("azide", click["greener_solvent_swap"].lower())

        cbz_deprot = ns["rdkit_classify_step"]("CNC(=O)OCc1ccccc1 >> CN")
        self.assertEqual(cbz_deprot["reaction_name"], "N-Cbz Deprotection (Hydrogenolysis)")
        self.assertEqual(cbz_deprot["safety_green_score"], 5)
        self.assertIn("N-Cbz", cbz_deprot["protecting_group_note"])

    def test_chiral_center_creation_detected(self):
        ns = _load_cell2_namespace()
        heur = ns["analyze_step_heuristics"]("CC(=O)c1ccccc1.NCC >> C[C@H](NCC)c1ccccc1")
        self.assertTrue(heur["is_chiral_transformation"])
        self.assertIn("Noyori", heur["stereochem_note"])

    def test_batch_classify_with_gemini_returns_dicts_and_handles_offline_fallback(self):
        # Offline (no client)
        ns_off = _load_cell2_namespace(client=None)
        with patch.dict(os.environ, {}, clear=True):
            cache_off = ns_off["batch_classify_with_gemini"](["A >> B"], use_gemini=True)
        self.assertEqual(cache_off, {})

        # Online mocked structured JSON
        payload = {
            "reactions": [
                {
                    "forward_smiles": "Brc1ccccc1.OB(O)c1ccccc1 >> c1ccc(-c2ccccc2)cc1",
                    "reaction_name": "Suzuki-Miyaura Cross-Coupling",
                    "typical_conditions": "Pd(dppf)Cl2, K2CO3, 2-MeTHF/H2O, 80 °C",
                    "selectivity_warning": "None",
                    "protecting_group_note": "None",
                    "safety_green_score": 4,
                    "greener_solvent_swap": "None",
                    "is_chiral_transformation": False,
                    "stereochem_note": "Achiral / Racemic",
                }
            ]
        }
        fake_client = SimpleNamespace(
            models=SimpleNamespace(
                generate_content=Mock(return_value=SimpleNamespace(text=json.dumps(payload)))
            )
        )
        ns_on = _load_cell2_namespace(client=fake_client)
        smi = "Brc1ccccc1.OB(O)c1ccccc1 >> c1ccc(-c2ccccc2)cc1"
        cache_on = ns_on["batch_classify_with_gemini"]([smi], use_gemini=True)
        self.assertIn(smi, cache_on)
        self.assertIsInstance(cache_on[smi], dict)
        self.assertEqual(cache_on[smi]["reaction_name"], "Suzuki-Miyaura Cross-Coupling")
        self.assertEqual(fake_client.models.generate_content.call_args.kwargs["model"], "gemini-3.8-flash")


class Pass2CustomStockAndAtomLockingTests(unittest.TestCase):
    def test_register_custom_lab_stock_loads_inchikeys_and_selects_zinc_plus_custom(self):
        tree3 = ast.parse(cell_source(3))
        fn_nodes = [
            n for n in tree3.body
            if isinstance(n, (ast.Assign, ast.FunctionDef))
        ]
        loaded_paths = []
        fake_stock = SimpleNamespace(load=Mock(), select=Mock())
        fake_finder = SimpleNamespace(stock=fake_stock)

        def fake_inchi_query(path):
            loaded_paths.append(path)
            with open(path, encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            return SimpleNamespace(rows=rows)

        ns = {
            "os": os,
            "csv": csv,
            "tempfile": tempfile,
            "logging": __import__("logging"),
            "Chem": _make_fake_chem(),
            "InMemoryInchiKeyQuery": fake_inchi_query,
            "finder": fake_finder,
        }
        exec(compile(ast.Module(body=fn_nodes, type_ignores=[]), str(NOTEBOOK), "exec"), ns)

        with redirect_stdout(io.StringIO()):
            res = ns["register_custom_lab_stock"](["smiles", "CCO", "c1ncsc1", "INVALID_SMI"], stock_name="lab_inventory")

        self.assertEqual(res["loaded_count"], 2)
        self.assertEqual(res["invalid_count"], 1)
        self.assertEqual(res["selected_stocks"], ["zinc", "lab_inventory"])
        fake_stock.load.assert_called_once()
        fake_stock.select.assert_called_once_with(["zinc", "lab_inventory"])
        self.assertEqual(len(fake_finder.custom_lab_stock_inchikeys), 2)
        if loaded_paths and os.path.exists(loaded_paths[0]):
            os.remove(loaded_paths[0])

    def test_set_target_smiles_locks_smarts_and_highlights_atoms(self):
        tree4 = ast.parse(cell_source(4))
        fn_nodes = [
            n for n in tree4.body
            if isinstance(n, ast.FunctionDef) and n.name in ("_resolve_frozen_atoms", "_set_target_smiles", "_load_custom_stock_callback")
        ]
        fake_finder = SimpleNamespace(target_smiles="")
        draw_mock = Mock(return_value="IMG")
        stock_reg_mock = Mock(return_value={"stock_name": "lab_inventory", "loaded_count": 2})

        ns = {
            "re": re,
            "Chem": _make_fake_chem(),
            "Draw": SimpleNamespace(MolToImage=draw_mock),
            "IPython": SimpleNamespace(display=SimpleNamespace(JSON=lambda d: d)),
            "display": Mock(),
            "finder": fake_finder,
            "register_custom_lab_stock": stock_reg_mock,
        }
        exec(compile(ast.Module(body=fn_nodes, type_ignores=[]), str(NOTEBOOK), "exec"), ns)

        with redirect_stdout(io.StringIO()):
            ns["_set_target_smiles"]("Cc1ncsc1C(=O)O", frozen_indices="c1ncsc1")

        self.assertEqual(fake_finder.target_smiles, "Cc1ncsc1C(=O)O")
        self.assertTrue(len(fake_finder.CustomFrozenAtoms) > 0)
        self.assertEqual(fake_finder.CustomFrozenSmarts, "c1ncsc1")
        self.assertIn("highlightAtoms", draw_mock.call_args.kwargs)

        cb_res = ns["_load_custom_stock_callback"]("SMILES,Name\nCCO,Ethanol\nc1ncsc1,Thiazole")
        self.assertEqual(cb_res["loaded_count"], 2)
        stock_reg_mock.assert_called_once_with(["CCO", "c1ncsc1"], stock_name="lab_inventory")


class Pass3PipelinePricingParetoAndELNTests(unittest.TestCase):
    def _build_cell5_namespace(self, routes, custom_lab_smis=None):
        ns = _load_cell2_namespace(client=None)
        fake_finder = SimpleNamespace(
            target_smiles="Cc1ncsc1C(=O)O",
            CustomFrozenAtoms=[],
            CustomFrozenSmarts=None,
            custom_lab_stock_inchikeys=set(),
            custom_lab_stock_smiles=set(custom_lab_smis or []),
            config=SimpleNamespace(search=SimpleNamespace(max_transforms=6, time_limit=120)),
            stock=SimpleNamespace(_items={}),
            prepare_tree=Mock(),
            tree_search=Mock(),
            build_routes=Mock(),
            extract_statistics=Mock(return_value={"is_solved": True}),
            routes=routes,
        )
        ns.update({
            "_html": html,
            "_md": SimpleNamespace(markdown=lambda txt, **_: f"<div>{txt}</div>"),
            "np": SimpleNamespace(asarray=lambda x, **_: x),
            "display": Mock(),
            "HTML": lambda s: s,
            "Descriptors": SimpleNamespace(ExactMolWt=lambda m: float(max(50, m.GetNumHeavyAtoms() * 14))),
            "_paretoset_fn": None,
            "finder": fake_finder,
            "CUSTOM_LAB_STOCK_INCHIKEYS": set(),
            "CUSTOM_LAB_STOCK_SMILES": set(custom_lab_smis or []),
            "_chem_mol": ns["Chem"].MolFromSmiles,
            "mol_or_rxn_to_svg_html": lambda *a, **k: '<div class="chem-card"><svg></svg></div>',
            "rdkit_predict_workup_properties": lambda *a: {"product": {"clogp": 1.0, "tpsa": 50, "hbd": 1, "hba": 2}, "advice": ["Neutral wash."]},
            "rdkit_predict_lcms_ions": lambda *a: [{
                "role": "product", "smiles": "CCO", "[M+H]+": 47.05, "[M+Na]+": 69.03,
                "[M+NH4]+": 64.07, "[M+MeCN+H]+": 88.08, "[M-H]-": 45.03,
                "isotope_note": "None",
            }],
        })
        tree5 = ast.parse(cell_source(5))
        widget_start = next(n.lineno for n in tree5.body if isinstance(n, ast.Import)
                            and any(alias.name == "ipywidgets" for alias in n.names))
        body5 = [
            n for n in tree5.body
            if n.lineno < widget_start and not isinstance(n, (ast.Import, ast.ImportFrom, ast.Try))
            and not (isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "LATEST_RETRO_RESULTS" and isinstance(n.value, ast.Call))
        ]
        exec(compile(ast.Module(body=body5, type_ignores=[]), str(NOTEBOOK), "exec"), ns)
        return ns

    def _make_mock_route(self, state_score, retro_steps, leaf_specs, is_solved=True):
        rxn_nodes = [
            SimpleNamespace(
                reaction_smiles=lambda s=rs: s,
                metadata={"template_code": f"T{i}", "policy_probability": 0.85},
            )
            for i, rs in enumerate(retro_steps, 1)
        ]
        leaves = [
            SimpleNamespace(smiles=smi, in_stock=in_st)
            for smi, in_st in leaf_specs
        ]
        rxn_tree = SimpleNamespace(
            is_solved=is_solved,
            reactions=lambda: rxn_nodes,
            leafs=lambda: leaves,
            in_stock=lambda leaf: leaf.in_stock,
            to_image=lambda: "TREE_IMG",
            depth=lambda node: 1,
        )
        return {"score": {"state score": state_score}, "reaction_tree": rxn_tree}

    def test_starting_materials_pareto_ranking_exclusion_and_eln(self):
        # Route A: 1-step Stille (high state score 0.92, toxic Sn -> safety 1, Pd catalyst)
        route_stille = self._make_mock_route(
            0.92,
            ["Cc1ncsc1C(=O)O >> c1ccc(Sn(CCCC)(CCCC)CCCC)cc1.Brc1ncsc1"],
            [("c1ccc(Sn(CCCC)(CCCC)CCCC)cc1", True), ("Brc1ncsc1", True)],
        )
        # Route B: 2-step Hantzsch + Ester hydrolysis (state score 0.84, green/safe -> safety 4-5, Pd-free, uses lab inventory leaf)
        route_green = self._make_mock_route(
            0.84,
            [
                "Cc1ncsc1C(=O)O >> Cc1ncsc1C(=O)OC",
                "Cc1ncsc1C(=O)OC >> CC(=S)N.ClCC(=O)OC",
            ],
            [("CC(=S)N", True), ("ClCC(=O)OC", True)],
        )

        ns = self._build_cell5_namespace(
            [route_stille, route_green],
            custom_lab_smis={"CC(=S)N", "ClCC(=O)OC"},
        )

        # MedChem fast prefers 1-step high state-score route
        res_fast = ns["run_retrosynthesis_pipeline"](
            ranking_mode="medchem_fast", target_scale_grams=2.5, use_gemini=False, render_output=False
        )
        self.assertEqual(len(res_fast["routes"]), 2)
        self.assertEqual(res_fast["routes"][0]["step_count"], 1)
        self.assertTrue(any(r["is_pareto_optimal"] for r in res_fast["routes"]))

        # Green/Process mode promotes the tin-free, lab-inventory route to #1
        res_green = ns["run_retrosynthesis_pipeline"](
            ranking_mode="green_safe", target_scale_grams=2.5, use_gemini=False, render_output=False
        )
        self.assertEqual(res_green["routes"][0]["step_count"], 2)
        self.assertEqual(res_green["routes"][0]["starting_materials"]["overall_cost_tier"], "$0 On-Shelf")

        # Excluding Palladium removes Stille route cleanly
        res_no_pd = ns["run_retrosynthesis_pipeline"](
            ranking_mode="balanced",
            exclude_reaction_names=["Palladium"],
            use_gemini=False,
            render_output=False,
        )
        self.assertEqual(len(res_no_pd["routes"]), 1)
        self.assertEqual(res_no_pd["filtered_out_routes"], 1)

        # ELN stoichiometry Markdown table scales accurately for 2.5g target
        eln_md = ns["generate_eln_markdown"](route_idx=1, target_scale_grams=2.5, assumed_step_yield=0.80)
        self.assertIn("Scaled Stoichiometry Table", eln_md)
        self.assertIn("2.50 g", eln_md)
        self.assertIn("Limiting Reactant", eln_md)
        self.assertIn("2D Structure", eln_md)
        self.assertIn("<svg>", eln_md)
        self.assertIn("[M+Na]+", eln_md)
        self.assertIn("cLogP", eln_md)

    def test_preset_switches_update_all_typeable_inputs_and_finder(self):
        class FakeText:
            def __init__(self, value=None, **kwargs):
                self._value = value
                self._handlers = []
                self.description = kwargs.get("description")

            @property
            def value(self):
                return self._value

            @value.setter
            def value(self, new):
                self._value = new
                for callback in self._handlers:
                    callback({"owner": self, "new": new})

            def observe(self, callback, names):
                self._handlers.append(callback)

        class FakeInt(FakeText):
            pass

        class FakeFloat(FakeText):
            pass

        nodes = ast.parse(cell_source(5)).body
        start = next(n.lineno for n in nodes if isinstance(n, ast.Assign) and
                     any(isinstance(t, ast.Name) and t.id == "_SEARCH_PRESETS" for t in n.targets))
        finder = SimpleNamespace(config=SimpleNamespace(
            search=SimpleNamespace(max_transforms=6, time_limit=120, iteration_limit=220,
                                   cutoff_cumulative=0.995),
            post_processing=SimpleNamespace(max_routes=10, min_routes=5),
        ))
        defaults = dict(max_transforms=6, time_limit=120, iteration_limit=220,
                        cutoff_cumulative=0.995, max_routes_to_extract=10, target_scale_grams=1.0)
        ns = dict(defaults, finder=finder, ranking_mode="green_safe",
                  RETRO_SEARCH_PARAMS=dict(defaults), display=Mock(),
                  widgets=SimpleNamespace(BoundedIntText=FakeInt, BoundedFloatText=FakeFloat,
                                          Dropdown=FakeText, VBox=lambda children: children))
        exec(compile(ast.Module(body=[n for n in nodes if n.lineno >= start],
                                type_ignores=[]), str(NOTEBOOK), "exec"), ns)
        self.assertIsInstance(ns["config_widgets"]["max_transforms"], FakeInt)
        self.assertIsInstance(ns["config_widgets"]["target_scale_grams"], FakeFloat)
        ns["ranking_mode_widget"].value = "process_scaleup"
        self.assertEqual(ns["RETRO_SEARCH_PARAMS"]["time_limit"], 300)
        self.assertEqual(ns["RETRO_SEARCH_PARAMS"]["target_scale_grams"], 10.0)
        self.assertEqual(finder.config.search.iteration_limit, 350)
        self.assertEqual(finder.config.post_processing.max_routes, 15)
        ns["config_widgets"]["max_transforms"].value = 7
        self.assertEqual(ns["RETRO_SEARCH_PARAMS"]["max_transforms"], 7)
        self.assertEqual(finder.config.search.max_transforms, 7)


class Pass4AgenticChatAndToolsTests(unittest.TestCase):
    def _load_cell6_namespace(self, latest_results=None, gemini_lookup=None):
        tree6 = ast.parse(cell_source(7))
        body6 = [
            n for n in tree6.body
            if not isinstance(n, (ast.Import, ast.ImportFrom))
            and not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call))
        ]
        types_stub = SimpleNamespace(
            Content=lambda **kw: SimpleNamespace(**kw),
            Part=SimpleNamespace(from_text=lambda text: SimpleNamespace(text=text)),
            Tool=lambda **kw: SimpleNamespace(**kw),
            GoogleSearch=lambda: "GOOGLE_SEARCH_OBJ",
            GenerateContentConfig=lambda **kw: SimpleNamespace(**kw),
            AutomaticFunctionCallingConfig=lambda **kw: SimpleNamespace(**kw),
        )
        ns = {
            "os": os,
            "re": re,
            "html": html,
            "inspect": inspect,
            "urllib": __import__("urllib"),
            "requests": SimpleNamespace(get=Mock(side_effect=OSError("offline"))),
            "markdown": SimpleNamespace(markdown=lambda text, **_: f"<p>{text}</p>"),
            "IPython": SimpleNamespace(display=SimpleNamespace(JSON=lambda d: d)),
            "output": SimpleNamespace(register_callback=Mock()),
            "userdata": SimpleNamespace(get=Mock(side_effect=KeyError("no colab secret"))),
            "genai": SimpleNamespace(Client=Mock()),
            "types": types_stub,
            "finder": SimpleNamespace(target_smiles="Cc1ncsc1C(=O)O", routes=[]),
            "LATEST_RETRO_RESULTS": latest_results or {},
            "gemini_lookup": gemini_lookup or {},
            "run_retrosynthesis_pipeline": Mock(return_value={
                "target_smiles": "Cc1ncsc1C(=O)O",
                "ranking_mode": "process_scaleup",
                "is_solved": True,
                "routes": [{
                    "composite_score": 0.91,
                    "state_score": 0.85,
                    "is_pareto_optimal": True,
                    "is_solved": True,
                    "step_count": 2,
                    "avg_safety_score": 4.5,
                    "selectivity_risk_count": 0,
                    "starting_materials": {"overall_cost_tier": "$ Low", "in_stock_count": 2, "total_leaves": 2, "precursors": []},
                    "steps": [{"step_index": 1, "reaction_name": "Hantzsch Thiazole Synthesis", "template_code": "T1", "forward_smiles": "A >> B", "typical_conditions": "EtOH", "safety_green_score": 5}],
                }],
            }),
            "generate_eln_markdown": Mock(return_value="| Step | Mass |\n|---|---|\n| 1 | 5.00 g |"),
        }
        exec(compile(ast.Module(body=body6, type_ignores=[]), str(NOTEBOOK), "exec"), ns)
        return ns

    def test_context_builder_uses_dict_keys_without_tuple_indexing_error(self):
        # Fallback branch with dict-valued gemini_lookup
        rxn_node = SimpleNamespace(reaction_smiles=lambda: "P >> R1.R2", metadata={"template_code": "T99"})
        rxn_tree = SimpleNamespace(is_solved=True, reactions=lambda: [rxn_node])
        ns = self._load_cell6_namespace(
            latest_results={},
            gemini_lookup={"R1.R2 >> P": {"reaction_name": "Suzuki-Miyaura Cross-Coupling", "typical_conditions": "Pd"}},
        )
        ns["finder"].routes = [{"score": {"state score": 0.88}, "reaction_tree": rxn_tree}]
        ctx = ns["_build_retrosynthesis_context"]()
        self.assertIn("Suzuki-Miyaura Cross-Coupling", ctx)

    def test_tool_calls_routed_separately_from_google_search(self):
        ns = self._load_cell6_namespace()

        # 1. Action request routes to Python tools (NO GoogleSearch mixed in)
        tool_call_resp = SimpleNamespace(
            text="",
            function_calls=[
                SimpleNamespace(
                    name="tool_rerun_retrosynthesis",
                    args={"ranking_mode": "process_scaleup", "exclude_reaction_keyword": "Palladium"},
                )
            ],
            candidates=[],
        )
        synth_resp = SimpleNamespace(text="Re-ranked routes without Palladium.", function_calls=None, candidates=[])
        recorded_configs = []

        def fake_generate(model, contents, config):
            recorded_configs.append(config)
            return tool_call_resp if len(recorded_configs) == 1 else synth_resp

        ns["genai"].Client = Mock(return_value=SimpleNamespace(models=SimpleNamespace(generate_content=fake_generate)))
        with patch.dict(os.environ, {"KDD_GEMINI_API_KEY": "dummy-test-key"}):
            out = ns["_send_chat_message"](
                "Can we re-run without Palladium in process_scaleup mode?",
                "flash-3.8",
                True,
            )

        self.assertIn("Agent Tool Executed", out["html"])
        self.assertEqual(recorded_configs[0].tools, ns["AGENT_PYTHON_TOOLS"])
        ns["run_retrosynthesis_pipeline"].assert_called_once()

        # 2. Literature request routes to GoogleSearch ONLY (no Python functions mixed in)
        search_configs = []
        lit_resp = SimpleNamespace(
            text="Published in J. Med. Chem. 2022.",
            function_calls=None,
            candidates=[
                SimpleNamespace(
                    grounding_metadata=SimpleNamespace(
                        grounding_chunks=[
                            SimpleNamespace(web=SimpleNamespace(uri="https://example.org/paper", title="J. Med. Chem."))
                        ]
                    )
                )
            ],
        )
        ns["genai"].Client = Mock(
            return_value=SimpleNamespace(
                models=SimpleNamespace(
                    generate_content=lambda model, contents, config: search_configs.append(config) or lit_resp
                )
            )
        )
        with patch.dict(os.environ, {"KDD_GEMINI_API_KEY": "dummy-test-key"}):
            lit_out = ns["_send_chat_message"](
                "Perform a literature search for the exact reaction conditions for Route #1.",
                "flash-3.8",
                True,
            )

        self.assertEqual(len(search_configs), 1)
        self.assertEqual(len(search_configs[0].tools), 1)
        self.assertEqual(search_configs[0].tools[0].google_search, "GOOGLE_SEARCH_OBJ")
        self.assertIn("Grounded Literature", lit_out["html"])

    def test_offline_fallback_executes_local_eln_recipe_without_api_key(self):
        ns = self._load_cell6_namespace()
        with patch.dict(os.environ, {}, clear=True):
            res = ns["_send_chat_message"](
                "Generate a 5.0g Stoichiometry & ELN Table for Route #1",
                "flash-3.8",
                True,
            )
        self.assertEqual(res["model"], "local-rdkit-agent")
        self.assertIn("5.00 g", res["html"])
        ns["generate_eln_markdown"].assert_called_once_with(
            route_idx=1, target_scale_grams=5.0, assumed_step_yield=0.8
        )

    def test_eight_tools_and_offline_chemistry_quick_actions(self):
        ns = self._load_cell6_namespace(latest_results={
            "routes": [{"steps": [{"forward_smiles": "CCO >> CC=O"}]}],
        })
        self.assertEqual(len(ns["AGENT_PYTHON_TOOLS"]), 8)
        ns["rdkit_enumerate_byproducts"] = Mock(return_value=[{
            "origin": "target product", "smiles": "CC=O", "formula": "C2H4O",
            "exact_mw": 44.0262, "svg_html": '<div class="chem-card"><svg></svg></div>',
        }])
        ns["rdkit_predict_lcms_ions"] = Mock(return_value=[{
            "role": "target product", "smiles": "CC=O", "formula": "C2H4O",
            "exact_mw": 44.0262, "[M+H]+": 45.0335, "[M+Na]+": 67.0154,
            "[M+NH4]+": 62.0600, "[M+MeCN+H]+": 86.0600, "[M-H]-": 43.0189,
            "isotope_note": "No Cl/Br M+2 signature",
        }])
        ns["mol_or_rxn_to_svg_html"] = lambda *a, **k: '<div class="chem-card"><svg></svg></div>'
        ns["_CHEM_SVG_CARDS"] = {'<div class="chem-card"><svg></svg></div>'}
        with patch.dict(os.environ, {}, clear=True):
            output = ns["_send_chat_message"](
                "Predict byproducts and LC-MS ions for Route #1 Step 1", "flash-3.8", True
            )
        self.assertIn("<svg>", output["html"])
        self.assertIn("45.0335", output["html"])
        self.assertIn("No Cl/Br", output["html"])
        escaped = ns["_render_chemistry_markdown"]('<script>alert(1)</script> `CC=O`')
        self.assertNotIn("<script>", escaped)
        self.assertNotIn("javascript:", ns["_render_chemistry_markdown"]("[click](javascript:alert(1))"))
        self.assertNotIn("https://example.org", ns["_render_chemistry_markdown"]("![pixel](https://example.org/pixel)"))


@unittest.skipUnless(importlib.util.find_spec("rdkit"), "real RDKit not installed")
class RDKitNativeChemistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdChemReactions, rdMolDescriptors
        from rdkit.Chem.Draw import rdMolDraw2D
        nodes = [
            node for node in ast.parse(cell_source(2)).body
            if isinstance(node, ast.FunctionDef) and node.name in (
                "has_fg", "_chem_mol", "mol_or_rxn_to_svg_html",
                "rdkit_enumerate_byproducts", "rdkit_predict_lcms_ions",
                "rdkit_predict_workup_properties",
            ) or isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "RAW_SMARTS" for t in node.targets
            )
        ]
        cls.ns = {"Chem": Chem, "Descriptors": Descriptors, "rdChemReactions": rdChemReactions,
                  "rdMolDescriptors": rdMolDescriptors, "rdMolDraw2D": rdMolDraw2D,
                  "_chem_html": html, "_CHEM_SVG_CARDS": set()}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(NOTEBOOK), "exec"), cls.ns)
        cls.ns["MEDCHEM_SMARTS"] = {
            key: [Chem.MolFromSmarts(sma) for sma in patterns.split("|")]
            for key, patterns in cls.ns["RAW_SMARTS"].items()
        }

    def test_validated_svg_and_side_product_ions(self):
        svg = self.ns["mol_or_rxn_to_svg_html"]("c1ncc[nH]1 >> c1ncc[nH]1", label="<script>bad</script>")
        self.assertIn("<svg", svg)
        self.assertNotIn("<script>", svg)
        self.assertIn("Invalid", self.ns["mol_or_rxn_to_svg_html"]("invalid"))
        acyl = "CC(=O)Cl.NCCO >> CC(=O)NCCO"
        products = self.ns["rdkit_enumerate_byproducts"](acyl)
        self.assertTrue(any(p["origin"] == "O-acylation" and "<svg" in p["svg_html"] for p in products))
        sulfur = self.ns["rdkit_enumerate_byproducts"]("CC(=O)Cl.NCCS >> CC(=O)NCCS")
        self.assertTrue(any(p["origin"] == "S-acylation" for p in sulfur))
        ions = self.ns["rdkit_predict_lcms_ions"](acyl)
        self.assertTrue(any(p["role"] == "O-acylation" for p in ions))
        self.assertTrue(all(p["[M+H]+"] > p["exact_mw"] > p["[M-H]-"] for p in ions))

    def test_suzuki_regioisomers_and_acid_workup(self):
        coupling = "Brc1cc(Cl)ccc1.OB(O)c1ccccc1 >> Brc1cccc(-c2ccccc2)c1"
        products = self.ns["rdkit_enumerate_byproducts"](coupling)
        self.assertGreaterEqual(len([p for p in products if p["origin"] == "halide regioisomer"]), 1)
        self.assertTrue(any(p["origin"] == "protodeboronation" for p in products))
        acid = self.ns["rdkit_predict_workup_properties"]("CC(=O)OC >> CC(=O)O")
        self.assertTrue(acid["product"]["acid"])
        self.assertIn("NaHCO3", " ".join(acid["advice"]))
        bromide = self.ns["rdkit_predict_lcms_ions"](coupling)
        self.assertTrue(any("Br" in item["isotope_note"] for item in bromide))

    def test_step_swap_sanitizes_and_updates_route_score(self):
        score = next(node for node in ast.parse(cell_source(5)).body
                     if isinstance(node, ast.FunctionDef) and node.name == "compute_composite_route_score")
        core = ast.parse(cell_source(2)).body
        extra = [node for node in core if isinstance(node, ast.FunctionDef)
                 and node.name in ("analyze_step_heuristics", "_match_rdkit_rule") or
                 isinstance(node, ast.Assign) and any(
                     isinstance(t, ast.Name) and t.id == "RDKIT_REACTION_RULES" for t in node.targets)]
        exec(compile(ast.Module(body=extra + [score], type_ignores=[]), str(NOTEBOOK), "exec"), self.ns)
        functions = [node for node in ast.parse(cell_source(7)).body
                     if isinstance(node, ast.FunctionDef) and node.name in (
                         "_route_step_forward_smiles", "tool_modify_and_validate_step")]
        namespace = dict(self.ns, LATEST_RETRO_RESULTS={
            "ranking_mode": "green_safe",
            "routes": [{"steps": [{
                "forward_smiles": "C[Sn](C)(C)c1ccccc1.Brc1ccccc1 >> c1ccc(-c2ccccc2)cc1",
                "reaction_name": "Stille Cross-Coupling", "typical_conditions": "Pd, DMF",
                "safety_green_score": 1,
            }], "step_count": 1, "state_score": 0.8, "avg_safety_score": 1.0,
                "selectivity_risk_count": 0, "composite_score": 0.5}],
        })
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
        result = namespace["tool_modify_and_validate_step"](1, 1, "stille_to_suzuki")
        self.assertIn("<svg", result)
        step = namespace["LATEST_RETRO_RESULTS"]["routes"][0]["steps"][0]
        self.assertIn("OB(O)", step["forward_smiles"])
        self.assertNotIn("Sn", step["forward_smiles"])
        self.assertIn("K2CO3", step["typical_conditions"])
        self.assertNotEqual(namespace["LATEST_RETRO_RESULTS"]["routes"][0]["composite_score"], 0.5)
        self.assertIn("Error:", namespace["tool_modify_and_validate_step"](1, 1, "custom_reactant", "invalid"))


if __name__ == "__main__":
    unittest.main()
