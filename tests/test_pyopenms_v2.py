import ast
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest


NOTEBOOK = Path(__file__).resolve().parents[1] / "pyOpenMS_v2.ipynb"
CELLS = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
VIEW_SOURCE = "".join(CELLS[7]["source"])


class Vector(list):
    def min(self):
        return min(self)

    def max(self):
        return max(self)

    def __ge__(self, value):
        return Mask(item >= value for item in self)

    def __le__(self, value):
        return Mask(item <= value for item in self)

    def __getitem__(self, index):
        if isinstance(index, (list, Mask)):
            return Vector(item for item, keep in zip(self, index) if keep)
        return super().__getitem__(index)


class Mask(list):
    def __and__(self, other):
        return Mask(left and right for left, right in zip(self, other))


def helpers():
    tree = ast.parse(VIEW_SOURCE)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    namespace = {
        "np": SimpleNamespace(asarray=lambda values, dtype=None: Vector(values),
                              isfinite=lambda values: Mask(math.isfinite(x) for x in values),
                              sum=sum),
        "pd": SimpleNamespace(isna=lambda value: value is None),
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    return namespace


class Spectrum:
    def __init__(self, level, rt, precursor=None, mzs=(), intensities=()):
        self.level, self.rt, self.precursor = level, rt, precursor
        self.mzs, self.intensities = Vector(mzs), Vector(intensities)

    def getMSLevel(self):
        return self.level

    def getRT(self):
        return self.rt

    def getPrecursors(self):
        return [] if self.precursor is None else [SimpleNamespace(getMZ=lambda: self.precursor)]

    def get_peaks(self):
        return self.mzs, self.intensities


class PeakViewTests(unittest.TestCase):
    def test_notebook_is_standalone_and_python_cells_parse(self):
        self.assertEqual(len(CELLS), 19)
        for cell in CELLS:
            if cell["cell_type"] == "code" and not any(
                line.lstrip().startswith(("!", "%")) for line in cell["source"]
            ):
                ast.parse("".join(cell["source"]))
        pipeline = "".join(CELLS[8]["source"])
        self.assertNotIn("assigned_card_offsets", pipeline)
        self.assertNotIn("range=[190, 310]", pipeline)
        self.assertIn("build_peak_explorer(", pipeline)

    def test_rt_bounds_use_data_and_ignore_non_finite_samples(self):
        bounds = helpers()["peak_rt_bounds"]
        self.assertEqual(bounds({"Time": [float("nan"), 42, 12, 24]}), (12.0, 42.0))
        self.assertEqual(bounds({"Time": [8, 8]}), (8.0, 9.0))
        self.assertIsNone(bounds({"Time": []}))

    def test_targets_use_assignment_or_dominant_ion(self):
        targets = helpers()["peak_targets"]
        self.assertEqual(targets({"dominant m/z": "100.123 (50%)"}, {}), [100.123])
        self.assertEqual(targets({"dominant m/z": "100.123 (50%)"},
                                 {"adducts": [{"obs_mz": 101.1}, {"obs_mz": 122.2}]}),
                         [101.1, 122.2])
        self.assertEqual(targets({"dominant m/z": "None"}, {}), [])

    def test_ms2_requires_window_and_precursor_match(self):
        scans = [Spectrum(2, 20, 200), Spectrum(2, 14, 100.001),
                 Spectrum(2, 13, 100), Spectrum(2, 12, None),
                 Spectrum(2, 22, 100), Spectrum(1, 11, 100)]
        row = {"peak start time": 10, "peak end time": 21, "retention_time": 13.2}
        matched = helpers()["matching_ms2"](scans, row, [100], ppm=20)
        self.assertEqual([scan.rt for scan in matched], [13, 14])
        self.assertEqual(helpers()["matching_ms2"](scans, row, []), [])
        self.assertEqual(helpers()["matching_ms2"](scans, {**row, "peak start time": None},
                                                    [100]), [])

    def test_xic_uses_ms1_only_and_ppm_window(self):
        scans = [Spectrum(1, 2, mzs=[99.998, 100, 100.001, 101],
                          intensities=[5, 10, 20, 50]),
                 Spectrum(2, 3, 100, [100], [1000]),
                 Spectrum(1, 4, mzs=[100.003], intensities=[70]),
                 Spectrum(1, 12, mzs=[100], intensities=[100])]
        times, intensities = helpers()["extract_xic"](scans, 100, (0, 10), ppm=20)
        self.assertEqual(times, [2.0, 4.0])
        self.assertEqual(intensities, [35.0, 0.0])
        self.assertEqual(helpers()["extract_xic"](scans, None, (0, 10)), ([], []))


if __name__ == "__main__":
    unittest.main()
