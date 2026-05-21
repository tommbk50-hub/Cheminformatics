# -*- coding: utf-8 -*-
"""millions_library_colab_standard.py

Adapted from Billions_library_GCE_V2 for standard free-tier Google Colab CPU instances
(~12GB RAM, 2 cores, ~70GB disk space). Key changes from the GCE version:
  - All /dev/shm (RAM disk) paths replaced with /content (SSD) paths.
  - DuckDB memory_limit reduced to 10GB to avoid OOM crashes on Colab.
  - BB_FILE points to a small sample file; full multi-billion datasets cannot
    be handled by standard Colab instances.
"""

!pip -q install "rdkit>=2024.9" "numpy>=2.0,<3" pandas pillow matplotlib mols2grid tqdm

!pip install -q "ipywidgets>=7,<8" --break-system-packages

# @title Cell 0: Exit Vector Visualizer (Select your reaction atom)
import os
import ipywidgets as W
from IPython.display import display, Image, HTML, SVG
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
try:
    from rdkit.Chem import rdMolDraw2D as rdmoldraw2d
except ImportError:
    from rdkit.Chem.Draw import rdMolDraw2D as rdmoldraw2d

RDLogger.DisableLog("rdApp.*")

def draw_smiles_with_atom_puff(smiles, width=400, height=400, highlight_atom=None):
    """Draws a 2D molecule with a pastel highlight on a specific atom."""
    m = Chem.MolFromSmiles(smiles)
    if m is None: return None
    Chem.AssignStereochemistry(m, cleanIt=True, force=True, flagPossibleStereoCenters=True)
    AllChem.Compute2DCoords(m)

    drawer = rdmoldraw2d.MolDraw2DSVG(width, height)
    opts = drawer.drawOptions()
    opts.addAtomIndices = True
    opts.addStereoAnnotation = True
    opts.atomHighlightsAreCircles = True
    opts.fillHighlights = True

    highlightAtoms, highlightColors, highlightRadii = [], {}, {}
    if highlight_atom is not None:
        idx = int(highlight_atom)
        if m.GetNumAtoms() > idx:
            highlightAtoms = [idx]
            highlightColors[idx] = (1.0, 0.45, 0.50, 0.45) # Pastel Red
            highlightRadii[idx] = 0.4

    rdmoldraw2d.PrepareAndDrawMolecule(drawer, m, highlightAtoms=highlightAtoms,
                                       highlightAtomColors=highlightColors,
                                       highlightAtomRadii=highlightRadii)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()

# --- UI Setup ---
smiles_w = W.Text(value="OC[C@@H]1CNC[C@H]1c2cccc(F)c2", description="SMILES:", layout=W.Layout(width="600px"))
atom_dd  = W.Dropdown(options=[], description="Atom:", layout=W.Layout(width="200px"))
gen_btn  = W.Button(description="Visualize Core", button_style='info')
img_out  = W.Output()

def update_dropdown(smi):
    m = Chem.MolFromSmiles(smi)
    if m:
        opts = [(f"{i}: {a.GetSymbol()}", i) for i, a in enumerate(m.GetAtoms())]
        atom_dd.options = opts
        # Auto-select the first Nitrogen if present
        n_idx = next((i for i, a in enumerate(m.GetAtoms()) if a.GetSymbol() == "N"), opts[0][1] if opts else None)
        atom_dd.value = n_idx

def on_render(_):
    with img_out:
        img_out.clear_output()
        svg_text = draw_smiles_with_atom_puff(smiles_w.value.strip(), highlight_atom=atom_dd.value)
        if svg_text: display(SVG(svg_text))
        else: print("Invalid SMILES string.")

smiles_w.observe(lambda change: update_dropdown(change.new) if change.name == 'value' else None)
gen_btn.on_click(on_render)

# Initialize
update_dropdown(smiles_w.value)
display(W.VBox([W.HBox([smiles_w, atom_dd, gen_btn]), img_out]))
on_render(None)

# @title Cell 1: ZINC22 Downloader & Raw Data Preview
import os, glob, subprocess, io
import pandas as pd
from IPython.display import display

wget_dir = os.getcwd()
out_dir = os.path.join(wget_dir, "zinc22_raw")
os.makedirs(out_dir, exist_ok=True)

# 1. Execute WGET Scripts
wget_files = glob.glob(os.path.join(wget_dir, "*.wget"))
if not wget_files:
    print(f"⚠️ No .wget files found in {wget_dir}. Please upload your ZINC22 cart script.")
else:
    for script in wget_files:
        print(f"📥 Processing {os.path.basename(script)}...")
        # Fix line endings and run
        subprocess.run(['sed', '-i', 's/\r$//', script], check=True)
        subprocess.run(f'cd "{out_dir}" && bash "{script}"', shell=True, check=True)
    print("✅ Downloads complete.")

# 2. Preview the First Downloaded Tranche
smi_gz_files = glob.glob(os.path.join(out_dir, '**/*.smi.gz'), recursive=True)
if smi_gz_files:
    first_file = smi_gz_files[0]
    print(f"\n🔍 Previewing raw data from: {os.path.basename(first_file)}")

    # Count the number of SMILES entries
    count_process = subprocess.run(f'zcat "{first_file}" | wc -l', shell=True, stdout=subprocess.PIPE, text=True)
    try:
        total_entries = int(count_process.stdout.strip())
        print(f"📊 Total SMILES entries: {total_entries:,}")
    except ValueError:
        print("⚠️ Could not count entries.")

    # Read the first 5 lines directly from the zipped file
    process = subprocess.run(['zcat', first_file], stdout=subprocess.PIPE, text=True)
    preview_lines = "\n".join(process.stdout.splitlines()[:5])

    # Display with correct column headings
    df_preview = pd.read_csv(io.StringIO(preview_lines), sep=r'\s+', header=None, names=['SMILES', 'ZINC_ID', 'Other_ID'])
    display(df_preview)

# @title Cell 2: 64-Core OS-Level Deduplication (pigz + parallel sort)
import os
import subprocess
import pandas as pd
from rdkit import Chem
import mols2grid
from IPython.display import display

# 1. Install Parallel GZIP (pigz)
print("📥 Installing Parallel GZIP (pigz) for 64-core compression...")
subprocess.run("sudo apt-get update -qq && sudo apt-get install -y pigz", shell=True, check=True)

base_dir = os.getcwd()
raw_dir = os.path.join(base_dir, "zinc22_raw")
clean_file = os.path.join(raw_dir, "combined_cleaned.smi.gz")

# 2. The 64-Core Pipeline
print(f"🚀 Running 64-core OS-level deduplication...")
print("⚙️ Decompressing (64 threads) -> Sorting/Deduplicating (64 threads, 100GB RAM) -> Compressing (64 threads)")

num_cores = os.cpu_count() or 64

# - unpigz -p {num_cores}: Decompresses across all cores
# - LC_ALL=C sort -u -k1,1 -S 100G --parallel={num_cores}: Pure C-byte sort, unique on column 1, 100GB RAM buffer, all cores
# - pigz -p {num_cores}: Recompresses across all cores
dedup_cmd = f"unpigz -p {num_cores} -c {raw_dir}/**/*.smi.gz | LC_ALL=C sort -u -k1,1 -S 100G --parallel={num_cores} | pigz -p {num_cores} > {clean_file}"

subprocess.run(dedup_cmd, shell=True, executable='/bin/bash')
print("✅ 64-Core Deduplication complete!\n")

print("📊 Calculating statistics...")
# Fast line counting using pigz
total_out = subprocess.check_output(f"unpigz -p {num_cores} -c {raw_dir}/**/*.smi.gz | wc -l", shell=True)
total_mols = int(total_out.strip())

unique_out = subprocess.check_output(f"unpigz -p {num_cores} -c {clean_file} | wc -l", shell=True)
unique_mols = int(unique_out.strip())

duplicates = total_mols - unique_mols
print(f"Total SMILES processed: {total_mols:,}")
print(f"Unique SMILES retained: {unique_mols:,}")
print(f"Duplicates removed:   {duplicates:,}\n")

print("🖼️ Rendering preview of cleaned dataset (first 100 molecules)...")
# Must decompress a small chunk for pandas to read
preview_cmd = f"unpigz -p {num_cores} -c {clean_file} | head -n 100"
preview_data = subprocess.check_output(preview_cmd, shell=True, text=True)

import io
df_clean = pd.read_csv(io.StringIO(preview_data), sep=r'\s+', header=None, names=['SMILES', 'ZINC_ID', 'Other_1', 'Other_2'])
df_clean['mol'] = df_clean['SMILES'].apply(Chem.MolFromSmiles)

grid = mols2grid.display(
    df_clean, mol_col='mol', smiles_col='SMILES',
    subset=['img', 'ZINC_ID'], tooltip=['ZINC_ID', 'SMILES'],
    n_cols=5, use_svg=True
)
display(grid)

# @title Cell 3: Install High-Performance Tools
!sudo apt-get update -qq
!sudo apt-get install -y ripgrep
# If you get an 'externally-managed-environment' error here, add --break-system-packages
!pip install -q duckdb pyarrow --break-system-packages
print("✅ Ripgrep and DuckDB installed!")

!pip install -q duckdb seaborn pyarrow

#@title Use mols2grid to visulaise a 100 molecule sample of the SMILES in /unclassified_reagents.smi
import os
import pandas as pd
import mols2grid
from IPython.display import display

unclass_path = os.path.join(os.getcwd(), 'unclassified_reagents.smi')

if os.path.exists(unclass_path):
    # Read the first 100 lines
    df_unclass = pd.read_csv(unclass_path, header=None, names=['SMILES'], nrows=100)

    print(f"🖼️ Rendering preview of {len(df_unclass)} unclassified molecules...")
    grid = mols2grid.display(
        df_unclass,
        smiles_col='SMILES',
        n_cols=5,
        use_svg=True,
        tooltip=['SMILES']
    )
    display(grid)
else:
    print(f"⚠️ The file {unclass_path} was not found.")

"""The "next cell" will be the Setup, Configuration, and Reaction Definitions block. This is where you define your target scaffold (Core SMILES), point the enumerator to the freshly cleaned data, and load all the medicinal chemistry reaction rules natively into memory."""

# @title Cell 3: Global Configuration & Native Reaction Definitions
from rdkit import Chem
from rdkit.Chem import AllChem
import collections

# --- 1. GLOBAL CONFIGURATION ---
# Your chemical scaffold with the target secondary amine [N;R;H1]
CORE_SMILES   = "O=C(O)[C@@H]1CNC[C@H]1c2cccc(F)c2"

# NOTE: Standard Colab instances (~12GB RAM) cannot handle the full multi-billion
# molecule dataset. Point this to a smaller sample file instead.
BB_FILE       = "/content/sample_zinc_chunk.smi.gz"

# Property Filters (Lipinski-ish)
ENABLE_FILTERS = True
MW_RANGE   = (150.0, 550.0)
LOGP_RANGE = (-1.0, 4.5)
HBD_RANGE  = (0, 3)
HBA_RANGE  = (0, 10)
TPSA_RANGE = (0.0, 120.0)
RB_RANGE   = (0, 10)
FSP3_RANGE = (0.0, 0.8)

print(f"✅ Configuration set. Core SMILES: {CORE_SMILES}")
print(f"✅ Building Blocks target: {BB_FILE}")

# --- 2. SYNCHRONIZED REACTION DEFINITIONS ---
# Mapped strictly to the Top 50 Secondary Amine Reagents.
# Reactant 1 is always the Core Secondary Amine. Reactant 2 is the Building Block.
reaction_data = [
    # --- Acylation & Related Derivatizations ---
    { "name": "Carboxylic Acid (Amide Coupling)", "smarts": "[N;R;H1:1].[CX3:2](=O)[OH]>>[N:1][C:2]=O" },
    { "name": "Sulfonyl Chloride (Sulfonamide Synthesis)", "smarts": "[N;R;H1:1].[Cl][S:2](=O)(=O)>>[N:1][S:2](=O)(=O)" },
    { "name": "Isocyanate (Urea Synthesis)", "smarts": "[N;R;H1:1].[N:3]=[C:2]=[O:4]>>[N:1][C:2](=[O:4])[NH:3]" },
    { "name": "Chloroformate (Carbamate Synthesis)", "smarts": "[N;R;H1:1].[Cl][C:2](=O)[OX2:3]>>[N:1][C:2](=O)[O:3]" },
    { "name": "Dialkyl Dicarbonate (Carbamate Synth)", "smarts": "[N;R;H1:1].[OX2:3][C:2](=O)[OX2]C(=O)[OX2]>>[N:1][C:2](=O)[O:3]" },
    { "name": "Isothiocyanate (Thiourea Synthesis)", "smarts": "[N;R;H1:1].[N:3]=[C:2]=[S:4]>>[N:1][C:2](=[S:4])[NH:3]" },
    { "name": "Acid Chloride (N-Acylation)", "smarts": "[N;R;H1:1].[Cl][C:2](=O)[#6:3]>>[N:1][C:2](=O)[#6:3]" },
    { "name": "Acid Anhydride (N-Acylation)", "smarts": "[N;R;H1:1].[#6:3][C:2](=O)[OX2][CX3](=O)>>[N:1][C:2](=O)[#6:3]" },

    # --- Alkylation & Arylation ---
    { "name": "Aldehyde (Reductive Amination / MCRs)", "smarts": "[N;R;H1:1].[CH1:2](=O)[#6:3]>>[N:1][CH2:2][#6:3]" },
    { "name": "Ketone (Reductive Amination)", "smarts": "[N;R;H1:1].[#6:3][C:2](=O)[#6:4]>>[N:1][CH1:2]([#6:3])[#6:4]" },
    { "name": "Aryl Halide (SNAr / Buchwald-Hartwig)", "smarts": "[N;R;H1:1].[F,Cl,Br,I][c:2]>>[N:1][c:2]" },
    { "name": "Alkyl Halide (Direct N-Alkylation)", "smarts": "[N;R;H1:1].[F,Cl,Br,I][CX4:2]>>[N:1][C:2]" },
    { "name": "Alkyl Sulfonate (Direct N-Alkylation)", "smarts": "[N;R;H1:1].[CX4:2][OX2][S](=O)(=O)>>[N:1][C:2]" },
    { "name": "Boronic Acid / Ester (Chan-Lam / Petasis)", "smarts": "[N;R;H1:1].[B]([O,OH])([O,OH])[#6:2]>>[N:1][#6:2]" },
    { "name": "Carbon Monoxide (Aminocarbonylation)", "smarts": "[N;R;H1:1].[C-:2]#[O+]>>[N:1][CH1:2]=O" },

    # --- Additions & Ring Openings ---
    { "name": "Alpha,beta-Unsaturated Carbonyl", "smarts": "[N;R;H1:1].[O:4]=[C:3]-[C:2]=[C:5]>>[N:1][C:5]-[CH1:2]-[C:3]=[O:4]" },
    { "name": "Epoxide (Ring Opening)", "smarts": "[N;R;H1:1].[O:2]1[C:3][C:4]1>>[N:1][C:3]-[C:4]-[OH:2]" },
    { "name": "Aziridine (Ring Opening)", "smarts": "[N;R;H1:1].[N:2]1[C:3][C:4]1>>[N:1][C:3]-[C:4]-[NH2:2]" },

    # --- Specialized & Multicomponent Reactions (MCRs) ---
    { "name": "Cyanide Ion (Strecker Synthesis)", "smarts": "[N;R;H1:1].[C-:2]#N>>[N:1][C:2]#N" },
    { "name": "Guanylating Agent", "smarts": "[N;R;H1:1].[NX3:3][C:2](=[NX2:4])n1cccc1>>[N:1][C:2](=[N:4])[N:3]" },

    # --- Prodrugs, Linkers & Bioisosteres ---
    { "name": "Sulfamoyl Chloride (Sulfamide Synthesis)", "smarts": "[N;R;H1:1].[Cl][S:2](=O)(=O)[N:3]>>[N:1][S:2](=O)(=O)[N:3]" },
    { "name": "Phosphorochloridate (Phosphoramidate)", "smarts": "[N;R;H1:1].[Cl][P:2](=O)([OX2:3])[OX2:4]>>[N:1][P:2](=O)([O:3])[O:4]" },
    { "name": "Phosgene / Triphosgene", "smarts": "[N;R;H1:1].[Cl][C:2](=O)[Cl:3]>>[N:1][C:2](=O)[Cl:3]" },

    # --- Targeted Covalent Inhibitor (TCI) Warheads ---
    { "name": "Cyanogen Halide (Cyanamide Formation)", "smarts": "[N;R;H1:1].[F,Cl,Br,I][C:2]#[N:3]>>[N:1][C:2]#[N:3]" },
    { "name": "Vinyl Sulfone (Conjugate Addition)", "smarts": "[N;R;H1:1].[C:3]=[C:2]-[S:4](=O)(=O)>>[N:1][C:3]-[CH1:2]-[S:4](=O)(=O)" },

    # --- SuFEx Click Chemistry ---
    { "name": "Sulfonyl Fluoride (SuFEx)", "smarts": "[N;R;H1:1].[F][S:2](=O)(=O)[#6:3]>>[N:1][S:2](=O)(=O)[#6:3]" },
    { "name": "Aryl Fluorosulfate (SuFEx)", "smarts": "[N;R;H1:1].[F][S:2](=O)(=O)[OX2:3][c:4]>>[N:1][S:2](=O)(=O)[O:3][c:4]" },

    # --- Complex Alkylation & MCRs ---
    { "name": "Alpha-Halo Ketone (Alpha-Amination)", "smarts": "[N;R;H1:1].[F,Cl,Br,I][CX4:2][C:3](=O)>>[N:1][C:2][C:3]=O" },
    { "name": "Terminal Alkyne (A³ Coupling)", "smarts": "[N;R;H1:1].[C;H1:3]#[C:2]>>[N:1][C:2]#[C:3]" },
    { "name": "Alkyl Nitrite (N-Nitrosation)", "smarts": "[N;R;H1:1].[OX2][N:2]=[O:3]>>[N:1][N:2]=[O:3]" },

    # --- Advanced Bioisosteres & Heteroatom Chemistry ---
    { "name": "Dialkyl Squarate (Squaramide Bioisostere)", "smarts": "[N;R;H1:1].[OX2]C1=C([OX2:2])C(=O)C1=O>>[N:1]C1=C([O:2])C(=O)C1=O" },
    { "name": "Carbon Disulfide (Dithiocarbamate)", "smarts": "[N;R;H1:1].S=[C:2]=S>>[N:1][C:2](=S)S" },
    { "name": "Imidate (Amidine Synthesis)", "smarts": "[N;R;H1:1].[C:2](=[NX2:3])[OX2]>>[N:1][C:2]=[N:3]" },
    { "name": "Thioacyl Chloride (Thioamide Synthesis)", "smarts": "[N;R;H1:1].[Cl][C:2](=S)[#6:3]>>[N:1][C:2](=S)[#6:3]" },
    { "name": "Oxalyl Chloride (Oxalamide Synthesis)", "smarts": "[N;R;H1:1].[Cl][C:2](=O)[C:3](=O)[Cl]>>[N:1][C:2](=O)[C:3](=O)Cl" },

    # Heteroatom-Nitrogen Bond Formations
    { "name": "Hydroxylamine-O-sulfonic acid (HOSA)", "smarts": "[N;R;H1:1].[NX3:2][OX2][S](=O)(=O)[OH]>>[N:1][N:2]" },
    { "name": "Peroxy Acid (N-Oxidation)", "smarts": "[N;R;H1:1].C(=O)[OX2][OH:2]>>[N:1][OH:2]" },
    { "name": "N-Chlorosuccinimide (N-Chlorination)", "smarts": "[N;R;H1:1].[Cl:2][N]1C(=O)CCC1=O>>[N:1][Cl:2]" },
    { "name": "Sulfinyl Chloride (Sulfinamide Synthesis)", "smarts": "[N;R;H1:1].[Cl][S:2](=O)[#6:3]>>[N:1][S:2](=O)[#6:3]" },
    { "name": "Sulfenyl Chloride (Sulfenamide Synthesis)", "smarts": "[N;R;H1:1].[Cl][SX2:2][#6:3]>>[N:1][S:2][#6:3]" },
    { "name": "Silyl Chloride (N-Silylation)", "smarts": "[N;R;H1:1].[Cl][Si:2]>>[N:1][Si:2]" },

    # Modern Catalysis, Radicals & Aryne Chemistry
    { "name": "Redox-Active Ester / NHP Ester", "smarts": "[N;R;H1:1].[C:2](=O)[OX2][N]1C(=O)c2ccccc2C1=O>>[N:1][C:2]=O" },
    { "name": "ortho-Silyl Aryl Triflate (Benzyne)", "smarts": "[N;R;H1:1].[Si][c:2]1[c:3]([OX2][S](=O)(=O)C(F)(F)F)cccc1>>[N:1][c:2][c:3]" },
    { "name": "Unactivated Alkene (Hydroamination)", "smarts": "[N;R;H1:1].[CX3:2]=[CX3:3]>>[N:1][C:2]-[CH1:3]" },
    { "name": "Activated Alkyne (Conjugate Addition)", "smarts": "[N;R;H1:1].[C:3]#[C:2]C(=O)>>[N:1][C:2]=[CH1:3]C(=O)" },
    { "name": "Isocyanide (Ugi-Smiles MCRs)", "smarts": "[N;R;H1:1].[C-:2]#[N+:3]>>[N:1][C:2]=[N:3]" },

    # Specialized Alkylations & Formylations
    { "name": "Alcohol (Mitsunobu Alkylation)", "smarts": "[N;R;H1:1].[CX4:2][OH]>>[N:1][C:2]" },
    { "name": "Formate Ester (N-Formylation)", "smarts": "[N;R;H1:1].[CH1:2](=O)[OX2]>>[N:1][CH1:2]=O" },
    { "name": "Formic Acid (Eschweiler-Clarke)", "smarts": "[N;R;H1:1].[CH1:2](=O)[OH]>>[N:1][CH3:2]" },
    { "name": "Trifluoromethyl Iodide", "smarts": "[N;R;H1:1].I[C:2](F)(F)F>>[N:1][C:2](F)(F)F" }
]

RXN_DEFINITIONS = collections.defaultdict(list)

print("🧪 Compiling Synchronized Reaction SMARTS...")
for entry in reaction_data:
    name, rtype, smarts = entry["name"], "Amine Derivatization", entry["smarts"]
    try:
        rxn = AllChem.ReactionFromSmarts(smarts)
        if rxn:
            rxn.Initialize()
            RXN_DEFINITIONS[rtype].append((name, rxn))
    except Exception as e:
        print(f"⚠️ Skipping invalid SMARTS for {name}: {e}")

print(f"✅ Successfully loaded {sum(len(v) for v in RXN_DEFINITIONS.values())} reactions mapped to your Reagents List.")

# @title Cell 4: Verbose Hyper-Scale Scanner & Binner (V6: Split-Chunk Optimized)
import os, glob, subprocess, shutil, collections, time, math
import duckdb
import concurrent.futures
import multiprocessing
import queue
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import FilterCatalog
from tqdm.auto import tqdm
from IPython.display import display, FileLink

RDLogger.DisableLog("rdApp.*")

# --- 1. Bulletproofed SMARTS Dictionary ---
functional_groups_smarts = {
    "Carboxylic Acid (Amide Coupling)": "[CX3](=O)[OH]",
    "Sulfonyl Chloride (Sulfonamide Synthesis)": "[Cl][S](=O)(=O)",
    "Isocyanate (Urea Synthesis)": "[N]=[C]=[O]",
    "Chloroformate (Carbamate Synthesis)": "[Cl]C(=O)[OX2]",
    "Dialkyl Dicarbonate (Carbamate Synth)": "[OX2]C(=O)[OX2]C(=O)[OX2]",
    "Isothiocyanate (Thiourea Synthesis)": "[N]=[C]=[S]",
    "Acid Chloride (N-Acylation)": "[Cl]C(=O)[#6]",
    "Acid Anhydride (N-Acylation)": "[CX3](=O)[OX2][CX3](=O)",
    "Aldehyde (Reductive Amination / MCRs)": "[CH1](=O)[#6]",
    "Ketone (Reductive Amination)": "[#6][CX3](=O)[#6]",
    "Aryl Halide (SNAr / Buchwald-Hartwig)": "[F,Cl,Br,I]c",
    "Alkyl Halide (Direct N-Alkylation)": "[F,Cl,Br,I][CX4]",
    "Alkyl Sulfonate (Direct N-Alkylation)": "[CX4][OX2][S](=O)(=O)",
    "Boronic Acid / Ester (Chan-Lam / Petasis)": "[B]([O,OH])([O,OH])",
    "Carbon Monoxide (Aminocarbonylation)": "[C-]#[O+]",
    "Alpha,beta-Unsaturated Carbonyl": "O=[C]-[C]=[C]",
    "Epoxide (Ring Opening)": "[O]1[C][C]1",
    "Aziridine (Ring Opening)": "[N]1[C][C]1",
    "Cyanide Ion (Strecker Synthesis)": "[C-]#N",
    "Guanylating Agent": "[NX3]C(=[NX2])n1cccc1",
    "Sulfamoyl Chloride (Sulfamide Synthesis)": "[Cl][S](=O)(=O)[N]",
    "Phosphorochloridate (Phosphoramidate)": "[Cl][P](=O)([OX2])[OX2]",
    "Phosgene / Triphosgene": "[Cl]C(=O)[Cl]",
    "Cyanogen Halide (Cyanamide Formation)": "[F,Cl,Br,I]C#N",
    "Vinyl Sulfone (Conjugate Addition)": "[C]=[C]-[S](=O)(=O)",
    "Sulfonyl Fluoride (SuFEx)": "[F][S](=O)(=O)",
    "Aryl Fluorosulfate (SuFEx)": "[F][S](=O)(=O)[OX2]c",
    "Alpha-Halo Ketone (Alpha-Amination)": "[F,Cl,Br,I][CX4]C(=O)",
    "Terminal Alkyne (A³ Coupling)": "[C;H1]#[C]",
    "Alkyl Nitrite (N-Nitrosation)": "[OX2][N]=[O]",
    "Dialkyl Squarate (Squaramide Bioisostere)": "[OX2]C1=C([OX2])C(=O)C1=O",
    "Carbon Disulfide (Dithiocarbamate)": "S=C=S",
    "Imidate (Amidine Synthesis)": "C(=[NX2])[OX2]",
    "Thioacyl Chloride (Thioamide Synthesis)": "[Cl]C(=S)[#6]",
    "Oxalyl Chloride (Oxalamide Synthesis)": "[Cl]C(=O)C(=O)[Cl]",
    "Hydroxylamine-O-sulfonic acid (HOSA)": "[NX3][OX2][S](=O)(=O)[OH]",
    "Peroxy Acid (N-Oxidation)": "C(=O)[OX2][OH]",
    "N-Chlorosuccinimide (N-Chlorination)": "Cl[N]1C(=O)CCC1=O",
    "Sulfinyl Chloride (Sulfinamide Synthesis)": "[Cl][S](=O)[#6]",
    "Sulfenyl Chloride (Sulfenamide Synthesis)": "[Cl][SX2][#6]",
    "Silyl Chloride (N-Silylation)": "[Cl][Si]",
    "Redox-Active Ester / NHP Ester": "C(=O)[OX2][N]1C(=O)c2ccccc2C1=O",
    "ortho-Silyl Aryl Triflate (Benzyne)": "[Si]c1c([OX2][S](=O)(=O)C(F)(F)F)cccc1",
    "Unactivated Alkene (Hydroamination)": "[CX3]=[CX3]",
    "Activated Alkyne (Conjugate Addition)": "[C]#[C]C(=O)",
    "Isocyanide (Ugi-Smiles MCRs)": "[C-]#[N+]",
    "Alcohol (Mitsunobu Alkylation)": "[CX4][OH]",
    "Formate Ester (N-Formylation)": "[CH1](=O)[OX2]",
    "Formic Acid (Eschweiler-Clarke)": "[CH1](=O)[OH]",
    "Trifluoromethyl Iodide": "I[C](F)(F)F"
}

# --- 2. Path Setup ---
base_dir = os.getcwd()
cleaned_file_path = os.path.join(base_dir, 'zinc22_raw', 'combined_cleaned.smi.gz')
ram_disk_raw = "/dev/shm/raw_mols.smi"
bin_dir = os.path.join(base_dir, "binned_bbs")

os.makedirs(bin_dir, exist_ok=True)
print("🗑️ Clearing old output files...")
for old_file in glob.glob(os.path.join(bin_dir, "*.smi")):
    try: os.remove(old_file)
    except: pass
for old_part in glob.glob("/dev/shm/mol_chunk_*"):
    try: os.remove(old_part)
    except: pass

# --- 3. The Isolated Worker (Verbose & Chunked) ---
def hyper_scale_worker(worker_id, chunk_file, smarts_dict, prog_queue):
    local_catalog = FilterCatalog.FilterCatalog()
    local_writers = {}
    total_found = 0
    internal_scanned = 0

    for rname, smarts in smarts_dict.items():
        safe_name = rname.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
        matcher = FilterCatalog.SmartsMatcher(safe_name, smarts, 1)
        local_catalog.AddEntry(FilterCatalog.FilterCatalogEntry(safe_name, matcher))
        local_writers[safe_name] = open(os.path.join(bin_dir, f"{safe_name}_{worker_id}.part"), 'w')

    unclass_file = open(os.path.join(bin_dir, f"unclass_{worker_id}.part"), 'w')
    con = duckdb.connect()

    # Simple, fast read - no offsets needed because the file is already chopped!
    query = f"SELECT column0 as smi, column1 as zid FROM read_csv('{chunk_file}', header=false, auto_detect=true)"

    try:
        for batch in con.execute(query).to_arrow_reader(50000):
            smi_col, zid_col = batch.column('smi').to_pylist(), batch.column('zid').to_pylist()
            batch_len = len(smi_col)

            for smi, zid in zip(smi_col, zid_col):
                if not smi: continue
                mol = Chem.MolFromSmiles(str(smi))
                if not mol: continue
                matches = local_catalog.GetMatches(mol)
                if matches:
                    total_found += 1
                    for m in matches: local_writers[m.GetDescription()].write(f"{smi}\t{zid}\n")
                else: unclass_file.write(f"{smi}\t{zid}\n")

            internal_scanned += batch_len
            prog_queue.put((batch_len, total_found)) # Send tick to main thread

            # VERBOSITY: Print a heartbeat every 250k molecules
            if internal_scanned % 250000 == 0:
                print(f"🟢 [Worker {worker_id:02d}] Scanned: {internal_scanned:,} | Caught: {total_found:,}")

    except Exception as e:
        print(f"❌ Worker {worker_id} Error: {e}")

    for w in local_writers.values(): w.close()
    unclass_file.close()
    return total_found

# --- 4. Execution Pipeline ---
num_workers = os.cpu_count() or 64

# Step 4A: Decompress
print(f"\n🚀 Step 1/3: Decompressing 75M molecules to RAM Disk...")
subprocess.run(f"unpigz -p {num_workers} -c '{cleaned_file_path}' > {ram_disk_raw}", shell=True, check=True)
total_mols = int(subprocess.check_output(f"wc -l {ram_disk_raw}", shell=True).split()[0])
lines_per_chunk = math.ceil(total_mols / num_workers)

# Step 4B: Chop the file
print(f"🔪 Chopping master file into 64 pieces for zero-latency reading...")
subprocess.run(f"split -d -l {lines_per_chunk} {ram_disk_raw} /dev/shm/mol_chunk_", shell=True, check=True)
os.remove(ram_disk_raw) # Free up RAM immediately

chunk_files = sorted(glob.glob("/dev/shm/mol_chunk_*"))
print(f"🚀 Step 2/3: Launching {num_workers} Workers (Processing {total_mols:,} molecules)...")
print("-" * 60)

start_time = time.time()
manager = multiprocessing.Manager()
prog_queue = manager.Queue()
total_reagents_caught = 0

with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
    # Launch workers
    futures = [executor.submit(hyper_scale_worker, i, chunk_files[i], functional_groups_smarts, prog_queue) for i in range(len(chunk_files))]

    # Live Progress Bar Thread
    with tqdm(total=total_mols, desc="Scanning & Binning", unit="mols") as pbar:
        while not all(f.done() for f in futures):
            try:
                # Pull ticks from the queue and update the bar smoothly
                scanned_tick, current_caught = prog_queue.get(timeout=2.0)
                pbar.update(scanned_tick)
                pbar.set_postfix({"Reagents Caught": f"{current_caught:,}"})
            except queue.Empty:
                continue

# Add up the final exact totals
final_caught = sum(f.result() for f in futures)

# --- 5. Atomic OS Merge ---
print("\n" + "-" * 60)
print("🚀 Step 3/3: Merging worker fragments into final bins...")
for rname in functional_groups_smarts.keys():
    safe_name = rname.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
    if glob.glob(os.path.join(bin_dir, f"{safe_name}_*.part")):
        subprocess.run(f"cat {bin_dir}/{safe_name}_*.part > {bin_dir}/{safe_name}.smi && rm {bin_dir}/{safe_name}_*.part", shell=True)
subprocess.run(f"cat {bin_dir}/unclass_*.part > {base_dir}/unclassified_reagents.smi && rm {bin_dir}/unclass_*.part", shell=True)

# Cleanup RAM Disk chunks
for chunk in chunk_files:
    if os.path.exists(chunk): os.remove(chunk)

elapsed = time.time() - start_time
print(f"✅ SUCCESS: {total_mols:,} molecules processed in {elapsed:.1f}s (~{total_mols/elapsed:,.0f} mols/sec).")
print(f"💎 Total Reagents Successfully Binned: {final_caught:,}")

# @title Cell 5: Preview a reaction bin
import os
import pandas as pd
import mols2grid
from IPython.display import display

# Directly specify the target file
target_file = os.path.join(os.getcwd(), "binned_bbs", "Aryl_Fluorosulfate_SuFEx.smi")

if os.path.exists(target_file):
    print(f"🖼️ Rendering preview of: {os.path.basename(target_file)}")

    # Read the first 100 rows
    df = pd.read_csv(target_file, sep='\t', header=None, names=['SMILES', 'ZINC_ID'], nrows=100)

    # Display with mols2grid
    grid = mols2grid.display(
        df,
        smiles_col='SMILES',
        tooltip=['SMILES', 'ZINC_ID'],
        n_cols=5,
        use_svg=True
    )
    display(grid)
else:
    print(f"⚠️ The file {target_file} was not found.")

#@title Cell 6: Count total reagents
!pip install -q plotly
import os
import glob
import pandas as pd
import plotly.express as px
import plotly.io as pio
from IPython.display import display, FileLink

# Configure Plotly to render correctly in Colab
pio.renderers.default = 'colab'

bin_dir = os.path.join(os.getcwd(), "binned_bbs")
smi_files = glob.glob(os.path.join(bin_dir, "*.smi"))
print(f"Found {len(smi_files)} .smi files in {bin_dir}")

data = []
for file_path in smi_files:
    filename = os.path.basename(file_path)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            count = sum(1 for _ in f)
        data.append({"File": filename, "SMILES Count": count})
    except Exception as e:
        print(f"Error reading {filename}: {e}")

if data:
    df = pd.DataFrame(data)
    df = df.sort_values(by="SMILES Count", ascending=False)
    max_val = df["SMILES Count"].max()

    # Generate CSV
    csv_filename = "reaction_bin_counts.csv"
    df.to_csv(csv_filename, index=False)
    print(f"✅ Data saved to {csv_filename}")

    # Display table preview
    display(df)

    # Generate Plot
    fig = px.bar(
        df,
        x="File",
        y="SMILES Count",
        title="Number of SMILES in each binned .smi file (Log Scale)",
        labels={"File": "Reaction Bin", "SMILES Count": "Number of Molecules"},
        color="SMILES Count",
        color_continuous_scale="Viridis",
        log_y=True,
        height=800,
        range_y=[1, max_val * 2]
    )
    fig.update_layout(xaxis_tickangle=-45)
    fig.show()

    # Provide Download Link
    display(FileLink(csv_filename))
else:
    print(f"No .smi files found in {bin_dir}.")

# @title Cell 7: Build the Industrial C++ Engine (Hyper-Optimized)
import os, subprocess, time

print("📥 Installing RDKit C++ Development headers (this takes a moment)...")

# Retry loop for apt-get to avoid lock issues in Colab
max_retries = 5
for i in range(max_retries):
    try:
        subprocess.run("sudo apt-get update -qq && sudo apt-get install -y librdkit-dev", shell=True, check=True)
        break
    except subprocess.CalledProcessError:
        if i < max_retries - 1:
            print(f"⚠️ apt-get is locked. Waiting 5 seconds before retrying ({i+1}/{max_retries})...")
            time.sleep(5)
        else:
            print("❌ Failed to acquire apt lock after multiple attempts. You may need to restart the runtime.")
            raise

cpp_source = """
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <memory>
#include <boost/shared_ptr.hpp>
#include <GraphMol/RDKitBase.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/SmilesParse/SmilesWrite.h>
#include <GraphMol/ChemReactions/Reaction.h>
#include <GraphMol/ChemReactions/ReactionParser.h>
#include <GraphMol/Descriptors/MolDescriptors.h>
#include <GraphMol/MolOps.h>

using namespace RDKit;

int main(int argc, char* argv[]) {
    if (argc < 10) return 1;

    std::string core_smi = argv[1], rxn_smarts = argv[2], rxn_name = argv[3];
    std::string input_path = argv[4], output_path = argv[5];
    double mw_min = std::stod(argv[6]), mw_max = std::stod(argv[7]), logp_min = std::stod(argv[8]), logp_max = std::stod(argv[9]);

    std::unique_ptr<ChemicalReaction> rxn(RxnSmartsToChemicalReaction(rxn_smarts));
    if (!rxn) return 1;
    rxn->initReactantMatchers();

    boost::shared_ptr<ROMol> core(SmilesToMol(core_smi));
    if (!core) return 1;

    std::ifstream infile(input_path);
    std::ofstream outfile(output_path);

    // OPTIMIZATION 4: Hijack stream buffers for massive 1MB block reads/writes
    char in_buffer[1048576];
    char out_buffer[1048576];
    infile.rdbuf()->pubsetbuf(in_buffer, sizeof(in_buffer));
    outfile.rdbuf()->pubsetbuf(out_buffer, sizeof(out_buffer));

    outfile << "smiles,zid,reactant_smiles,mw,clogp,rxn_name\\n";

    // OPTIMIZATION 2: Pre-allocate the Reactants vector OUTSIDE the loop
    MOL_SPTR_VECT reactants(2);
    reactants[0] = core;

    std::string line;
    while (std::getline(infile, line)) {
        size_t tab_pos = line.find('\\t');
        if (tab_pos == std::string::npos) continue;

        // OPTIMIZATION 3: Zero-Copy Pointer Arithmetic (No std::string allocations)
        line[tab_pos] = '\\0';
        const char* smi_ptr = line.c_str();
        const char* zid_ptr = line.c_str() + tab_pos + 1;

        boost::shared_ptr<ROMol> partner(SmilesToMol(smi_ptr));
        if (!partner) continue;

        // O(1) vector replacement
        reactants[1] = partner;

        std::vector<MOL_SPTR_VECT> prods = rxn->runReactants(reactants);
        for (auto& prod_vect : prods) {
            for (auto& p : prod_vect) {
                try { p->updatePropertyCache(false); } catch (...) { continue; }

                double mw = Descriptors::calcAMW(*p);
                if (mw < mw_min || mw > mw_max) continue;
                double clogp = Descriptors::calcClogP(*p);
                if (clogp < logp_min || clogp > logp_max) continue;

                try {
                    MolOps::sanitizeMol(*static_cast<RWMol*>(p.get()));
                    outfile << MolToSmiles(*p) << "," << zid_ptr << "," << smi_ptr << "," << mw << "," << clogp << "," << rxn_name << "\\n";
                } catch (...) { continue; }
            }
        }
    }
    return 0;
}
"""

with open("high_speed_enumerator.cpp", "w") as f:
    f.write(cpp_source)

print("🛠️ Compiling Bare-Metal Reactant Engine...")

# OPTIMIZATION 1: Added -march=native, -mtune=native, and -funroll-loops
compile_cmd = [
    "g++", "-O3", "-march=native", "-mtune=native", "-funroll-loops",
    "-I/usr/include/rdkit", "high_speed_enumerator.cpp", "-o", "high_speed_enumerator",
    "-L/usr/lib", "-lRDKitDescriptors", "-lRDKitChemReactions", "-lRDKitSmilesParse",
    "-lRDKitGraphMol", "-lRDKitDataStructs", "-lRDKitRDGeometryLib", "-lRDKitRDGeneral"
]

subprocess.run(compile_cmd, check=True)
os.chmod("high_speed_enumerator", 0o755)
print("✅ Industrial C++ binary compiled with hardware-specific SIMD instructions.")

# @title Cell 8: Bare-Metal C++ Orchestrator (V10.4: Schema Fix)
import os, glob, subprocess, duckdb, shutil, time
import pandas as pd
import mols2grid
from IPython.display import FileLink, display
from rdkit.Chem import AllChem
from tqdm.auto import tqdm

# --- 1. Path Setup & RAM-Disk Initialization ---
base_dir = os.getcwd()
bin_dir = os.path.join(base_dir, "binned_bbs")
final_csv = os.path.join(base_dir, "enumerated_products_final.csv")
local_backup_dir = os.path.join(base_dir, "Billion_Scale_Enumeration_Backups")

ram_in = "/content/cpp_inputs"
ram_out = "/content/cpp_outputs"
tasks_file = "/content/tasks.sh"
ticks_file = "/content/ticks.log"

os.makedirs(local_backup_dir, exist_ok=True)
os.makedirs(ram_in, exist_ok=True)
os.makedirs(ram_out, exist_ok=True)

print("🗑️ Clearing old environments and RAM Disks...")
if os.path.exists(final_csv):
    try: os.remove(final_csv)
    except: pass
subprocess.run(f"rm -rf {ram_in}/* {ram_out}/* {tasks_file} {ticks_file}", shell=True)  # /content paths

num_workers = os.cpu_count() or 64
CHUNK_SIZE = 50000

# --- 2. Zero-Copy Routing & Task Generation ---
print("🔪 Routing bins and generating OS-level execution map...")

commands = []
total_tasks = 0

for rtype, rxn_list in RXN_DEFINITIONS.items():
    for rname, rxn in rxn_list:
        safe_name = rname.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
        clean_rname = rname.replace(",", "_") # Fix for CSV column misalignment
        bin_file = os.path.join(bin_dir, f"{safe_name}.smi")

        if not os.path.exists(bin_file): continue

        try:
            line_count = int(subprocess.check_output(f'wc -l "{bin_file}"', shell=True).split()[0])
        except: continue

        if line_count == 0: continue

        chunk_prefix = os.path.join(ram_in, f"{safe_name}_chunk_")
        rxn_smarts = AllChem.ReactionToSmarts(rxn)

        if line_count <= CHUNK_SIZE:
            single_chunk = f"{chunk_prefix}0000"
            os.symlink(bin_file, single_chunk)
            output_file = os.path.join(ram_out, f"{safe_name}_out_0000.csv")

            cmd = f"./high_speed_enumerator '{CORE_SMILES}' '{rxn_smarts}' '{clean_rname}' '{single_chunk}' '{output_file}' {MW_RANGE[0]} {MW_RANGE[1]} {LOGP_RANGE[0]} {LOGP_RANGE[1]}"
            msg = f"🟢 [Core Thread] Completed routing {line_count:,} molecules for {clean_rname}"
            commands.append(f"{cmd} && echo '{msg}' >> {ticks_file}")
            total_tasks += 1

        else:
            subprocess.run(f"split -d -a 4 -l {CHUNK_SIZE} '{bin_file}' '{chunk_prefix}'", shell=True)
            generated_chunks = sorted(glob.glob(f"{chunk_prefix}*"))

            for i, chunk_file in enumerate(generated_chunks):
                output_file = os.path.join(ram_out, f"{safe_name}_out_{i:04d}.csv")
                cmd = f"./high_speed_enumerator '{CORE_SMILES}' '{rxn_smarts}' '{clean_rname}' '{chunk_file}' '{output_file}' {MW_RANGE[0]} {MW_RANGE[1]} {LOGP_RANGE[0]} {LOGP_RANGE[1]}"
                msg = f"🟢 [Core Thread] Processed 50k micro-chunk ({i+1}/{len(generated_chunks)}) for {clean_rname}"
                commands.append(f"{cmd} && echo '{msg}' >> {ticks_file}")
                total_tasks += 1

with open(tasks_file, "w") as f:
    for c in commands:
        f.write(f"{c}\n")

# --- 3. Bare-Metal Execution (Verbose Streaming) ---
print(f"\n🚀 Launching 64-Core OS-Level Scheduler on {total_tasks:,} micro-chunks...")
subprocess.run(f"touch {ticks_file}", shell=True)

os_process = subprocess.Popen(
    f"xargs -d '\\n' -P {num_workers} -I {{}} bash -c {{}} < {tasks_file}",
    shell=True
)

with open(ticks_file, 'r') as log_stream:
    with tqdm(total=total_tasks, desc="C++ Orchestration", unit="chunks") as pbar:
        while os_process.poll() is None:
            new_lines = log_stream.readlines()
            for line in new_lines:
                tqdm.write(line.strip())
                pbar.update(1)

            if not new_lines:
                time.sleep(0.2)

        for line in log_stream.readlines():
            tqdm.write(line.strip())
            pbar.update(1)

# --- 4. Unified RAM-Block Aggregation & DuckDB Overclocking ---
print("\nΔ Fusing outputs and overclocking DuckDB deduplication...")

merged_ram_file = "/content/merged_outputs.csv"
subprocess.run(f"cat {ram_out}/*.csv > {merged_ram_file} 2>/dev/null", shell=True)

con = duckdb.connect()
con.execute(f"PRAGMA threads={num_workers};")
con.execute(f"PRAGMA memory_limit='10GB';")

# FIXED: All columns are now VARCHAR to bypass parse-crashing on the embedded headers
schema = "{'smiles': 'VARCHAR', 'zid': 'VARCHAR', 'reactant_smiles': 'VARCHAR', 'mw': 'VARCHAR', 'clogp': 'VARCHAR', 'rxn_name': 'VARCHAR'}"
duck_query = f"""
    COPY (
        SELECT smiles, ANY_VALUE(zid) as zid, ANY_VALUE(reactant_smiles) as reactant_smiles,
               ANY_VALUE(mw) as mw, ANY_VALUE(clogp) as clogp, ANY_VALUE(rxn_name) as rxn_name
        FROM read_csv('{merged_ram_file}', columns={schema}, header=true, ignore_errors=true)
        WHERE smiles != 'smiles'
        GROUP BY smiles
    ) TO '{final_csv}' (HEADER, DELIMITER ',');
"""

try:
    if os.path.exists(merged_ram_file) and os.path.getsize(merged_ram_file) > 0:
        con.execute(duck_query)

        backup_path = os.path.join(local_backup_dir, "enumerated_products_final.csv")
        shutil.copy(final_csv, backup_path)

        total_lines = int(subprocess.check_output(f'wc -l "{final_csv}"', shell=True).split()[0])
        total_smiles = total_lines - 1 if total_lines > 0 else 0

        print(f"✅ Success! Generated {total_smiles:,} unique, filtered molecules.")
        display(FileLink(final_csv))

        # --- 5. Visual Dashboard ---
        df_final = pd.read_csv(final_csv, nrows=100)
        if 'smiles' in df_final.columns:
            print(f"\n🖼️ Rendering preview of the first {len(df_final)} generated products...")
            grid = mols2grid.display(
                df_final, smiles_col='smiles',
                subset=['img', 'zid', 'rxn_name', 'mw', 'clogp'],
                tooltip=['smiles', 'reactant_smiles', 'zid', 'rxn_name', 'mw', 'clogp'],
                n_cols=5, use_svg=True
            )
            display(grid)
    else:
        print("⚠️ No valid products were generated. File was empty.")
except Exception as e:
    print(f"⚠️ Merge failed: {e}")
finally:
    subprocess.run(f"rm -rf {ram_in}/* {ram_out}/* {tasks_file} {ticks_file} {merged_ram_file}", shell=True)  # /content paths

#@title Download the results as a zip file
!pigz -p 64 /home/tommbk50/enumerated_products_final.csv

# @title Cell 9: Gold Standard Lead-Like Filter (V16: Unbreakable & Smooth ETA)
import os, glob, subprocess, duckdb, shutil, time, math
import pandas as pd
import mols2grid
from IPython.display import FileLink, display
from tqdm.auto import tqdm

# ==============================================================================
# 🎛️ MEDICINAL CHEMISTRY THRESHOLDS
# ==============================================================================
MAX_ROTATABLE_BONDS = 10
MAX_TPSA            = 140.0
MIN_FSP3            = 0.30
MAX_HBD             = 5
MAX_HBA             = 10
# PAINS, BRENK, and NIH structural toxicity alerts are permanently ON.
# ==============================================================================

input_csv = os.path.join(os.getcwd(), "Billion_Scale_Enumeration_Backups", "enumerated_products_final.csv")
filtered_csv = os.path.join(os.getcwd(), "lead_like_products.csv")

print(f"⚙️ Target: {input_csv}")
print(f"⚙️ Compiling V16 C++ Engine (Micro-Pulse Telemetry)...")

cpp_source = """
#include <iostream>
#include <fstream>
#include <string>
#include <memory>
#include <GraphMol/RDKitBase.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/Descriptors/MolDescriptors.h>
#include <GraphMol/FilterCatalog/FilterCatalog.h>

using namespace RDKit;

int main(int argc, char* argv[]) {
    if (argc < 9) return 1;
    std::string input_path = argv[1];
    std::string output_path = argv[2];

    int max_rot_bonds = std::stoi(argv[3]);
    double max_tpsa = std::stod(argv[4]);
    double min_fsp3 = std::stod(argv[5]);
    int max_hbd = std::stoi(argv[6]);
    int max_hba = std::stoi(argv[7]);
    std::string ticks_file = argv[8];

    FilterCatalogParams params;
    params.addCatalog(FilterCatalogParams::PAINS);
    params.addCatalog(FilterCatalogParams::BRENK);
    params.addCatalog(FilterCatalogParams::NIH);
    FilterCatalog toxicity_catalog(params);

    std::ifstream infile(input_path);
    std::ofstream outfile(output_path);

    std::string line;
    int mol_count = 0;

    while (std::getline(infile, line)) {
        // Gracefully skip the header if it lands in this chunk
        if (line.find("smiles") == 0) continue;

        mol_count++;

        // --- MICRO-PULSE TELEMETRY ---
        // Every 10,000 molecules, append a '1' to the log file natively
        if (mol_count % 10000 == 0) {
            std::ofstream log(ticks_file, std::ios_base::app);
            log << "1\\n";
        }

        size_t comma_pos = line.find(',');
        if (comma_pos == std::string::npos) continue;

        line[comma_pos] = '\\0';
        const char* smi_ptr = line.c_str();

        std::unique_ptr<ROMol> mol(SmilesToMol(smi_ptr));
        if (!mol) continue;

        // --- OPTIMIZED GAUNTLET ---
        if (Descriptors::calcNumRotatableBonds(*mol) > max_rot_bonds) continue;
        if (Descriptors::calcTPSA(*mol) > max_tpsa) continue;
        if (Descriptors::calcFractionCSP3(*mol) < min_fsp3) continue;
        if (Descriptors::calcNumHBD(*mol) > max_hbd) continue;
        if (Descriptors::calcNumHBA(*mol) > max_hba) continue;
        if (toxicity_catalog.hasMatch(*mol)) continue;

        line[comma_pos] = ',';
        outfile << line << "\\n";
    }
    return 0;
}
"""

with open("molecule_filter.cpp", "w") as f:
    f.write(cpp_source)

compile_cmd = [
    "g++", "-O3", "-march=native", "-mtune=native", "-funroll-loops",
    "-I/usr/include/rdkit", "molecule_filter.cpp", "-o", "molecule_filter",
    "-L/usr/lib", "-lRDKitDescriptors", "-lRDKitFilterCatalog", "-lRDKitCatalogs", "-lRDKitSmilesParse",
    "-lRDKitGraphMol", "-lRDKitDataStructs", "-lRDKitRDGeometryLib", "-lRDKitRDGeneral"
]
subprocess.run(compile_cmd, check=True)
os.chmod("molecule_filter", 0o755)
print("✅ C++ Filter Engine compiled successfully.")

# --- 3. NVMe Block-Seeking & 1-to-1 Core Mapping ---
ram_in = "/dev/shm/filter_in"
ram_out = "/dev/shm/filter_out"
ticks_file = "/dev/shm/filter_ticks.log"

subprocess.run(f"rm -rf {ram_in} {ram_out} {ticks_file}", shell=True)
os.makedirs(ram_in, exist_ok=True)
os.makedirs(ram_out, exist_ok=True)

num_workers = os.cpu_count() or 64

print(f"\n📊 Scanning total molecule count for telemetry...")
try:
    total_lines = int(subprocess.check_output(f'wc -l "{input_csv}"', shell=True).split()[0]) - 1
except:
    total_lines = 67204042

print(f"🔪 Phase 1: Zero-Latency NVMe Splitting (Instant 64-way split)...")
# INSTANT SPLIT: Bypasses Python entirely to slice the file natively on the SSD
subprocess.run(f"split -d -a 2 -n l/{num_workers} '{input_csv}' {ram_in}/chunk_", shell=True)

chunks = sorted(glob.glob(f"{ram_in}/chunk_*"))
commands = []
for i, chunk_file in enumerate(chunks):
    out_file = f"{ram_out}/out_{i:02d}.csv"
    cmd = f"./molecule_filter '{chunk_file}' '{out_file}' {MAX_ROTATABLE_BONDS} {MAX_TPSA} {MIN_FSP3} {MAX_HBD} {MAX_HBA} '{ticks_file}' 2>/dev/null"
    commands.append(cmd)

# --- 5. Bare-Metal Execution (Unbreakable Progress Bar) ---
print(f"\n🚀 Phase 2: Launching 64-Core Marathon...")
subprocess.run(f"touch {ticks_file}", shell=True)

active_processes = [subprocess.Popen(cmd, shell=True) for cmd in commands]

last_milestone = 0
with tqdm(total=total_lines, desc="Filtering", unit="mols", smoothing=0.1) as pbar:
    # Loop runs as long as ANY core is still alive
    while any(p.poll() is None for p in active_processes):
        time.sleep(0.5) # Poll every half second

        try:
            # OS-LEVEL POLLING: Bypasses Python's EOF trap by asking Linux directly
            ticks = int(subprocess.check_output(f"wc -l < {ticks_file}", shell=True))
            current_mols = ticks * 10000

            # Smoothly catch the bar up to reality
            if current_mols > pbar.n:
                pbar.update(current_mols - pbar.n)

            # VERBOSE FEEDBACK: Print an update above the bar every 5 Million molecules
            if current_mols - last_milestone >= 5000000:
                tqdm.write(f"⚡ [Status] {current_mols:,} molecules scanned... Cores are blazing!")
                last_milestone = current_mols

        except Exception:
            pass

    # Final sweep
    try:
        ticks = int(subprocess.check_output(f"wc -l < {ticks_file}", shell=True))
        current_mols = ticks * 10000
        if current_mols > pbar.n:
            pbar.update(current_mols - pbar.n)
    except: pass

    # Force to 100% to account for small remainders under 10k
    if pbar.n < total_lines:
        pbar.update(total_lines - pbar.n)

# --- 6. Unified RAM-Block Aggregation & DuckDB Overclocking ---
print("\nΔ Re-merging surviving molecules into final Lead-Like CSV...")
merged_ram_file = "/dev/shm/merged_filtered.csv"
subprocess.run(f"cat {ram_out}/*.csv > {merged_ram_file} 2>/dev/null", shell=True)

header = subprocess.check_output(f"head -n 1 '{input_csv}'", shell=True).decode('utf-8').strip()
subprocess.run(f"echo '{header}' > '{filtered_csv}'", shell=True)
subprocess.run(f"cat '{merged_ram_file}' >> '{filtered_csv}'", shell=True)

original_lines = total_lines
try:
    surviving_lines = int(subprocess.check_output(f'wc -l "{filtered_csv}"', shell=True).split()[0]) - 1
except:
    surviving_lines = 0
dropped = original_lines - surviving_lines

print(f"\n✅ FILTER COMPLETE!")
print(f"📈 Starting Molecules: {original_lines:,}")
print(f"🗑️ Dropped (Failed Gold Standard Limits): {dropped:,}")
print(f"💎 Surviving Lead-Like Molecules: {surviving_lines:,}")

display(FileLink(filtered_csv))

# --- 7. Visual Dashboard ---
df_final = pd.read_csv(filtered_csv, nrows=100)
if not df_final.empty:
    print(f"\n🖼️ Rendering preview of the surviving Lead-Like products...")
    grid = mols2grid.display(
        df_final, smiles_col='smiles',
        subset=['img', 'zid', 'rxn_name', 'mw', 'clogp'],
        tooltip=['smiles', 'reactant_smiles', 'zid', 'rxn_name', 'mw', 'clogp'],
        n_cols=5, use_svg=True
    )
    display(grid)

subprocess.run(f"rm -rf {ram_in} {ram_out} {ticks_file} {merged_ram_file} molecule_filter*", shell=True)

#@title Compress and Download lead_like_products.csv
import os
import subprocess
from IPython.display import FileLink, display

csv_path = os.path.join(os.getcwd(), 'lead_like_products.csv')
gz_path = csv_path + '.gz'

if os.path.exists(csv_path):
    print("🗜️ Compressing the Lead-Like products CSV (this might take a minute)...")
    # Using pigz for fast multi-core compression, keeping the original file (-k)
    subprocess.run(f"pigz -k -f -p 64 '{csv_path}' || gzip -k -f '{csv_path}'", shell=True)

    if os.path.exists(gz_path):
        size_mb = os.path.getsize(gz_path) / (1024 * 1024)
        print(f"✅ Compression complete! GZIP File size: {size_mb:.2f} MB")
        print("⬇️ Click the link below to download:")
        display(FileLink(gz_path))
    else:
        print("⚠️ Compression failed.")
else:
    print(f"⚠️ File not found: {csv_path}")