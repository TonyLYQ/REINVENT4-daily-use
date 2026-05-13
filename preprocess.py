#!/usr/bin/env python
"""Prepare REINVENT input SMILES from SDF/SMI files.

Examples
--------
python preprocess.py scaffold compounds.sdf
python preprocess.py sidechain compounds.smi --r-group-num 1 --seed 42 --unique-template-num 100
"""

from __future__ import annotations

import argparse
import bisect
import math
import random
import re
import sys
from dataclasses import dataclass
from itertools import accumulate
from pathlib import Path
from typing import Iterable, Sequence

try:
    from rdkit import Chem
    from rdkit.Chem import rdchem
except ModuleNotFoundError:
    Chem = None
    rdchem = None


SUPPORTED_SDF_SUFFIXES = {".sdf", ".sd"}
SUPPORTED_SMI_SUFFIXES = {".smi", ".smiles"}
SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class EditSite:
    kind: str  # "add" or "replace"
    atom_idx: int
    anchor_idx: int


def warn(message: str) -> None:
    print(f"WARNING: {message}", file=sys.stderr)


def require_rdkit() -> None:
    if Chem is None or rdchem is None:
        raise SystemExit(
            "RDKit is required. Activate the REINVENT environment before running this script."
        )


def prepare_mol(mol: Chem.Mol) -> Chem.Mol | None:
    """Return a sanitized molecule with explicit hydrogens removed."""
    if mol is None:
        return None

    mol = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(mol)
        mol = Chem.RemoveHs(mol, sanitize=True)
        Chem.SanitizeMol(mol)
    except Exception:
        return None

    return mol


def read_sdf(path: Path) -> list[Chem.Mol]:
    mols: list[Chem.Mol] = []
    supplier = Chem.SDMolSupplier(str(path), removeHs=False)

    for idx, mol in enumerate(supplier, start=1):
        prepared = prepare_mol(mol)
        if prepared is None:
            warn(f"Skipped invalid molecule {idx} in {path}")
            continue
        mols.append(prepared)

    return mols


def read_smi(path: Path) -> list[Chem.Mol]:
    mols: list[Chem.Mol] = []

    with path.open() as handle:
        for idx, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            smiles = stripped.replace(",", " ").split()[0]
            mol = Chem.MolFromSmiles(smiles)
            prepared = prepare_mol(mol)
            if prepared is None:
                warn(f"Skipped invalid SMILES on line {idx} in {path}: {smiles}")
                continue
            mols.append(prepared)

    return mols


def read_molecules(path: Path) -> list[Chem.Mol]:
    suffix = path.suffix.lower()

    if suffix in SUPPORTED_SDF_SUFFIXES:
        return read_sdf(path)
    if suffix in SUPPORTED_SMI_SUFFIXES:
        return read_smi(path)

    raise ValueError(f"Unsupported input format for {path}. Use .sdf, .sd, .smi, or .smiles.")


def canonicalize_mol(mol: Chem.Mol) -> str | None:
    prepared = prepare_mol(mol)
    if prepared is None:
        return None
    return Chem.MolToSmiles(prepared, canonical=True, isomericSmiles=True)


def bracket_dummy_atoms(smiles: str) -> str:
    """Prefer [*] over bare * for generated LibInvent templates."""
    return re.sub(r"(?<!\[)\*(?![:\]])", "[*]", smiles)


def atom_has_hydrogen(atom: rdchem.Atom) -> bool:
    explicit_h_count = sum(1 for nbr in atom.GetNeighbors() if nbr.GetAtomicNum() == 1)
    return max(atom.GetTotalNumHs(), explicit_h_count) > 0


def find_edit_sites(mol: Chem.Mol) -> list[EditSite]:
    sites: list[EditSite] = []

    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        if atomic_num <= 1:
            continue

        atom_idx = atom.GetIdx()

        if atom_has_hydrogen(atom):
            sites.append(EditSite("add", atom_idx=atom_idx, anchor_idx=atom_idx))

        if atom.GetDegree() == 1:
            bond = atom.GetBonds()[0]
            if bond.GetBondType() == rdchem.BondType.SINGLE:
                neighbor = atom.GetNeighbors()[0]
                if neighbor.GetAtomicNum() > 1:
                    sites.append(
                        EditSite("replace", atom_idx=atom_idx, anchor_idx=neighbor.GetIdx())
                    )

    return sites


def has_conflict(sites: Sequence[EditSite]) -> bool:
    """Avoid impossible edits and multiple attachment points on one anchor atom."""
    replaced_atoms = {site.atom_idx for site in sites if site.kind == "replace"}
    anchors = [site.anchor_idx for site in sites]

    if len(anchors) != len(set(anchors)):
        return True

    for site in sites:
        if site.kind == "add" and site.atom_idx in replaced_atoms:
            return True
        if site.anchor_idx in replaced_atoms:
            return True

    return False


def apply_sites(mol: Chem.Mol, sites: Sequence[EditSite]) -> str | None:
    rw_mol = Chem.RWMol(mol)

    for site in sites:
        if site.kind == "replace":
            dummy = Chem.Atom(0)
            dummy.SetNoImplicit(True)
            rw_mol.ReplaceAtom(site.atom_idx, dummy, preserveProps=False)

    for site in sites:
        if site.kind == "add":
            dummy = Chem.Atom(0)
            dummy.SetNoImplicit(True)
            dummy_idx = rw_mol.AddAtom(dummy)
            rw_mol.AddBond(site.anchor_idx, dummy_idx, rdchem.BondType.SINGLE)

    edited = rw_mol.GetMol()

    try:
        Chem.SanitizeMol(edited)
    except Exception:
        return None

    smiles = Chem.MolToSmiles(edited, canonical=True, isomericSmiles=True)
    return bracket_dummy_atoms(smiles)


def unrank_combination(n: int, k: int, index: int) -> tuple[int, ...]:
    """Return the lexicographic combination at zero-based index."""
    if k < 0 or k > n:
        raise ValueError("Invalid combination size")

    result: list[int] = []
    start = 0

    for position in range(k):
        remaining = k - position - 1
        for candidate in range(start, n - remaining):
            count = math.comb(n - candidate - 1, remaining)
            if index < count:
                result.append(candidate)
                start = candidate + 1
                break
            index -= count

    return tuple(result)


def randomized_indices(total: int, rng: random.Random) -> Iterable[int]:
    if total <= 0:
        return
    if total == 1:
        yield 0
        return

    start = rng.randrange(total)
    step = rng.randrange(1, total)
    while math.gcd(step, total) != 1:
        step = rng.randrange(1, total)

    for offset in range(total):
        yield (start + offset * step) % total


def make_sidechain_templates(
    mols: Sequence[Chem.Mol], r_group_num: int, seed: int, unique_template_num: int
) -> list[str]:
    candidate_sets: list[list[EditSite]] = []
    combination_counts: list[int] = []

    for mol in mols:
        sites = find_edit_sites(mol)
        candidate_sets.append(sites)
        combination_counts.append(math.comb(len(sites), r_group_num) if len(sites) >= r_group_num else 0)

    total_combinations = sum(combination_counts)
    if total_combinations == 0:
        return []

    cumulative_counts = list(accumulate(combination_counts))
    rng = random.Random(seed)
    templates: list[str] = []
    seen: set[str] = set()

    for global_index in randomized_indices(total_combinations, rng):
        mol_idx = bisect.bisect_right(cumulative_counts, global_index)
        previous_count = cumulative_counts[mol_idx - 1] if mol_idx > 0 else 0
        local_index = global_index - previous_count

        sites = candidate_sets[mol_idx]
        selected_indices = unrank_combination(len(sites), r_group_num, local_index)
        selected_sites = [sites[idx] for idx in selected_indices]

        if has_conflict(selected_sites):
            continue

        template = apply_sites(mols[mol_idx], selected_sites)
        if template and template not in seen:
            templates.append(template)
            seen.add(template)

            if len(templates) >= unique_template_num:
                break

    return templates


def output_path_for(input_path: Path, mode: str, output_dir: Path) -> Path:
    return output_dir / f"{input_path.stem}_{mode}.smi"


def write_smiles(path: Path, smiles: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for smi in smiles:
            handle.write(f"{smi}\n")


def run_scaffold(input_path: Path, output_dir: Path) -> Path:
    mols = read_molecules(input_path)
    smiles = [smi for mol in mols if (smi := canonicalize_mol(mol))]
    output_path = output_path_for(input_path, "scaffold", output_dir)
    write_smiles(output_path, smiles)
    return output_path


def run_sidechain(
    input_path: Path, output_dir: Path, r_group_num: int, seed: int, unique_template_num: int
) -> Path:
    mols = read_molecules(input_path)
    templates = make_sidechain_templates(mols, r_group_num, seed, unique_template_num)
    output_path = output_path_for(input_path, "sidechain", output_dir)
    write_smiles(output_path, templates)
    return output_path


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def existing_input(value: str) -> Path:
    path = Path(value)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"input file does not exist: {value}")
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"input path is not a file: {value}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate REINVENT scaffold or sidechain template SMILES from SDF/SMI files."
    )
    parser.add_argument(
        "mode",
        choices=("sidechain", "scaffold"),
        help="sidechain: create [*]-annotated templates; scaffold: canonicalize input SMILES.",
    )
    parser.add_argument("inputs", nargs="+", type=existing_input, help="Input .sdf/.sd/.smi files.")
    parser.add_argument(
        "--output-dir",
        default=SCRIPT_DIR / "inputs",
        type=Path,
        help="Directory for generated .smi files. Defaults to this repository's inputs/ directory.",
    )
    parser.add_argument(
        "--r-group-num",
        type=positive_int,
        help="Number of [*] attachment points to add/replace per sidechain template.",
    )
    parser.add_argument("--seed", type=int, help="Random seed for sidechain template selection.")
    parser.add_argument(
        "--unique-template-num",
        type=positive_int,
        help="Maximum number of unique sidechain templates to write per input file.",
    )

    args = parser.parse_args()

    if args.mode == "sidechain":
        missing = [
            name
            for name, value in (
                ("--r-group-num", args.r_group_num),
                ("--seed", args.seed),
                ("--unique-template-num", args.unique_template_num),
            )
            if value is None
        ]
        if missing:
            parser.error(f"sidechain mode requires: {', '.join(missing)}")

    return args


def main() -> None:
    args = parse_args()
    require_rdkit()
    output_dir = args.output_dir

    for input_path in args.inputs:
        if args.mode == "scaffold":
            output_path = run_scaffold(input_path, output_dir)
        else:
            output_path = run_sidechain(
                input_path,
                output_dir,
                args.r_group_num,
                args.seed,
                args.unique_template_num,
            )

        print(output_path)


if __name__ == "__main__":
    main()
