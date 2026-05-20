#!/usr/bin/env python3
"""Filter PubChem patent-associated molecules from a SMILES CSV.

The input CSV must contain a column named ``SMILES``.  The script queries
PubChem through PubChemPy for each unique SMILES string and removes molecules
whose matched PubChem CID has ``PatentID`` cross-references.

Usage:
    python patent_filter.py /path/to/output.csv
    python patent_filter.py /path/to/output.csv --broad

Output:
    /path/to/output_patent_filtered.csv

Note:
    PubChem PatentID cross-references indicate that a molecule appears in
    patent-associated PubChem records.  This is a conservative cheminformatics
    filter, not a legal determination of active patent protection.

    With ``--broad``, the script removes every molecule that can be matched to
    any PubChem CID, without checking PatentID cross-references.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import pubchempy as pcp


MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 1.0
REQUEST_DELAY_SECONDS = 0.2


class PubChemQueryError(RuntimeError):
    """Raised when a PubChem request fails in a way that makes filtering unsafe."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read a CSV containing a SMILES column, remove molecules with PubChem "
            "patent associations, and write *_patent_filtered.csv next to it."
        )
    )
    parser.add_argument("csv_path", help="Path to the input CSV file containing a SMILES column.")
    parser.add_argument(
        "--broad",
        action="store_true",
        help=(
            "Broad filter mode: remove a row if its SMILES can be matched to any PubChem CID. "
            "Without this flag, rows are removed only when matched CIDs have PatentID xrefs."
        ),
    )
    return parser.parse_args()


def make_output_path(csv_path: Path) -> Path:
    return csv_path.with_name(f"{csv_path.stem}_patent_filtered{csv_path.suffix}")


def find_smiles_column(columns: list[str]) -> str:
    if "SMILES" in columns:
        return "SMILES"

    case_insensitive_matches = [column for column in columns if column.lower() == "smiles"]
    if len(case_insensitive_matches) == 1:
        return case_insensitive_matches[0]

    raise ValueError("Input CSV must contain a column named 'SMILES'.")


def pubchem_call_with_retries(description: str, func: Any) -> Any:
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return func()
        except (pcp.NotFoundError, pcp.BadRequestError):
            return None
        except (pcp.ServerBusyError, pcp.ServerError, pcp.TimeoutError, OSError) as exc:
            last_error = exc
        except pcp.PubChemPyError as exc:
            raise PubChemQueryError(f"PubChemPy failed while {description}: {exc}") from exc

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS * attempt)

    raise PubChemQueryError(
        f"PubChem request failed after {MAX_RETRIES} attempts while {description}: {last_error}"
    )


def get_cids_for_smiles(smiles: str) -> list[int]:
    compounds = pubchem_call_with_retries(
        f"looking up CIDs for SMILES {smiles!r}",
        lambda: pcp.get_compounds(smiles, namespace="smiles"),
    )

    if not compounds:
        return []

    cids: set[int] = set()
    for compound in compounds:
        try:
            cid = int(compound.cid)
        except (TypeError, ValueError):
            continue

        if cid > 0:
            cids.add(cid)

    return sorted(cids)


def get_patent_ids_for_cid(cid: int) -> list[str]:
    response = pubchem_call_with_retries(
        f"looking up PatentID xrefs for CID {cid}",
        lambda: pcp.request(cid, namespace="cid", operation="xrefs/PatentID"),
    )

    if response is None:
        return []

    try:
        payload = json.loads(response.read().decode("utf-8"))
    except (AttributeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PubChemQueryError(f"PubChem returned invalid PatentID JSON for CID {cid}: {exc}") from exc

    patent_ids: set[str] = set()
    information = payload.get("InformationList", {}).get("Information", [])

    for item in information:
        patent_value = item.get("PatentID", [])

        if isinstance(patent_value, list):
            patent_ids.update(str(patent_id) for patent_id in patent_value if patent_id)
        elif patent_value:
            patent_ids.add(str(patent_value))

    return sorted(patent_ids)


def should_remove_smiles(smiles: str, *, broad: bool) -> bool:
    cids = get_cids_for_smiles(smiles)

    if broad:
        return bool(cids)

    for cid in cids:
        patent_ids = get_patent_ids_for_cid(cid)
        if patent_ids:
            return True

        time.sleep(REQUEST_DELAY_SECONDS)

    return False


def normalize_smiles_value(value: Any) -> str:
    if pd.isna(value):
        return ""

    return str(value).strip()


def build_patent_flags(smiles_values: pd.Series, *, broad: bool) -> dict[str, bool]:
    unique_smiles = sorted({normalize_smiles_value(value) for value in smiles_values})
    unique_smiles = [smiles for smiles in unique_smiles if smiles]

    patent_flags: dict[str, bool] = {"": False}
    total = len(unique_smiles)

    for index, smiles in enumerate(unique_smiles, start=1):
        patent_flags[smiles] = should_remove_smiles(smiles, broad=broad)

        if broad:
            status = "cid-matched" if patent_flags[smiles] else "kept"
        else:
            status = "patent-associated" if patent_flags[smiles] else "kept"

        print(f"[{index}/{total}] {status}: {smiles}", file=sys.stderr)
        time.sleep(REQUEST_DELAY_SECONDS)

    return patent_flags


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv_path).expanduser().resolve()

    if not csv_path.exists():
        print(f"Error: input CSV does not exist: {csv_path}", file=sys.stderr)
        return 1

    if csv_path.suffix.lower() != ".csv":
        print(f"Error: input file must be a CSV file: {csv_path}", file=sys.stderr)
        return 1

    try:
        data = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        smiles_column = find_smiles_column(list(data.columns))
        patent_flags = build_patent_flags(data[smiles_column], broad=args.broad)
    except (OSError, ValueError, PubChemQueryError, pd.errors.ParserError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    normalized_smiles = data[smiles_column].map(normalize_smiles_value)
    patented_mask = normalized_smiles.map(lambda smiles: patent_flags.get(smiles, False))
    filtered_data = data.loc[~patented_mask].copy()

    output_path = make_output_path(csv_path)
    filtered_data.to_csv(output_path, index=False)

    print(f"Input rows: {len(data)}", file=sys.stderr)
    if args.broad:
        print(f"Removed CID-matched rows: {int(patented_mask.sum())}", file=sys.stderr)
    else:
        print(f"Removed patent-associated rows: {int(patented_mask.sum())}", file=sys.stderr)
    print(f"Output rows: {len(filtered_data)}", file=sys.stderr)
    print(f"Wrote: {output_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
