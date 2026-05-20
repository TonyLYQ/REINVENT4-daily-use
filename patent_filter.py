#!/usr/bin/env python3
"""Filter PubChem patent-associated molecules from a SMILES CSV.

The input CSV must contain a column named ``SMILES``.  The script queries
PubChem PUG-REST for each unique SMILES string and removes rows whose matched
PubChem CID has ``PatentID`` cross-references.

Usage:
    python patent_filter.py /path/to/output.csv

Output:
    /path/to/output_patent_filtered.csv

Note:
    PubChem PatentID cross-references indicate that a molecule appears in
    patent-associated PubChem records.  This is a conservative cheminformatics
    filter, not a legal determination of active patent protection.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests


PUBCHEM_PUG_REST = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
REQUEST_TIMEOUT_SECONDS = 30
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 1.0
REQUEST_DELAY_SECONDS = 0.2


class PubChemQueryError(RuntimeError):
    """Raised when a PubChem request fails in a way that makes filtering unsafe."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read a CSV containing a SMILES column, remove molecules with PubChem "
            "PatentID cross-references, and write *_patent_filtered.csv next to it."
        )
    )
    parser.add_argument("csv_path", help="Path to the input CSV file containing a SMILES column.")
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


def request_json(
    session: requests.Session,
    method: str,
    url: str,
    *,
    empty_status_codes: set[int] | None = None,
    **kwargs: Any,
) -> dict[str, Any] | None:
    last_error: Exception | None = None
    empty_status_codes = empty_status_codes or set()

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.request(
                method,
                url,
                timeout=REQUEST_TIMEOUT_SECONDS,
                **kwargs,
            )
        except requests.RequestException as exc:
            last_error = exc
        else:
            if response.status_code in empty_status_codes:
                return None

            if response.status_code in {429, 500, 502, 503, 504}:
                last_error = PubChemQueryError(
                    f"PubChem temporary error {response.status_code}: {response.text[:200]}"
                )
            elif not response.ok:
                raise PubChemQueryError(
                    f"PubChem request failed with HTTP {response.status_code}: "
                    f"{response.text[:500]}"
                )
            else:
                try:
                    return response.json()
                except ValueError as exc:
                    raise PubChemQueryError(
                        f"PubChem returned non-JSON content for {url}: {response.text[:500]}"
                    ) from exc

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS * attempt)

    raise PubChemQueryError(f"PubChem request failed after {MAX_RETRIES} attempts: {last_error}")


def get_cids_for_smiles(session: requests.Session, smiles: str) -> list[int]:
    payload = request_json(
        session,
        "POST",
        f"{PUBCHEM_PUG_REST}/compound/smiles/cids/JSON",
        empty_status_codes={400, 404},
        data={"smiles": smiles},
    )

    if payload is None:
        return []

    cids = payload.get("IdentifierList", {}).get("CID", [])
    return [int(cid) for cid in cids]


def get_patent_ids_for_cid(session: requests.Session, cid: int) -> list[str]:
    payload = request_json(
        session,
        "GET",
        f"{PUBCHEM_PUG_REST}/compound/cid/{cid}/xrefs/PatentID/JSON",
        empty_status_codes={404},
    )

    if payload is None:
        return []

    patent_ids: set[str] = set()
    information = payload.get("InformationList", {}).get("Information", [])

    for item in information:
        patent_value = item.get("PatentID", [])

        if isinstance(patent_value, list):
            patent_ids.update(str(patent_id) for patent_id in patent_value if patent_id)
        elif patent_value:
            patent_ids.add(str(patent_value))

    return sorted(patent_ids)


def has_pubchem_patent_reference(session: requests.Session, smiles: str) -> bool:
    cids = get_cids_for_smiles(session, smiles)

    for cid in cids:
        patent_ids = get_patent_ids_for_cid(session, cid)
        if patent_ids:
            return True

        time.sleep(REQUEST_DELAY_SECONDS)

    return False


def normalize_smiles_value(value: Any) -> str:
    if pd.isna(value):
        return ""

    return str(value).strip()


def build_patent_flags(smiles_values: pd.Series) -> dict[str, bool]:
    unique_smiles = sorted({normalize_smiles_value(value) for value in smiles_values})
    unique_smiles = [smiles for smiles in unique_smiles if smiles]

    patent_flags: dict[str, bool] = {"": False}

    with requests.Session() as session:
        session.headers.update(
            {
                "User-Agent": (
                    "patent-filter/1.0 "
                    "(PubChem PUG-REST; filters SMILES with PatentID xrefs)"
                )
            }
        )

        total = len(unique_smiles)
        for index, smiles in enumerate(unique_smiles, start=1):
            patent_flags[smiles] = has_pubchem_patent_reference(session, smiles)
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
        patent_flags = build_patent_flags(data[smiles_column])
    except (OSError, ValueError, PubChemQueryError, pd.errors.ParserError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    normalized_smiles = data[smiles_column].map(normalize_smiles_value)
    patented_mask = normalized_smiles.map(lambda smiles: patent_flags.get(smiles, False))
    filtered_data = data.loc[~patented_mask].copy()

    output_path = make_output_path(csv_path)
    filtered_data.to_csv(output_path, index=False)

    print(f"Input rows: {len(data)}", file=sys.stderr)
    print(f"Removed patent-associated rows: {int(patented_mask.sum())}", file=sys.stderr)
    print(f"Output rows: {len(filtered_data)}", file=sys.stderr)
    print(f"Wrote: {output_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
