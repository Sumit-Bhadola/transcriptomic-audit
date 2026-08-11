"""Data loading for the transcriptomic reproducibility audit.

Handles TCGA-BRCA (UCSC Xena) with Parquet caching, so the slow TSV
parse happens exactly once.

Usage:
    python src/data.py
"""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

META_COLS = {"patient", "tss", "sample_type", "plate", "subtype"}

# The Xena BRCA clinical matrix contains three columns mentioning PAM50.
# This is the one with RNAseq-based calls and the widest sample coverage.
LABEL_COL = "PAM50Call_RNAseq"


def _find(*names: str) -> Path:
    """Locate a raw file, tolerating .gz variants."""
    for name in names:
        for candidate in (RAW / name, RAW / f"{name}.gz"):
            if candidate.exists():
                return candidate
    available = sorted(p.name for p in RAW.iterdir()) if RAW.exists() else []
    raise FileNotFoundError(
        f"Could not find any of {names} in {RAW}.\nFiles present: {available}"
    )


def parse_barcode(barcode: str) -> dict:
    """Split a TCGA barcode into its batch-relevant components.

    TCGA-A2-A0T2-01A-11R-A084-07
         ^^         ^^      ^^^^
         TSS        type    plate
    """
    p = barcode.split("-")
    return {
        "patient": "-".join(p[:3]),
        "tss": p[1] if len(p) > 1 else None,
        "sample_type": p[3][:2] if len(p) > 3 else None,
        "plate": p[5] if len(p) > 5 else None,
    }


def load_tcga(force: bool = False) -> pd.DataFrame:
    """Return a samples x genes frame with subtype and batch columns."""
    cache = PROCESSED / "tcga_brca.parquet"
    if cache.exists() and not force:
        return pd.read_parquet(cache)

    expr_path = _find("HiSeqV2", "HiSeqV2_PANCAN")
    clin_path = _find("BRCA_clinicalMatrix")

    print(f"Reading {expr_path.name} ...")
    expr = pd.read_csv(expr_path, sep="\t", index_col=0).T
    expr.index.name = "sampleID"

    print(f"Reading {clin_path.name} ...")
    clin = pd.read_csv(clin_path, sep="\t", index_col=0, low_memory=False)

    # Prefer the exact RNAseq call. Never take the first fuzzy match:
    # "Integrated_Clusters_with_PAM50__nature2012" also contains "PAM50"
    # and would silently give you the wrong labels.
    if LABEL_COL in clin.columns:
        label_col = LABEL_COL
    else:
        candidates = [c for c in clin.columns if "PAM50" in c.upper()]
        if not candidates:
            raise KeyError(
                "No PAM50 column found. Available columns:\n"
                + "\n".join(sorted(clin.columns))
            )
        label_col = candidates[0]
        print(f"WARNING: {LABEL_COL} missing, falling back to {label_col}")
    print(f"Using label column: {label_col}")

    df = expr.join(clin[[label_col]], how="inner")
    df = df[df[label_col].notna()]
    df = df.rename(columns={label_col: "subtype"})

    meta = pd.DataFrame([parse_barcode(s) for s in df.index], index=df.index)
    df = pd.concat([meta, df], axis=1)

    # Primary solid tumour only (drops normals and metastases)
    df = df[df["sample_type"] == "01"]

    gene_cols = [c for c in df.columns if c not in META_COLS]
    df[gene_cols] = df[gene_cols].astype("float32")

    # Xena HiSeqV2 is already log2(norm_count + 1). Fail loudly otherwise.
    vmax = float(df[gene_cols].to_numpy().max())
    if vmax > 40:
        raise ValueError(
            f"Values look unlogged (max={vmax:.1f}). Do NOT log-transform "
            "again without checking the dataset description on Xena."
        )

    df.to_parquet(cache)
    print(f"Cached to {cache}")
    return df


def split_xy(df: pd.DataFrame):
    """Separate expression from labels and grouping variables.

    Returns (X, y, tss, patient). Pass tss or patient as `groups` to
    StratifiedGroupKFold once you build the corrected protocol.
    """
    genes = [c for c in df.columns if c not in META_COLS]
    return df[genes], df["subtype"], df["tss"], df["patient"]


if __name__ == "__main__":
    df = load_tcga()
    X, y, tss, patient = split_xy(df)

    print(f"\nsamples: {len(df):,}   genes: {X.shape[1]:,}")
    print(f"patients: {patient.nunique():,}   sites: {tss.nunique()}")
    print(f"duplicate patients: {len(df) - patient.nunique()}")

    print("\nsubtype counts:")
    print(y.value_counts())
    ratio = y.value_counts().max() / y.value_counts().min()
    print(f"\nimbalance ratio: {ratio:.1f}x")

    print("\ntop 10 sites:")
    print(tss.value_counts().head(10))