"""
scripts/train_full_dataset.py — Ingest raw CIC-IDS2017 CSVs and train full models.

Automates:
  1. Auto-discovery of downloaded Kaggle/UNB CSV files or ZIP archives in data/raw or ~/Downloads.
  2. Memory-safe ingestion across all 8 working-day traffic files.
  3. Preprocessing, infinite/NaN sanitization, and 66-feature schema alignment.
  4. End-to-end retraining of Binary, Multiclass (13 classes), and Isolation Forest models.
  5. Updating model weights in models/ and performance metrics in reports/.

Usage:
  python3 -m scripts.train_full_dataset
  python3 -m scripts.train_full_dataset --sample-size 50000   # Fast high-precision training
  python3 -m scripts.train_full_dataset --full                # Train on all 2.83M rows
"""

from __future__ import annotations
import argparse
import logging
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DATA_DIR,
    MODELS_DIR,
    RAW_DATA_DIR,
    PROCESSED_CSV,
    REPORTS_DIR,
    RANDOM_SEED,
    setup_logging,
)
from src.data_loader import discover_csv_files, load_single_csv, _normalize_column_name
from src.preprocess import run_preprocessing
from src.train import run_training as run_binary_training
from src.multiclass import run_multiclass_training
from src.anomaly import run_anomaly_experiment

logger = setup_logging()
log = logging.getLogger("ai_nids.train_full_dataset")


def discover_and_extract_downloads(target_dir: Path) -> List[Path]:
    """Look in ~/Downloads and data/raw for Kaggle or UNB CIC-IDS2017 archives."""
    target_dir.mkdir(parents=True, exist_ok=True)
    existing_csvs = list(target_dir.glob("*.csv"))
    if existing_csvs:
        log.info("Found %d CSV file(s) already in %s", len(existing_csvs), target_dir)
        return existing_csvs

    # Check user Downloads folder if accessible
    downloads_dir = Path.home() / "Downloads"
    if downloads_dir.exists():
        zip_candidates = list(downloads_dir.glob("*network-intrusion*.zip")) + \
                         list(downloads_dir.glob("*archive*.zip")) + \
                         list(downloads_dir.glob("*CIC*.zip")) + \
                         list(downloads_dir.glob("*MachineLearningCSV*.zip"))

        for zf_path in zip_candidates:
            try:
                log.info("Found potential dataset archive in Downloads: %s", zf_path.name)
                with zipfile.ZipFile(zf_path, "r") as zf:
                    csv_members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
                    if csv_members:
                        log.info("Extracting %d CSV files from %s to %s...", len(csv_members), zf_path.name, target_dir)
                        for m in csv_members:
                            filename = Path(m).name
                            with zf.open(m) as src_file, open(target_dir / filename, "wb") as dst_file:
                                shutil.copyfileobj(src_file, dst_file)
            except Exception as e:
                log.warning("Could not extract %s: %e", zf_path, e)

        # Also check for unzipped CSVs in Downloads
        csv_candidates = list(downloads_dir.glob("*WorkingHours*.csv")) + \
                         list(downloads_dir.glob("*ISCX.csv"))
        for c in csv_candidates:
            log.info("Copying %s to %s", c.name, target_dir)
            shutil.copy2(c, target_dir / c.name)

    return list(target_dir.glob("*.csv"))


def load_balanced_sample_from_csvs(
    csv_files: List[Path],
    samples_per_file: int = 15000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Load a balanced, representative subset across all working-day capture files."""
    rng = np.random.RandomState(random_seed)
    dfs = []
    total_loaded = 0

    log.info("Ingesting %d CIC-IDS2017 CSV file(s) (target ~%d samples/file)...", len(csv_files), samples_per_file)
    for i, fp in enumerate(csv_files, 1):
        try:
            log.info("[%d/%d] Reading %s...", i, len(csv_files), fp.name)
            # Read header first
            df_full = pd.read_csv(fp, nrows=100)
            norm_cols = [_normalize_column_name(c) for c in df_full.columns]
            
            # Find label column
            lbl_candidates = [c for c in df_full.columns if c.strip().lower() in ["label", "class"]]
            if not lbl_candidates:
                log.warning("Skipping %s: no label column found", fp.name)
                continue
            lbl_col = lbl_candidates[0]

            # Read file in chunks to maintain low memory footprint
            chunks = []
            rows_collected = 0
            for chunk in pd.read_csv(fp, chunksize=25000, low_memory=False):
                # Sample proportionally
                if samples_per_file and len(chunk) > samples_per_file:
                    sampled = chunk.sample(n=min(len(chunk), samples_per_file // 2), random_state=rng)
                else:
                    sampled = chunk
                chunks.append(sampled)
                rows_collected += len(sampled)
                if samples_per_file and rows_collected >= samples_per_file:
                    break

            if chunks:
                df_file = pd.concat(chunks, ignore_index=True)
                df_file["_source_file"] = fp.name
                dfs.append(df_file)
                total_loaded += len(df_file)
                log.info("  &rarr; Sampled %d rows from %s (Total so far: %d)", len(df_file), fp.name, total_loaded)
        except Exception as e:
            log.error("Failed to read %s: %s", fp.name, e)

    if not dfs:
        raise ValueError("Could not load any data from the provided CSV files.")

    combined = pd.concat(dfs, ignore_index=True)
    log.info("Combined multi-file dataset: %d rows, %d columns.", len(combined), len(combined.columns))
    return combined


def main():
    parser = argparse.ArgumentParser(description="AI-NIDS Full Dataset Training Pipeline")
    parser.add_argument("--data-dir", type=str, default=str(RAW_DATA_DIR), help="Directory containing raw CIC-IDS2017 CSVs")
    parser.add_argument("--sample-size", type=int, default=25000, help="Samples to extract per CSV file (0 for full)")
    parser.add_argument("--full", action="store_true", help="Ingest complete dataset without subsampling")
    args = parser.parse_args()

    raw_dir = Path(args.data_dir)
    csv_files = discover_and_extract_downloads(raw_dir)

    if not csv_files:
        print("\n" + "=" * 70)
        print("  NO RAW CIC-IDS2017 CSV FILES FOUND IN data/raw/")
        print("=" * 70)
        print("To train on the official CIC-IDS2017 dataset from Kaggle / UNB:")
        print("  1. Download the dataset from Kaggle (chethuhn/network-intrusion-dataset)")
        print("     or UNB: http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/MachineLearningCSV.zip")
        print(f"  2. Extract the CSV files into: {raw_dir.resolve()}")
        print("  3. Run this script again: python3 -m scripts.train_full_dataset")
        print("=" * 70 + "\n")
        sys.exit(1)

    print(f"\nDiscovered {len(csv_files)} CSV files in {raw_dir}:")
    for f in csv_files:
        print(f"  • {f.name}")

    samples_per_file = 0 if args.full else args.sample_size
    raw_df = load_balanced_sample_from_csvs(csv_files, samples_per_file=samples_per_file)

    print("\nRunning standard preprocessing pipeline (Sanitization, NaN/Inf handling, Label Hygiene)...")
    processed_df, summary = run_preprocessing(raw_df)
    print(f"Preprocessing complete. Clean feature matrix: {len(processed_df)} rows × {len(processed_df.columns)} columns.")

    print("\n" + "=" * 60)
    print("  TRAINING STAGE 1: SUPERVISED BINARY INTRUSION DETECTOR")
    print("=" * 60)
    b_results, b_name, b_meta = run_binary_training(processed_df, force_synthetic=False)

    print("\n" + "=" * 60)
    print("  TRAINING STAGE 2: SUPERVISED MULTICLASS CLASSIFIER (13 CLASSES)")
    print("=" * 60)
    m_results, m_name, m_meta = run_multiclass_training(processed_df, force_synthetic=False)

    print("\n" + "=" * 60)
    print("  TRAINING STAGE 3: UNSUPERVISED ISOLATION FOREST ANOMALY SCORER")
    print("=" * 60)
    from src.feature_engineering import separate_features_target, build_label_mapping
    from sklearn.model_selection import train_test_split
    from src.config import BINARY_TARGET_COLUMN, ORIGINAL_LABEL_COLUMN
    
    clean_df, feature_names = separate_features_target(processed_df)
    y_bin = processed_df[BINARY_TARGET_COLUMN]
    y_multi = processed_df[ORIGINAL_LABEL_COLUMN] if ORIGINAL_LABEL_COLUMN in processed_df.columns else None
    
    X_tr, X_te, y_tr, y_te = train_test_split(clean_df, y_bin, test_size=0.2, random_state=RANDOM_SEED, stratify=y_bin)
    y_te_multi = y_multi.loc[X_te.index] if y_multi is not None else None
    
    a_pipeline, a_info, a_results = run_anomaly_experiment(
        X_tr, y_tr, X_te, y_te,
        y_test_multiclass=y_te_multi,
        feature_names=feature_names,
        save_artifacts=True,
    )

    print("\n" + "=" * 70)
    print("  RETRAINING COMPLETE! All models updated in models/")
    print("=" * 70)
    print(f"  • Binary Model Pipeline:      {MODELS_DIR / 'binary_intrusion_pipeline.joblib'}")
    print(f"  • Multiclass Model Pipeline:  {MODELS_DIR / 'multiclass_intrusion_pipeline.joblib'}")
    print(f"  • Isolation Forest Anomaly:   {MODELS_DIR / 'isolation_forest.joblib'}")
    print(f"  • Reports & Telemetry:        {REPORTS_DIR}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
