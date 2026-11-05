#!/usr/bin/env python3
"""
Data Preprocessing for Bengaluru Air Quality Analysis
"""
import pandas as pd
import numpy as np
import os
from glob import glob

# Set directories
raw_dir = '../data/raw'
processed_dir = '../data/processed'
os.makedirs(processed_dir, exist_ok=True)

# Load all blr-* files (newer format)
files = glob(os.path.join(raw_dir, 'blr-*.csv'))
print(f"Found {len(files)} files")

all_dfs = []
for f in files:
    try:
        df = pd.read_csv(f)
        print(f"Loaded {os.path.basename(f)}: {df.shape}")
        all_dfs.append(df)
    except Exception as e:
        print(f"Error loading {f}: {e}")

# Combine all data
if all_dfs:
    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\nCombined shape: {combined.shape}")
    
    # Save combined raw data
    combined.to_csv(os.path.join(processed_dir, 'combined_raw.csv'), index=False)
    print("Saved combined_raw.csv")
    
    # Basic info
    print("\n=== Data Info ===")
    print(combined.info())
    print("\n=== Missing Values ===")
    print(combined.isnull().sum())
else:
    print("No data loaded!")
