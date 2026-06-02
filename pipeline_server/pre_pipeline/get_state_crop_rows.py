#!/usr/bin/env python3
import argparse
import pandas as pd

parser = argparse.ArgumentParser(
    description="Filter rows by Crop and/or StateName. If not provided, all values are included."
)

parser.add_argument("--input", required=True, help="Input CSV file")
parser.add_argument("--output", required=True, help="Output CSV file")
parser.add_argument("--crop", help="Regex pattern to match crop names (optional)")
parser.add_argument("--state", help="Comma-separated state name(s) to filter (optional)")
parser.add_argument("--district", help="Comma-separated district name(s) to filter (optional)")
parser.add_argument("--domains", help="Comma-separated QueryType values to filter (optional)")

args = parser.parse_args()

# Read CSV
df = pd.read_csv(
    args.input,
    low_memory=False,
    encoding="utf-8",
    encoding_errors="ignore"
)

# Ensure required columns exist
if "Crop" not in df.columns:
    raise ValueError("Column 'Crop' not found")

if "StateName" not in df.columns:
    raise ValueError("Column 'StateName' not found")

# Normalize dataframe columns
df["Crop"] = (
    df["Crop"]
    .astype(str)
    .str.strip()
    .str.lower()
)

df["StateName"] = (
    df["StateName"]
    .astype(str)
    .str.strip()
    .str.lower()
)

# Start with all rows
df_filtered = df.copy()

# Apply Crop filter if provided
if args.crop:
    df_filtered = df_filtered[df_filtered["Crop"].str.contains(args.crop, case=False, regex=True, na=False)]

# Apply State filter if provided
if args.state:
    target_states = [s.strip().lower() for s in args.state.split(",") if s.strip()]
    df_filtered = df_filtered[df_filtered["StateName"].isin(target_states)]

# Apply District filter if provided
if args.district:
    if "DistrictName" not in df_filtered.columns:
        raise ValueError("Column 'DistrictName' not found")
    target_districts = [d.strip().lower() for d in args.district.split(",") if d.strip()]
    df_filtered["DistrictName"] = df_filtered["DistrictName"].astype(str).str.strip().str.lower()
    df_filtered = df_filtered[df_filtered["DistrictName"].isin(target_districts)]

# Apply Domain (QueryType) filter if provided
if args.domains:
    if "QueryType" not in df_filtered.columns:
        raise ValueError("Column 'QueryType' not found")
    target_domains = [d.strip().lower() for d in args.domains.split(",") if d.strip()]
    df_filtered["QueryType"] = df_filtered["QueryType"].astype(str).str.strip().str.lower()
    df_filtered = df_filtered[df_filtered["QueryType"].isin(target_domains)]

# Save output
df_filtered.to_csv(args.output, index=False)

print(
    f"{len(df_filtered):,} rows written to {args.output}"
)