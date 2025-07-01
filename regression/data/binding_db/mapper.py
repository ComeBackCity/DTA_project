import pandas as pd
from collections import defaultdict

def map_failed_sequences_by_identity(csv_path, output_path="missing_sequence_to_sequence_x_mapping.csv"):
    df = pd.read_csv(csv_path)

    # Group by sequence only among missing ones
    seq_to_keys = defaultdict(list)

    for _, row in df.iterrows():
        seq = str(row["protein_sequence"]).strip()
        key = row["sequence_key"]
        seq_to_keys[seq].append(key)

    # Keep only those sequences with >1 failures (shared)
    filtered = {seq: keys for seq, keys in seq_to_keys.items() if len(keys) > 1}

    # Convert to DataFrame
    rows = []
    for keys in filtered.values():
        row = {f"sequence_key_{i}": key for i, key in enumerate(keys)}
        rows.append(row)

    if rows:
        out_df = pd.DataFrame(rows)
        out_df.to_csv(output_path, index=False)
        print(f"✅ Mapping saved to {output_path} with {len(out_df)} grouped sequences.")
    else:
        print("✅ No duplicate failed sequences found.")

map_failed_sequences_by_identity("missing_triplet_to_seqid_mapping.csv")