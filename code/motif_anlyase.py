import pandas as pd
import torch
from collections import defaultdict
from util import backbone_one_hot, dinuc_shuffle
import json
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def intervals_overlap(a, b):
    """Return True if two [start, stop) intervals have any overlap (at least 1 bp)."""
    start1, end1 = a
    start2, end2 = b
    return max(start1, start2) < min(end1, end2)

def compute_effect(promoter_seq, motif_start, motif_stop, predictor):
    """计算给定 motif 区间的功能效应"""
    original_motif = promoter_seq[motif_start-1:motif_stop]
    original_score = seq_get_score(promoter_seq, predictor)
    
    shuffled_motif = dinuc_shuffle(original_motif)
    perturbed_seq = promoter_seq[:motif_start-1] + shuffled_motif + promoter_seq[motif_stop:]
    perturbed_score = seq_get_score(perturbed_seq, predictor)
    
    return original_score - perturbed_score

def backbone_one_hot(seq):
    charmap = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
    encoded = np.zeros([len(charmap), len(seq)])
    for i in range(len(seq)):
        if seq[i] == 'M':
            encoded[:, i] = np.random.rand(4)
        else:
            if seq[i] != 'N':
                encoded[charmap[seq[i]], i] = 1
    return encoded

def seq_get_score(seq, model):
    one_hot = [backbone_one_hot(seq)]
    one_hot = torch.tensor(one_hot).float().to(device)
    pred = model(one_hot).squeeze(0).cpu().numpy()
    return pred

def process_fimo_with_effect(
    fimo_df,
    promoter_seq,
    predictor
):
    # --- Step 1: Load and filter FIMO ---
    df = fimo_df.copy()
    if df.empty:
        return [], []
    df['start'] = pd.to_numeric(df['start'], errors='coerce')
    df['stop'] = pd.to_numeric(df['stop'], errors='coerce')
    df = df.dropna(subset=['start', 'stop'])
    df['start'] = df['start'].astype(int)
    df['stop'] = df['stop'].astype(int)
    df = df.sort_values(by='p-value').reset_index(drop=True)

    # --- Step 2: Precompute effect for all motifs (optional but efficient) ---
    print("Computing functional effects for all candidate motifs...")
    effects = []
    for _, row in df.iterrows():
        eff = compute_effect(promoter_seq, row['start'], row['stop'], predictor)
        effects.append(eff)
    df['effect'] = effects

    # --- Step 3: Greedy selection with conflict resolution by effect ---
    kept = []  # list of indices in df that are kept

    for i in range(len(df)):
        current_row = df.iloc[i]
        current_interval = (current_row['start'], current_row['stop'])
        current_effect = current_row['effect']

        # Find all kept intervals that conflict (>=40% overlap)
        conflicts = []
        non_conflicts = []
        for j in kept:
            kept_row = df.iloc[j]
            kept_interval = (kept_row['start'], kept_row['stop'])
            if intervals_overlap(current_interval, kept_interval):
                conflicts.append(j)
            else:
                non_conflicts.append(j)

        if not conflicts:
            # No conflict, add it
            kept.append(i)
        else:
            # Get max effect among conflicts
            max_conflict_effect = max(df.iloc[j]['effect'] for j in conflicts)
            if current_effect > max_conflict_effect:
                # Replace all conflicting motifs with current
                kept = non_conflicts + [i]
                print(f"🔁 Replaced {len(conflicts)} motif(s) with stronger motif at [{current_row['start']}, {current_row['stop']}] (effect={current_effect:.3f})")
            else:
                # Skip current
                pass

    # --- Step 4: Extract final starts/stops and sort by start ---
    final_df = df.iloc[kept].copy()
    final_df = final_df.sort_values(by='start').reset_index(drop=True)

    starts = final_df['start'].astype(float).tolist()
    stops = final_df['stop'].astype(float).tolist()

    print(f"✅ Done. {len(starts)} motifs retained.")
    return starts, stops


def read_fasta_as_dict(fasta_path):
    seqs = {}
    current_id = None
    current_seq = []
    with open(fasta_path, 'r') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if current_id is not None:
                    seqs[current_id] = ''.join(current_seq).upper()
                current_id = line[1:].strip().split()[0]
                current_seq = []
            else:
                current_seq.append(line)
    if current_id is not None:
        seqs[current_id] = ''.join(current_seq).upper()
    return seqs

if __name__ == "__main__":
    fimo_file = "/home/zjli/experiment_2025_2/data/batch_enhancer/69_fimo.tsv"
    fasta_path = "/home/zjli/experiment_2025_2/data/batch_enhancer/K562_DMSO_top100_300bp.fasta"
    predictor_path = "/home/zjli/experiment_2025_2/model/results2/model/flexibledenselstm_K562_expr_flexibledenselstm.pth"
    output_jsonl = "/home/zjli/experiment_2025_2/data/batch_enhancer/K562_enhancer_motif.jsonl"

    # Load predictor

    predictor = torch.load(predictor_path).to(device)
    predictor.eval()
    for p in predictor.parameters():
        p.requires_grad = False

    # Load multi-sequence inputs
    fimo_df = pd.read_csv(fimo_file, sep='\t', comment='#')
    fasta_map = read_fasta_as_dict(fasta_path)

    # Run pipeline for each sequence_name and save JSONL
    total = 0
    with open(output_jsonl, 'w') as fw:
        for seq_name, sub_df in fimo_df.groupby('sequence_name'):
            seq_key = str(seq_name)
            if seq_key not in fasta_map:
                print(f"⚠️ sequence_name={seq_key} not found in FASTA, skip.")
                continue

            starts, stops = process_fimo_with_effect(
                fimo_df=sub_df,
                promoter_seq=fasta_map[seq_key],
                predictor=predictor
            )

            record = {
                "sequence_name": seq_key,
                "starts": starts,
                "stops": stops
            }
            fw.write(json.dumps(record) + '\n')
            total += 1

    print(f"✅ All done. {total} sequences saved to: {output_jsonl}")