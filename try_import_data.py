"""Read sequences from a FASTA file and print a small safe summary.

This script is defensive: it checks how many sequences were found
before indexing and prints up to the first 3 sequences.
"""
from Bio import SeqIO
import pandas as pd
from pathlib import Path

fasta_path = Path(__file__).parent / "data" / "test1.fasta"
if not fasta_path.exists():
    raise FileNotFoundError(f"Fasta file not found: {fasta_path}")

# 1. 读取FASTA文件
sequences = [str(rec.seq) for rec in SeqIO.parse(str(fasta_path), "fasta")]

print(f"读取了 {len(sequences)} 条序列")
if len(sequences) == 0:
    print("No sequences found.")
else:
    print("前三条（如果有）：")
    for i, seq in enumerate(sequences[:3], start=1):
        print(f"{i}: {seq}")

data = pd.Series(sequences)
print("Pandas Series preview:")
print(data.head())
