from Bio import SeqIO
import numpy as np

# 1. 读取FASTA文件
sequences = []
for record in SeqIO.parse("./data/test1.fasta", "fasta"):
    sequences.append(str(record.seq))

print(f"读取了 {len(sequences)} 条序列")
# print(f"第一条序列：{sequences[0][:50]}..."
print("前三条：")
print(sequences[0])
print(sequences[1])
print(sequences[2])