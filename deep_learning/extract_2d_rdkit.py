import os
import torch
import numpy as np
from Bio import SeqIO
from rdkit import Chem
from tqdm import tqdm
from pathlib import Path

path = Path('D:/Python/dachuang2026')

# ==========================================
# 配置参数
# ==========================================
FASTA_FILES = [
    path / "data/train_data.fasta",  # 请根据你的实际路径修改
    path / "data/test1.fasta",
    path / "data/test2.fasta"
]

# 特征保存的根目录
SAVE_DIR = path / "data/features"
os.makedirs(SAVE_DIR, exist_ok=True)

MAX_ATOMS = 100  # 我们之前模型设定的最大原子数 (不足补0，超出截断)
FEATURE_DIM = 21  # 方案书设定的 21 维理化特征


# ==========================================
# 定义 21维 原子特征提取函数
# ==========================================
def atom_features(atom):
    """提取单个原子的 21 维特征 (简化示例，涵盖核心理化属性)"""
    # 1. 原子类型 (One-hot 模拟, 常见于肽段的 C, N, O, S, H 等)
    symbol = atom.GetSymbol()
    types = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I']
    type_idx = types.index(symbol) if symbol in types else len(types)
    type_one_hot = [1 if i == type_idx else 0 for i in range(len(types) + 1)]  # 10维

    # 2. 杂化状态 (SP, SP2, SP3 等)
    hybridization = atom.GetHybridization()
    hyb_types = [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3]
    hyb_one_hot = [1 if hybridization == h else 0 for h in hyb_types]  # 3维

    # 3. 其他数值特征 (8维)
    degree = atom.GetDegree()  # 连接数
    formal_charge = atom.GetFormalCharge()  # 形式电荷
    num_hs = atom.GetTotalNumHs()  # 连接的氢原子数
    is_aromatic = 1 if atom.GetIsAromatic() else 0  # 是否在芳香环上
    mass = atom.GetMass()  # 原子质量
    vdw_radius = Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum())  # 范德华半径
    is_in_ring = 1 if atom.IsInRing() else 0  # 是否成环
    implicit_valence = atom.GetValence(Chem.ValenceType.IMPLICIT) # 隐式价数 (新版语法)

    other_features = [degree, formal_charge, num_hs, is_aromatic, mass, vdw_radius, is_in_ring, implicit_valence]

    # 组合成 21 维 (10 + 3 + 8 = 21)
    return type_one_hot + hyb_one_hot + other_features


# ==========================================
# 主流程：提取并保存
# ==========================================
def extract_and_save_2d():
    for fasta_file in FASTA_FILES:
        if not os.path.exists(fasta_file):
            continue

        print(f"\n开始处理 2D 特征: {fasta_file}")
        records = list(SeqIO.parse(fasta_file, "fasta"))

        for record in tqdm(records, desc="Extracting 2D"):
            safe_id = record.id.replace("|", "_").replace("/", "_")
            save_path = os.path.join(SAVE_DIR, f"{safe_id}_2d.pt")

            # 支持断点续传
            if os.path.exists(save_path):
                continue

            seq = str(record.seq).upper()

            # RDKit 直接将氨基酸序列转为分子对象
            mol = Chem.MolFromSequence(seq)

            # 初始化 (FEATURE_DIM, MAX_ATOMS) 的全零张量
            feat_2d = np.zeros((FEATURE_DIM, MAX_ATOMS), dtype=np.float32)

            if mol is not None:
                # 遍历分子中的每一个原子
                atoms = mol.GetAtoms()
                num_atoms_to_process = min(len(atoms), MAX_ATOMS)

                for i in range(num_atoms_to_process):
                    atom_feat = atom_features(atoms[i])
                    feat_2d[:, i] = atom_feat  # 填充每一列

            # 转为 PyTorch 张量并保存
            feat_2d_tensor = torch.from_numpy(feat_2d)
            torch.save(feat_2d_tensor, save_path)


if __name__ == "__main__":
    extract_and_save_2d()
    print("\n🎉 所有序列的 2D RDKit 特征提取完毕！")