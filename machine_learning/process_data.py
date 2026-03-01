import warnings
import json
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# 全局配置
warnings.filterwarnings('ignore')
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 正常显示中文
plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号

DEFAULT_PATH = Path('D:/Python/dachuang2026')  # 显示项目路径
AMINO_ACIDS = 'ARNDCEQGHILKMFPSTWYV'
ALL_DIPEPTIDES = [a + b for a in AMINO_ACIDS for b in AMINO_ACIDS]  # 400种二肽组合


def read_fasta(fname):
    """读取FASTA文件并转化为DataFrame"""
    records = []
    for rec in SeqIO.parse(fname, "fasta"):
        records.append((rec.id, str(rec.seq).upper()))
        # .upper()确保序列大写，避免后续处理中的大小写问题
    return pd.DataFrame(records, columns=["Id", "Sequence"])


def get_dipeptide_freq_dict(seq):
    """高效计算二肽频率"""
    if len(seq) < 2:
        return {}
    dipeptides = [seq[i:i + 2] for i in range(len(seq) - 1)]
    total = len(dipeptides)
    return {k: v / total for k, v in Counter(dipeptides).items()}


def get_termini_composition(seq):
    """提取末端5个氨基酸的组成特征"""
    features = {}
    n_term = seq[:5] if len(seq) >= 5 else seq
    c_term = seq[-5:] if len(seq) >= 5 else seq

    for aa in AMINO_ACIDS:
        features[f'Nterm_{aa}'] = n_term.count(aa) / len(n_term) if n_term else 0
        features[f'Cterm_{aa}'] = c_term.count(aa) / len(c_term) if c_term else 0
    return features


def process_data(dic, feature_config=None):
    """
    针对肽毒性预测优化的特征工程（含二级结构语法修正与两亲性特征）
    :param dic: 原始 DataFrame，需包含 'Id' (可选) 和 'Sequence' 列
    :param feature_config: 字典类型，用于预测模式下对齐特征列
    """
    # 1. 标签提取与清理
    if 'Id' in dic.columns:
        dic['toxicity'] = dic['Id'].apply(lambda x: x.split('|')[1] if '|' in x else '0')
        dic = dic.drop(columns=['Id'])

    # 确保序列为大写，防止计算出错
    dic['Sequence'] = dic['Sequence'].str.upper()

    # 2. 基础物理化学特征
    dic['length'] = dic['Sequence'].apply(len)

    # 质量计算优化：使用字典映射提升速度
    df_aa = pd.read_csv(DEFAULT_PATH / 'data/amino_acids.csv').set_index('Called')
    mass_dict = df_aa['Mass'].to_dict()
    hydrone_mass = 18.01528

    dic['mass'] = dic['Sequence'].apply(
        lambda seq: sum(mass_dict.get(aa, 0) for aa in seq) - (len(seq) - 1) * hydrone_mass
    )

    # 3. 创建 ProteinAnalysis 对象缓存（增加空序列检查）
    analyzers = dic['Sequence'].apply(lambda x: ProteinAnalysis(x) if len(x) > 0 else None)

    # 4. 理化指标计算
    dic['hydrophobicity'] = analyzers.apply(lambda x: x.gravy() if x else 0)
    dic['isoelectric_point'] = analyzers.apply(lambda x: x.isoelectric_point() if x else 7.0)
    dic['charge_at_pH7.4'] = analyzers.apply(lambda x: x.charge_at_pH(7.4) if x else 0.0)

    # 5. 二级结构特征（语法修正版）
    # 获取 (helix, turn, sheet) 元组列表
    ss_results = analyzers.apply(lambda x: x.secondary_structure_fraction() if x else (0.0, 0.0, 0.0))
    # 转换为 DataFrame 并合并，确保索引对齐
    ss_df = pd.DataFrame(ss_results.tolist(), columns=['helix', 'turn', 'sheet'], index=dic.index)
    dic = pd.concat([dic, ss_df], axis=1)

    # 6. 两亲性模拟特征 (新增)
    # 通过比较序列两端的疏水性差异来模拟两亲性
    dic['amphiphilicity_index'] = dic['Sequence'].apply(
        lambda seq: abs(ProteinAnalysis(seq[:len(seq) // 2]).gravy() -
                        ProteinAnalysis(seq[len(seq) // 2:]).gravy()) if len(seq) >= 4 else 0.0
    )

    # 7. 氨基酸组成频率 (AAC)
    for aa in AMINO_ACIDS:
        dic[f'comp_{aa}'] = dic['Sequence'].apply(lambda seq: seq.count(aa) / len(seq) if len(seq) > 0 else 0)

    # 8. 二肽频率 (DPC) 与特征对齐逻辑
    all_freqs = [get_dipeptide_freq_dict(seq) for seq in dic['Sequence']]

    if feature_config is None:
        # 训练模式：计算方差并筛选前 50 个特征
        dipeptide_variance = {}
        for dip in ALL_DIPEPTIDES:
            vals = [f.get(dip, 0) for f in all_freqs]
            if np.var(vals) > 0:
                dipeptide_variance[dip] = np.var(vals)

        top_dips = sorted(dipeptide_variance.items(), key=lambda x: x[1], reverse=True)[:50]
        selected_dipeptides = [d[0] for d in top_dips]
        current_config = {'top_dipeptides': selected_dipeptides}
    else:
        # 预测模式：使用传入的配置，确保列名一致
        selected_dipeptides = feature_config.get('top_dipeptides', [])
        current_config = feature_config

    for dip in selected_dipeptides:
        dic[f'dip_{dip}'] = [f.get(dip, 0) for f in all_freqs]

    # 9. 序列复杂度与分组特征
    dic['seq_complexity'] = dic['Sequence'].apply(lambda seq: len(set(seq)) / len(seq) if len(seq) > 0 else 0)
    dic['charged_ratio'] = dic['Sequence'].apply(
        lambda seq: sum(seq.count(aa) for aa in 'KRHDE') / len(seq) if len(seq) > 0 else 0
    )

    # 10. 末端组成特征合并优化
    termini_features = dic['Sequence'].apply(get_termini_composition)
    termini_df = pd.DataFrame(termini_features.tolist(), index=dic.index)
    dic = pd.concat([dic, termini_df], axis=1)

    # 11. 毒性相关 Motif
    toxic_motifs = ['RR', 'KK', 'RK', 'KR', 'CXC', 'CC', 'RGD']
    for motif in toxic_motifs:
        dic[f'motif_{motif}'] = dic['Sequence'].apply(
            lambda seq: seq.count(motif) / max(1, len(seq) - len(motif) + 1)
        )

    # 输出统计信息
    print(f"处理完成！样本数: {len(dic)}")
    feature_cols = [col for col in dic.columns if col not in ['Sequence', 'toxicity']]
    print(f"特征维度: {len(feature_cols)}")

    return dic, current_config


if __name__ == "__main__":
    # 执行流程
    print("开始读取数据...")
    dic = read_fasta(DEFAULT_PATH / 'data/test2.fasta')
    print(dic.head())

    print("\n开始处理数据...")
    processed_train, train_config = process_data(dic)
    print(processed_train.head())

    # 保存结果
    output_csv_path = DEFAULT_PATH / 'data/processed_train_data_done.csv'
    output_json_path = DEFAULT_PATH / 'data/feature_config.json'

    processed_train.to_csv(output_csv_path, index=False)
    with open(output_json_path, 'w') as f:
        json.dump(train_config, f)

    print(f"\n数据已保存至:\nCSV: {output_csv_path}\nJSON: {output_json_path}")
    print(f"最终数据形状: {processed_train.shape}")