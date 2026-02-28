from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')
plt.rcParams["font.sans-serif"] = ["SimHei"] # 正常显示中文
plt.rcParams["axes.unicode_minus"] = False # 正常显示负号

path=Path('/')
amino_acids = 'ARNDCEQGHILKMFPSTWYV'
all_dipeptides = [a + b for a in amino_acids for b in amino_acids]

def read_fasta(fname):
    with open(fname, "r") as f:
        records = ((rec.id, str(rec.seq))
                   for rec in SeqIO.parse(fname, "fasta"))
    return pd.DataFrame(records, columns=["Id", "Sequence"])

def get_dipeptide_freq_dict(seq):
    if len(seq) < 2:
        return {}
    dipeptides = [seq[i:i + 2] for i in range(len(seq) - 1)]
    total = len(dipeptides)
    freq_dict = Counter(dipeptides)
    return {k: v / total for k, v in freq_dict.items()}

def get_termini_composition(seq):
    """提取末端5个氨基酸的组成特征"""
    features = {}

    # N端前5个氨基酸（如果长度足够）
    n_term = seq[:5] if len(seq) >= 5 else seq
    for aa in amino_acids:
        features[f'Nterm_{aa}'] = n_term.count(aa) / len(n_term) if n_term else 0

    # C端后5个氨基酸（如果长度足够）
    c_term = seq[-5:] if len(seq) >= 5 else seq
    for aa in amino_acids:
        features[f'Cterm_{aa}'] = c_term.count(aa) / len(c_term) if c_term else 0

    return features

def process_data(dic):
    """
    针对小样本数据的优化版特征工程
    特征数控制在200维以内
    """
    # 1. 从Id列中提取标签
    dic['toxicity'] = dic['Id'].apply(lambda x: x.split('|')[1])
    dic=dic.drop(columns=['Id'])
    # 2. 计算长度
    dic['length'] = dic['Sequence'].apply(len)
    # 3. 计算质量，假设线性
    df=pd.read_csv(path / 'data/amino_acids.csv')
    """
     'Amino Acids: Formula, Molecular Weight', WebQC.Org, 10 February 2026, https://zh.webqc.org/aminoacids.php
    """
    df.set_index('Called', inplace=True)
    hydrone_mass = 18.01528
    dic['mass'] = dic['Sequence'].apply(
        lambda seq: sum(df.loc[aa, 'Mass'] for aa in seq) - (len(seq) - 1) * hydrone_mass
    )
    # 4. 创建ProteinAnalysis对象缓存（关键优化！）
    analyzers = dic['Sequence'].apply(ProteinAnalysis)

    # 5. 计算疏水性（GRAVY）
    dic['hydrophobicity'] = analyzers.apply(lambda x: x.gravy())

    # 6. 计算等电点（整个序列，不是平均！）
    dic['isoelectric_point'] = analyzers.apply(lambda x: x.isoelectric_point())

    # 7. 计算人体pH下的电荷
    dic['charge_at_pH7.4'] = analyzers.apply(lambda x: x.charge_at_pH(7.4))

    # 8. 计算二级结构倾向
    ss_fractions = analyzers.apply(lambda x: x.secondary_structure_fraction())
    dic['helix'] = ss_fractions.apply(lambda x: x[0])  # α-螺旋
    dic['turn'] = ss_fractions.apply(lambda x: x[1])   # 转角
    dic['sheet'] = ss_fractions.apply(lambda x: x[2])  # β-折叠

    # 9. 计算氨基酸组成
    for aa in amino_acids:
        dic[f'comp_{aa}'] = dic['Sequence'].apply(lambda seq, a=aa: seq.count(a) / len(seq))

    # 10. 计算二肽频率
    all_freqs = []
    for seq in dic['Sequence']:
        all_freqs.append(get_dipeptide_freq_dict(seq))

    # 计算每个二肽的方差
    dipeptide_variance = {}
    for dip in all_dipeptides:
        values = [freq.get(dip, 0) for freq in all_freqs]
        if np.var(values) > 0:  # 只保留有变化的特征
            dipeptide_variance[dip] = np.var(values)

    # 选择方差最大的前50个二肽
    top_dipeptides = sorted(dipeptide_variance.items(),
                            key=lambda x: x[1], reverse=True)[:50]
    selected_dipeptides = [d[0] for d in top_dipeptides]

    # 计算选中的二肽特征
    for dip in selected_dipeptides:
        dic[f'dip_{dip}'] = [freq.get(dip, 0) for freq in all_freqs]

    # 11. 理化分组特征简化版
    property_groups = {
        'hydrophobic': 'AILMFWYV',  # 疏水
        'hydrophilic': 'RKNDEQ',  # 亲水
        'neutral': 'GSTCP',  # 中性
        'charged': 'KRHDE',  # 带电
    }

    # 计算每组氨基酸的整体比例
    for group_name, aas in property_groups.items():
        dic[f'group_{group_name}_ratio'] = dic['Sequence'].apply(
            lambda seq: sum(seq.count(aa) for aa in aas) / len(seq)
        )

    # 12. 序列复杂度分析
    dic['seq_complexity'] = dic['Sequence'].apply(
        lambda seq: len(set(seq)) / len(seq)  # 运用集合类型计算独特氨基酸比例
    )
    dic['charged_ratio'] = dic['Sequence'].apply(
        lambda seq: sum(seq.count(aa) for aa in 'KRHDE') / len(seq)  # 带电氨基酸比例连带分析
    )

    # 13. 末端特征
    termini_features = dic['Sequence'].apply(get_termini_composition)
    termini_df = pd.DataFrame(termini_features.tolist())

    # 添加末端特征到主DataFrame
    for col in termini_df.columns:
        dic[col] = termini_df[col].values

    # 14. 简单已知毒性相关特征
    toxic_motifs = ['RR', 'KK', 'RK', 'KR', 'CXC', 'CC', 'RGD', 'KGD']
    for motif in toxic_motifs:
        if len(motif) == 2:  # 二肽
            dic[f'motif_{motif}'] = dic['Sequence'].apply(
                lambda seq: seq.count(motif) / max(1, len(seq) - 1)
            )
        else:  # 三肽或其他
            dic[f'motif_{motif}'] = dic['Sequence'].apply(
                lambda seq: seq.count(motif) / max(1, len(seq) - len(motif) + 1)
            )

    # 15. 输出简单的统计信息
    print(f"\n处理完成！")
    print(f"样本数: {len(dic)}")
    feature_cols = [col for col in dic.columns if col not in ['Sequence', 'toxicity']]
    print(f"特征数: {len(feature_cols)}")
    print(f"样本/特征比: {len(dic) / len(feature_cols):.1f}:1")

    return dic

if __name__ == "__main__":
    train_data = read_fasta(path / 'data/test2.fasta')
    processed_train = process_data(train_data)
    processed_train.to_csv(path / 'data/processed_test2.csv', index=False)