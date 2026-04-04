import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Bio import SeqIO
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, ConcatDataset # 引入 ConcatDataset

path = Path('D:/Python/dachuang2026')
# ==========================================
# 1. 真实 FASTA 数据集加载器
# 读取你的 .fasta 文件并转化为模型输入
# ==========================================

class RealFastaDataset(Dataset):
    def __init__(self, fasta_files, max_seq_len=50):
        super(RealFastaDataset, self).__init__()
        self.max_seq_len = max_seq_len
        self.sequences = []
        self.labels = []
        self.ids = []

        # 【修复点】：不管传入的是普通的字符串路径，还是 WindowsPath 对象，
        # 只要它不是一个列表或元组，我们就把它强行变成列表。
        if not isinstance(fasta_files, (list, tuple)):
            fasta_files = [fasta_files]

        for fasta_file in fasta_files:
            # 安全起见：把可能存在的 WindowsPath 对象强制转换回标准字符串
            fasta_file_str = str(fasta_file)
            print(f"正在加载数据集: {fasta_file_str} ...")

            for record in SeqIO.parse(fasta_file_str, "fasta"):
                seq = str(record.seq).upper()
                label = 1.0 if "tox" in record.description.lower() or "|1" in record.description else 0.0

                self.sequences.append(seq)
                self.labels.append(label)
                self.ids.append(record.id)

        self.num_samples = len(self.sequences)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 1. 解析出安全的 ID (和提取脚本保持一致)
        seq_id = self.ids[idx]
        safe_id = seq_id.replace("|", "_").replace("/", "_")

        label = torch.tensor([self.labels[idx]], dtype=torch.float32)

        # 2. 【核心修改】读取真实的 1D 特征！
        # 注意：确保这里的路径与你电脑上 .pt 文件的真实路径一致
        feat_1d = torch.load(path / f"data/features/{safe_id}_1d.pt")

        # 3. 2D 和 3D 暂时还保留随机数占位符（我们接下来解决）
        feat_2d = torch.load(path / f"data/features/{safe_id}_2d.pt")
        feat_3d_n = torch.zeros(self.max_seq_len, 45)
        feat_3d_c = torch.zeros(self.max_seq_len, 3)
        plddt = torch.tensor([50.0], dtype=torch.float32)

        return feat_1d, feat_2d, feat_3d_n, feat_3d_c, plddt, label

# class RealFastaDataset(Dataset):
#     def __init__(self, fasta_file, max_seq_len=50):
#         super(RealFastaDataset, self).__init__()
#         self.max_seq_len = max_seq_len
#         self.sequences = []
#         self.labels = []
#         self.ids = []
#
#
#
#         # 解析 FASTA 文件
#         print(f"正在加载数据集: {fasta_file} ...")
#         for record in SeqIO.parse(fasta_file, "fasta"):
#             seq = str(record.seq).upper()
#
#             # 【注意】这里假设 FASTA 的 header 包含了标签，例如 ">seq1|1" 或 ">seq1 tox"
#             # 如果你的 header 没有标签，你需要根据你的实际情况修改这里的解析逻辑
#             # 这里做一个通用防呆处理：如果解析不到，默认设为 0 (非毒性)
#             label = 1.0 if "tox" in record.description.lower() or "|1" in record.description else 0.0
#
#             self.sequences.append(seq)
#             self.labels.append(label)
#
#         self.num_samples = len(self.sequences)
#         print(f"成功加载 {self.num_samples} 条多肽序列。")
#
#     def __len__(self):
#         return self.num_samples
#
#     def __getitem__(self, idx):
#         seq = self.sequences[idx]
#         label = torch.tensor([self.labels[idx]], dtype=torch.float32)
#
#         # -----------------------------------------------------------
#         # 【特征工程占位区】
#         # 在真实项目中，你需要在这里把序列(seq)扔进 ESM2/RDKit/ESMFold。
#         # 为了演示代码能直接运行，这里针对该序列生成对应维度的随机特征。
#         # -----------------------------------------------------------
#
#         # 1D: 模拟 ESM2/ProtT5 提取的 Embedding (真实情况应为预计算好的向量加载)
#         # 固定长度 padding 逻辑模拟
#         actual_len = min(len(seq), self.max_seq_len)
#         feat_1d = torch.zeros(self.max_seq_len, 1024)
#         feat_1d[:actual_len, :] = torch.randn(actual_len, 1024)
#
#         # 2D: 模拟 RDKit 提取的原子级特征图
#         feat_2d = torch.randn(21, 100)
#
#         # 3D: 模拟 ESMFold 预测的节点特征和坐标，以及 pLDDT 评分
#         feat_3d_n = torch.zeros(self.max_seq_len, 45)
#         feat_3d_n[:actual_len, :] = torch.randn(actual_len, 45)
#         feat_3d_c = torch.zeros(self.max_seq_len, 3)
#         feat_3d_c[:actual_len, :] = torch.randn(actual_len, 3)
#
#         # 模拟 ESMFold 跑出来的结构置信度 pLDDT (0-100)
#         plddt = torch.tensor([np.random.uniform(40, 95)], dtype=torch.float32)
#
#         return feat_1d, feat_2d, feat_3d_n, feat_3d_c, plddt, label


# ==========================================
# 2. MSCF-Tox 模型定义 (保持不变，使用前文的架构)
# ==========================================
class Extractor1D(nn.Module):
    def __init__(self, input_dim=1280, hidden_dim=256):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim // 2, bidirectional=True, batch_first=True)
        self.transformer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.transformer(out)
        return out.mean(dim=1)


class Extractor2D(nn.Module):
    def __init__(self, in_channels=21, hidden_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        return self.conv(x).squeeze(-1)


class Extractor3D(nn.Module):
    def __init__(self, node_dim=45, hidden_dim=256):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(node_dim, 128), nn.ReLU(), nn.Linear(128, hidden_dim), nn.ReLU())

    def forward(self, nodes, coords):
        return self.fc(nodes).mean(dim=1)


class ConfidenceGating(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 3))

    def forward(self, plddt):
        return F.softmax(self.mlp(plddt), dim=-1)


class MSCFTox(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.ext_1d = Extractor1D(hidden_dim=hidden_dim)
        self.ext_2d = Extractor2D(hidden_dim=hidden_dim)
        self.ext_3d = Extractor3D(hidden_dim=hidden_dim)
        self.gating = ConfidenceGating()
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, x_1d, x_2d, x_3d_n, x_3d_c, plddt):
        f_1d = self.ext_1d(x_1d)
        f_2d = self.ext_2d(x_2d)
        f_3d = self.ext_3d(x_3d_n, x_3d_c)
        weights = self.gating(plddt)
        stacked_features = torch.stack([f_1d * weights[:, 0:1], f_2d * weights[:, 1:2], f_3d * weights[:, 2:3]], dim=1)
        query = (f_1d * weights[:, 0:1]).unsqueeze(1)
        attn_out, _ = self.cross_attn(query, stacked_features, stacked_features)
        return self.classifier(attn_out.squeeze(1))


# ==========================================
# 3. 训练与测试主循环
# ==========================================
def train_and_evaluate():
    print("=== 初始化 MSCF-Tox 深度学习平台 ===")

    # 将这里替换为你上传的文件名
    TRAIN_FASTA = path / "data/train_data.fasta"
    TEST_FASTA_1 = path / "data/test1.fasta"
    TEST_FASTA_2 = path / "data/test2.fasta"

    try:
        # 1. 加载训练集
        train_dataset = RealFastaDataset(TRAIN_FASTA)

        # 2. 分别加载两个测试集
        test_dataset_1 = RealFastaDataset(TEST_FASTA_1)
        test_dataset_2 = RealFastaDataset(TEST_FASTA_2)

        # 3. 核心步骤：将两个测试集无缝拼接成一个大数据集
        combined_test_dataset = ConcatDataset([test_dataset_1, test_dataset_2])
        print(f"成功合并测试集！当前测试集总样本数: {len(combined_test_dataset)}")

    except FileNotFoundError as e:
        print(f"错误：未找到 FASTA 文件！请确保文件名一致且在同级目录下。详细信息: {e}")
        return

    # 生成 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    # 使用合并后的 Dataset 生成 DataLoader
    test_loader = DataLoader(combined_test_dataset, batch_size=16, shuffle=False)

    model = MSCFTox(hidden_dim=256)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    num_epochs = 10
    print("\n=== 开始训练 ===")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            x_1d, x_2d, x_3d_n, x_3d_c, plddt, labels = batch
            optimizer.zero_grad()
            outputs = model(x_1d, x_2d, x_3d_n, x_3d_c, plddt)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}] | Training Loss: {epoch_loss / len(train_loader):.4f}")

    print("\n=== 开始在测试集上评估 ===")
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in test_loader:
            x_1d, x_2d, x_3d_n, x_3d_c, plddt, labels = batch
            outputs = model(x_1d, x_2d, x_3d_n, x_3d_c, plddt)
            preds = (outputs >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    print(f"测试集准确率 (Accuracy): {correct / total * 100:.2f}%")


if __name__ == "__main__":
    train_and_evaluate()

"""
=== 初始化 MSCF-Tox 深度学习平台 ===
正在加载数据集: D:\Python\dachuang2026\data\train_data.fasta ...
成功加载 6387 条多肽序列。
正在加载数据集: D:\Python\dachuang2026\data\test1.fasta ...
成功加载 1126 条多肽序列。
正在加载数据集: D:\Python\dachuang2026\data\test2.fasta ...
成功加载 582 条多肽序列。
成功合并测试集！当前测试集总样本数: 1708

=== 开始训练 ===
Epoch [1/10] | Training Loss: 0.6082
Epoch [2/10] | Training Loss: 0.6009
Epoch [3/10] | Training Loss: 0.5993
Epoch [4/10] | Training Loss: 0.6000
Epoch [5/10] | Training Loss: 0.5998
Epoch [6/10] | Training Loss: 0.5993
Epoch [7/10] | Training Loss: 0.5987
Epoch [8/10] | Training Loss: 0.5976
Epoch [9/10] | Training Loss: 0.5976
Epoch [10/10] | Training Loss: 0.5985

=== 开始在测试集上评估 ===
测试集准确率 (Accuracy): 78.57%
"""