import os
import torch
from transformers import AutoTokenizer, EsmModel
from Bio import SeqIO
from tqdm import tqdm
from pathlib import Path

path = Path('D:/Python/dachuang2026')

# ==========================================
# 配置参数
# ==========================================
# 你的 FASTA 文件路径
FASTA_FILES = [
    path / "data/train_data.fasta",  # 请根据你的实际路径修改
    path / "data/test1.fasta",
    path / "data/test2.fasta"
]

# 特征保存的根目录
SAVE_DIR = path / "data/features"
os.makedirs(SAVE_DIR, exist_ok=True)

# 模型选择：如果 650M 显存不够(报 OOM)，可暂时换成 facebook/esm2_t12_35M_UR50D 跑通流程
MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
MAX_SEQ_LEN = 50  # 截断或填充的最大长度
EMBED_DIM = 1280  # 650M 模型的输出维度是 1280

# ==========================================
# 初始化 ESM-2 模型
# ==========================================
print(f"正在加载模型: {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = EsmModel.from_pretrained(MODEL_NAME)

# 自动使用 GPU 加速（如果有的话）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()  # 设置为推理模式
print(f"模型加载完毕！当前使用设备: {device}")


# ==========================================
# 核心特征提取函数
# ==========================================
def extract_and_save_features():
    for fasta_file in FASTA_FILES:
        if not os.path.exists(fasta_file):
            print(f"跳过: 未找到文件 {fasta_file}")
            continue

        print(f"\n开始处理文件: {fasta_file}")
        records = list(SeqIO.parse(fasta_file, "fasta"))

        # 使用 tqdm 增加进度条，防止看着像死机
        for record in tqdm(records, desc="Extracting 1D"):
            # 获取序列和安全的 ID 作为文件名
            seq = str(record.seq).upper()
            # 替换掉 ID 中可能导致文件路径错误的特殊字符
            safe_id = record.id.replace("|", "_").replace("/", "_")
            save_path = os.path.join(SAVE_DIR, f"{safe_id}_1d.pt")

            # 断点续传支持：如果文件已经存在，直接跳过
            if os.path.exists(save_path):
                continue

            # 1. Tokenizer 编码
            inputs = tokenizer(seq, return_tensors="pt", add_special_tokens=True).to(device)

            # 2. 模型推理 (关闭梯度，节省显存)
            with torch.no_grad():
                outputs = model(**inputs)
                # outputs.last_hidden_state 的形状是 (1, seq_len + 2, 1280)
                # 因为首尾分别加了 <cls> 和 <eos> token

            # 3. 去除首尾的特殊 token，提取纯氨基酸特征
            embeddings = outputs.last_hidden_state[0, 1:-1, :]  # 形状: (seq_len, 1280)

            # 4. 填充 (Padding) 或 截断 (Truncation) 对齐到 50 长度
            actual_len = embeddings.shape[0]
            feat_1d = torch.zeros(MAX_SEQ_LEN, EMBED_DIM)  # 初始化全 0 矩阵

            valid_len = min(actual_len, MAX_SEQ_LEN)
            feat_1d[:valid_len, :] = embeddings[:valid_len, :].cpu()  # 移回 CPU 并赋值

            # 5. 保存张量到本地
            torch.save(feat_1d, save_path)


if __name__ == "__main__":
    extract_and_save_features()
    print("\n🎉 所有序列的 1D ESM-2 特征提取完毕！")