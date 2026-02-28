"""
随机森林毒性预测模型训练与预测脚本
用于训练和评估蛋白质毒性预测模型
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from sklearn.preprocessing import StandardScaler
import warnings
from machine_learning.process_data import process_data, read_fasta
import joblib

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def prepare_features(df):
    """
    准备特征和标签（标签已经是'0'和'1'字符串）

    Parameters:
    -----------
    df : pandas.DataFrame
        包含特征和标签的数据框

    Returns:
    --------
    X : pandas.DataFrame
        特征数据
    y_numeric : pandas.Series
        数值型标签（0/1）
    """
    # 分离特征和标签
    X = df.drop(['Sequence', 'toxicity'], axis=1)
    y = df['toxicity']

    # 将字符串'0'/'1'转换为整数
    y_numeric = y.astype(int)

    return X, y_numeric


def train_random_forest(X_train, y_train):
    """
    训练随机森林模型（不使用交叉验证）

    Parameters:
    -----------
    X_train : pandas.DataFrame
        训练特征数据
    y_train : pandas.Series
        训练标签数据

    Returns:
    --------
    rf_model : RandomForestClassifier
        训练好的随机森林模型
    scaler : StandardScaler
        标准化器
    """
    # 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # 训练随机森林模型
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )

    rf_model.fit(X_train_scaled, y_train)

    return rf_model, scaler


def evaluate_model(model, X_test, y_test):
    """
    评估模型在测试集上的性能

    Parameters:
    -----------
    model : RandomForestClassifier
        训练好的模型
    X_test : array-like
        测试特征数据
    y_test : array-like
        测试标签数据

    Returns:
    --------
    y_pred : array
        预测标签
    y_pred_proba : array
        预测概率
    """
    # 预测
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # 计算指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')

    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"F1分数 (F1-score): {f1:.4f}")

    # 详细分类报告
    print("\n详细分类报告:")
    print(classification_report(y_test, y_pred, target_names=['无毒(0)', '有毒(1)']))

    return y_pred, y_pred_proba


def plot_confusion_matrix(y_test, y_pred, save_path):
    """
    绘制并保存混淆矩阵

    Parameters:
    -----------
    y_test : array-like
        真实标签
    y_pred : array-like
        预测标签
    save_path : Path or str
        保存路径
    """
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['无毒(0)', '有毒(1)'],
                yticklabels=['无毒(0)', '有毒(1)'])
    plt.title('混淆矩阵 - 测试集')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.show()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"混淆矩阵已保存至: {save_path}")


def plot_feature_importance(model, feature_names, save_path, top_n=20):
    """
    绘制并保存特征重要性图

    Parameters:
    -----------
    model : RandomForestClassifier
        训练好的模型
    feature_names : list
        特征名称列表
    save_path : Path or str
        保存路径
    top_n : int
        显示最重要的前N个特征
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    # 取前top_n个特征
    top_indices = indices[:top_n]
    top_importances = importances[top_indices]
    top_names = [feature_names[i] for i in top_indices]

    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), top_importances[::-1])
    plt.yticks(range(top_n), top_names[::-1])
    plt.xlabel('重要性')
    plt.title(f'前{top_n}个最重要的特征')
    plt.show()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"特征重要性图已保存至: {save_path}")


def save_predictions(model, X_test, y_test, y_pred, save_path):
    """
    保存详细的预测结果

    Parameters:
    -----------
    model : RandomForestClassifier
        训练好的模型
    X_test : array-like
        测试特征数据
    y_test : array-like
        真实标签
    y_pred : array-like
        预测标签
    save_path : Path or str
        保存路径
    """
    y_pred_proba = model.predict_proba(X_test)

    results = pd.DataFrame({
        '真实标签': y_test,
        '预测标签': y_pred,
        '预测正确': y_test == y_pred,
        '无毒概率': y_pred_proba[:, 0],
        '有毒概率': y_pred_proba[:, 1]
    })

    # 添加预测结果解释
    results['真实类别'] = results['真实标签'].map({0: '无毒', 1: '有毒'})
    results['预测类别'] = results['预测标签'].map({0: '无毒', 1: '有毒'})

    results.to_csv(save_path, index=False)
    print(f"详细预测结果已保存至: {save_path}")

    # 显示错误分类的样本
    errors = results[results['预测正确'] == False]
    if len(errors) > 0:
        print(f"\n错误分类的样本数: {len(errors)} ({len(errors) / len(y_test) * 100:.1f}%)")
        print("\n错误分类示例:")
        print(errors[['真实类别', '预测类别', '无毒概率', '有毒概率']].head())


def load_and_process_data(train_file, test_files):
    """
    加载并处理训练和测试数据

    Parameters:
    -----------
    train_file : Path
        训练数据文件路径
    test_files : list
        测试数据文件路径列表

    Returns:
    --------
    X_train, y_train, X_test, y_test, feature_names : tuple
        处理好的训练和测试数据
    """
    # 1. 加载并处理训练数据
    print("加载训练数据...")
    train_data = read_fasta(train_file)
    processed_train = process_data(train_data)

    # 2. 准备训练特征和标签
    X_train = processed_train.drop(['Sequence', 'toxicity'], axis=1)
    y_train = processed_train['toxicity'].astype(int)

    print(f"\n训练集信息:")
    print(f"  特征维度: {X_train.shape[1]}")
    print(f"  训练集标签分布:")
    print(f"    无毒 (0): {(y_train == 0).sum()} 样本 ({((y_train == 0).sum() / len(y_train) * 100):.1f}%)")
    print(f"    有毒 (1): {(y_train == 1).sum()} 样本 ({((y_train == 1).sum() / len(y_train) * 100):.1f}%)")

    # 3. 加载并处理测试数据
    print("\n加载测试数据...")
    test_data_list = []
    for test_file in test_files:
        test_data = read_fasta(test_file)
        test_data_list.append(test_data)

    test_data = pd.concat(test_data_list, ignore_index=True)
    processed_test = process_data(test_data)

    print(f"测试集样本数: {len(processed_test)}")

    # 4. 准备测试特征和标签
    X_test = processed_test.drop(['Sequence', 'toxicity'], axis=1)
    y_test = processed_test['toxicity'].astype(int)

    # 确保测试集的特征列与训练集一致
    missing_cols = set(X_train.columns) - set(X_test.columns)
    for col in missing_cols:
        X_test[col] = 0

    X_test = X_test[X_train.columns]  # 保持列顺序一致

    return X_train, y_train, X_test, y_test, X_train.columns.tolist()


def main():
    """
    主函数：执行完整的训练和预测流程
    """
    # 定义路径
    path = Path('/')
    train_file = path / 'data/train_data.fasta'
    test_files = [path / 'data/test1.fasta', path / 'data/test2.fasta']

    # 创建结果目录
    results_dir = path / 'results'
    results_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("随机森林毒性预测模型训练与评估")
    print("=" * 60)

    # 1. 加载和处理数据
    X_train, y_train, X_test, y_test, feature_names = load_and_process_data(train_file, test_files)

    # 2. 训练模型
    print("\n训练随机森林模型...")
    rf_model, scaler = train_random_forest(X_train, y_train)

    # 3. 标准化测试数据
    X_test_scaled = scaler.transform(X_test)

    # 4. 在测试集上进行预测
    y_pred = rf_model.predict(X_test_scaled)
    y_pred_proba = rf_model.predict_proba(X_test_scaled)

    # 5. 计算评估指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    correct_predictions = (y_test == y_pred).sum()
    total_predictions = len(y_test)
    correct_rate = correct_predictions / total_predictions

    print("\n" + "=" * 40)
    print("模型评估结果:")
    print("=" * 40)
    print(f"  预测正确数: {correct_predictions}/{total_predictions}")
    print(f"  预测正确率: {correct_rate:.4f} ({correct_rate * 100:.2f}%)")
    print(f"  预测错误数: {total_predictions - correct_predictions}/{total_predictions}")
    print(f"  预测错误率: {1 - correct_rate:.4f} ({(1 - correct_rate) * 100:.2f}%)")
    print("-" * 40)
    print(f"  准确率 (Accuracy):  {accuracy:.4f}")
    print(f"  精确率 (Precision): {precision:.4f}")
    print(f"  召回率 (Recall):    {recall:.4f}")
    print(f"  F1分数 (F1-score):  {f1:.4f}")

    # 6. 绘制混淆矩阵
    print("\n生成可视化图表...")
    plot_confusion_matrix(y_test, y_pred, results_dir / 'confusion_matrix.png')

    # 7. 绘制特征重要性图
    plot_feature_importance(rf_model, feature_names, results_dir / 'feature_importance.png')

    # 8. 保存详细预测结果
    save_predictions(rf_model, X_test_scaled, y_test, y_pred, results_dir / 'detailed_predictions.csv')

    # 9. 保存模型和相关文件
    joblib.dump(rf_model, results_dir / 'random_forest_model.joblib')
    joblib.dump(scaler, results_dir / 'scaler.pkl')
    print(f"\n模型文件已保存至: {results_dir / 'random_forest_model.joblib'}")
    print(f"标准化器已保存至: {results_dir / 'scaler.pkl'}")

    # 10. 结果汇总
    print(f"\n📊 结果汇总:")
    print(f"  - 训练集大小: {len(X_train)} 个样本")
    print(f"  - 测试集大小: {len(X_test)} 个样本")
    print(f"  - 特征数量: {X_train.shape[1]}")
    print(f"  - 准确率: {accuracy:.4f}")
    print(f"  - F1分数: {f1:.4f}")
    print(f"\n📁 结果保存位置: {results_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()