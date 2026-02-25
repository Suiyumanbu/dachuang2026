# predict.py

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # <--- 添加这一行，在导入pyplot之前
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from sklearn.preprocessing import StandardScaler
import warnings
from process_data import process_data, read_fasta  # 导入处理数据的函数
import joblib

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 定义路径
path = Path('D:\Python\dachuang2026')


def prepare_features(df):
    """
    准备特征和标签（标签已经是'0'和'1'字符串）
    """
    # 分离特征和标签
    X = df.drop(['Sequence', 'toxicity'], axis=1)
    y = df['toxicity']

    # 将字符串'0'/'1'转换为整数
    y_numeric = y.astype(int)

    # 检查标签值是否合法
    unique_values = y_numeric.unique()
    if set(unique_values) - {0, 1}:
        print(f"警告: 发现非0/1的标签值: {unique_values}")

    print(f"标签分布:")
    print(f"  类别0 (无毒): {(y_numeric == 0).sum()} 个样本 ({((y_numeric == 0).sum()/len(y_numeric)*100):.1f}%)")
    print(f"  类别1 (有毒): {(y_numeric == 1).sum()} 个样本 ({((y_numeric == 1).sum()/len(y_numeric)*100):.1f}%)")

    return X, y_numeric

def train_random_forest(X_train, y_train):
    """
    训练随机森林模型（不使用交叉验证）
    """
    print("\n" + "=" * 50)
    print("开始训练随机森林模型...")
    print("=" * 50)

    # 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # 训练随机森林模型
    print("\n训练模型...")
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

    print("模型训练完成！")

    return rf_model, scaler


def evaluate_model(model, X_test, y_test):
    """
    评估模型在测试集上的性能
    """
    print("\n" + "=" * 50)
    print("测试集评估结果")
    print("=" * 50)

    # 预测
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # 计算指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')  # 二分类用binary
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
    """
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['无毒(0)', '有毒(1)'],
                yticklabels=['无毒(0)', '有毒(1)'])
    plt.title('混淆矩阵 - 测试集')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    # 计算并显示额外指标
    tn, fp, fn, tp = cm.ravel()
    print(f"\n混淆矩阵详情:")
    print(f"  真阴性 (TN): {tn} (正确预测为无毒)")
    print(f"  假阳性 (FP): {fp} (无毒被误判为有毒)")
    print(f"  假阴性 (FN): {fn} (有毒被误判为无毒)")
    print(f"  真阳性 (TP): {tp} (正确预测为有毒)")


def plot_feature_importance(model, feature_names, save_path, top_n=20):
    """
    绘制并保存特征重要性图
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
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"特征重要性图已保存至: {save_path}")

    # 打印最重要的几个特征
    print(f"\n前{top_n}个最重要的特征:")
    for i, (name, importance) in enumerate(zip(top_names, top_importances)):
        print(f"{i + 1}. {name}: {importance:.4f}")


def save_predictions(model, X_test, y_test, y_pred, save_path):
    """
    保存详细的预测结果
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
        print(f"\n错误分类的样本数: {len(errors)} ({len(errors)/len(y_test)*100:.1f}%)")
        print("\n错误分类示例:")
        print(errors[['真实类别', '预测类别', '无毒概率', '有毒概率']].head())


def main():
    """
    主函数：使用train.fasta训练，用test.fasta测试
    """
    print("=" * 60)
    print("蛋白质毒性预测 - 随机森林模型")
    print("(标签格式: 0=无毒, 1=有毒)")
    print("=" * 60)

    # 创建结果目录
    results_dir = path / 'results'
    results_dir.mkdir(exist_ok=True)

    # 1. 加载并处理训练数据
    print("\n【1】加载训练数据...")
    train_file = path / 'data/train_data.fasta'
    if not train_file.exists():
        print(f"错误: 找不到训练文件 {train_file}")
        return

    train_data = read_fasta(train_file)
    print(f"  训练集样本数: {len(train_data)}")
    processed_train = process_data(train_data)

    # 2. 准备训练特征和标签
    print("\n【2】准备训练特征...")
    # 分离特征和标签
    X_train = processed_train.drop(['Sequence', 'toxicity'], axis=1)
    y_train = processed_train['toxicity'].astype(int)  # 字符串'0'/'1'转整数

    # 显示训练集标签分布
    print(f"  特征维度: {X_train.shape[1]}")
    print(f"  训练集标签分布:")
    print(f"    无毒 (0): {(y_train == 0).sum()} 样本 ({((y_train == 0).sum() / len(y_train) * 100):.1f}%)")
    print(f"    有毒 (1): {(y_train == 1).sum()} 样本 ({((y_train == 1).sum() / len(y_train) * 100):.1f}%)")

    # 3. 训练模型
    print("\n【3】训练随机森林模型...")
    # 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # 训练模型
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )

    model.fit(X_train_scaled, y_train)
    print("  模型训练完成！")

    # 4. 加载并处理测试数据
    print("\n【4】加载测试数据...")
    test_file = path / 'data/test1.fasta'
    if not test_file.exists():
        print(f"错误: 找不到测试文件 {test_file}")
        return

    test_data = read_fasta(test_file)
    print(f"  测试集样本数: {len(test_data)}")
    processed_test = process_data(test_data)

    # 5. 准备测试特征和标签
    print("\n【5】准备测试特征...")
    X_test = processed_test.drop(['Sequence', 'toxicity'], axis=1)
    y_test = processed_test['toxicity'].astype(int)

    # 显示测试集标签分布
    print(f"  测试集标签分布:")
    print(f"    无毒 (0): {(y_test == 0).sum()} 样本 ({((y_test == 0).sum() / len(y_test) * 100):.1f}%)")
    print(f"    有毒 (1): {(y_test == 1).sum()} 样本 ({((y_test == 1).sum() / len(y_test) * 100):.1f}%)")

    # 确保测试集的特征列与训练集一致
    missing_cols = set(X_train.columns) - set(X_test.columns)
    for col in missing_cols:
        X_test[col] = 0

    X_test = X_test[X_train.columns]  # 保持列顺序一致

    # 标准化测试数据
    X_test_scaled = scaler.transform(X_test)

    # 6. 在测试集上进行预测
    print("\n【6】在测试集上进行预测...")
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)

    # 【7】模型评估结果
    print("\n【7】模型评估结果")
    print("-" * 40)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # 计算预测正确率
    correct_predictions = (y_test == y_pred).sum()
    total_predictions = len(y_test)
    correct_rate = correct_predictions / total_predictions

    print(f"  预测正确数: {correct_predictions}/{total_predictions}")
    print(f"  预测正确率: {correct_rate:.4f} ({correct_rate * 100:.2f}%)")
    print(f"  预测错误数: {total_predictions - correct_predictions}/{total_predictions}")
    print(f"  预测错误率: {1 - correct_rate:.4f} ({(1 - correct_rate) * 100:.2f}%)")
    print("-" * 40)
    print(f"  准确率 (Accuracy):  {accuracy:.4f}")
    print(f"  精确率 (Precision): {precision:.4f}")
    print(f"  召回率 (Recall):    {recall:.4f}")
    print(f"  F1分数 (F1-score):  {f1:.4f}")

    # 8. 混淆矩阵
    print("\n【8】混淆矩阵")
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print("-" * 40)
    print(f"              预测无毒    预测有毒")
    print(f"  实际无毒    {tn:6d}      {fp:6d}")
    print(f"  实际有毒    {fn:6d}      {tp:6d}")
    print("-" * 40)

    # 计算更多指标
    sensitivity = tp / (tp + fn)  # 召回率/敏感度
    specificity = tn / (tn + fp)  # 特异性
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # 阳性预测值
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # 阴性预测值

    print(f"\n  敏感度 (Sensitivity): {sensitivity:.4f} (正确识别有毒的能力)")
    print(f"  特异性 (Specificity): {specificity:.4f} (正确识别无毒的能力)")
    print(f"  阳性预测值 (PPV):     {ppv:.4f} (预测为有毒中实际有毒的比例)")
    print(f"  阴性预测值 (NPV):     {npv:.4f} (预测为无毒中实际无毒的比例)")

    # 9. 保存可视化结果
    print("\n【9】保存可视化结果...")

    # 绘制并保存混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['无毒(0)', '有毒(1)'],
                yticklabels=['无毒(0)', '有毒(1)'])
    plt.title('混淆矩阵 - 测试集')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.tight_layout()
    plt.savefig(results_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close('all')
    print(f"  混淆矩阵已保存: {results_dir / 'confusion_matrix.png'}")

    # 绘制特征重要性
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:20]  # 取前20个

    plt.figure(figsize=(10, 8))
    plt.barh(range(20), importances[indices][::-1])
    plt.yticks(range(20), [X_train.columns[i] for i in indices][::-1])
    plt.xlabel('重要性')
    plt.title('前20个最重要的特征')
    plt.tight_layout()
    plt.savefig(results_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close('all')
    print(f"  特征重要性图已保存: {results_dir / 'feature_importance.png'}")

    # 10. 保存详细预测结果
    print("\n【10】保存预测结果...")

    results_df = pd.DataFrame({
        '真实标签': y_test,
        '预测标签': y_pred,
        '预测正确': y_test == y_pred,
        '无毒概率': y_pred_proba[:, 0],
        '有毒概率': y_pred_proba[:, 1]
    })

    # 添加可读性更好的类别列
    results_df['真实类别'] = results_df['真实标签'].map({0: '无毒', 1: '有毒'})
    results_df['预测类别'] = results_df['预测标签'].map({0: '无毒', 1: '有毒'})

    # 添加原始序列信息（可选）
    if 'Sequence' in processed_test.columns:
        results_df['序列'] = processed_test['Sequence'].values

    results_df.to_csv(results_dir / 'test_predictions.csv', index=False)
    print(f"  预测结果已保存: {results_dir / 'test_predictions.csv'}")

    # 显示错误分类的样本
    errors = results_df[results_df['预测正确'] == False]
    if len(errors) > 0:
        print(f"\n  错误分类样本数: {len(errors)} ({len(errors) / len(y_test) * 100:.1f}%)")
        print("\n  错误分类示例（前5个）:")
        print(errors[['真实类别', '预测类别', '无毒概率', '有毒概率']].head())

    # 11. 保存模型和相关文件
    print("\n【11】保存模型...")
    joblib.dump(model, results_dir / 'random_forest_model.pkl')
    joblib.dump(scaler, results_dir / 'scaler.pkl')

    # 保存特征名称
    pd.Series(X_train.columns).to_csv(results_dir / 'feature_names.csv', index=False)

    # 保存模型配置
    model_config = {
        'model_type': 'RandomForestClassifier',
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'class_weight': 'balanced',
        'random_state': 42,
        'n_features': X_train.shape[1],
        'features': list(X_train.columns)
    }

    import json
    with open(results_dir / 'model_config.json', 'w', encoding='utf-8') as f:
        json.dump(model_config, f, indent=2, ensure_ascii=False)

    print(f"  模型文件已保存: {results_dir / 'random_forest_model.pkl'}")
    print(f"  标准化器已保存: {results_dir / 'scaler.pkl'}")
    print(f"  特征列表已保存: {results_dir / 'feature_names.csv'}")
    print(f"  模型配置已保存: {results_dir / 'model_config.json'}")

    # 12. 输出总结
    print("\n" + "=" * 60)
    print("✅ 模型训练和测试完成！")
    print("=" * 60)
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
    plt.close('all')
"""
============================================================
蛋白质毒性预测 - 随机森林模型
(标签格式: 0=无毒, 1=有毒)
============================================================

【1】加载训练数据...
  训练集样本数: 6387

处理完成！
样本数: 6387
特征数: 132
样本/特征比: 48.4:1

【2】准备训练特征...
  特征维度: 132
  训练集标签分布:
    无毒 (0): 4569 样本 (71.5%)
    有毒 (1): 1818 样本 (28.5%)

【3】训练随机森林模型...
  模型训练完成！

【4】加载测试数据...
  测试集样本数: 1126

处理完成！
样本数: 1126
特征数: 132
样本/特征比: 8.5:1

【5】准备测试特征...
  测试集标签分布:
    无毒 (0): 806 样本 (71.6%)
    有毒 (1): 320 样本 (28.4%)

【6】在测试集上进行预测...

【7】模型评估结果
----------------------------------------
  预测正确数: 1023/1126
  预测正确率: 0.9085 (90.85%)
  预测错误数: 103/1126
  预测错误率: 0.0915 (9.15%)
----------------------------------------
  准确率 (Accuracy):  0.9085
  精确率 (Precision): 0.8678
  召回率 (Recall):    0.8000
  F1分数 (F1-score):  0.8325

【8】混淆矩阵
----------------------------------------
              预测无毒    预测有毒
  实际无毒       767          39
  实际有毒        64         256
----------------------------------------

  敏感度 (Sensitivity): 0.8000 (正确识别有毒的能力)
  特异性 (Specificity): 0.9516 (正确识别无毒的能力)
  阳性预测值 (PPV):     0.8678 (预测为有毒中实际有毒的比例)
  阴性预测值 (NPV):     0.9230 (预测为无毒中实际无毒的比例)

============================================================
✅ 模型训练和测试完成！
============================================================

📊 结果汇总:
  - 训练集大小: 6387 个样本
  - 测试集大小: 1126 个样本
  - 特征数量: 132
  - 准确率: 0.9085
  - F1分数: 0.8325
============================================================
"""