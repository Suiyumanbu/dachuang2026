# filename: toxicity_regression.py
"""
毒性肽回归模型训练与评估
用于预测毒性概率或毒性强度等连续值
导入process_data.py中的特征处理函数
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy import stats

# 导入自定义模块
# 假设process_data.py在当前目录或Python路径中
try:
    from process_data import read_fasta, process_data

    print("成功导入 process_data 模块")
except ImportError as e:
    print(f"导入 process_data 失败: {e}")
    print("请确保 process_data.py 在同一目录下")
    sys.exit(1)

# 回归模型
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# 模型评估与选择
from sklearn.model_selection import (cross_val_score, KFold, train_test_split,
                                     GridSearchCV, RandomizedSearchCV)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             explained_variance_score, median_absolute_error,
                             mean_squared_log_error)
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor

warnings.filterwarnings('ignore')
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 设置路径
# path = Path('D:/Coding_in_Python/deep_learning_porject_1')
path = Path('./data')

# ==================== 数据准备 ====================

def prepare_regression_data(df, target_column='toxicity'):
    """
    准备回归数据，将分类标签转换为连续值
    如果有真实的连续值目标，可以直接使用
    """
    print("准备回归数据...")

    # 使用原有的特征处理函数
    df_processed = process_data(df)

    # 提取特征和目标
    if target_column in df_processed.columns:
        # 如果有目标列，直接使用
        y = df_processed[target_column].astype(float)
        print(f"使用现有目标列: {target_column}")
    else:
        # 否则从Id中提取或生成模拟的连续值
        print("未找到目标列，从Id中提取或生成连续值...")
        if 'Id' in df.columns and '|' in df['Id'].iloc[0]:
            # 尝试从Id中提取连续值
            try:
                y = df['Id'].apply(lambda x: float(x.split('|')[1]))
                print("从Id中提取连续值成功")
            except:
                # 如果失败，基于序列特征生成模拟的连续值
                print("生成模拟的连续值用于演示...")
                y = generate_simulated_target(df_processed)
        else:
            # 生成模拟的连续值
            print("生成模拟的连续值用于演示...")
            y = generate_simulated_target(df_processed)

    # 删除不需要的列
    X = df_processed.drop(['Sequence', 'toxicity', target_column]
                          if target_column in df_processed.columns
                          else ['Sequence', 'toxicity'], axis=1)

    print(f"特征矩阵形状: {X.shape}")
    print(f"目标变量范围: [{y.min():.4f}, {y.max():.4f}]")
    print(f"目标变量均值: {y.mean():.4f} ± {y.std():.4f}")

    return X, y


def generate_synthetic_regression_data(n_samples=1000, noise_level=0.1):
    """
    生成合成的回归数据用于演示
    当没有真实数据时使用

    Parameters:
    -----------
    n_samples : int
        生成的样本数量
    noise_level : float
        噪声水平，控制目标值的随机波动程度

    Returns:
    --------
    pd.DataFrame
        包含Id、Sequence和target的DataFrame
    """
    np.random.seed(42)  # 设置随机种子，确保结果可重复

    # 氨基酸字母表
    amino_acids = 'ARNDCEQGHILKMFPSTWYV'

    sequences = []
    targets = []

    for i in range(n_samples):
        # 随机生成序列长度（5-50个氨基酸）
        length = np.random.randint(5, 51)

        # 随机生成氨基酸序列
        seq = ''.join(np.random.choice(list(amino_acids), length))
        sequences.append(seq)

        # 基于序列特征生成目标值（模拟毒性分数）
        # 计算一些与毒性相关的特征
        c_count = seq.count('C') / length  # 半胱氨酸比例
        hydrophobic_count = sum(seq.count(aa) for aa in 'AILMFWYV') / length  # 疏水氨基酸比例
        charged_count = sum(seq.count(aa) for aa in 'KRHDE') / length  # 带电氨基酸比例

        # 生成目标值（0-1之间的连续值）
        # 使用线性组合 + 噪声
        target = (0.3 * c_count +
                  0.4 * hydrophobic_count +
                  0.3 * charged_count +
                  noise_level * np.random.randn())

        # 确保目标值在0-1范围内
        target = np.clip(target, 0, 1)
        targets.append(target)

    # 创建DataFrame
    df = pd.DataFrame({
        'Id': [f'synthetic_{i}' for i in range(n_samples)],
        'Sequence': sequences,
        'target': targets
    })

    print(f"生成合成数据完成:")
    print(f"  样本数: {len(df)}")
    print(f"  目标值范围: [{df['target'].min():.4f}, {df['target'].max():.4f}]")
    print(f"  目标值均值: {df['target'].mean():.4f} ± {df['target'].std():.4f}")

    return df


def generate_simulated_target(df_processed):
    """
    基于序列特征生成模拟的连续目标值
    仅用于演示，实际应用中应使用真实数据
    """
    np.random.seed(42)

    # 使用一些特征生成目标值
    features = df_processed.columns

    # 寻找可能的特征
    c_content = None
    hydrophobic = None
    charged = None

    for col in features:
        if 'comp_C' in col:
            c_content = df_processed[col].values
        elif 'group_hydrophobic_ratio' in col:
            hydrophobic = df_processed[col].values
        elif 'charged_ratio' in col:
            charged = df_processed[col].values

    if c_content is None:
        c_content = np.random.random(len(df_processed))
    if hydrophobic is None:
        hydrophobic = np.random.random(len(df_processed))
    if charged is None:
        charged = np.random.random(len(df_processed))

    # 生成目标值 (0-1范围)
    target = (0.3 * c_content +
              0.4 * hydrophobic +
              0.3 * charged +
              0.05 * np.random.randn(len(df_processed)))

    # 确保在合理范围内
    target = np.clip(target, 0, 1)

    return target


# ==================== 回归模型定义 ====================

def get_regression_models():
    """
    定义多个回归模型用于比较
    """
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.01),
        'Elastic Net': ElasticNet(alpha=0.01, l1_ratio=0.5),
        'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=15,
                                               random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5,
                                                       learning_rate=0.1, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1,
                                random_state=42, n_jobs=-1),
        'LightGBM': LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.1,
                                  random_state=42, n_jobs=-1, verbose=-1),
        'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1),
        'KNN': KNeighborsRegressor(n_neighbors=5)
    }
    return models


def get_model_param_grids():
    """
    为不同模型定义参数网格
    """
    param_grids = {
        'Ridge': {
            'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
        },
        'Lasso': {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
        },
        'Elastic Net': {
            'alpha': [0.001, 0.01, 0.1, 1.0],
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
        },
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10]
        },
        'Gradient Boosting': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2]
        },
        'XGBoost': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0]
        },
        'LightGBM': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15],
            'learning_rate': [0.01, 0.1, 0.2],
            'num_leaves': [31, 50, 100]
        },
        'SVR': {
            'C': [0.1, 1.0, 10.0, 100.0],
            'gamma': ['scale', 'auto', 0.1, 0.01],
            'epsilon': [0.01, 0.1, 0.2]
        },
        'KNN': {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance']
        }
    }
    return param_grids


# ==================== 模型训练与评估 ====================

def evaluate_regression_models(X, y, models=None, cv=5, test_size=0.2):
    """
    评估多个回归模型的性能
    """
    if models is None:
        models = get_regression_models()

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []

    print("\n" + "=" * 80)
    print("回归模型性能比较")
    print("=" * 80)

    for name, model in models.items():
        print(f"\n训练模型: {name}")

        try:
            # 交叉验证
            cv_scores = cross_val_score(model, X_train_scaled, y_train,
                                        cv=cv, scoring='r2')

            # 训练模型
            model.fit(X_train_scaled, y_train)

            # 预测
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)

            # 计算指标
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)

            # 保存结果
            results.append({
                'Model': name,
                'CV R2_mean': cv_scores.mean(),
                'CV R2_std': cv_scores.std(),
                'Train R2': train_r2,
                'Test R2': test_r2,
                'Train RMSE': train_rmse,
                'Test RMSE': test_rmse,
                'Train MAE': train_mae,
                'Test MAE': test_mae
            })

            print(f"  CV R2: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            print(f"  Test R2: {test_r2:.4f}")
            print(f"  Test RMSE: {test_rmse:.4f}")
            print(f"  Test MAE: {test_mae:.4f}")

        except Exception as e:
            print(f"  模型 {name} 训练失败: {e}")

    # 创建结果DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Test R2', ascending=False)

    print("\n" + "=" * 80)
    print("模型排名 (按Test R2)")
    print("=" * 80)
    print(results_df[['Model', 'Test R2', 'Test RMSE', 'Test MAE']].to_string(index=False))

    return results_df, scaler


def optimize_best_model(X, y, best_model_name, results_df, cv=5):
    """
    对最佳模型进行超参数优化
    """
    print(f"\n{'=' * 80}")
    print(f"优化最佳模型: {best_model_name}")
    print(f"{'=' * 80}")

    # 获取参数网格
    param_grids = get_model_param_grids()

    if best_model_name not in param_grids:
        print(f"未找到 {best_model_name} 的参数网格")
        return None, None

    # 获取模型
    models = get_regression_models()
    base_model = models[best_model_name]

    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 创建Pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', base_model)
    ])

    # 参数网格（需要添加前缀）
    param_grid = {f'regressor__{key}': value
                  for key, value in param_grids[best_model_name].items()}

    # 网格搜索
    print("正在进行网格搜索...")
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=cv, scoring='r2',
        n_jobs=-1, verbose=1
    )

    grid_search.fit(X_train, y_train)

    print(f"\n最佳参数: {grid_search.best_params_}")
    print(f"最佳交叉验证 R2: {grid_search.best_score_:.4f}")

    # 在测试集上评估
    y_pred = grid_search.predict(X_test)
    test_r2 = r2_score(y_test, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"测试集 R2: {test_r2:.4f}")
    print(f"测试集 RMSE: {test_rmse:.4f}")

    return grid_search.best_estimator_, grid_search.best_params_


# ==================== 可视化 ====================

def plot_regression_results(y_test, y_pred, model_name, feature_importance=None, feature_names=None):
    """
    可视化回归结果
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. 预测值 vs 真实值
    ax1 = axes[0, 0]
    ax1.scatter(y_test, y_pred, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             'r--', lw=2, label='理想预测')
    ax1.set_xlabel('真实值')
    ax1.set_ylabel('预测值')
    ax1.set_title(f'{model_name}: 预测值 vs 真实值')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 残差分布
    ax2 = axes[0, 1]
    residuals = y_test - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax2.axhline(y=0, color='r', linestyle='--', lw=2)
    ax2.set_xlabel('预测值')
    ax2.set_ylabel('残差')
    ax2.set_title('残差图')
    ax2.grid(True, alpha=0.3)

    # 3. 残差直方图
    ax3 = axes[1, 0]
    ax3.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    ax3.set_xlabel('残差')
    ax3.set_ylabel('频数')
    ax3.set_title('残差分布')
    ax3.axvline(x=0, color='r', linestyle='--', lw=2)

    # 4. 特征重要性（如果有）
    ax4 = axes[1, 1]
    if feature_importance is not None and feature_names is not None:
        # 获取Top特征
        indices = np.argsort(feature_importance)[-15:]
        top_features = [feature_names[i] for i in indices]
        top_importance = feature_importance[indices]

        colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
        ax4.barh(range(len(top_features)), top_importance, color=colors)
        ax4.set_yticks(range(len(top_features)))
        ax4.set_yticklabels(top_features)
        ax4.set_xlabel('特征重要性')
        ax4.set_title('特征重要性 (Top 15)')
    else:
        # 显示评估指标
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        metrics_text = f'R² Score: {r2:.4f}\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}'
        ax4.text(0.5, 0.5, metrics_text, ha='center', va='center',
                 transform=ax4.transAxes, fontsize=14,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax4.set_title('模型评估指标')
        ax4.axis('off')

    plt.tight_layout()
    plt.show()


def plot_model_comparison(results_df):
    """
    绘制模型比较图
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. R2分数比较
    ax1 = axes[0, 0]
    models = results_df['Model'].values
    x = range(len(models))
    ax1.bar(x, results_df['Test R2'].values, alpha=0.7, color='steelblue')
    ax1.set_xlabel('模型')
    ax1.set_ylabel('R² 分数')
    ax1.set_title('各模型测试集 R² 分数')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.axhline(y=0, color='r', linestyle='-', linewidth=0.5)

    # 2. RMSE比较
    ax2 = axes[0, 1]
    ax2.bar(x, results_df['Test RMSE'].values, alpha=0.7, color='coral')
    ax2.set_xlabel('模型')
    ax2.set_ylabel('RMSE')
    ax2.set_title('各模型测试集 RMSE')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45, ha='right')

    # 3. MAE比较
    ax3 = axes[1, 0]
    ax3.bar(x, results_df['Test MAE'].values, alpha=0.7, color='lightgreen')
    ax3.set_xlabel('模型')
    ax3.set_ylabel('MAE')
    ax3.set_title('各模型测试集 MAE')
    ax3.set_xticks(x)
    ax3.set_xticklabels(models, rotation=45, ha='right')

    # 4. 交叉验证结果
    ax4 = axes[1, 1]
    ax4.errorbar(x, results_df['CV R2_mean'].values,
                 yerr=results_df['CV R2_std'].values,
                 fmt='o-', capsize=5, color='purple')
    ax4.set_xlabel('模型')
    ax4.set_ylabel('交叉验证 R²')
    ax4.set_title('5折交叉验证 R² (均值 ± 标准差)')
    ax4.set_xticks(x)
    ax4.set_xticklabels(models, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ==================== 预测新数据 ====================

# 修改后的 predict_new_data 函数
def predict_new_data(model, scaler, new_data_file, output_file=None,
                     feature_columns=None, expected_features=None):
    """
    对新数据进行预测

    Parameters:
    -----------
    model : 训练好的模型
    scaler : 训练时使用的标准化器
    new_data_file : str or Path
        新数据文件路径
    output_file : str or Path, optional
        输出文件路径
    feature_columns : list, optional
        训练时使用的特征列名
    expected_features : list, optional
        模型期望的特征名称列表
    """
    print(f"\n{'=' * 80}")
    print(f"对新数据进行预测: {new_data_file}")
    print(f"{'=' * 80}")

    # 读取新数据
    new_df = read_fasta(new_data_file)
    print(f"新数据形状: {new_df.shape}")

    # 处理特征（但需要固定二肽选择）
    print("处理新数据特征...")

    # 使用全局的氨基酸列表
    amino_acids = 'ARNDCEQGHILKMFPSTWYV'

    # 复制数据
    dic = new_df.copy()

    # 基础特征（与训练集保持一致）
    dic['length'] = dic['Sequence'].apply(len)

    # 氨基酸组成（20维）
    for aa in amino_acids:
        dic[f'comp_{aa}'] = dic['Sequence'].apply(
            lambda seq, a=aa: seq.count(a) / len(seq) if len(seq) > 0 else 0
        )

    # 基础理化性质
    def safe_protein_analysis(seq):
        try:
            if len(seq) < 2:
                return None
            return ProteinAnalysis(seq)
        except:
            return None

    analyzers = dic['Sequence'].apply(safe_protein_analysis)
    dic['hydrophobicity'] = analyzers.apply(lambda x: x.gravy() if x is not None else 0)
    dic['charge_at_pH7.4'] = analyzers.apply(lambda x: x.charge_at_pH(7.4) if x is not None else 0)
    dic['isoelectric_point'] = analyzers.apply(lambda x: x.isoelectric_point() if x is not None else 7.0)

    # 重要：这里需要使用训练集确定的二肽特征，而不是重新选择
    if expected_features is not None:
        # 从expected_features中提取二肽特征
        dipeptide_features = [f for f in expected_features if f.startswith('dip_')]

        # 计算这些特定的二肽特征
        def get_specific_dipeptide_freq(seq, dip):
            """计算特定二肽的频率"""
            if len(seq) < 2:
                return 0
            dipeptides = [seq[i:i + 2] for i in range(len(seq) - 1)]
            total = len(dipeptides)
            return dipeptides.count(dip) / total if total > 0 else 0

        for feat in dipeptide_features:
            dip = feat.replace('dip_', '')
            dic[feat] = dic['Sequence'].apply(lambda seq, d=dip: get_specific_dipeptide_freq(seq, d))

    # 理化分组特征
    property_groups = {
        'hydrophobic': 'AILMFWYV',
        'hydrophilic': 'RKNDEQ',
        'neutral': 'GSTCP',
        'charged': 'KRHDE',
    }

    for group_name, aas in property_groups.items():
        dic[f'group_{group_name}_ratio'] = dic['Sequence'].apply(
            lambda seq: sum(seq.count(aa) for aa in aas) / len(seq) if len(seq) > 0 else 0
        )

    # 序列复杂度特征
    dic['seq_complexity'] = dic['Sequence'].apply(
        lambda seq: len(set(seq)) / len(seq) if len(seq) > 0 else 0
    )
    dic['charged_ratio'] = dic['Sequence'].apply(
        lambda seq: sum(seq.count(aa) for aa in 'KRHDE') / len(seq) if len(seq) > 0 else 0
    )

    # 毒性模体特征
    toxic_motifs = ['RR', 'KK', 'RK', 'KR', 'CXC', 'CC', 'RGD', 'KGD']
    for motif in toxic_motifs:
        if len(motif) == 2:
            dic[f'motif_{motif}'] = dic['Sequence'].apply(
                lambda seq: seq.count(motif) / max(1, len(seq) - 1)
            )
        else:
            dic[f'motif_{motif}'] = dic['Sequence'].apply(
                lambda seq: seq.count(motif) / max(1, len(seq) - len(motif) + 1)
            )

    # 选择与训练集相同的特征列
    if expected_features is not None:
        # 只保留模型训练时使用的特征
        available_features = [f for f in expected_features if f in dic.columns]
        missing_features = set(expected_features) - set(available_features)

        if missing_features:
            print(f"警告: 以下特征在测试集中缺失，将填充为0: {missing_features}")
            for feat in missing_features:
                dic[feat] = 0

        X_new = dic[expected_features]
    else:
        # 如果没有指定特征，删除非特征列
        X_new = dic.drop(['Sequence', 'Id'] if 'Id' in dic.columns else ['Sequence'],
                         axis=1, errors='ignore')

    print(f"测试集特征矩阵形状: {X_new.shape}")

    # 确保所有特征都是数值型
    X_new = X_new.astype(float)

    # 标准化
    X_new_scaled = scaler.transform(X_new)

    # 预测
    predictions = model.predict(X_new_scaled)

    # 创建结果DataFrame
    results = pd.DataFrame({
        'Id': new_df['Id'],
        'Sequence': new_df['Sequence'],
        'Predicted_Toxicity_Score': predictions
    })

    # 添加置信区间（如果是随机森林等模型）
    if hasattr(model, 'estimators_') and hasattr(model, 'named_steps'):
        # 对于Pipeline中的随机森林
        rf_model = model.named_steps['regressor']
        if hasattr(rf_model, 'estimators_'):
            all_preds = np.array([tree.predict(X_new_scaled)
                                  for tree in rf_model.estimators_])
            results['Prediction_Std'] = all_preds.std(axis=0)
            results['Prediction_CI_95_lower'] = (results['Predicted_Toxicity_Score'] -
                                                 1.96 * results['Prediction_Std'])
            results['Prediction_CI_95_upper'] = (results['Predicted_Toxicity_Score'] +
                                                 1.96 * results['Prediction_Std'])

    # 添加分类标签（使用0.5作为阈值）
    results['Predicted_Class'] = (results['Predicted_Toxicity_Score'] > 0.5).astype(int)

    # 统计信息
    print(f"\n预测结果统计:")
    print(f"  预测值范围: [{predictions.min():.4f}, {predictions.max():.4f}]")
    print(f"  预测值均值: {predictions.mean():.4f} ± {predictions.std():.4f}")
    print(
        f"  预测为毒性的样本数: {sum(results['Predicted_Class'] == 1)} ({sum(results['Predicted_Class'] == 1) / len(results) * 100:.1f}%)")
    print(
        f"  预测为非毒性的样本数: {sum(results['Predicted_Class'] == 0)} ({sum(results['Predicted_Class'] == 0) / len(results) * 100:.1f}%)")

    # 显示高风险样本
    high_risk = results[results['Predicted_Toxicity_Score'] > 0.8]
    if len(high_risk) > 0:
        print(f"\n高风险样本 (毒性分数 > 0.8):")
        for idx, row in high_risk.head(10).iterrows():
            print(f"  {row['Id']}: 分数={row['Predicted_Toxicity_Score']:.4f}")

    # 保存结果
    if output_file:
        results.to_csv(output_file, index=False)
        print(f"\n预测结果已保存至: {output_file}")

    return results


# ==================== 主程序 ====================

if __name__ == "__main__":
    print("=" * 80)
    print("毒性肽回归模型训练与评估")
    print("=" * 80)

    # 1. 加载数据
    print("\n1. 加载训练数据...")
    train_file = path / 'data/train_data.fasta'

    if train_file.exists():
        train_df = read_fasta(train_file)
        print(f"训练数据形状: {train_df.shape}")

        # 准备回归数据
        X, y = prepare_regression_data(train_df)
    else:
        print(f"训练文件不存在: {train_file}")
        print("使用合成数据演示...")
        # 生成合成数据用于演示
        synthetic_df = generate_synthetic_regression_data(n_samples=2000)
        X, y = prepare_regression_data(synthetic_df, target_column='target')

    # 2. 评估多个回归模型
    print("\n2. 评估多个回归模型...")
    results_df, scaler = evaluate_regression_models(X, y, cv=5)

    # 3. 可视化模型比较
    print("\n3. 可视化模型比较...")
    plot_model_comparison(results_df)

    # 4. 选择最佳模型进行优化
    best_model_name = results_df.iloc[0]['Model']
    print(f"\n4. 优化最佳模型: {best_model_name}")

    # 检查是否需要对最佳模型进行优化
    if best_model_name in get_model_param_grids():
        best_model, best_params = optimize_best_model(X, y, best_model_name, results_df)

        if best_model is not None:
            # 5. 在测试集上评估最佳模型
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # 预测
            y_pred = best_model.predict(X_test)

            # 可视化结果
            print("\n5. 可视化最佳模型结果...")

            # 尝试获取特征重要性
            feature_importance = None
            if hasattr(best_model.named_steps['regressor'], 'feature_importances_'):
                feature_importance = best_model.named_steps['regressor'].feature_importances_
            elif hasattr(best_model.named_steps['regressor'], 'coef_'):
                feature_importance = np.abs(best_model.named_steps['regressor'].coef_)

            plot_regression_results(y_test, y_pred, best_model_name,
                                    feature_importance, X.columns.tolist())
    else:
        print(f"{best_model_name} 不支持超参数优化或未定义参数网格")

        # 使用默认模型进行预测
        models = get_regression_models()
        default_model = models[best_model_name]

        # 训练
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 训练和预测
        default_model.fit(X_train_scaled, y_train)
        y_pred = default_model.predict(X_test_scaled)

        # 可视化
        plot_regression_results(y_test, y_pred, best_model_name)

        # 6. 对测试数据进行预测（如果有）
        print("\n6. 对测试数据进行预测...")
        test_file = path / 'test1.fasta'
        if test_file.exists():
            # 获取训练集的特征列名
            expected_features = X.columns.tolist()  # X是在前面定义的训练集特征

            # 使用最佳模型进行预测
            if 'best_model' in locals():
                predict_new_data(best_model, scaler, test_file,
                                 path / 'test1_regression_predictions.csv',
                                 expected_features=expected_features)
            else:
                # 使用默认模型
                models = get_regression_models()
                default_model = models[best_model_name]

                # 训练完整模型
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                default_model.fit(X_scaled, y)

                predict_new_data(default_model, scaler, test_file,
                                 path / 'test1_regression_predictions.csv',
                                 expected_features=expected_features)
        else:
            print(f"测试文件不存在: {test_file}")

    print("\n" + "=" * 80)
    print("回归分析完成！")
    print("=" * 80)