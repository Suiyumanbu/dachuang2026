#!/usr/bin/env python3
"""
肽序列理化性质分析工具
=========================================

该模块用于计算肽/蛋白质序列的各种理化性质，包括：
- 基本序列信息
- 分子量及组成
- 氨基酸组成分析
- 电荷性质（等电点、pH-电荷曲线）
- 结构性质（疏水性、二级结构倾向）
- 光谱性质（消光系数、吸光度）
- 稳定性预测

作者: BioPython 示例
版本: 1.0.0
"""

import os
import sys
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns

from Bio.Seq import Seq
from Bio.SeqUtils import molecular_weight
from Bio.SeqUtils.ProtParam import ProteinAnalysis


# ============================================================================
# 数据类定义
# ============================================================================

@dataclass
class AminoAcidInfo:
    """氨基酸基本信息"""
    one_letter: str
    three_letter: str
    name: str
    category: str
    molecular_weight: float
    hydropathy_index: float  # Kyte-Doolittle疏水指数


@dataclass
class PeptideProperties:
    """肽序列的所有计算结果"""
    # 序列信息
    sequence: str
    name: str
    length: int

    # 分子量
    molecular_weight_average: float
    molecular_weight_monoisotopic: float
    average_residue_weight: float

    # 氨基酸组成
    aa_counts: Dict[str, int]
    aa_percentages: Dict[str, float]
    category_composition: Dict[str, Dict[str, float]]

    # 电荷性质
    isoelectric_point: float
    acidic_residues: int
    basic_residues: int
    net_charge_pH7: int
    charge_at_ph: Dict[float, float]

    # 结构性质
    gravy_score: float
    aromaticity: float
    aliphatic_index: float
    helix_fraction: float
    turn_fraction: float
    sheet_fraction: float

    # 光谱性质
    extinction_coeff_reduced: float
    extinction_coeff_oxidized: float
    absorbance_280: float
    trp_count: int
    tyr_count: int
    cys_count: int

    # 稳定性
    instability_index: float
    is_stable: bool
    half_life_prediction: str
    n_terminal_aa: str


# ============================================================================
# 氨基酸常量数据库
# ============================================================================

class AminoAcidDatabase:
    """氨基酸特性数据库"""

    # 标准20种氨基酸的一字母编码
    STANDARD_AMINO_ACIDS = set('ACDEFGHIKLMNPQRSTVWY')

    # 氨基酸分类系统
    CATEGORIES = {
        '酸性': ['D', 'E'],
        '碱性': ['R', 'K', 'H'],
        '极性不带电': ['S', 'T', 'N', 'Q'],
        '非极性疏水': ['A', 'V', 'L', 'I', 'M', 'F', 'W', 'P', 'G'],
        '芳香族': ['F', 'W', 'Y'],
        '含硫': ['C', 'M'],
        '脂肪族': ['A', 'V', 'L', 'I'],
        '小分子': ['A', 'G', 'S', 'N', 'D', 'T', 'V'],
        '大分子': ['R', 'K', 'E', 'Q', 'M', 'W', 'Y', 'F', 'L', 'I']
    }

    # Kyte-Doolittle疏水指数
    HYDROPATHY = {
        'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
        'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
        'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
        'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
    }

    # 摩尔消光系数 (280 nm, M⁻¹ cm⁻¹)
    EXTINCTION_COEFFICIENTS = {
        'W': 5500,  # 色氨酸
        'Y': 1490,  # 酪氨酸
        'C': 125  # 半胱氨酸（还原型）
    }

    # N端氨基酸对应的哺乳动物细胞半衰期
    HALF_LIFE_RULES = {
        'M': '>20小时（稳定）',
        'S': '>20小时（稳定）',
        'A': '>20小时（稳定）',
        'T': '>20小时（稳定）',
        'V': '>20小时（稳定）',
        'G': '>20小时（稳定）',
        'R': '2小时（不稳定）',
        'K': '2小时（不稳定）',
        'F': '2分钟（极不稳定）',
        'L': '2分钟（极不稳定）',
        'Y': '2分钟（极不稳定）',
        'W': '2分钟（极不稳定）',
        'D': '2分钟（极不稳定）',
        'E': '2分钟（极不稳定）',
        'P': '2分钟（极不稳定）',
        'I': '>20小时（稳定）',
        'H': '>20小时（稳定）',
        'N': '>20小时（稳定）',
        'Q': '>20小时（稳定）',
        'C': '>20小时（稳定）'
    }


# ============================================================================
# 核心计算类
# ============================================================================

class PeptideAnalyzer:
    """
    肽序列分析器 - 核心计算引擎

    使用方法:
    >>> analyzer = PeptideAnalyzer("ACDEFGHIKL")
    >>> results = analyzer.analyze_all()
    >>> print(results.molecular_weight_average)
    """

    def __init__(self, sequence: str, name: str = "未命名肽"):
        """
        初始化分析器

        参数:
            sequence: 氨基酸序列（一字母代码）
            name: 序列名称（用于报告）

        异常:
            ValueError: 序列包含无效氨基酸
        """
        self.sequence = sequence.strip().upper()
        self.name = name

        self._validate_sequence()

        # 使用Biopython的分析引擎
        self._biopython_analyzer = ProteinAnalysis(self.sequence)

        # 预计算常用值以提高性能
        self._length = len(self.sequence)
        self._aa_list = list(self.sequence)

    def _validate_sequence(self) -> None:
        """验证序列是否包含非标准氨基酸"""
        invalid_chars = set(self.sequence) - AminoAcidDatabase.STANDARD_AMINO_ACIDS
        if invalid_chars:
            raise ValueError(
                f"序列包含非标准氨基酸: {invalid_chars}\n"
                f"有效氨基酸: {sorted(AminoAcidDatabase.STANDARD_AMINO_ACIDS)}"
            )
        if not self.sequence:
            raise ValueError("序列不能为空")

    def analyze_all(self) -> PeptideProperties:
        """
        执行完整的性质分析

        返回:
            PeptideProperties: 包含所有计算结果的完整数据类
        """
        return PeptideProperties(
            # 序列信息
            sequence=self.sequence,
            name=self.name,
            length=self._length,

            # 分子量相关
            molecular_weight_average=self._calculate_molecular_weight_average(),
            molecular_weight_monoisotopic=self._calculate_molecular_weight_monoisotopic(),
            average_residue_weight=self._calculate_average_residue_weight(),

            # 氨基酸组成
            aa_counts=self._get_aa_counts(),
            aa_percentages=self._get_aa_percentages(),
            category_composition=self._get_category_composition(),

            # 电荷性质
            isoelectric_point=self._calculate_isoelectric_point(),
            acidic_residues=self._count_acidic_residues(),
            basic_residues=self._count_basic_residues(),
            net_charge_pH7=self._calculate_net_charge_at_pH7(),
            charge_at_ph=self._calculate_charge_curve(),

            # 结构性质
            gravy_score=self._calculate_gravy(),
            aromaticity=self._calculate_aromaticity(),
            aliphatic_index=self._calculate_aliphatic_index(),
            helix_fraction=self._get_secondary_structure_fraction()[0],
            turn_fraction=self._get_secondary_structure_fraction()[1],
            sheet_fraction=self._get_secondary_structure_fraction()[2],

            # 光谱性质
            extinction_coeff_reduced=self._get_extinction_coefficient()[0],
            extinction_coeff_oxidized=self._get_extinction_coefficient()[1],
            absorbance_280=self._calculate_absorbance_280(),
            trp_count=self.sequence.count('W'),
            tyr_count=self.sequence.count('Y'),
            cys_count=self.sequence.count('C'),

            # 稳定性
            instability_index=self._calculate_instability_index(),
            is_stable=self._is_stable(),
            half_life_prediction=self._predict_half_life(),
            n_terminal_aa=self.sequence[0] if self.sequence else ''
        )

    # ------------------------------------------------------------------------
    # 分子量计算方法
    # ------------------------------------------------------------------------

    def _calculate_molecular_weight_average(self) -> float:
        """计算平均分子量（使用氨基酸平均原子质量）"""
        return round(molecular_weight(self.sequence, seq_type="protein"), 2)

    def _calculate_molecular_weight_monoisotopic(self) -> float:
        """计算单同位素分子量（使用最轻同位素）"""
        return round(molecular_weight(
            self.sequence,
            seq_type="protein",
            monoisotopic=True
        ), 2)

    def _calculate_average_residue_weight(self) -> float:
        """计算平均残基分子量"""
        return round(
            self._calculate_molecular_weight_average() / self._length,
            2
        )

    # ------------------------------------------------------------------------
    # 氨基酸组成分析方法
    # ------------------------------------------------------------------------

    def _get_aa_counts(self) -> Dict[str, int]:
        """获取各氨基酸计数"""
        return self._biopython_analyzer.count_amino_acids()

    def _get_aa_percentages(self) -> Dict[str, float]:
        """获取各氨基酸百分比（0-100）"""
        raw_percentages = self._biopython_analyzer.get_amino_acids_percent()
        return {
            aa: round(percent * 100, 2)
            for aa, percent in raw_percentages.items()
        }

    def _get_category_composition(self) -> Dict[str, Dict[str, float]]:
        """获取各类别氨基酸的计数和百分比"""
        composition = {}
        aa_counts = self._get_aa_counts()

        for category, aas in AminoAcidDatabase.CATEGORIES.items():
            count = sum(aa_counts.get(aa, 0) for aa in aas)
            percent = (count / self._length * 100) if self._length > 0 else 0

            composition[category] = {
                'count': count,
                'percentage': round(percent, 2)
            }

        return composition

    # ------------------------------------------------------------------------
    # 电荷性质计算方法
    # ------------------------------------------------------------------------

    def _calculate_isoelectric_point(self) -> float:
        """计算等电点(pI)"""
        try:
            return round(self._biopython_analyzer.isoelectric_point(), 2)
        except Exception as e:
            print(f"警告: 等电点计算失败 - {e}")
            return 0.0

    def _count_acidic_residues(self) -> int:
        """计数酸性残基 (D, E)"""
        return self.sequence.count('D') + self.sequence.count('E')

    def _count_basic_residues(self) -> int:
        """计数碱性残基 (R, K, H)"""
        return self.sequence.count('R') + self.sequence.count('K') + self.sequence.count('H')

    def _calculate_net_charge_at_pH7(self) -> int:
        """计算pH7.0时的理论净电荷"""
        return self._count_basic_residues() - self._count_acidic_residues()

    def _calculate_charge_at_pH(self, ph: float) -> Optional[float]:
        """计算指定pH下的净电荷"""
        try:
            return round(self._biopython_analyzer.charge_at_pH(ph), 3)
        except Exception:
            return None

    def _calculate_charge_curve(self, ph_points: List[float] = None) -> Dict[float, float]:
        """计算pH-电荷曲线数据点"""
        if ph_points is None:
            ph_points = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 7.4, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]

        charge_dict = {}
        for ph in ph_points:
            charge = self._calculate_charge_at_pH(ph)
            if charge is not None:
                charge_dict[ph] = charge

        return charge_dict

    # ------------------------------------------------------------------------
    # 结构性质计算方法
    # ------------------------------------------------------------------------

    def _calculate_gravy(self) -> float:
        """计算GRAVY值（平均疏水性）"""
        return round(self._biopython_analyzer.gravy(), 3)

    def _calculate_aromaticity(self) -> float:
        """计算芳香性值"""
        return round(self._biopython_analyzer.aromaticity(), 3)

    def _calculate_aliphatic_index(self) -> float:
        """计算脂肪族指数"""
        try:
            return round(self._biopython_analyzer.aliphatic_index(), 2)
        except Exception:
            return 0.0

    def _get_secondary_structure_fraction(self) -> Tuple[float, float, float]:
        """获取二级结构倾向分数"""
        try:
            helix, turn, sheet = self._biopython_analyzer.secondary_structure_fraction()
            return (round(helix, 3), round(turn, 3), round(sheet, 3))
        except Exception:
            return (0.0, 0.0, 0.0)

    # ------------------------------------------------------------------------
    # 光谱性质计算方法
    # ------------------------------------------------------------------------

    def _get_extinction_coefficient(self) -> Tuple[float, float]:
        """获取摩尔消光系数"""
        try:
            reduced, oxidized = self._biopython_analyzer.molar_extinction_coefficient()
            return (round(reduced, 0), round(oxidized, 0))
        except Exception:
            return (0.0, 0.0)

    def _calculate_absorbance_280(self) -> float:
        """
        计算1 mg/mL溶液在280nm的吸光度

        基于公式: A280 = (5500*nW + 1490*nY + 125*nC) / 分子量
        """
        mw = self._calculate_molecular_weight_average()
        if mw == 0:
            return 0.0

        w_count = self.sequence.count('W')
        y_count = self.sequence.count('Y')
        c_count = self.sequence.count('C')

        ext_coeff = (
                w_count * AminoAcidDatabase.EXTINCTION_COEFFICIENTS['W'] +
                y_count * AminoAcidDatabase.EXTINCTION_COEFFICIENTS['Y'] +
                c_count * AminoAcidDatabase.EXTINCTION_COEFFICIENTS['C']
        )

        # 转换为1 mg/mL的吸光度
        absorbance = ext_coeff / mw * 10  # 单位: (mg/mL)^-1 cm^-1
        return round(absorbance, 3)

    # ------------------------------------------------------------------------
    # 稳定性预测方法
    # ------------------------------------------------------------------------

    def _calculate_instability_index(self) -> float:
        """计算不稳定指数"""
        try:
            return round(self._biopython_analyzer.instability_index(), 2)
        except Exception:
            return 0.0

    def _is_stable(self) -> bool:
        """
        根据不稳定指数判断稳定性

        返回:
            True: 稳定 (指数 < 40)
            False: 不稳定 (指数 >= 40)
        """
        return self._calculate_instability_index() < 40

    def _predict_half_life(self) -> str:
        """预测哺乳动物细胞中的半衰期"""
        if not self.sequence:
            return "未知"

        n_term = self.sequence[0]
        return AminoAcidDatabase.HALF_LIFE_RULES.get(
            n_term,
            "未知（基于N端氨基酸无法预测）"
        )


# ============================================================================
# 报告生成器
# ============================================================================

class ReportGenerator:
    """报告生成器 - 负责格式化和输出结果"""

    def __init__(self, results: PeptideProperties):
        """
        初始化报告生成器

        参数:
            results: 肽分析结果数据对象
        """
        self.results = results

    def print_summary(self) -> None:
        """打印简要摘要"""
        print("\n" + "=" * 60)
        print(f" 肽序列分析摘要 - {self.results.name}")
        print("=" * 60)

        print(f"序列: {self.results.sequence}")
        print(f"长度: {self.results.length} aa")
        print(f"分子量: {self.results.molecular_weight_average:,.2f} Da")
        print(f"等电点(pI): {self.results.isoelectric_point}")
        print(f"净电荷(pH7): {self.results.net_charge_pH7:+d}")
        print(f"疏水性(GRAVY): {self.results.gravy_score}")
        print(f"稳定性: {'稳定' if self.results.is_stable else '不稳定'}")
        print(f"A280吸光度: {self.results.absorbance_280} (1 mg/mL)")

    def print_detailed_report(self) -> None:
        """打印详细分析报告"""
        print("\n" + "=" * 80)
        print(f" 肽序列详细分析报告 - {self.results.name}")
        print("=" * 80)

        self._print_sequence_info()
        self._print_molecular_properties()
        self._print_aa_composition()
        self._print_charge_properties()
        self._print_structural_properties()
        self._print_spectral_properties()
        self._print_stability_properties()

        print("\n" + "=" * 80)
        print(" 报告生成完成")
        print("=" * 80)

    def _print_sequence_info(self) -> None:
        """打印序列基本信息"""
        print("\n【序列信息】")
        print(f"  名称: {self.results.name}")
        print(f"  序列: {self.results.sequence}")
        print(f"  长度: {self.results.length} 个氨基酸")
        print(f"  N端: {self.results.n_terminal_aa}")
        print(f"  C端: {self.results.sequence[-1] if self.results.sequence else ''}")

    def _print_molecular_properties(self) -> None:
        """打印分子量相关性质"""
        print("\n【分子量性质】")
        print(f"  平均分子量: {self.results.molecular_weight_average:,.2f} Da")
        print(f"  单同位素分子量: {self.results.molecular_weight_monoisotopic:,.2f} Da")
        print(f"  平均残基分子量: {self.results.average_residue_weight:.2f} Da")

    def _print_aa_composition(self) -> None:
        """打印氨基酸组成"""
        print("\n【氨基酸组成】")

        # 显示分类组成
        print("  按类别:")
        for category, data in self.results.category_composition.items():
            if data['count'] > 0:
                print(f"    {category}: {data['count']} ({data['percentage']}%)")

        # 显示详细的氨基酸组成
        print("\n  详细组成:")
        sorted_aas = sorted(self.results.aa_counts.keys())
        for aa in sorted_aas:
            if self.results.aa_counts[aa] > 0:
                print(f"    {aa}: {self.results.aa_counts[aa]} ({self.results.aa_percentages[aa]}%)")

    def _print_charge_properties(self) -> None:
        """打印电荷性质"""
        print("\n【电荷性质】")
        print(f"  等电点(pI): {self.results.isoelectric_point}")
        print(f"  酸性残基(D+E): {self.results.acidic_residues}")
        print(f"  碱性残基(R+K+H): {self.results.basic_residues}")
        print(f"  理论净电荷(pH7): {self.results.net_charge_pH7:+d}")

        # 显示关键pH点的电荷
        key_ph = [3.0, 5.0, 7.0, 7.4, 8.0, 9.0]
        print("  不同pH下的电荷:")
        for ph in key_ph:
            charge = self.results.charge_at_ph.get(ph, None)
            if charge is not None:
                print(f"    pH {ph:3.1f}: {charge:+6.3f}")

    def _print_structural_properties(self) -> None:
        """打印结构性质"""
        print("\n【结构性质】")
        print(f"  疏水性(GRAVY): {self.results.gravy_score}")
        print(f"    解释: {'疏水' if self.results.gravy_score > 0 else '亲水'}")
        print(f"  芳香性: {self.results.aromaticity}")
        print(f"  脂肪族指数: {self.results.aliphatic_index}")
        print(f"  二级结构倾向:")
        print(f"    α-螺旋: {self.results.helix_fraction}")
        print(f"    β-折叠: {self.results.sheet_fraction}")
        print(f"    转角: {self.results.turn_fraction}")

    def _print_spectral_properties(self) -> None:
        """打印光谱性质"""
        print("\n【光谱性质】")
        print(f"  摩尔消光系数(还原型): {self.results.extinction_coeff_reduced:,.0f} M⁻¹ cm⁻¹")
        print(f"  摩尔消光系数(氧化型): {self.results.extinction_coeff_oxidized:,.0f} M⁻¹ cm⁻¹")
        print(f"  A280吸光度(1 mg/mL): {self.results.absorbance_280}")
        print(f"  生色团组成:")
        print(f"    色氨酸(W): {self.results.trp_count} 个")
        print(f"    酪氨酸(Y): {self.results.tyr_count} 个")
        print(f"    半胱氨酸(C): {self.results.cys_count} 个")

    def _print_stability_properties(self) -> None:
        """打印稳定性预测"""
        print("\n【稳定性预测】")
        print(f"  不稳定指数: {self.results.instability_index}")
        print(f"  稳定性判断: {'稳定' if self.results.is_stable else '不稳定'}")
        print(f"  哺乳动物细胞半衰期: {self.results.half_life_prediction}")

    def to_dataframe(self) -> pd.DataFrame:
        """将结果转换为pandas DataFrame"""
        data = {
            '属性': [],
            '值': [],
            '单位/说明': []
        }

        # 添加所有属性
        self._add_to_dataframe(data, '序列', self.results.sequence, '')
        self._add_to_dataframe(data, '长度', self.results.length, 'aa')
        self._add_to_dataframe(data, '平均分子量', self.results.molecular_weight_average, 'Da')
        self._add_to_dataframe(data, '单同位素分子量', self.results.molecular_weight_monoisotopic, 'Da')
        self._add_to_dataframe(data, '等电点(pI)', self.results.isoelectric_point, '')
        self._add_to_dataframe(data, '净电荷(pH7)', self.results.net_charge_pH7, '')
        self._add_to_dataframe(data, '疏水性(GRAVY)', self.results.gravy_score, '')
        self._add_to_dataframe(data, '芳香性', self.results.aromaticity, '')
        self._add_to_dataframe(data, '不稳定指数', self.results.instability_index, '')
        self._add_to_dataframe(data, '稳定性', '稳定' if self.results.is_stable else '不稳定', '')
        self._add_to_dataframe(data, 'A280吸光度', self.results.absorbance_280, '(mg/mL)⁻¹ cm⁻¹')

        return pd.DataFrame(data)

    def _add_to_dataframe(self, df: Dict, prop: str, value: any, unit: str) -> None:
        """辅助方法：向DataFrame添加一行"""
        df['属性'].append(prop)
        df['值'].append(value)
        df['单位/说明'].append(unit)

    def save_to_csv(self, filename: str) -> None:
        """保存结果为CSV文件"""
        df = self.to_dataframe()
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"结果已保存至: {filename}")

    def save_to_text(self, filename: str) -> None:
        """保存结果为文本文件"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"肽序列分析报告 - {self.results.name}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"序列: {self.results.sequence}\n")
            f.write(f"长度: {self.results.length}\n\n")

            for prop, value in self._get_all_properties_as_dict().items():
                f.write(f"{prop}: {value}\n")

        print(f"文本报告已保存至: {filename}")

    def _get_all_properties_as_dict(self) -> Dict:
        """将所有属性转换为字典"""
        return {
            '分子量(平均)': f"{self.results.molecular_weight_average:,.2f} Da",
            '分子量(单同位素)': f"{self.results.molecular_weight_monoisotopic:,.2f} Da",
            '等电点': f"{self.results.isoelectric_point}",
            '疏水性(GRAVY)': f"{self.results.gravy_score}",
            '不稳定指数': f"{self.results.instability_index}",
            'A280吸光度': f"{self.results.absorbance_280}",
        }


# ============================================================================
# 可视化工具
# ============================================================================

class Visualizer:
    """可视化工具 - 创建分析图表"""

    def __init__(self, results: PeptideProperties):
        """
        初始化可视化器

        参数:
            results: 肽分析结果数据对象
        """
        self.results = results

        # 设置中文字体支持
        self._setup_chinese_font()

        # 设置Seaborn样式
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 11

    def _setup_chinese_font(self) -> None:
        """设置中文字体支持"""
        try:
            # 尝试设置支持中文的字体
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
        except:
            pass

    def plot_aa_composition(self, save_path: Optional[str] = None) -> None:
        """
        绘制氨基酸组成图

        参数:
            save_path: 保存路径，None则显示不保存
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'氨基酸组成分析 - {self.results.name}', fontsize=14, fontweight='bold')

        # 左图：氨基酸计数条形图
        aas = []
        counts = []
        colors = []

        for aa, count in sorted(self.results.aa_counts.items()):
            if count > 0:
                aas.append(aa)
                counts.append(count)
                # 根据氨基酸性质设置颜色
                if aa in AminoAcidDatabase.CATEGORIES['酸性']:
                    colors.append('#FF6B6B')  # 红色
                elif aa in AminoAcidDatabase.CATEGORIES['碱性']:
                    colors.append('#4ECDC4')  # 青色
                elif aa in AminoAcidDatabase.CATEGORIES['非极性疏水']:
                    colors.append('#FFD93D')  # 黄色
                else:
                    colors.append('#6BCF7F')  # 绿色

        bars = ax1.bar(aas, counts, color=colors, edgecolor='black', linewidth=0.5)
        ax1.set_xlabel('氨基酸', fontsize=12)
        ax1.set_ylabel('计数', fontsize=12)
        ax1.set_title('氨基酸出现频率', fontsize=12)
        ax1.tick_params(axis='both', labelsize=10)

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                     f'{int(height)}', ha='center', va='bottom', fontsize=9)

        # 右图：类别组成饼图
        categories = []
        percentages = []

        for category, data in self.results.category_composition.items():
            if data['count'] > 0 and category in ['酸性', '碱性', '非极性疏水', '极性不带电']:
                categories.append(category)
                percentages.append(data['percentage'])

        if percentages:
            wedges, texts, autotexts = ax2.pie(
                percentages,
                labels=categories,
                autopct='%1.1f%%',
                startangle=90,
                colors=['#FF6B6B', '#4ECDC4', '#FFD93D', '#6BCF7F']
            )
            ax2.set_title('氨基酸类别分布', fontsize=12)

            # 设置字体大小
            for text in texts + autotexts:
                text.set_fontsize(10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"图表已保存至: {save_path}")
        plt.show()

    def plot_charge_profile(self, save_path: Optional[str] = None) -> None:
        """
        绘制电荷-pH曲线

        参数:
            save_path: 保存路径，None则显示不保存
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'电荷性质分析 - {self.results.name}', fontsize=14, fontweight='bold')

        # 左图：电荷-pH曲线
        ph_values = sorted(self.results.charge_at_ph.keys())
        charge_values = [self.results.charge_at_ph[ph] for ph in ph_values]

        ax1.plot(ph_values, charge_values, 'b-', linewidth=2.5, marker='o', markersize=4)
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.7, linewidth=1)
        ax1.axvline(x=self.results.isoelectric_point, color='red',
                    linestyle='--', alpha=0.7, linewidth=1.5,
                    label=f'pI = {self.results.isoelectric_point}')

        ax1.set_xlabel('pH', fontsize=12)
        ax1.set_ylabel('净电荷', fontsize=12)
        ax1.set_title('pH-电荷曲线', fontsize=12)
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 14)

        # 右图：电荷分布柱状图
        categories = ['酸性 (-)', '碱性 (+)', '净电荷 (pH7)']
        values = [
            -self.results.acidic_residues,
            self.results.basic_residues,
            self.results.net_charge_pH7
        ]

        colors = ['#FF6B6B' if v < 0 else '#4ECDC4' for v in values]
        bars = ax2.bar(categories, values, color=colors, edgecolor='black', linewidth=0.8)

        ax2.set_xlabel('电荷类型', fontsize=12)
        ax2.set_ylabel('电荷值', fontsize=12)
        ax2.set_title('电荷分布', fontsize=12)
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
        ax2.grid(True, alpha=0.3, axis='y')

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2.,
                     height + (0.5 if height >= 0 else -0.8),
                     f'{int(height)}', ha='center', va='bottom' if height >= 0 else 'top',
                     fontsize=11, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"图表已保存至: {save_path}")
        plt.show()

    def plot_property_radar(self, save_path: Optional[str] = None) -> None:
        """
        绘制性质雷达图

        参数:
            save_path: 保存路径，None则显示不保存
        """
        # 选择要显示的性质
        categories = ['分子量', '疏水性', '芳香性', '脂肪族指数', '稳定性', '等电点']

        # 归一化处理
        mw_norm = min(self.results.molecular_weight_average / 20000, 1.0)  # 假设最大20000
        gravy_norm = (self.results.gravy_score + 4.5) / 9.0  # 范围 -4.5 到 4.5
        aroma_norm = min(self.results.aromaticity * 2, 1.0)  # 最大0.5
        aliphatic_norm = min(self.results.aliphatic_index / 200, 1.0)  # 最大约200
        stability_norm = 1.0 - min(self.results.instability_index / 100, 1.0)  # 越大越稳定
        pi_norm = self.results.isoelectric_point / 14.0  # 范围 0-14

        values = [mw_norm, gravy_norm, aroma_norm, aliphatic_norm, stability_norm, pi_norm]

        # 闭合雷达图
        values += values[:1]
        categories += categories[:1]

        # 创建雷达图
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        ax.plot(angles, values, 'o-', linewidth=2, color='#4ECDC4')
        ax.fill(angles, values, alpha=0.25, color='#4ECDC4')

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories[:-1], fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_title(f'性质雷达图 - {self.results.name}', fontsize=14, fontweight='bold', pad=20)
        ax.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"雷达图已保存至: {save_path}")
        plt.show()


# ============================================================================
# 主程序入口
# ============================================================================

def main():
    """主程序入口 - 演示使用示例"""

    print("=" * 60)
    print("  肽序列理化性质分析工具 v1.0")
    print("=" * 60)

    # 示例肽序列库
    example_peptides = {
        "胰岛素A链": "GIVEQCCTSICSLYQLENYCN",
        "胰高血糖素": "HSQGTFTSDYSKYLDSRRAQDFVQWLMNT",
        "蜂毒肽": "GIGAVLKVLTTGLPALISWIKRKRQQ",
        "抗菌肽LL-37": "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES",
        "标准测试肽": "ACDEFGHIKLMNPQRSTVWY"  # 包含20种氨基酸
    }

    # 选择要分析的肽
    peptide_name = "标准测试肽"
    peptide_sequence = example_peptides[peptide_name]

    print(f"\n正在分析: {peptide_name}")
    print(f"序列: {peptide_sequence}")
    print(f"长度: {len(peptide_sequence)} aa")

    try:
        # 1. 创建分析器并执行计算
        print("\n[1/4] 正在计算理化性质...")
        analyzer = PeptideAnalyzer(peptide_sequence, peptide_name)
        results = analyzer.analyze_all()

        # 2. 生成报告
        print("[2/4] 正在生成分析报告...")
        report = ReportGenerator(results)
        report.print_summary()
        report.print_detailed_report()

        # 3. 保存数据
        print("\n[3/4] 正在保存分析结果...")
        report.save_to_csv(f"{peptide_name}_properties.csv")
        report.save_to_text(f"{peptide_name}_report.txt")

        # 4. 创建可视化
        print("[4/4] 正在生成可视化图表...")
        viz = Visualizer(results)
        viz.plot_aa_composition(f"{peptide_name}_aa_composition.png")
        viz.plot_charge_profile(f"{peptide_name}_charge_profile.png")
        viz.plot_property_radar(f"{peptide_name}_radar_chart.png")

        print("\n" + "=" * 60)
        print("✅ 分析完成！")
        print("=" * 60)

        # 批量分析所有示例肽
        print("\n📊 批量分析所有示例肽序列:")
        print("-" * 60)

        batch_results = []
        for name, seq in example_peptides.items():
            print(f"  正在分析: {name}...", end="")
            analyzer = PeptideAnalyzer(seq, name)
            res = analyzer.analyze_all()
            batch_results.append({
                '名称': name,
                '长度': res.length,
                '分子量(Da)': f"{res.molecular_weight_average:,.0f}",
                '等电点': res.isoelectric_point,
                '疏水性': res.gravy_score,
                '稳定性': '稳定' if res.is_stable else '不稳定',
                'A280': res.absorbance_280
            })
            print(" 完成")

        # 显示批量结果
        df_batch = pd.DataFrame(batch_results)
        print("\n批量分析结果汇总:")
        print(df_batch.to_string(index=False))
        df_batch.to_csv('batch_peptide_analysis.csv', index=False, encoding='utf-8-sig')
        print("\n批量分析结果已保存至: batch_peptide_analysis.csv")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()