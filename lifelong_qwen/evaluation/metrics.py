"""
评估指标模块，实现各种评估指标的计算。
"""

import logging
from typing import Dict, List, Optional, Union, Any

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

logger = logging.getLogger(__name__)

def calculate_metrics(
    predictions: List[int],
    references: List[int],
    metrics: List[str] = ["accuracy"],
) -> Dict[str, float]:
    """
    计算评估指标。
    
    Args:
        predictions: 预测结果
        references: 参考答案
        metrics: 要计算的指标列表
        
    Returns:
        指标结果字典
    """
    results = {}
    
    # 确保预测和参考长度相同
    if len(predictions) != len(references):
        logger.warning(f"预测长度 ({len(predictions)}) 与参考长度 ({len(references)}) 不匹配")
        # 截断到相同长度
        min_len = min(len(predictions), len(references))
        predictions = predictions[:min_len]
        references = references[:min_len]
    
    # 计算准确率
    if "accuracy" in metrics:
        results["accuracy"] = accuracy_score(references, predictions)
    
    # 计算精确率、召回率和 F1 分数
    if any(m in metrics for m in ["precision", "recall", "f1"]):
        precision, recall, f1, _ = precision_recall_fscore_support(
            references, predictions, average="macro"
        )
        
        if "precision" in metrics:
            results["precision"] = precision
        
        if "recall" in metrics:
            results["recall"] = recall
        
        if "f1" in metrics:
            results["f1"] = f1
    
    # 计算困惑度（基于交叉熵损失）
    if "perplexity" in metrics and "loss" in results:
        results["perplexity"] = np.exp(results["loss"])
    
    return results

def calculate_forgetting(
    current_loss: float,
    baseline_loss: float,
    domains: List[str] = ["general"],
) -> Dict[str, float]:
    """
    计算遗忘指标。
    
    Args:
        current_loss: 当前模型的损失
        baseline_loss: 基线模型的损失
        domains: 评估的领域
        
    Returns:
        遗忘指标结果字典
    """
    # 遗忘指标：当前损失与基线损失的比值
    # 值越大表示遗忘越严重
    forgetting_ratio = current_loss / baseline_loss if baseline_loss > 0 else 1.0
    
    # 归一化到 [0, 1] 范围，其中 0 表示无遗忘，1 表示完全遗忘
    # 使用 sigmoid 函数进行归一化
    normalized_forgetting = 2 / (1 + np.exp(-forgetting_ratio + 1)) - 1
    
    # 为每个领域分配相同的遗忘指标
    # 在实际应用中，可能需要为不同领域计算不同的遗忘指标
    results = {domain: normalized_forgetting for domain in domains}
    
    # 添加总体遗忘指标
    results["overall"] = normalized_forgetting
    
    return results

def calculate_knowledge_retention(
    old_accuracy: float,
    new_accuracy: float,
) -> float:
    """
    计算知识保留率。
    
    Args:
        old_accuracy: 旧任务的准确率
        new_accuracy: 新任务的准确率
        
    Returns:
        知识保留率
    """
    # 知识保留率：新任务准确率与旧任务准确率的比值
    # 值越接近 1 表示保留率越高
    retention = new_accuracy / old_accuracy if old_accuracy > 0 else 0.0
    
    # 限制在 [0, 1] 范围内
    retention = max(0.0, min(1.0, retention))
    
    return retention

def calculate_backward_transfer(
    old_performance: Dict[str, float],
    new_performance: Dict[str, float],
) -> Dict[str, float]:
    """
    计算反向迁移指标。
    
    Args:
        old_performance: 学习新任务前的性能
        new_performance: 学习新任务后的性能
        
    Returns:
        反向迁移指标结果字典
    """
    results = {}
    
    # 计算每个指标的反向迁移
    for metric in old_performance:
        if metric in new_performance:
            # 反向迁移：学习新任务后旧任务性能的变化
            # 正值表示积极迁移，负值表示消极迁移（遗忘）
            results[metric] = new_performance[metric] - old_performance[metric]
    
    return results 