"""
EWC 正则化器模块，实现 Elastic Weight Consolidation 算法。

EWC (Elastic Weight Consolidation) 是一种缓解灾难性遗忘的方法，
通过估计参数的重要性，并在学习新任务时对重要参数施加更强的正则化约束。
"""

import os
import logging
from typing import Dict, List, Optional, Union, Tuple, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)

class EWCRegularizer:
    """
    EWC 正则化器，用于缓解灾难性遗忘。
    
    实现了 Elastic Weight Consolidation 算法，通过估计参数的重要性，
    并在学习新任务时对重要参数施加更强的正则化约束。
    """
    
    def __init__(self, model: nn.Module):
        """
        初始化 EWC 正则化器。
        
        Args:
            model: 模型
        """
        self.model = model
        self.params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        self.importance = {}  # 参数重要性
        self.old_params = {}  # 旧参数值
        
        # 初始化参数重要性和旧参数值
        for n, p in self.params.items():
            self.importance[n] = torch.zeros_like(p)
            self.old_params[n] = p.clone().detach()
    
    def update_importance(self, model: nn.Module, dataset: Dataset, trainer) -> None:
        """
        更新参数重要性。
        
        Args:
            model: 模型
            dataset: 数据集
            trainer: 训练器
        """
        logger.info("计算参数重要性")
        
        # 保存当前参数值
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.params:
                self.old_params[n] = p.clone().detach()
        
        # 创建数据加载器
        dataloader = DataLoader(
            dataset,
            batch_size=8,
            shuffle=True,
        )
        
        # 计算 Fisher 信息矩阵（参数重要性）
        model.eval()
        for batch in tqdm(dataloader, desc="计算参数重要性"):
            # 使用训练器的数据处理逻辑
            batch = trainer._prepare_inputs(batch)
            
            # 前向传播
            outputs = model(**batch)
            loss = outputs.loss
            
            # 计算梯度
            model.zero_grad()
            loss.backward()
            
            # 累积 Fisher 信息矩阵
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None and n in self.importance:
                    # Fisher 信息矩阵是梯度的平方
                    self.importance[n] += p.grad.detach().pow(2)
        
        # 归一化 Fisher 信息矩阵
        for n in self.importance:
            self.importance[n] /= len(dataloader)
        
        logger.info("参数重要性计算完成")
    
    def penalty(self, model: nn.Module) -> torch.Tensor:
        """
        计算 EWC 惩罚项。
        
        Args:
            model: 模型
            
        Returns:
            EWC 惩罚项
        """
        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.importance and n in self.old_params:
                # EWC 惩罚项: 0.5 * importance * (current_param - old_param)^2
                loss += 0.5 * (self.importance[n] * (p - self.old_params[n]).pow(2)).sum()
        
        return loss
    
    def save(self, path: str) -> None:
        """
        保存 EWC 参数重要性和旧参数值。
        
        Args:
            path: 保存路径
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 将参数重要性和旧参数值转换为 CPU 张量
        importance_cpu = {n: p.cpu() for n, p in self.importance.items()}
        old_params_cpu = {n: p.cpu() for n, p in self.old_params.items()}
        
        # 保存参数重要性和旧参数值
        torch.save({
            'importance': importance_cpu,
            'old_params': old_params_cpu,
        }, path)
        
        logger.info(f"EWC 参数重要性已保存到 {path}")
    
    def load(self, path: str) -> None:
        """
        加载 EWC 参数重要性和旧参数值。
        
        Args:
            path: 加载路径
        """
        if not os.path.exists(path):
            logger.warning(f"EWC 参数重要性文件不存在: {path}")
            return
        
        # 加载参数重要性和旧参数值
        checkpoint = torch.load(path, map_location='cpu')
        
        # 将参数重要性和旧参数值转换为与模型相同的设备
        device = next(self.model.parameters()).device
        self.importance = {n: p.to(device) for n, p in checkpoint['importance'].items()}
        self.old_params = {n: p.to(device) for n, p in checkpoint['old_params'].items()}
        
        logger.info(f"EWC 参数重要性已加载自 {path}")
        
        # 检查参数名称是否匹配
        current_params = set(self.params.keys())
        loaded_params = set(self.importance.keys())
        
        if current_params != loaded_params:
            missing = current_params - loaded_params
            extra = loaded_params - current_params
            
            if missing:
                logger.warning(f"缺少参数重要性: {missing}")
            
            if extra:
                logger.warning(f"多余参数重要性: {extra}")
                
                # 移除多余的参数重要性
                for n in extra:
                    del self.importance[n]
                    del self.old_params[n] 