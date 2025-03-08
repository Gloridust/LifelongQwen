"""
经验回放缓冲区模块，实现样本存储和采样功能。

经验回放是一种缓解灾难性遗忘的方法，通过在学习新任务时混合使用旧任务的样本，
帮助模型保持对旧任务的记忆。
"""

import os
import json
import random
import logging
from typing import List, Dict, Any, Optional, Union
from collections import deque

import torch
import numpy as np
from datasets import Dataset

logger = logging.getLogger(__name__)

class ReplayBuffer:
    """
    经验回放缓冲区，用于存储和采样训练样本。
    
    支持多种采样策略：
    - uniform: 均匀采样
    - prioritized: 优先级采样（基于样本重要性）
    - recent: 优先采样最近添加的样本
    """
    
    def __init__(self, capacity: int = 10000, sampling_strategy: str = "uniform"):
        """
        初始化经验回放缓冲区。
        
        Args:
            capacity: 缓冲区容量
            sampling_strategy: 采样策略，可选 "uniform", "prioritized", "recent"
        """
        self.capacity = capacity
        self.sampling_strategy = sampling_strategy
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)  # 样本优先级
        self.sample_count = deque(maxlen=capacity)  # 样本被采样次数
    
    def add_sample(self, sample: Dict[str, Any], priority: float = 1.0) -> None:
        """
        添加单个样本到缓冲区。
        
        Args:
            sample: 样本数据
            priority: 样本优先级
        """
        self.buffer.append(sample)
        self.priorities.append(priority)
        self.sample_count.append(0)
    
    def add_samples(self, samples: Union[List[Dict[str, Any]], Dataset], priorities: Optional[List[float]] = None) -> None:
        """
        添加多个样本到缓冲区。
        
        Args:
            samples: 样本数据列表或 Dataset 对象
            priorities: 样本优先级列表
        """
        # 如果是 Dataset 对象，转换为列表
        if isinstance(samples, Dataset):
            samples = [samples[i] for i in range(len(samples))]
        
        # 如果没有提供优先级，使用默认值
        if priorities is None:
            priorities = [1.0] * len(samples)
        elif len(priorities) != len(samples):
            logger.warning(f"优先级列表长度 ({len(priorities)}) 与样本列表长度 ({len(samples)}) 不匹配，使用默认优先级")
            priorities = [1.0] * len(samples)
        
        # 添加样本
        for sample, priority in zip(samples, priorities):
            self.add_sample(sample, priority)
        
        logger.info(f"已添加 {len(samples)} 个样本到经验回放缓冲区")
    
    def sample(self, n: int) -> List[Dict[str, Any]]:
        """
        从缓冲区采样 n 个样本。
        
        Args:
            n: 采样数量
            
        Returns:
            采样的样本列表
        """
        if n <= 0:
            return []
        
        if len(self.buffer) == 0:
            logger.warning("经验回放缓冲区为空，无法采样")
            return []
        
        # 确保采样数量不超过缓冲区大小
        n = min(n, len(self.buffer))
        
        # 根据采样策略选择样本
        if self.sampling_strategy == "uniform":
            # 均匀采样
            indices = random.sample(range(len(self.buffer)), n)
        elif self.sampling_strategy == "prioritized":
            # 优先级采样
            priorities = np.array(self.priorities)
            probabilities = priorities / np.sum(priorities)
            indices = np.random.choice(len(self.buffer), n, p=probabilities, replace=False)
        elif self.sampling_strategy == "recent":
            # 优先采样最近添加的样本
            indices = range(max(0, len(self.buffer) - n), len(self.buffer))
        else:
            # 默认使用均匀采样
            logger.warning(f"未知的采样策略: {self.sampling_strategy}，使用均匀采样")
            indices = random.sample(range(len(self.buffer)), n)
        
        # 更新样本被采样次数
        for i in indices:
            self.sample_count[i] += 1
        
        # 返回采样的样本
        return [self.buffer[i] for i in indices]
    
    def size(self) -> int:
        """
        获取缓冲区当前大小。
        
        Returns:
            缓冲区大小
        """
        return len(self.buffer)
    
    def is_full(self) -> bool:
        """
        检查缓冲区是否已满。
        
        Returns:
            是否已满
        """
        return len(self.buffer) >= self.capacity
    
    def clear(self) -> None:
        """清空缓冲区。"""
        self.buffer.clear()
        self.priorities.clear()
        self.sample_count.clear()
        logger.info("经验回放缓冲区已清空")
    
    def save(self, path: str) -> None:
        """
        保存缓冲区到文件。
        
        Args:
            path: 保存路径
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 将缓冲区数据转换为可序列化的格式
        data = {
            "capacity": self.capacity,
            "sampling_strategy": self.sampling_strategy,
            "buffer": list(self.buffer),
            "priorities": list(self.priorities),
            "sample_count": list(self.sample_count),
        }
        
        # 保存到文件
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
        
        logger.info(f"经验回放缓冲区已保存到 {path}")
    
    def load(self, path: str) -> None:
        """
        从文件加载缓冲区。
        
        Args:
            path: 加载路径
        """
        if not os.path.exists(path):
            logger.warning(f"经验回放缓冲区文件不存在: {path}")
            return
        
        # 从文件加载数据
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 更新缓冲区
        self.capacity = data["capacity"]
        self.sampling_strategy = data["sampling_strategy"]
        
        # 重新创建缓冲区
        self.buffer = deque(data["buffer"], maxlen=self.capacity)
        self.priorities = deque(data["priorities"], maxlen=self.capacity)
        self.sample_count = deque(data["sample_count"], maxlen=self.capacity)
        
        logger.info(f"经验回放缓冲区已加载自 {path}，包含 {len(self.buffer)} 个样本") 