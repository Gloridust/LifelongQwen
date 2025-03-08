"""
适配器管理器模块，用于管理多个适配器。
"""

import os
import json
import logging
from typing import List, Dict, Optional, Union, Set

import torch
from peft import PeftModel, PeftConfig

logger = logging.getLogger(__name__)

class AdapterManager:
    """
    适配器管理器，负责管理多个适配器的加载、切换和合并。
    """
    
    def __init__(self, adapter_base_path: str):
        """
        初始化适配器管理器。
        
        Args:
            adapter_base_path: 适配器基础路径
        """
        self.adapter_base_path = adapter_base_path
        self.active_adapters: Set[str] = set()
        self.adapter_metadata: Dict[str, Dict] = {}
        self._load_metadata()
    
    def _load_metadata(self) -> None:
        """加载所有适配器的元数据。"""
        if not os.path.exists(self.adapter_base_path):
            os.makedirs(self.adapter_base_path, exist_ok=True)
            return
        
        # 遍历适配器目录
        for domain in os.listdir(self.adapter_base_path):
            domain_path = os.path.join(self.adapter_base_path, domain)
            if not os.path.isdir(domain_path):
                continue
            
            # 检查适配器配置文件
            adapter_config_path = os.path.join(domain_path, "adapter_config.json")
            if not os.path.exists(adapter_config_path):
                continue
            
            # 加载适配器元数据
            try:
                with open(adapter_config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # 加载额外元数据（如果存在）
                metadata_path = os.path.join(domain_path, "metadata.json")
                metadata = {}
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                
                # 合并配置和元数据
                self.adapter_metadata[domain] = {
                    "config": config,
                    "metadata": metadata,
                    "path": domain_path,
                }
                logger.info(f"已加载适配器元数据: {domain}")
            except Exception as e:
                logger.warning(f"加载适配器 {domain} 元数据失败: {e}")
    
    def get_available_adapters(self) -> List[str]:
        """获取所有可用的适配器列表。"""
        return list(self.adapter_metadata.keys())
    
    def get_adapter_path(self, domain: str) -> Optional[str]:
        """获取特定领域适配器的路径。"""
        if domain in self.adapter_metadata:
            return self.adapter_metadata[domain]["path"]
        return None
    
    def get_adapter_metadata(self, domain: str) -> Optional[Dict]:
        """获取特定领域适配器的元数据。"""
        if domain in self.adapter_metadata:
            return self.adapter_metadata[domain]["metadata"]
        return None
    
    def activate_adapters(self, model: PeftModel, domains: List[str]) -> PeftModel:
        """
        激活指定领域的适配器。
        
        Args:
            model: PEFT 模型
            domains: 要激活的领域列表
            
        Returns:
            更新后的模型
        """
        # 检查所有领域是否都有对应的适配器
        for domain in domains:
            if domain not in self.adapter_metadata:
                logger.warning(f"适配器 {domain} 不存在，将被跳过")
        
        # 过滤出有效的领域
        valid_domains = [d for d in domains if d in self.adapter_metadata]
        if not valid_domains:
            logger.warning("没有有效的适配器可激活")
            return model
        
        # 设置活跃适配器
        self.active_adapters = set(valid_domains)
        
        # 如果只有一个适配器，直接激活
        if len(valid_domains) == 1:
            adapter_name = valid_domains[0]
            logger.info(f"激活单个适配器: {adapter_name}")
            model.set_adapter(adapter_name)
            return model
        
        # 如果有多个适配器，需要合并或切换
        logger.info(f"激活多个适配器: {', '.join(valid_domains)}")
        
        # 目前简单实现为使用第一个适配器
        # 注意：实际应用中可能需要更复杂的适配器合并策略
        model.set_adapter(valid_domains[0])
        logger.warning(f"当前使用第一个适配器 {valid_domains[0]}，未实现多适配器合并")
        
        return model
    
    def save_adapter_metadata(self, domain: str, metadata: Dict) -> None:
        """
        保存适配器元数据。
        
        Args:
            domain: 领域名称
            metadata: 元数据字典
        """
        domain_path = os.path.join(self.adapter_base_path, domain)
        if not os.path.exists(domain_path):
            logger.warning(f"适配器 {domain} 目录不存在，无法保存元数据")
            return
        
        metadata_path = os.path.join(domain_path, "metadata.json")
        try:
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            # 更新内存中的元数据
            if domain in self.adapter_metadata:
                self.adapter_metadata[domain]["metadata"] = metadata
            else:
                self._load_metadata()  # 重新加载所有元数据
                
            logger.info(f"已保存适配器 {domain} 的元数据")
        except Exception as e:
            logger.error(f"保存适配器 {domain} 元数据失败: {e}")
    
    def register_new_adapter(self, domain: str, adapter_path: str, metadata: Optional[Dict] = None) -> None:
        """
        注册新的适配器。
        
        Args:
            domain: 领域名称
            adapter_path: 适配器路径
            metadata: 元数据字典
        """
        if not os.path.exists(adapter_path):
            logger.error(f"适配器路径 {adapter_path} 不存在")
            return
        
        # 检查适配器配置
        adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
        if not os.path.exists(adapter_config_path):
            logger.error(f"适配器配置文件不存在: {adapter_config_path}")
            return
        
        # 加载适配器配置
        try:
            with open(adapter_config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 保存元数据
            if metadata:
                metadata_path = os.path.join(adapter_path, "metadata.json")
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            # 更新内存中的元数据
            self.adapter_metadata[domain] = {
                "config": config,
                "metadata": metadata or {},
                "path": adapter_path,
            }
            
            logger.info(f"已注册新适配器: {domain}")
        except Exception as e:
            logger.error(f"注册适配器 {domain} 失败: {e}")
    
    def remove_adapter(self, domain: str) -> bool:
        """
        移除适配器。
        
        Args:
            domain: 领域名称
            
        Returns:
            是否成功移除
        """
        if domain not in self.adapter_metadata:
            logger.warning(f"适配器 {domain} 不存在，无法移除")
            return False
        
        # 从活跃适配器中移除
        if domain in self.active_adapters:
            self.active_adapters.remove(domain)
        
        # 从元数据中移除
        del self.adapter_metadata[domain]
        logger.info(f"已移除适配器: {domain}")
        
        return True 