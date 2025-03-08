"""
配置工具模块，实现配置加载和保存功能。
"""

import os
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def load_config(config_path: str, default_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    加载配置文件。
    
    Args:
        config_path: 配置文件路径
        default_config: 默认配置
        
    Returns:
        配置字典
    """
    # 如果配置文件不存在，使用默认配置
    if not os.path.exists(config_path):
        logger.warning(f"配置文件不存在: {config_path}，使用默认配置")
        
        if default_config:
            # 保存默认配置
            save_config(config_path, default_config)
            return default_config.copy()
        else:
            return {}
    
    # 加载配置文件
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        logger.info(f"已加载配置文件: {config_path}")
        return config
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        
        if default_config:
            logger.warning("使用默认配置")
            return default_config.copy()
        else:
            return {}

def save_config(config_path: str, config: Dict[str, Any]) -> bool:
    """
    保存配置文件。
    
    Args:
        config_path: 配置文件路径
        config: 配置字典
        
    Returns:
        是否成功
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    # 保存配置文件
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        logger.info(f"已保存配置文件: {config_path}")
        return True
    except Exception as e:
        logger.error(f"保存配置文件失败: {e}")
        return False

def update_config(config_path: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    更新配置文件。
    
    Args:
        config_path: 配置文件路径
        updates: 更新的配置
        
    Returns:
        更新后的配置字典
    """
    # 加载现有配置
    config = load_config(config_path, {})
    
    # 更新配置
    config.update(updates)
    
    # 保存更新后的配置
    save_config(config_path, config)
    
    return config 