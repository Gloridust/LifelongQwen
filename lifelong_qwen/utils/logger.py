"""
日志工具模块，实现日志配置功能。
"""

import os
import sys
import logging
from datetime import datetime
from typing import Optional

def setup_logger(
    name: Optional[str] = None,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console: bool = True,
) -> logging.Logger:
    """
    设置日志记录器。
    
    Args:
        name: 日志记录器名称
        level: 日志级别
        log_file: 日志文件路径
        console: 是否输出到控制台
        
    Returns:
        日志记录器
    """
    # 获取日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 清除现有的处理器
    logger.handlers = []
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 添加文件处理器
    if log_file:
        # 确保日志目录存在
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # 添加控制台处理器
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger

def get_default_log_file() -> str:
    """
    获取默认日志文件路径。
    
    Returns:
        默认日志文件路径
    """
    log_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(log_dir, f"lifelong_qwen_{timestamp}.log") 