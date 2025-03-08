"""
知识管理器模块，实现知识管理功能。
"""

import os
import json
import logging
import shutil
from typing import Dict, List, Optional, Union, Any

from ..models import AdapterManager, ModelConfig
from ..inference.knowledge_router import KnowledgeRouter

logger = logging.getLogger(__name__)

def manage_knowledge(args):
    """
    管理知识的主函数。
    
    Args:
        args: 命令行参数
    """
    logger.info(f"开始知识管理模式: {args.action}")
    
    # 根据操作类型调用相应的函数
    if args.action == "add":
        if not args.domain or not args.data:
            logger.error("添加知识需要指定领域和数据路径")
            return False
        
        return add_knowledge(args.domain, args.data)
    
    elif args.action == "delete":
        if not args.domain:
            logger.error("删除知识需要指定领域")
            return False
        
        return delete_knowledge(args.domain)
    
    elif args.action == "update":
        if not args.domain or not args.data:
            logger.error("更新知识需要指定领域和数据路径")
            return False
        
        return update_knowledge(args.domain, args.data)
    
    elif args.action == "list":
        return list_knowledge(args.domain)
    
    else:
        logger.error(f"未知的知识管理操作: {args.action}")
        return False

def add_knowledge(domain: str, data_path: str) -> bool:
    """
    添加知识。
    
    Args:
        domain: 知识领域
        data_path: 知识数据路径
        
    Returns:
        是否成功
    """
    logger.info(f"添加知识: 领域={domain}, 数据路径={data_path}")
    
    # 检查数据路径是否存在
    if not os.path.exists(data_path):
        logger.error(f"数据路径不存在: {data_path}")
        return False
    
    # 创建知识目录
    knowledge_dir = os.path.join("data", "knowledge_base", domain)
    os.makedirs(knowledge_dir, exist_ok=True)
    
    # 如果数据路径是目录，复制整个目录
    if os.path.isdir(data_path):
        # 复制目录中的所有文件
        for filename in os.listdir(data_path):
            src_file = os.path.join(data_path, filename)
            dst_file = os.path.join(knowledge_dir, filename)
            
            if os.path.isfile(src_file):
                shutil.copy2(src_file, dst_file)
                logger.info(f"复制文件: {src_file} -> {dst_file}")
    else:
        # 复制单个文件
        filename = os.path.basename(data_path)
        dst_file = os.path.join(knowledge_dir, filename)
        shutil.copy2(data_path, dst_file)
        logger.info(f"复制文件: {data_path} -> {dst_file}")
    
    # 初始化知识路由器并添加文档
    knowledge_router = KnowledgeRouter(
        knowledge_base_path=os.path.join("data", "knowledge_base")
    )
    
    # 添加文档到向量数据库
    knowledge_router.add_documents(knowledge_dir)
    
    logger.info(f"成功添加知识: 领域={domain}")
    return True

def delete_knowledge(domain: str) -> bool:
    """
    删除知识。
    
    Args:
        domain: 知识领域
        
    Returns:
        是否成功
    """
    logger.info(f"删除知识: 领域={domain}")
    
    # 检查知识目录是否存在
    knowledge_dir = os.path.join("data", "knowledge_base", domain)
    if not os.path.exists(knowledge_dir):
        logger.warning(f"知识目录不存在: {knowledge_dir}")
        return False
    
    # 删除知识目录
    shutil.rmtree(knowledge_dir)
    logger.info(f"已删除知识目录: {knowledge_dir}")
    
    # 检查是否需要重建向量数据库
    vector_db_dir = os.path.join("data", "knowledge_base", "vector_db")
    if os.path.exists(vector_db_dir):
        logger.info("删除向量数据库，将在下次添加知识时重建")
        shutil.rmtree(vector_db_dir)
    
    logger.info(f"成功删除知识: 领域={domain}")
    return True

def update_knowledge(domain: str, data_path: str) -> bool:
    """
    更新知识。
    
    Args:
        domain: 知识领域
        data_path: 知识数据路径
        
    Returns:
        是否成功
    """
    logger.info(f"更新知识: 领域={domain}, 数据路径={data_path}")
    
    # 先删除旧知识
    delete_knowledge(domain)
    
    # 再添加新知识
    return add_knowledge(domain, data_path)

def list_knowledge(domain: Optional[str] = None) -> bool:
    """
    列出知识。
    
    Args:
        domain: 知识领域，如果为 None，则列出所有领域
        
    Returns:
        是否成功
    """
    knowledge_base_dir = os.path.join("data", "knowledge_base")
    
    # 确保知识库目录存在
    if not os.path.exists(knowledge_base_dir):
        logger.warning(f"知识库目录不存在: {knowledge_base_dir}")
        return False
    
    # 如果指定了领域，只列出该领域的知识
    if domain:
        domain_dir = os.path.join(knowledge_base_dir, domain)
        if not os.path.exists(domain_dir):
            logger.warning(f"领域目录不存在: {domain_dir}")
            return False
        
        print(f"\n领域: {domain}")
        for item in os.listdir(domain_dir):
            item_path = os.path.join(domain_dir, item)
            if os.path.isfile(item_path):
                size = os.path.getsize(item_path)
                print(f"  - {item} ({size} 字节)")
    else:
        # 列出所有领域
        domains = [d for d in os.listdir(knowledge_base_dir) if os.path.isdir(os.path.join(knowledge_base_dir, d)) and d != "vector_db"]
        
        if not domains:
            print("\n知识库为空")
            return True
        
        print("\n知识库领域:")
        for domain in domains:
            domain_dir = os.path.join(knowledge_base_dir, domain)
            files = [f for f in os.listdir(domain_dir) if os.path.isfile(os.path.join(domain_dir, f))]
            print(f"  - {domain} ({len(files)} 个文件)")
    
    # 列出适配器信息
    model_config = ModelConfig()
    adapter_manager = AdapterManager(model_config.adapter_save_dir)
    adapters = adapter_manager.get_available_adapters()
    
    print("\n可用适配器:")
    if adapters:
        for adapter in adapters:
            metadata = adapter_manager.get_adapter_metadata(adapter)
            created_at = metadata.get("created_at", "未知") if metadata else "未知"
            print(f"  - {adapter} (创建时间: {created_at})")
    else:
        print("  无")
    
    return True 