"""
推理引擎模块，实现模型推理功能。
"""

import os
import logging
from typing import Dict, List, Optional, Union, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..models import ModelConfig, load_model_with_adapters, AdapterManager
from .knowledge_router import KnowledgeRouter
from ..training.data_processor import prepare_inference_data

logger = logging.getLogger(__name__)

def run_inference(args):
    """
    运行推理的主函数。
    
    Args:
        args: 命令行参数
    """
    logger.info(f"开始推理模式")
    
    # 创建模型配置
    model_config = ModelConfig()
    
    # 初始化适配器管理器
    adapter_manager = AdapterManager(model_config.adapter_save_dir)
    
    # 获取可用的适配器
    available_adapters = adapter_manager.get_available_adapters()
    logger.info(f"可用的适配器: {', '.join(available_adapters)}")
    
    # 检查指定的领域是否可用
    valid_domains = [d for d in args.domains if d in available_adapters]
    if not valid_domains:
        logger.warning(f"指定的领域 {args.domains} 不可用，将使用基础模型")
    
    # 获取适配器路径
    adapter_paths = [adapter_manager.get_adapter_path(d) for d in valid_domains]
    
    # 加载模型和分词器
    logger.info(f"加载模型和适配器: {', '.join(valid_domains)}")
    model, tokenizer = load_model_with_adapters(
        model_config=model_config,
        adapter_paths=adapter_paths,
        load_in_8bit=True,  # 使用8位量化以节省内存
    )
    
    # 初始化知识路由器
    knowledge_router = None
    if args.use_rag:
        logger.info("初始化知识路由器")
        knowledge_router = KnowledgeRouter()
    
    # 处理输入
    input_text = args.input
    if os.path.exists(input_text):
        # 如果输入是文件路径，读取文件内容
        with open(input_text, 'r', encoding='utf-8') as f:
            input_text = f.read().strip()
    
    # 生成响应
    logger.info("生成响应")
    response = generate_response(
        model=model,
        tokenizer=tokenizer,
        input_text=input_text,
        knowledge_router=knowledge_router,
        temperature=args.temperature,
        max_length=args.max_length,
    )
    
    # 输出响应
    print(f"\n输入: {input_text}\n")
    print(f"响应: {response}\n")
    
    return response

def generate_response(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_text: str,
    knowledge_router: Optional[KnowledgeRouter] = None,
    temperature: float = 0.7,
    max_length: int = 512,
) -> str:
    """
    生成响应。
    
    Args:
        model: 模型
        tokenizer: 分词器
        input_text: 输入文本
        knowledge_router: 知识路由器
        temperature: 生成温度
        max_length: 最大生成长度
        
    Returns:
        生成的响应
    """
    # 如果启用了 RAG，使用知识路由器增强输入
    if knowledge_router is not None:
        logger.info("使用 RAG 增强输入")
        augmented_input = knowledge_router.augment_query(input_text)
        logger.debug(f"增强后的输入: {augmented_input}")
        input_text = augmented_input
    
    # 准备输入
    inputs = prepare_inference_data(input_text, tokenizer_name_or_path=tokenizer.name_or_path)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    
    # 生成参数
    gen_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_length,
        "temperature": temperature,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    
    # 生成响应
    with torch.no_grad():
        output_ids = model.generate(**gen_kwargs)
    
    # 解码响应
    response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    
    return response 