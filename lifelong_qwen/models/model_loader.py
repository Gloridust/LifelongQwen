"""
模型加载器模块，负责加载基础模型和适配器。
"""

import os
import logging
from typing import List, Dict, Optional, Union, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig

from .config import ModelConfig

logger = logging.getLogger(__name__)

def load_base_model(
    model_config: ModelConfig,
    device_map: str = "auto",
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    加载基础 Qwen 模型。
    
    Args:
        model_config: 模型配置对象
        device_map: 设备映射策略
        load_in_8bit: 是否以8位精度加载模型
        load_in_4bit: 是否以4位精度加载模型
        
    Returns:
        加载的模型和分词器
    """
    logger.info(f"正在加载基础模型: {model_config.base_model_name_or_path}")
    
    quantization_config = None
    if load_in_4bit:
        logger.info("使用4位量化加载模型")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif load_in_8bit:
        logger.info("使用8位量化加载模型")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_config.base_model_name_or_path,
        trust_remote_code=True,
        device_map=device_map,
        quantization_config=quantization_config,
    )
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.base_model_name_or_path,
        trust_remote_code=True,
    )
    
    logger.info(f"基础模型加载完成")
    return model, tokenizer

def load_model_with_adapters(
    model_config: ModelConfig,
    adapter_paths: List[str],
    device_map: str = "auto",
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    加载基础模型并应用多个适配器。
    
    Args:
        model_config: 模型配置对象
        adapter_paths: 适配器路径列表
        device_map: 设备映射策略
        load_in_8bit: 是否以8位精度加载模型
        load_in_4bit: 是否以4位精度加载模型
        
    Returns:
        加载的模型和分词器
    """
    # 加载基础模型
    model, tokenizer = load_base_model(
        model_config=model_config,
        device_map=device_map,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
    )
    
    # 加载适配器
    if adapter_paths:
        logger.info(f"正在加载 {len(adapter_paths)} 个适配器")
        for i, adapter_path in enumerate(adapter_paths):
            adapter_name = os.path.basename(adapter_path)
            logger.info(f"加载适配器 {i+1}/{len(adapter_paths)}: {adapter_name}")
            
            # 使用 PeftModel 加载适配器
            model = PeftModel.from_pretrained(
                model,
                adapter_path,
                adapter_name=adapter_name,
            )
    
    return model, tokenizer 