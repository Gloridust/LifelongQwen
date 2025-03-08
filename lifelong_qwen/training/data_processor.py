"""
数据处理器模块，实现数据加载和预处理功能。
"""

import os
import json
import logging
from typing import Dict, List, Optional, Union, Any

import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

def prepare_training_data(data_path: str, tokenizer_name_or_path: str = "Qwen/Qwen-2.5-3B", max_length: int = 1024) -> Dataset:
    """
    准备训练数据。
    
    Args:
        data_path: 数据路径，支持 jsonl, json, csv, txt 格式
        tokenizer_name_or_path: 分词器名称或路径
        max_length: 最大序列长度
        
    Returns:
        处理后的数据集
    """
    logger.info(f"准备训练数据: {data_path}")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)
    
    # 根据文件扩展名加载数据
    file_ext = os.path.splitext(data_path)[1].lower()
    
    if file_ext == '.jsonl':
        # JSONL 格式
        dataset = load_dataset('json', data_files=data_path)['train']
    elif file_ext == '.json':
        # JSON 格式
        dataset = load_dataset('json', data_files=data_path)['train']
    elif file_ext == '.csv':
        # CSV 格式
        dataset = load_dataset('csv', data_files=data_path)['train']
    elif file_ext == '.txt':
        # 文本格式，每行一个样本
        with open(data_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        dataset = Dataset.from_dict({
            'text': lines
        })
    else:
        # 尝试使用 datasets 库自动加载
        try:
            dataset = load_dataset(data_path)['train']
        except Exception as e:
            logger.error(f"无法加载数据: {e}")
            raise ValueError(f"不支持的数据格式: {file_ext}")
    
    logger.info(f"原始数据集大小: {len(dataset)}")
    
    # 检查数据格式
    if 'text' in dataset.column_names:
        # 单文本格式
        format_type = 'text'
    elif all(col in dataset.column_names for col in ['input', 'output']):
        # 输入-输出格式
        format_type = 'input_output'
    elif all(col in dataset.column_names for col in ['instruction', 'input', 'output']):
        # 指令-输入-输出格式
        format_type = 'instruction_input_output'
    elif all(col in dataset.column_names for col in ['instruction', 'output']):
        # 指令-输出格式
        format_type = 'instruction_output'
    else:
        # 未知格式，尝试查找可能的文本列
        text_columns = [col for col in dataset.column_names if 'text' in col.lower()]
        if text_columns:
            format_type = 'text'
            dataset = dataset.rename_column(text_columns[0], 'text')
        else:
            logger.error(f"无法识别数据格式，列名: {dataset.column_names}")
            raise ValueError("不支持的数据格式")
    
    logger.info(f"识别的数据格式: {format_type}")
    
    # 根据数据格式处理
    def format_data(examples):
        if format_type == 'text':
            texts = examples['text']
        elif format_type == 'input_output':
            texts = [f"输入: {inp}\n输出: {out}" for inp, out in zip(examples['input'], examples['output'])]
        elif format_type == 'instruction_input_output':
            texts = [f"指令: {ins}\n输入: {inp}\n输出: {out}" 
                    for ins, inp, out in zip(examples['instruction'], examples['input'], examples['output'])]
        elif format_type == 'instruction_output':
            texts = [f"指令: {ins}\n输出: {out}" for ins, out in zip(examples['instruction'], examples['output'])]
        else:
            raise ValueError(f"不支持的数据格式: {format_type}")
        
        # 分词
        tokenized = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        
        # 准备输入和标签
        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask
        labels = input_ids.clone()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    
    # 应用数据处理
    processed_dataset = dataset.map(
        format_data,
        batched=True,
        remove_columns=dataset.column_names,
    )
    
    logger.info(f"处理后的数据集大小: {len(processed_dataset)}")
    
    return processed_dataset

def prepare_evaluation_data(data_path: str, tokenizer_name_or_path: str = "Qwen/Qwen-2.5-3B", max_length: int = 1024) -> Dataset:
    """
    准备评估数据。
    
    Args:
        data_path: 数据路径
        tokenizer_name_or_path: 分词器名称或路径
        max_length: 最大序列长度
        
    Returns:
        处理后的数据集
    """
    # 评估数据处理与训练数据类似，但可能需要不同的处理逻辑
    # 这里简单复用训练数据处理函数
    return prepare_training_data(data_path, tokenizer_name_or_path, max_length)

def prepare_inference_data(text: str, tokenizer_name_or_path: str = "Qwen/Qwen-2.5-3B", max_length: int = 1024) -> Dict[str, torch.Tensor]:
    """
    准备推理数据。
    
    Args:
        text: 输入文本
        tokenizer_name_or_path: 分词器名称或路径
        max_length: 最大序列长度
        
    Returns:
        处理后的数据
    """
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)
    
    # 分词
    tokenized = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    
    return {
        "input_ids": tokenized.input_ids,
        "attention_mask": tokenized.attention_mask,
    } 