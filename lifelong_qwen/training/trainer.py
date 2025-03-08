"""
训练器模块，实现模型训练和增量学习功能。
"""

import os
import logging
import time
from typing import Dict, List, Optional, Union, Tuple, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftModel,
)
from tqdm import tqdm

from ..models import ModelConfig, load_base_model
from .ewc import EWCRegularizer
from .replay_buffer import ReplayBuffer
from .data_processor import prepare_training_data

logger = logging.getLogger(__name__)

def train_model(args):
    """
    训练模型的主函数，支持增量学习。
    
    Args:
        args: 命令行参数
    """
    logger.info(f"开始训练模式，领域: {args.domain}")
    
    # 创建模型配置
    model_config = ModelConfig()
    model_config.lora_r = args.lora_r
    model_config.lora_alpha = args.lora_alpha
    model_config.lora_dropout = args.lora_dropout
    model_config.ewc_lambda = args.ewc_lambda
    
    # 准备训练数据
    logger.info(f"准备训练数据: {args.data}")
    train_dataset = prepare_training_data(args.data)
    
    # 加载基础模型
    logger.info("加载基础模型")
    model, tokenizer = load_base_model(
        model_config=model_config,
        load_in_8bit=True,  # 使用8位量化以节省内存
    )
    
    # 配置 LoRA
    logger.info("配置 LoRA 适配器")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=model_config.lora_r,
        lora_alpha=model_config.lora_alpha,
        lora_dropout=model_config.lora_dropout,
        target_modules=model_config.lora_target_modules,
    )
    
    # 应用 LoRA 配置
    model = get_peft_model(model, peft_config)
    
    # 初始化经验回放缓冲区
    replay_buffer = ReplayBuffer(
        capacity=10000,  # 可根据需要调整
        sampling_strategy="uniform",
    )
    
    # 检查是否有之前的训练数据需要回放
    replay_data_path = os.path.join("data", "replay", f"{args.domain}_replay.json")
    if os.path.exists(replay_data_path):
        logger.info(f"加载经验回放数据: {replay_data_path}")
        replay_buffer.load(replay_data_path)
    
    # 初始化 EWC 正则化器
    ewc_regularizer = None
    ewc_checkpoint_path = os.path.join("models", "checkpoints", f"{args.domain}_ewc.pt")
    if os.path.exists(ewc_checkpoint_path) and args.ewc_lambda > 0:
        logger.info(f"加载 EWC 参数重要性: {ewc_checkpoint_path}")
        ewc_regularizer = EWCRegularizer(model)
        ewc_regularizer.load(ewc_checkpoint_path)
    
    # 准备训练参数
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, args.domain),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=model_config.learning_rate,
        weight_decay=model_config.weight_decay,
        warmup_ratio=model_config.warmup_ratio,
        logging_dir=os.path.join("logs", "tensorboard"),
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        fp16=True,
        report_to="tensorboard",
    )
    
    # 创建数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # 如果有经验回放数据，合并到训练数据中
    if replay_buffer.size() > 0 and args.replay_ratio > 0:
        replay_samples = replay_buffer.sample(
            int(len(train_dataset) * args.replay_ratio)
        )
        logger.info(f"合并 {len(replay_samples)} 个经验回放样本到训练数据中")
        # 注意：这里假设 train_dataset 支持 extend 方法
        # 实际实现可能需要根据数据集类型进行调整
        train_dataset.extend(replay_samples)
    
    # 创建自定义训练器
    class LifelongTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            """
            重写计算损失函数，添加 EWC 正则化。
            """
            outputs = model(**inputs)
            loss = outputs.loss
            
            # 应用 EWC 正则化
            if ewc_regularizer is not None:
                ewc_loss = ewc_regularizer.penalty(model)
                loss = loss + args.ewc_lambda * ewc_loss
                self.log({"ewc_loss": ewc_loss.item()})
            
            return (loss, outputs) if return_outputs else loss
    
    # 创建训练器
    trainer = LifelongTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # 开始训练
    logger.info("开始训练")
    trainer.train()
    
    # 保存模型
    output_dir = os.path.join(args.output_dir, args.domain)
    logger.info(f"保存模型到 {output_dir}")
    model.save_pretrained(output_dir)
    
    # 更新 EWC 参数重要性
    if args.ewc_lambda > 0:
        logger.info("计算参数重要性")
        if ewc_regularizer is None:
            ewc_regularizer = EWCRegularizer(model)
        
        # 使用训练数据计算参数重要性
        ewc_regularizer.update_importance(model, train_dataset, trainer)
        
        # 保存 EWC 参数重要性
        os.makedirs(os.path.dirname(ewc_checkpoint_path), exist_ok=True)
        ewc_regularizer.save(ewc_checkpoint_path)
    
    # 更新经验回放缓冲区
    logger.info("更新经验回放缓冲区")
    replay_buffer.add_samples(train_dataset.select(range(min(1000, len(train_dataset)))))
    
    # 保存经验回放缓冲区
    os.makedirs(os.path.dirname(replay_data_path), exist_ok=True)
    replay_buffer.save(replay_data_path)
    
    logger.info(f"训练完成，模型已保存到 {output_dir}")
    
    # 添加适配器元数据
    metadata_path = os.path.join(output_dir, "metadata.json")
    import json
    metadata = {
        "domain": args.domain,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "training_samples": len(train_dataset),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "replay_ratio": args.replay_ratio,
        "ewc_lambda": args.ewc_lambda,
    }
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    return output_dir 