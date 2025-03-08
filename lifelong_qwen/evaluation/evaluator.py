"""
评估器模块，实现模型评估功能。
"""

import os
import json
import logging
from typing import Dict, List, Optional, Union, Any

import torch
import numpy as np
from tqdm import tqdm
from datasets import Dataset

from ..models import ModelConfig, load_model_with_adapters, AdapterManager
from ..training.data_processor import prepare_evaluation_data
from .metrics import calculate_metrics, calculate_forgetting

logger = logging.getLogger(__name__)

def evaluate_model(args):
    """
    评估模型的主函数。
    
    Args:
        args: 命令行参数
    """
    logger.info(f"开始评估模式")
    
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
    
    # 准备评估数据
    logger.info(f"准备评估数据: {args.test_data}")
    eval_dataset = prepare_evaluation_data(args.test_data, tokenizer_name_or_path=tokenizer.name_or_path)
    
    # 执行评估
    logger.info("开始评估")
    results = run_evaluation(
        model=model,
        tokenizer=tokenizer,
        eval_dataset=eval_dataset,
        metrics=args.metrics,
        domains=valid_domains,
    )
    
    # 保存评估结果
    logger.info(f"保存评估结果到 {args.output}")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 打印评估结果
    print("\n评估结果:")
    for metric, value in results["metrics"].items():
        print(f"{metric}: {value}")
    
    if "forgetting" in results:
        print("\n遗忘评估:")
        for domain, forgetting in results["forgetting"].items():
            print(f"{domain}: {forgetting}")
    
    return results

def run_evaluation(
    model,
    tokenizer,
    eval_dataset: Dataset,
    metrics: List[str] = ["accuracy"],
    domains: List[str] = ["general"],
) -> Dict[str, Any]:
    """
    运行评估。
    
    Args:
        model: 模型
        tokenizer: 分词器
        eval_dataset: 评估数据集
        metrics: 评估指标
        domains: 评估的领域
        
    Returns:
        评估结果
    """
    # 设置模型为评估模式
    model.eval()
    
    # 准备结果字典
    results = {
        "metrics": {},
        "domains": domains,
    }
    
    # 计算每个样本的损失和预测
    losses = []
    predictions = []
    references = []
    
    # 批量处理评估数据
    batch_size = 8
    for i in tqdm(range(0, len(eval_dataset), batch_size), desc="评估进度"):
        batch = eval_dataset[i:min(i + batch_size, len(eval_dataset))]
        
        # 将批次数据移动到模型所在的设备
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        labels = batch["labels"].to(model.device)
        
        # 前向传播
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
        
        # 收集损失和预测
        losses.append(outputs.loss.item())
        
        # 对于生成任务，可以使用模型生成输出并与参考进行比较
        # 这里简化为使用下一个 token 的预测作为评估
        logits = outputs.logits
        pred = torch.argmax(logits, dim=-1)
        
        # 只考虑有标签的位置（非填充位置）
        mask = labels != -100
        pred_filtered = pred[mask]
        labels_filtered = labels[mask]
        
        predictions.extend(pred_filtered.cpu().numpy())
        references.extend(labels_filtered.cpu().numpy())
    
    # 计算评估指标
    metric_results = calculate_metrics(
        predictions=predictions,
        references=references,
        metrics=metrics,
    )
    
    # 添加平均损失
    metric_results["loss"] = np.mean(losses)
    
    # 将评估指标添加到结果字典
    results["metrics"] = metric_results
    
    # 如果需要计算遗忘，加载基线模型并比较
    if "forgetting" in metrics:
        logger.info("计算遗忘指标")
        
        # 加载基线模型（基础模型，无适配器）
        baseline_model, _ = load_model_with_adapters(
            model_config=ModelConfig(),
            adapter_paths=[],
            load_in_8bit=True,
        )
        
        # 计算基线模型的损失
        baseline_losses = []
        for i in tqdm(range(0, len(eval_dataset), batch_size), desc="基线评估"):
            batch = eval_dataset[i:min(i + batch_size, len(eval_dataset))]
            
            # 将批次数据移动到模型所在的设备
            input_ids = batch["input_ids"].to(baseline_model.device)
            attention_mask = batch["attention_mask"].to(baseline_model.device)
            labels = batch["labels"].to(baseline_model.device)
            
            # 前向传播
            with torch.no_grad():
                outputs = baseline_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
            
            # 收集损失
            baseline_losses.append(outputs.loss.item())
        
        # 计算遗忘指标
        forgetting_results = calculate_forgetting(
            current_loss=np.mean(losses),
            baseline_loss=np.mean(baseline_losses),
            domains=domains,
        )
        
        # 将遗忘指标添加到结果字典
        results["forgetting"] = forgetting_results
    
    return results 