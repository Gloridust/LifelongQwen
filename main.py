#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LifelongQwen: 基于 Qwen-2.5-3b 的终生学习 LLM 系统

该系统实现了终生学习能力，使模型能够持续获取新知识，同时保留已学知识。
主要特点:
- 增量学习: 从新数据中学习，无需完全重训练
- 知识保留: 学习新知识时不遗忘旧知识
- 知识整合: 有效结合新旧知识
- 适应性: 适应数据分布变化
- 资源效率: 在计算和内存限制下高效运行
"""

import os
import sys
import time
import json
import threading
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join('logs', f'lifelong_qwen_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'))
    ]
)
logger = logging.getLogger(__name__)

# 确保日志目录存在
os.makedirs('logs', exist_ok=True)

# 确保对话历史目录存在
os.makedirs('data/conversation_history', exist_ok=True)

# 全局变量
last_interaction_time = time.time()
is_training = False
conversation_history = []
IDLE_THRESHOLD = 15  # 闲置15秒后开始训练

def save_conversation(user_input: str, model_response: str) -> None:
    """
    保存对话历史到文件。
    
    Args:
        user_input: 用户输入
        model_response: 模型响应
    """
    global conversation_history
    
    # 添加到内存中的对话历史
    conversation_history.append({
        "input": user_input,
        "output": model_response,
        "timestamp": datetime.now().isoformat()
    })
    
    # 保存到文件
    history_file = os.path.join('data/conversation_history', f'history_{datetime.now().strftime("%Y%m%d")}.jsonl')
    
    try:
        with open(history_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps({
                "input": user_input,
                "output": model_response,
                "timestamp": datetime.now().isoformat()
            }, ensure_ascii=False) + '\n')
    except Exception as e:
        logger.error(f"保存对话历史失败: {e}")

def prepare_training_data() -> str:
    """
    准备训练数据，将对话历史转换为训练数据格式。
    
    Returns:
        训练数据文件路径
    """
    # 获取所有对话历史文件
    history_dir = 'data/conversation_history'
    history_files = [os.path.join(history_dir, f) for f in os.listdir(history_dir) if f.endswith('.jsonl')]
    
    if not history_files:
        logger.warning("没有找到对话历史文件")
        return None
    
    # 创建训练数据目录
    train_data_dir = 'data/train_data'
    os.makedirs(train_data_dir, exist_ok=True)
    
    # 生成训练数据文件名
    train_data_file = os.path.join(train_data_dir, f'conversation_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jsonl')
    
    # 合并所有对话历史
    all_conversations = []
    for file in history_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        all_conversations.append(json.loads(line))
        except Exception as e:
            logger.error(f"读取对话历史文件 {file} 失败: {e}")
    
    # 如果没有对话历史，返回 None
    if not all_conversations:
        logger.warning("没有有效的对话历史数据")
        return None
    
    # 将对话历史转换为训练数据格式并保存
    try:
        with open(train_data_file, 'w', encoding='utf-8') as f:
            for conv in all_conversations:
                f.write(json.dumps({
                    "instruction": "请根据用户输入生成回复",
                    "input": conv["input"],
                    "output": conv["output"]
                }, ensure_ascii=False) + '\n')
        
        logger.info(f"已生成训练数据文件: {train_data_file}")
        return train_data_file
    except Exception as e:
        logger.error(f"生成训练数据失败: {e}")
        return None

def training_thread_function() -> None:
    """训练线程函数，在系统闲置时自动训练。"""
    global is_training
    
    logger.info("开始准备训练数据")
    is_training = True
    
    try:
        # 准备训练数据
        train_data_file = prepare_training_data()
        
        if train_data_file:
            # 创建训练参数
            class Args:
                pass
            
            args = Args()
            args.data = train_data_file
            args.domain = 'conversation'
            args.epochs = 1
            args.batch_size = 4
            args.replay_ratio = 0.3
            args.ewc_lambda = 0.1
            args.output_dir = 'models'
            args.lora_r = 8
            args.lora_alpha = 16
            args.lora_dropout = 0.05
            
            # 导入训练模块
            from lifelong_qwen.training import train_model
            
            logger.info("开始训练历史对话数据")
            train_model(args)
            logger.info("训练完成")
        else:
            logger.warning("没有可用的训练数据，跳过训练")
    except Exception as e:
        logger.error(f"训练过程出错: {e}")
    finally:
        is_training = False

def check_idle_and_train() -> None:
    """检查系统是否闲置，如果闲置则开始训练。"""
    global last_interaction_time, is_training
    
    while True:
        current_time = time.time()
        idle_time = current_time - last_interaction_time
        
        # 如果系统闲置超过阈值且当前没有训练任务在运行
        if idle_time > IDLE_THRESHOLD and not is_training:
            logger.info(f"系统已闲置 {idle_time:.1f} 秒，开始训练")
            
            # 创建并启动训练线程
            training_thread = threading.Thread(target=training_thread_function)
            training_thread.daemon = True
            training_thread.start()
            
            # 等待训练完成
            training_thread.join()
            
            # 更新最后交互时间，避免训练完成后立即再次训练
            last_interaction_time = time.time()
        
        # 休眠一段时间再检查
        time.sleep(1)

def interactive_mode() -> None:
    """交互式模式，用户可以直接与模型对话。"""
    global last_interaction_time
    
    # 导入推理模块
    from lifelong_qwen.inference import run_inference, generate_response
    from lifelong_qwen.models import ModelConfig, load_model_with_adapters, AdapterManager
    
    # 创建模型配置
    model_config = ModelConfig()
    
    # 初始化适配器管理器
    adapter_manager = AdapterManager(model_config.adapter_save_dir)
    
    # 获取可用的适配器
    available_adapters = adapter_manager.get_available_adapters()
    logger.info(f"可用的适配器: {', '.join(available_adapters) if available_adapters else '无'}")
    
    # 默认使用所有可用的适配器
    adapter_paths = [adapter_manager.get_adapter_path(d) for d in available_adapters]
    
    # 加载模型和分词器
    logger.info("加载模型和适配器")
    model, tokenizer = load_model_with_adapters(
        model_config=model_config,
        adapter_paths=adapter_paths,
        load_in_8bit=True,  # 使用8位量化以节省内存
    )
    
    # 启动闲置检测线程
    idle_thread = threading.Thread(target=check_idle_and_train)
    idle_thread.daemon = True
    idle_thread.start()
    
    print("\n欢迎使用 LifelongQwen 终生学习系统！")
    print("输入 'exit' 或 'quit' 退出系统。")
    
    while True:
        try:
            # 获取用户输入
            user_input = input("\n用户: ")
            
            # 更新最后交互时间
            last_interaction_time = time.time()
            
            # 检查是否退出
            if user_input.lower() in ['exit', 'quit']:
                print("再见！")
                break
            
            # 如果正在训练，等待训练完成
            if is_training:
                print("系统正在学习中，请稍等...")
                while is_training:
                    time.sleep(0.5)
            
            # 生成响应
            print("思考中...")
            response = generate_response(
                model=model,
                tokenizer=tokenizer,
                input_text=user_input,
                temperature=0.7,
                max_length=512,
            )
            
            # 打印响应
            print(f"\n助手: {response}")
            
            # 保存对话历史
            save_conversation(user_input, response)
            
        except KeyboardInterrupt:
            print("\n再见！")
            break
        except Exception as e:
            logger.error(f"处理用户输入时出错: {e}")
            print(f"抱歉，出现了一个错误: {e}")

def main():
    """主函数，处理命令行参数并调用相应的功能模块"""
    parser = argparse.ArgumentParser(description='LifelongQwen: 基于 Qwen-2.5-3b 的终生学习 LLM 系统')
    
    # 创建子命令
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 训练命令
    train_parser = subparsers.add_parser('train', help='训练或增量学习模式')
    train_parser.add_argument('--data', type=str, required=True, help='训练数据路径')
    train_parser.add_argument('--domain', type=str, default='general', help='知识领域标识')
    train_parser.add_argument('--epochs', type=int, default=3, help='训练轮次')
    train_parser.add_argument('--batch-size', type=int, default=4, help='批次大小')
    train_parser.add_argument('--replay-ratio', type=float, default=0.3, help='经验回放样本比例')
    train_parser.add_argument('--ewc-lambda', type=float, default=0.1, help='EWC正则化强度')
    train_parser.add_argument('--output-dir', type=str, default='models', help='模型输出目录')
    train_parser.add_argument('--lora-r', type=int, default=8, help='LoRA秩')
    train_parser.add_argument('--lora-alpha', type=int, default=16, help='LoRA alpha参数')
    train_parser.add_argument('--lora-dropout', type=float, default=0.05, help='LoRA dropout率')
    
    # 推理命令
    infer_parser = subparsers.add_parser('infer', help='推理模式')
    infer_parser.add_argument('--input', type=str, help='输入文本或文件路径')
    infer_parser.add_argument('--domains', type=str, nargs='+', default=['general'], help='使用的知识领域')
    infer_parser.add_argument('--use-rag', action='store_true', help='是否使用RAG增强')
    infer_parser.add_argument('--temperature', type=float, default=0.7, help='生成温度')
    infer_parser.add_argument('--max-length', type=int, default=512, help='最大生成长度')
    
    # 评估命令
    eval_parser = subparsers.add_parser('evaluate', help='评估模式')
    eval_parser.add_argument('--test-data', type=str, required=True, help='测试数据路径')
    eval_parser.add_argument('--domains', type=str, nargs='+', default=['general'], help='评估的知识领域')
    eval_parser.add_argument('--metrics', type=str, nargs='+', default=['accuracy', 'forgetting'], help='评估指标')
    eval_parser.add_argument('--output', type=str, default='evaluation_results.json', help='评估结果输出路径')
    
    # 知识管理命令
    knowledge_parser = subparsers.add_parser('knowledge', help='知识管理模式')
    knowledge_parser.add_argument('--action', choices=['add', 'delete', 'update', 'list'], required=True, help='知识管理操作')
    knowledge_parser.add_argument('--domain', type=str, help='知识领域')
    knowledge_parser.add_argument('--data', type=str, help='知识数据路径')
    
    # 解析参数
    args = parser.parse_args()
    
    # 如果没有指定命令，进入交互式模式
    if not args.command:
        interactive_mode()
        return
    
    # 导入相关模块
    if args.command == 'train':
        from lifelong_qwen.training import train_model
        train_model(args)
    elif args.command == 'infer':
        from lifelong_qwen.inference import run_inference
        run_inference(args)
    elif args.command == 'evaluate':
        from lifelong_qwen.evaluation import evaluate_model
        evaluate_model(args)
    elif args.command == 'knowledge':
        from lifelong_qwen.knowledge_management import manage_knowledge
        manage_knowledge(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.exception(f"程序执行出错: {e}")
        sys.exit(1)
