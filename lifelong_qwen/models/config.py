"""
模型配置模块，定义模型配置类。
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union

@dataclass
class ModelConfig:
    """模型配置类，存储模型相关的配置信息。"""
    
    # 基础模型配置
    base_model_name_or_path: str = "Qwen/Qwen-2.5-3B"
    model_revision: Optional[str] = None
    
    # LoRA 配置
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    
    # 训练配置
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    
    # 生成配置
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    
    # 终生学习配置
    ewc_lambda: float = 0.1  # EWC 正则化强度
    replay_ratio: float = 0.3  # 经验回放样本比例
    
    # 系统路径
    model_cache_dir: str = field(default_factory=lambda: os.path.join(os.getcwd(), "models", "cache"))
    adapter_save_dir: str = field(default_factory=lambda: os.path.join(os.getcwd(), "models", "adapters"))
    
    def __post_init__(self):
        """初始化后的处理，确保目录存在。"""
        os.makedirs(self.model_cache_dir, exist_ok=True)
        os.makedirs(self.adapter_save_dir, exist_ok=True)
    
    def get_adapter_path(self, domain: str) -> str:
        """获取特定领域适配器的保存路径。"""
        return os.path.join(self.adapter_save_dir, domain)
    
    def to_dict(self) -> Dict:
        """将配置转换为字典。"""
        return {k: v for k, v in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> "ModelConfig":
        """从字典创建配置对象。"""
        return cls(**config_dict)
    
    @classmethod
    def from_pretrained(cls, config_path: str) -> "ModelConfig":
        """从文件加载配置。"""
        import json
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def save_pretrained(self, config_path: str) -> None:
        """保存配置到文件。"""
        import json
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2) 