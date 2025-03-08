"""
训练相关模块，包括增量学习、经验回放和灾难性遗忘缓解机制。
"""

from .trainer import train_model
from .ewc import EWCRegularizer
from .replay_buffer import ReplayBuffer
from .data_processor import prepare_training_data

__all__ = [
    'train_model',
    'EWCRegularizer',
    'ReplayBuffer',
    'prepare_training_data',
]
