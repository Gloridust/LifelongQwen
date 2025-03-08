"""
推理相关模块，包括模型推理和知识路由。
"""

from .inference_engine import run_inference, generate_response
from .knowledge_router import KnowledgeRouter

__all__ = [
    'run_inference',
    'generate_response',
    'KnowledgeRouter',
]
