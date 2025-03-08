"""
知识路由器模块，实现知识检索和增强功能。
"""

import os
import logging
from typing import Dict, List, Optional, Union, Any

import torch
import numpy as np
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, DirectoryLoader

logger = logging.getLogger(__name__)

class KnowledgeRouter:
    """
    知识路由器，负责知识检索和增强。
    
    使用向量数据库存储和检索知识，支持 RAG (Retrieval-Augmented Generation)。
    """
    
    def __init__(
        self,
        knowledge_base_path: str = "data/knowledge_base",
        embedding_model_name: str = "BAAI/bge-small-zh",
        top_k: int = 3,
    ):
        """
        初始化知识路由器。
        
        Args:
            knowledge_base_path: 知识库路径
            embedding_model_name: 嵌入模型名称
            top_k: 检索的文档数量
        """
        self.knowledge_base_path = knowledge_base_path
        self.embedding_model_name = embedding_model_name
        self.top_k = top_k
        
        # 确保知识库目录存在
        os.makedirs(knowledge_base_path, exist_ok=True)
        
        # 初始化嵌入模型
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        )
        
        # 初始化向量数据库
        self.vector_db = None
        self._init_vector_db()
    
    def _init_vector_db(self) -> None:
        """初始化向量数据库。"""
        # 检查持久化的向量数据库是否存在
        persist_directory = os.path.join(self.knowledge_base_path, "vector_db")
        
        if os.path.exists(persist_directory) and os.listdir(persist_directory):
            logger.info(f"加载现有向量数据库: {persist_directory}")
            self.vector_db = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings,
            )
        else:
            logger.info("向量数据库不存在，将在首次添加文档时创建")
            self.vector_db = None
    
    def add_documents(self, documents_path: str) -> None:
        """
        添加文档到知识库。
        
        Args:
            documents_path: 文档路径，可以是单个文件或目录
        """
        logger.info(f"添加文档到知识库: {documents_path}")
        
        # 加载文档
        if os.path.isdir(documents_path):
            # 加载目录中的所有文本文件
            loader = DirectoryLoader(
                documents_path,
                glob="**/*.txt",
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"},
            )
        else:
            # 加载单个文件
            loader = TextLoader(documents_path, encoding="utf-8")
        
        documents = loader.load()
        logger.info(f"加载了 {len(documents)} 个文档")
        
        # 分割文档
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
        logger.info(f"分割为 {len(chunks)} 个文本块")
        
        # 创建或更新向量数据库
        persist_directory = os.path.join(self.knowledge_base_path, "vector_db")
        
        if self.vector_db is None:
            logger.info(f"创建新的向量数据库: {persist_directory}")
            self.vector_db = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=persist_directory,
            )
            self.vector_db.persist()
        else:
            logger.info("更新现有向量数据库")
            self.vector_db.add_documents(chunks)
            self.vector_db.persist()
    
    def search_knowledge(self, query: str) -> List[str]:
        """
        搜索知识库。
        
        Args:
            query: 查询文本
            
        Returns:
            相关文档列表
        """
        if self.vector_db is None:
            logger.warning("向量数据库为空，无法搜索知识")
            return []
        
        # 搜索相关文档
        docs = self.vector_db.similarity_search(query, k=self.top_k)
        
        # 提取文档内容
        results = [doc.page_content for doc in docs]
        
        return results
    
    def augment_query(self, query: str) -> str:
        """
        增强查询，添加相关知识。
        
        Args:
            query: 原始查询
            
        Returns:
            增强后的查询
        """
        # 搜索相关知识
        knowledge = self.search_knowledge(query)
        
        if not knowledge:
            logger.info("未找到相关知识，使用原始查询")
            return query
        
        # 构建增强查询
        augmented_query = f"""请基于以下信息回答问题：

问题：{query}

相关信息：
{chr(10).join([f"- {k}" for k in knowledge])}

请根据上述信息提供准确、全面的回答。如果相关信息不足以回答问题，请说明并尽可能给出合理的回答。
"""
        
        return augmented_query 