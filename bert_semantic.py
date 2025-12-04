#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BERT语义理解模块 - 用于意图识别和语义相似度计算
"""

import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class BERTSemanticEngine:
    """基于BERT的语义理解引擎"""
    
    def __init__(self, model_name: str = 'bert-base-chinese'):
        """
        初始化BERT模型
        
        Args:
            model_name: BERT模型名称
        """
        logger.info(f"正在加载BERT模型: {model_name}")
        
        try:
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.model = BertModel.from_pretrained(model_name)
            self.model.eval()  # 设置为评估模式
            
            # 检查是否有GPU可用
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            
            logger.info(f"BERT模型加载成功，使用设备: {self.device}")
            
            # 预定义意图模板及其embeddings
            self.intent_templates = {
                'symptom': [
                    '有什么症状', '症状表现', '临床表现', '会出现什么',
                    '什么感觉', '有哪些不适', '身体会怎样'
                ],
                'treatment': [
                    '怎么治疗', '如何治', '治疗方法', '怎么办',
                    '怎样医治', '如何医', '吃什么药'
                ],
                'cause': [
                    '什么原因', '为什么', '怎么引起', '病因',
                    '为何会得', '导致因素', '什么导致'
                ],
                'examination': [
                    '做什么检查', '检查项目', '需要检查', '怎么确诊',
                    '如何诊断', '查什么', '检测方法'
                ],
                'complication': [
                    '并发症', '会引起什么', '后果', '严重吗',
                    '会恶化吗', '会转移吗', '会扩散吗'
                ],
                'prevention': [
                    '如何预防', '怎么预防', '预防方法', '怎样避免',
                    '如何避免', '防止方法', '注意什么'
                ],
                'department': [
                    '挂什么科', '看哪个科', '去什么科室', '哪个科',
                    '看什么科', '挂号', '就诊科室'
                ]
            }
            
            # 预计算意图模板的embeddings
            logger.info("预计算意图模板embeddings...")
            self.intent_embeddings = self._precompute_intent_embeddings()
            logger.info("意图模板embeddings计算完成")
            
        except Exception as e:
            logger.error(f"BERT模型加载失败: {e}")
            raise
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        获取文本的BERT embedding
        
        Args:
            text: 输入文本
            
        Returns:
            embedding向量
        """
        with torch.no_grad():
            # 分词和编码
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=128
            ).to(self.device)
            
            # 获取BERT输出
            outputs = self.model(**inputs)
            
            # 使用[CLS] token的embedding作为句子表示
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            return embedding.squeeze()
    
    def _precompute_intent_embeddings(self) -> Dict[str, np.ndarray]:
        """预计算所有意图模板的embeddings"""
        intent_embeddings = {}
        
        for intent, templates in self.intent_templates.items():
            # 计算每个模板的embedding
            embeddings = []
            for template in templates:
                emb = self.get_embedding(template)
                embeddings.append(emb)
            
            # 取平均作为该意图的代表embedding
            intent_embeddings[intent] = np.mean(embeddings, axis=0)
        
        return intent_embeddings
    
    def predict_intent(self, question: str, threshold: float = 0.6) -> Tuple[str, float]:
        """
        使用语义相似度预测意图
        
        Args:
            question: 用户问题
            threshold: 相似度阈值
            
        Returns:
            (intent, confidence) 意图和置信度
        """
        # 获取问题的embedding
        question_emb = self.get_embedding(question)
        
        # 计算与各意图的余弦相似度
        similarities = {}
        for intent, intent_emb in self.intent_embeddings.items():
            similarity = self._cosine_similarity(question_emb, intent_emb)
            similarities[intent] = similarity
        
        # 找到最相似的意图
        best_intent = max(similarities, key=similarities.get)
        best_score = similarities[best_intent]
        
        # 如果相似度低于阈值，返回通用意图
        if best_score < threshold:
            return 'general', best_score
        
        return best_intent, best_score
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        计算两个文本的语义相似度
        
        Args:
            text1: 文本1
            text2: 文本2
            
        Returns:
            相似度分数 [0, 1]
        """
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)
        
        return self._cosine_similarity(emb1, emb2)
    
    def find_most_similar(self, query: str, candidates: List[str], top_k: int = 3) -> List[Tuple[str, float]]:
        """
        从候选列表中找到最相似的文本
        
        Args:
            query: 查询文本
            candidates: 候选文本列表
            top_k: 返回前k个最相似的
            
        Returns:
            [(text, score), ...] 排序后的结果
        """
        query_emb = self.get_embedding(query)
        
        similarities = []
        for candidate in candidates:
            cand_emb = self.get_embedding(candidate)
            sim = self._cosine_similarity(query_emb, cand_emb)
            similarities.append((candidate, sim))
        
        # 按相似度降序排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def semantic_search(self, query: str, knowledge_items: Dict[str, List[str]]) -> List[Tuple[str, str, float]]:
        """
        在知识库中进行语义搜索
        
        Args:
            query: 查询问题
            knowledge_items: 知识条目字典 {category: [items]}
            
        Returns:
            [(category, item, score), ...] 最相关的知识
        """
        query_emb = self.get_embedding(query)
        
        results = []
        for category, items in knowledge_items.items():
            for item in items:
                item_emb = self.get_embedding(item)
                score = self._cosine_similarity(query_emb, item_emb)
                results.append((category, item, score))
        
        # 按相似度排序
        results.sort(key=lambda x: x[2], reverse=True)
        
        return results[:5]  # 返回前5个最相关的


# 单例模式
_bert_engine = None

def get_bert_engine() -> BERTSemanticEngine:
    """获取BERT引擎单例"""
    global _bert_engine
    if _bert_engine is None:
        _bert_engine = BERTSemanticEngine()
    return _bert_engine


if __name__ == "__main__":
    # 测试代码
    print("测试BERT语义理解模块")
    print("=" * 60)
    
    engine = BERTSemanticEngine()
    
    # 测试意图识别
    test_questions = [
        "乙肝有什么症状啊？",
        "脂肪肝应该怎么治疗呢？",
        "肝硬化是由什么原因造成的？",
        "检查肝癌需要做哪些项目？",
        "得了丙肝会有什么严重后果吗？",
        "平时怎样预防酒精肝？",
        "乙肝应该看什么科室？"
    ]
    
    print("\n意图识别测试:")
    print("-" * 60)
    for q in test_questions:
        intent, confidence = engine.predict_intent(q)
        print(f"问题: {q}")
        print(f"意图: {intent} (置信度: {confidence:.3f})\n")
    
    # 测试语义相似度
    print("\n语义相似度测试:")
    print("-" * 60)
    text1 = "乙肝有什么症状"
    text2 = "乙型肝炎的临床表现是什么"
    text3 = "脂肪肝怎么治疗"
    
    sim12 = engine.compute_similarity(text1, text2)
    sim13 = engine.compute_similarity(text1, text3)
    
    print(f"'{text1}' vs '{text2}': {sim12:.3f}")
    print(f"'{text1}' vs '{text3}': {sim13:.3f}")
    print("test")
