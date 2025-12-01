#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
知识库模块 - 管理肝病知识图谱和实体识别
"""

import json
import re
from typing import Optional, Dict, List, Any


class KnowledgeBase:
    """肝病知识库"""
    
    def __init__(self, kg_path: str = 'liver_kg.json'):
        """
        初始化知识库
        
        Args:
            kg_path: 知识图谱JSON文件路径
        """
        with open(kg_path, 'r', encoding='utf-8') as f:
            self.kg = json.load(f)
        
        # 构建别名映射表
        self.alias_map = {}
        for disease, info in self.kg.items():
            # 疾病名本身
            self.alias_map[disease] = disease
            # 别名
            for alias in info.get('aliases', []):
                self.alias_map[alias.lower()] = disease
        
        # 按长度排序，优先匹配长的名称
        self.sorted_aliases = sorted(
            self.alias_map.keys(), 
            key=len, 
            reverse=True
        )
    
    def extract_disease(self, text: str) -> Optional[str]:
        """
        从文本中提取疾病名称
        
        Args:
            text: 输入文本
            
        Returns:
            疾病名称，未找到返回None
        """
        text_lower = text.lower()
        
        # 按长度从长到短匹配，避免"乙肝"被"肝"匹配
        for alias in self.sorted_aliases:
            if alias in text_lower:
                return self.alias_map[alias]
        
        return None
    
    def get_disease_info(self, disease: str, field: Optional[str] = None) -> Any:
        """
        获取疾病信息
        
        Args:
            disease: 疾病名称
            field: 字段名（symptom/treatment等），None表示返回全部
            
        Returns:
            疾病信息
        """
        if disease not in self.kg:
            return None
        
        disease_info = self.kg[disease]
        
        if field is None:
            return disease_info
        
        return disease_info.get(field, None)
    
    def get_all_diseases(self) -> List[str]:
        """获取所有疾病列表"""
        return list(self.kg.keys())
    
    def search_by_keyword(self, keyword: str) -> Dict[str, List[str]]:
        """
        根据关键词搜索相关疾病
        
        Args:
            keyword: 搜索关键词
            
        Returns:
            {疾病名: [匹配的字段列表]}
        """
        results = {}
        keyword_lower = keyword.lower()
        
        for disease, info in self.kg.items():
            matches = []
            
            # 搜索各个字段
            for field, value in info.items():
                if field == 'aliases':
                    continue
                
                if isinstance(value, list):
                    if any(keyword_lower in str(v).lower() for v in value):
                        matches.append(field)
                elif isinstance(value, str):
                    if keyword_lower in value.lower():
                        matches.append(field)
            
            if matches:
                results[disease] = matches
        
        return results


# 单例模式
_kb = None

def get_knowledge_base() -> KnowledgeBase:
    """获取知识库实例"""
    global _kb
    if _kb is None:
        _kb = KnowledgeBase()
    return _kb


def extract_disease(text: str) -> Optional[str]:
    """便捷函数：提取疾病"""
    return get_knowledge_base().extract_disease(text)


if __name__ == "__main__":
    # 测试
    kb = KnowledgeBase()
    
    print("=== 测试疾病提取 ===")
    test_texts = [
        "大三阳有什么症状？",
        "乙型肝炎怎么治疗？",
        "脂肪肝是什么原因？",
        "NAFLD需要做什么检查？",
        "肝硬变会有什么并发症？",
        "我想问问HBV能治好吗？",
        "酒精性肝病能预防吗？",
    ]
    
    for text in test_texts:
        disease = kb.extract_disease(text)
        print(f"文本: {text}")
        print(f"识别疾病: {disease}\n")
    
    print("\n=== 测试知识查询 ===")
    disease = "乙肝"
    print(f"\n{disease}的症状:")
    print(kb.get_disease_info(disease, 'symptom'))
    
    print(f"\n{disease}的治疗:")
    print(kb.get_disease_info(disease, 'treatment'))
    
    print("\n=== 测试关键词搜索 ===")
    results = kb.search_by_keyword("黄疸")
    print(f"\n包含'黄疸'的疾病:")
    for disease, fields in results.items():
        print(f"  {disease}: {fields}")
