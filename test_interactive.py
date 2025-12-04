#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试交互式推理功能
"""

from knowledge_base import KnowledgeBase

def test_interactive_inference():
    kb = KnowledgeBase()
    
    print("测试1: 提取症状")
    symptoms = kb._extract_symptoms_from_text("我经常感到疲劳，右上腹疼痛，有时还恶心")
    print(f"提取到的症状: {symptoms}")
    
    print("\n测试2: 候选疾病")
    candidates = kb._get_candidate_diseases(symptoms)
    print(f"候选疾病: {candidates}")
    
    print("\n测试3: 交互式推理")
    result = kb.interactive_disease_inference("我经常感到疲劳，右上腹疼痛，有时还恶心")
    print(f"推理结果: {result}")

if __name__ == "__main__":
    test_interactive_inference()