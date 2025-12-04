#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试心脏病知识库
验证系统对新领域的适应性
"""

from knowledge_base import KnowledgeBase

def test_heart_disease_kg():
    """测试心脏病知识库"""
    print("=" * 60)
    print("心脏病知识库测试")
    print("=" * 60)
    
    # 加载心脏病知识库
    heart_kg = KnowledgeBase('heart_disease_kg.json')
    
    print("1. 测试疾病列表:")
    diseases = heart_kg.get_all_diseases()
    for disease in diseases:
        print(f"  - {disease}")
    
    print("\n2. 测试症状提取:")
    text = "我最近总是胸闷，有时候还会心悸"
    symptoms = heart_kg._extract_symptoms_from_text(text)
    print(f"用户描述: {text}")
    print(f"提取症状: {symptoms}")
    
    print("\n3. 测试疾病推断:")
    inferred_disease = heart_kg.infer_disease_by_symptoms(text)
    print(f"推断疾病: {inferred_disease}")
    
    print("\n4. 测试交互式推理:")
    result = heart_kg.interactive_disease_inference("我走路时会气短，晚上睡觉经常咳嗽")
    print(f"推理结果: {result['answer']}")
    
    print("\n5. 测试知识查询:")
    if diseases:
        disease_info = heart_kg.get_disease_info(diseases[0])
        print(f"\n{diseases[0]}的信息:")
        print(f"  描述: {disease_info.get('description', 'N/A')}")
        print(f"  症状: {', '.join(disease_info.get('symptom', []))}")
        print(f"  治疗: {', '.join(disease_info.get('treatment', []))}")

if __name__ == "__main__":
    test_heart_disease_kg()