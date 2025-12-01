#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速测试BERT模型
用于验证模型是否正确加载和使用
"""

from qa_engine import QAEngine
import os

def main():
    print("="*60)
    print("快速测试 - BERT意图分类模型")
    print("="*60)
    
    # 初始化引擎（使用BERT）
    kg_path = os.path.join(os.path.dirname(__file__), 'liver_kg.json')
    engine = QAEngine(kg_path, use_bert_model=True)
    
    # 测试问题
    test_questions = [
        "我最近感觉很疲劳，是不是乙肝？",
        "脂肪肝应该如何治疗啊？",
        "肝硬化通常是什么导致的？",
        "检查肝癌需要做哪些项目？",
        "丙肝会不会引起其他疾病？",
        "怎样才能预防酒精肝？",
        "大三阳应该挂哪个科室？"
    ]
    
    print("\n开始测试...\n")
    
    for i, question in enumerate(test_questions, 1):
        print(f"{i}. 问题: {question}")
        result = engine.answer_question(question)
        
        print(f"   疾病: {result['disease']}")
        print(f"   意图: {result['intent_chinese']} ({result['intent']})")
        print(f"   使用模型: {result.get('model_used', '未知')}")
        print(f"   答案: {result['answer'][:80]}...")
        print()
    
    print("="*60)
    print("✓ 测试完成！")
    print("="*60)

if __name__ == "__main__":
    main()
