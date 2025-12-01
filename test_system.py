#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试BERT模型和规则方法的对比
"""

import os
import sys

print("="*70)
print("肝病智能问答系统 - 测试工具")
print("="*70)

# 检查模型是否存在
model_exists = os.path.exists('models/intent_model')

if model_exists:
    print("\n✓ 检测到 BERT 模型")
    print("  位置: models/intent_model/")
else:
    print("\n✗ BERT 模型未找到")
    print("  请先运行: python3 train_intent.py")
    print("\n将使用规则方法进行测试...")

# 测试问题
test_questions = [
    "乙肝有什么症状?",
    "脂肪肝怎么治疗?",
    "肝硬化是什么原因引起的?",
    "肝癌需要做什么检查?",
    "丙肝会有什么并发症?",
    "如何预防酒精肝?",
    "乙肝挂什么科?",
]

print("\n" + "="*70)
print("开始测试...")
print("="*70)

# 导入问答引擎
from qa_engine import QAEngine

# 测试BERT模型（如果存在）
if model_exists:
    print("\n【测试 BERT 模型】")
    print("-"*70)
    try:
        engine_bert = QAEngine(use_bert_model=True)
        
        if engine_bert.use_bert_model:
            print("✓ BERT模型加载成功\n")
            
            for i, question in enumerate(test_questions, 1):
                result = engine_bert.answer_question(question)
                print(f"{i}. 问题: {question}")
                print(f"   疾病: {result.get('disease')}")
                print(f"   意图: {result.get('intent_chinese')} ({result.get('intent')})")
                print(f"   模型: {result.get('model_used')}")
                print()
        else:
            print("✗ BERT模型加载失败\n")
    except Exception as e:
        print(f"✗ BERT模型测试出错: {e}\n")

# 测试规则方法
print("\n【测试规则方法】")
print("-"*70)
try:
    engine_rule = QAEngine(use_bert_model=False)
    print("✓ 规则分类器加载成功\n")
    
    for i, question in enumerate(test_questions, 1):
        result = engine_rule.answer_question(question)
        print(f"{i}. 问题: {question}")
        print(f"   疾病: {result.get('disease')}")
        print(f"   意图: {result.get('intent_chinese')} ({result.get('intent')})")
        print(f"   模型: {result.get('model_used')}")
        print()
except Exception as e:
    print(f"✗ 规则方法测试出错: {e}\n")

print("="*70)
print("测试完成！")
print("="*70)

if not model_exists:
    print("\n提示：要使用 BERT 模型，请执行以下步骤：")
    print("  1. 安装依赖: pip install transformers torch pandas scikit-learn tqdm")
    print("  2. 训练模型: python3 train_intent.py")
    print("  3. 重新测试: python3 test_system.py")
