#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
交互式疾病推理系统测试脚本
测试多轮对话的疾病推理功能
"""

from qa_engine import QAEngine

def test_interactive_inference():
    """测试交互式推理功能"""
    print("=" * 60)
    print("交互式疾病推理系统测试")
    print("=" * 60)
    
    # 创建问答引擎实例
    engine = QAEngine(use_bert_model=False)  # 使用规则方法以便测试
    
    # 模拟多轮对话
    context = None
    
    # 测试用例1：逐步推理
    test_cases = [
        "我经常感到疲劳，右上腹疼痛，有时还恶心",
        "是的，我确实有黄疸，皮肤和眼睛都有点黄",
        "没有蜘蛛痣，但我最近体重下降了不少"
    ]
    
    print("测试场景：用户逐步提供症状信息")
    print("-" * 50)
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n第{i}轮对话:")
        print(f"用户: {case}")
        print("-" * 30)
        
        result = engine.answer_question(case)
        
        if result['success']:
            print(f"系统: {result['answer']}")
            if 'suggestions' in result and result['suggestions']:
                print("\n建议问题:")
                for j, suggestion in enumerate(result['suggestions'], 1):
                    print(f"  {j}. {suggestion}")
        else:
            print(f"系统: {result['answer']}")
        
        print()

    # 测试用例2：直接描述多个症状
    print("\n" + "=" * 60)
    print("测试场景：用户一次性描述多个症状")
    print("-" * 60)
    
    case = "我最近皮肤发黄，眼睛也黄，特别累，而且右上腹疼痛，食欲不振"
    print(f"用户: {case}")
    print("-" * 30)
    
    result = engine.answer_question(case)
    
    if result['success']:
        print(f"系统: {result['answer']}")
        if 'suggestions' in result and result['suggestions']:
            print("\n建议问题:")
            for j, suggestion in enumerate(result['suggestions'], 1):
                print(f"  {j}. {suggestion}")
    else:
        print(f"系统: {result['answer']}")

if __name__ == "__main__":
    test_interactive_inference()