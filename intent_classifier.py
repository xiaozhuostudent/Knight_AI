#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
意图分类器 - 基于规则的轻量级实现
支持意图：symptom(症状)、treatment(治疗)、cause(病因)、examination(检查)、
         complication(并发症)、prevention(预防)、department(科室)
"""

import re
from typing import Literal

IntentType = Literal["symptom", "treatment", "cause", "examination", 
                     "complication", "prevention", "department", "general"]


class IntentClassifier:
    """基于关键词匹配的意图分类器"""
    
    def __init__(self):
        """初始化意图关键词映射"""
        self.intent_patterns = {
            'symptom': [
                r'症状|表现|感觉|反应|会.*样|有.*感|不舒服|难受',
                r'什么.*表现|有哪些.*症状'
            ],
            'treatment': [
                r'治疗|治|医治|怎么办|如何.*好|能.*好吗|怎样.*治|怎么.*治',
                r'吃.*药|用.*药|药物|medication|吃什么|用什么'
            ],
            'cause': [
                r'病因|原因|为什么|怎么.*的|怎样.*的|导致|引起|得.*病',
                r'为啥|咋.*的'
            ],
            'examination': [
                r'检查|查.*什么|做.*检查|化验|检测|诊断',
                r'需要.*查|要.*查|去.*查'
            ],
            'complication': [
                r'并发症|后果|影响|严重|会.*坏|危害|会.*什么|风险',
                r'不治.*会|会不会.*成'
            ],
            'prevention': [
                r'预防|避免|防止|注意|如何.*免|怎么.*防|防.*法',
                r'预防.*措施|怎样.*防'
            ],
            'department': [
                r'科室|挂.*科|看.*科|去.*科|什么科|哪个科',
                r'挂号.*科'
            ]
        }
        
        # 编译正则表达式
        self.compiled_patterns = {
            intent: [re.compile(pattern) for pattern in patterns]
            for intent, patterns in self.intent_patterns.items()
        }
    
    def predict_intent(self, question: str) -> IntentType:
        """
        预测问题的意图
        
        Args:
            question: 用户问题
            
        Returns:
            意图类型
        """
        # 清理问题
        question = question.strip()
        
        # 按优先级匹配（某些意图更具体）
        priority_order = [
            'examination', 'treatment', 'complication', 
            'prevention', 'cause', 'symptom', 'department'
        ]
        
        # 存储匹配结果和分数
        matches = {}
        
        for intent in priority_order:
            patterns = self.compiled_patterns.get(intent, [])
            for pattern in patterns:
                if pattern.search(question):
                    matches[intent] = matches.get(intent, 0) + 1
        
        # 返回匹配次数最多的意图
        if matches:
            return max(matches.items(), key=lambda x: x[1])[0]
        
        # 默认返回general
        return 'general'
    
    def get_intent_chinese(self, intent: IntentType) -> str:
        """获取意图的中文名称"""
        intent_map = {
            'symptom': '症状',
            'treatment': '治疗',
            'cause': '病因',
            'examination': '检查',
            'complication': '并发症',
            'prevention': '预防',
            'department': '科室',
            'general': '概览'
        }
        return intent_map.get(intent, '未知')


# 单例模式
_classifier = None

def get_classifier() -> IntentClassifier:
    """获取分类器实例"""
    global _classifier
    if _classifier is None:
        _classifier = IntentClassifier()
    return _classifier


def predict_intent(question: str) -> IntentType:
    """便捷函数：预测意图"""
    return get_classifier().predict_intent(question)


if __name__ == "__main__":
    # 测试
    classifier = IntentClassifier()
    
    test_cases = [
        "大三阳有什么症状？",
        "乙肝怎么治疗？",
        "脂肪肝是什么原因引起的？",
        "肝硬化需要做什么检查？",
        "肝癌会有什么并发症？",
        "怎么预防酒精肝？",
        "肝病应该挂什么科？",
    ]
    
    for question in test_cases:
        intent = classifier.predict_intent(question)
        print(f"问题: {question}")
        print(f"意图: {intent} ({classifier.get_intent_chinese(intent)})\n")
