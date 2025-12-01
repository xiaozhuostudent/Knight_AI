#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
问答引擎 - 整合意图识别、实体抽取和知识查询，生成智能回答
支持BERT模型和规则回退
"""

from typing import Dict, Any
from intent_classifier import IntentClassifier, IntentType
from knowledge_base import KnowledgeBase
import random
import os
import logging

logger = logging.getLogger(__name__)


class QAEngine:
    """智能问答引擎"""
    
    def __init__(self, kg_path: str = 'liver_kg.json', use_bert_model: bool = True):
        """初始化问答引擎
        
        Args:
            kg_path: 知识图谱JSON文件路径
            use_bert_model: 是否使用BERT模型进行意图识别
        """
        self.kb = KnowledgeBase(kg_path)
        self.use_bert_model = use_bert_model
        self.bert_classifier = None
        
        # 尝试加载BERT模型
        if use_bert_model:
            try:
                from transformers import pipeline
                model_path = 'models/intent_model'
                
                if os.path.exists(model_path):
                    logger.info(f"加载BERT意图分类模型: {model_path}")
                    self.bert_classifier = pipeline(
                        'text-classification',
                        model=model_path,
                        tokenizer=model_path,
                        device=-1  # 使用CPU
                    )
                    # 加载标签映射
                    import json
                    with open(os.path.join(model_path, 'label_map.json'), 'r', encoding='utf-8') as f:
                        label_map = json.load(f)
                        self.id2label = label_map['id2label']
                    logger.info("✓ BERT模型加载成功")
                else:
                    logger.warning(f"BERT模型不存在: {model_path}，回退到规则方法")
                    self.use_bert_model = False
            except Exception as e:
                logger.warning(f"BERT模型加载失败: {e}，回退到规则方法")
                self.use_bert_model = False
        
        # 如果不使用BERT或加载失败，使用规则分类器
        if not self.use_bert_model:
            logger.info("使用规则意图分类器")
            self.classifier = IntentClassifier()
        
        # 回答模板
        self.templates = {
            'symptom': [
                "{disease}的主要症状包括：{content}",
                "患{disease}后，通常会出现以下症状：{content}",
                "{disease}患者常见的临床表现有：{content}"
            ],
            'treatment': [
                "{disease}的治疗方案包括：\n{content}\n\n建议在医生指导下进行规范治疗。",
                "针对{disease}，临床上主要采取以下治疗措施：\n{content}\n\n请务必遵医嘱治疗。",
                "{disease}的治疗方法有：\n{content}\n\n温馨提示：不同病情需要个体化治疗方案。"
            ],
            'cause': [
                "{disease}的主要病因有：{content}",
                "导致{disease}的常见原因包括：{content}",
                "{disease}通常由以下因素引起：{content}"
            ],
            'examination': [
                "确诊{disease}通常需要进行以下检查：\n{content}\n\n建议到正规医院的{department}就诊。",
                "{disease}的常规检查项目包括：\n{content}\n\n可以挂{department}进行详细检查。",
                "针对{disease}，医生可能会建议做这些检查：\n{content}"
            ],
            'complication': [
                "{disease}如果不及时治疗，可能会出现以下并发症：{content}\n\n因此早发现早治疗非常重要！",
                "{disease}的潜在并发症包括：{content}\n\n定期复查和规范治疗可以有效降低并发症风险。",
                "{disease}可能导致的严重后果有：{content}\n\n请重视疾病管理，防止病情恶化。"
            ],
            'prevention': [
                "预防{disease}，建议采取以下措施：\n{content}",
                "要避免{disease}，可以这样做：\n{content}\n\n预防胜于治疗！",
                "{disease}的预防方法包括：\n{content}\n\n保持健康生活方式很重要。"
            ],
            'department': [
                "{disease}建议到{content}就诊。",
                "如果怀疑有{disease}，可以挂{content}。",
                "{disease}通常由{content}负责诊治。"
            ]
        }
    
    def predict_intent(self, question: str) -> str:
        """
        预测问题意图
        
        Args:
            question: 用户问题
            
        Returns:
            意图标签
        """
        if self.use_bert_model and self.bert_classifier:
            # 使用BERT模型
            result = self.bert_classifier(question, top_k=1)[0]
            label_id = result['label'].split('_')[-1]  # 提取LABEL_0 -> 0
            intent = self.id2label.get(label_id, 'general')
            logger.debug(f"BERT预测意图: {intent} (置信度: {result['score']:.3f})")
            return intent
        else:
            # 使用规则方法
            intent = self.classifier.predict_intent(question)
            logger.debug(f"规则预测意图: {intent}")
            return intent
    
    def _format_list(self, items: list, use_numbers: bool = False) -> str:
        """格式化列表为文本"""
        if not items:
            return "暂无相关信息"
        
        if use_numbers:
            return "\n".join(f"{i}. {item}" for i, item in enumerate(items, 1))
        else:
            return "、".join(items)
    
    def _generate_answer(self, disease: str, intent: IntentType, info: Any) -> str:
        """
        生成自然语言答案
        
        Args:
            disease: 疾病名称
            intent: 意图类型
            info: 从知识库查询到的信息
            
        Returns:
            生成的答案
        """
        if info is None:
            return f"抱歉，我暂时没有{disease}关于{self.classifier.get_intent_chinese(intent)}的信息。"
        
        # 获取科室信息（用于某些模板）
        department = self.kb.get_disease_info(disease, 'department') or "相应科室"
        
        # 格式化内容
        if isinstance(info, list):
            # 对于治疗、检查等，使用编号列表
            if intent in ['treatment', 'examination', 'complication', 'prevention']:
                content = self._format_list(info, use_numbers=True)
            else:
                content = self._format_list(info, use_numbers=False)
        else:
            content = str(info)
        
        # 选择模板
        if intent in self.templates:
            template = random.choice(self.templates[intent])
            return template.format(
                disease=disease,
                content=content,
                department=department
            )
        
        # 通用回答
        intent_chinese_map = {
            'symptom': '症状',
            'treatment': '治疗',
            'cause': '病因',
            'examination': '检查',
            'complication': '并发症',
            'prevention': '预防',
            'department': '科室'
        }
        intent_name = intent_chinese_map.get(intent, '相关信息')
        return f"{disease}的{intent_name}：{content}"
    
    def _generate_overview(self, disease: str) -> str:
        """生成疾病概览"""
        info = self.kb.get_disease_info(disease)
        if not info:
            return f"抱歉，我没有找到关于{disease}的信息。"
        
        overview = f"📋 **{disease}概览**\n\n"
        
        # 疾病描述
        if 'description' in info:
            overview += f"💬 {info['description']}\n\n"
        
        # 主要症状
        if 'symptom' in info:
            symptoms = info['symptom'][:5]  # 显示前5个
            overview += f"🔸 **主要症状**：{self._format_list(symptoms)}\n\n"
        
        # 常见病因
        if 'cause' in info:
            causes = info['cause'][:3]  # 显示前3个
            overview += f"🔹 **常见病因**：{self._format_list(causes)}\n\n"
        
        # 推荐科室
        if 'department' in info:
            overview += f"🏥 **就诊科室**：{info['department']}\n\n"
        
        overview += "💡 输入更具体的问题（如症状、治疗、预防等）可获取详细信息。"
        
        return overview
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        回答用户问题
        
        Args:
            question: 用户问题
            
        Returns:
            {
                'success': bool,
                'answer': str,
                'disease': str,
                'intent': str,
                'suggestions': List[str]  # 相关问题建议
            }
        """
        # 1. 提取疾病
        disease = self.kb.extract_disease(question)
        
        if not disease:
            # 未识别到疾病
            diseases = self.kb.get_all_diseases()
            return {
                'success': False,
                'answer': f"抱歉，我没有识别出具体的疾病名称。\n\n我目前可以回答关于以下肝病的问题：\n" +
                         "\n".join(f"• {d}" for d in diseases),
                'disease': None,
                'intent': None,
                'suggestions': [f"{d}有什么症状？" for d in diseases[:3]]
            }
        
        # 2. 识别意图
        intent = self.predict_intent(question)
        
        # 3. 查询知识库
        if intent == 'general':
            # 概览查询
            answer = self._generate_overview(disease)
        else:
            # 具体字段查询
            info = self.kb.get_disease_info(disease, intent)
            answer = self._generate_answer(disease, intent, info)
        
        # 4. 生成相关问题建议
        suggestions = self._generate_suggestions(disease, intent)
        
        # 获取中文意图名称
        if self.use_bert_model:
            intent_chinese_map = {
                'symptom': '症状',
                'treatment': '治疗',
                'cause': '病因',
                'examination': '检查',
                'complication': '并发症',
                'prevention': '预防',
                'department': '科室',
                'general': '概览'
            }
            intent_chinese = intent_chinese_map.get(intent, '未知')
        else:
            intent_chinese = self.classifier.get_intent_chinese(intent)
        
        return {
            'success': True,
            'answer': answer,
            'disease': disease,
            'intent': intent,
            'intent_chinese': intent_chinese,
            'suggestions': suggestions,
            'model_used': 'BERT' if self.use_bert_model else '规则'
        }
    
    def _generate_suggestions(self, disease: str, current_intent: IntentType) -> list:
        """生成相关问题建议"""
        intents = ['symptom', 'treatment', 'cause', 'examination', 'prevention']
        
        # 排除当前意图
        intents = [i for i in intents if i != current_intent]
        
        # 随机选择3个
        selected = random.sample(intents, min(3, len(intents)))
        
        question_templates = {
            'symptom': f"{disease}有什么症状？",
            'treatment': f"{disease}怎么治疗？",
            'cause': f"{disease}是什么原因引起的？",
            'examination': f"{disease}需要做什么检查？",
            'prevention': f"如何预防{disease}？",
            'complication': f"{disease}会有什么并发症？"
        }
        
        return [question_templates[i] for i in selected]


# 单例模式
_engine = None

def get_qa_engine() -> QAEngine:
    """获取问答引擎实例"""
    global _engine
    if _engine is None:
        _engine = QAEngine()
    return _engine


def answer_question(question: str) -> Dict[str, Any]:
    """便捷函数：回答问题"""
    return get_qa_engine().answer_question(question)


if __name__ == "__main__":
    # 测试
    engine = QAEngine()
    
    test_questions = [
        "大三阳有什么症状？",
        "乙肝怎么治疗？",
        "脂肪肝是什么原因引起的？",
        "肝硬化需要做什么检查？",
        "肝癌会有什么并发症？",
        "如何预防酒精肝？",
        "丙肝应该挂什么科？",
        "告诉我乙肝的情况",
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*60}")
        print(f"问题 {i}: {question}")
        print('='*60)
        
        result = engine.answer_question(question)
        
        if result['success']:
            print(f"识别疾病: {result['disease']}")
            print(f"问题意图: {result['intent_chinese']}")
            print(f"\n回答:\n{result['answer']}")
            print(f"\n相关问题:")
            for suggestion in result['suggestions']:
                print(f"  • {suggestion}")
        else:
            print(result['answer'])
