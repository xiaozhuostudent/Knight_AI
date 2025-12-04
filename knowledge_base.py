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
    
    def infer_disease_by_symptoms(self, text: str) -> Optional[str]:
        """
        根据用户描述的症状推断可能的疾病（动态版本）
        
        Args:
            text: 用户描述症状的文本
            
        Returns:
            最可能的疾病名称，未找到返回None
        """
        text_lower = text.lower()
        
        # 动态统计每个疾病的匹配次数
        disease_scores = {}
        
        # 遍历所有疾病，检查其症状是否在用户描述中提及
        for disease, info in self.kg.items():
            symptoms = info.get('symptom', [])
            matched_count = 0
            
            # 棑查每个症状是否在用户描述中
            for symptom in symptoms:
                # 使用更宽松的匹配方式
                if symptom.lower() in text_lower or text_lower in symptom.lower():
                    matched_count += 1
            
            # 如果有匹配的症状，记录匹配数量
            if matched_count > 0:
                disease_scores[disease] = matched_count
        
        # 如果有匹配，返回得分最高的疾病
        if disease_scores:
            # 只有当至少有一个症状匹配时才返回结果
            best_match = max(disease_scores, key=disease_scores.get)
            if disease_scores[best_match] >= 1:  # 至少有一个症状匹配
                return best_match
        
        return None

    def interactive_disease_inference(self, text: str, previous_context: dict = None) -> dict:
        """
        基于知识图谱的交互式疾病推理系统
        支持多轮对话，逐步缩小范围
        
        Args:
            text: 用户当前输入的症状描述
            previous_context: 上一轮的推理上下文，包含已确认症状和候选疾病
            
        Returns:
            包含推理结果、下一步建议的字典
        """
        # 初始化或继承上下文
        if previous_context is None:
            context = {
                "confirmed_symptoms": [],      # 已确认的症状列表
                "excluded_symptoms": [],       # 已排除的症状列表  
                "conversation_round": 0,       # 对话轮次
                "possible_diseases": [],       # 当前候选疾病
                "previous_question": None      # 上一轮问的问题
            }
        else:
            context = previous_context.copy()
        
        context["conversation_round"] += 1
        
        # 1. 从当前文本中提取新症状
        new_symptoms = self._extract_symptoms_from_text(text)
        
        # 2. 处理用户对上一轮问题的回答（确认或排除症状）
        if context["previous_question"] and "question_type" in context["previous_question"]:
            self._process_user_response(text, context, new_symptoms)
        
        # 3. 更新已确认症状列表（合并新识别的症状）
        for symptom in new_symptoms:
            if symptom not in context["confirmed_symptoms"]:
                context["confirmed_symptoms"].append(symptom)
        
        # 4. 获取候选疾病
        candidate_diseases = self._get_candidate_diseases(context["confirmed_symptoms"])
        
        if not candidate_diseases:
            return self._build_no_result_response(context)
        
        # 5. 分析候选疾病的差异，生成鉴别诊断问题
        analysis_result = self._analyze_disease_differences(candidate_diseases, context)
        
        # 6. 更新上下文并返回结果
        context["possible_diseases"] = candidate_diseases
        context["previous_question"] = analysis_result.get("next_question", {})
        
        return self._build_final_response(analysis_result, context)
    
    def _extract_symptoms_from_text(self, text: str) -> list:
        """从文本中提取症状关键词（动态版本）"""
        found_symptoms = []
        text_lower = text.lower()
        
        # 动态收集所有疾病的症状作为关键词
        all_symptoms = set()
        for disease, info in self.kg.items():
            symptoms = info.get('symptom', [])
            for symptom in symptoms:
                # 添加症状的各个组成部分作为关键词
                parts = symptom.split('、')
                all_symptoms.update(parts)
                all_symptoms.add(symptom)
        
        # 检查文本中是否包含这些症状关键词
        for symptom in all_symptoms:
            if symptom.lower() in text_lower and symptom not in found_symptoms:
                found_symptoms.append(symptom)
        
        return found_symptoms
    
    def _process_user_response(self, text: str, context: dict, new_symptoms: list):
        """处理用户对特定症状问题的回答"""
        previous_question = context["previous_question"]
        
        if previous_question["question_type"] == "symptom_confirmation":
            symptom = previous_question["symptom"]
            
            # 简单的情感分析：用户是否确认有这个症状
            confirmation_words = ["是", "有", "对的", "确实", "会", "经常", "一直", "总是"]
            denial_words = ["没有", "不是", "不", "没", "无", "不会", "从来不"]
            
            text_lower = text.lower()
            
            if any(word in text_lower for word in confirmation_words):
                if symptom not in context["confirmed_symptoms"]:
                    context["confirmed_symptoms"].append(symptom)
            elif any(word in text_lower for word in denial_words):
                if symptom not in context["excluded_symptoms"]:
                    context["excluded_symptoms"].append(symptom)
    
    def _get_candidate_diseases(self, confirmed_symptoms: list) -> list:
        """根据已确认的症状获取候选疾病"""
        if not confirmed_symptoms:
            return []
        
        # 统计每个疾病匹配的症状数量
        disease_scores = {}
        
        for disease, info in self.kg.items():
            symptoms = info.get('symptom', [])
            matched_count = 0
            
            for symptom in confirmed_symptoms:
                # 检查症状是否在疾病症状列表中
                if any(symptom in s or s in symptom for s in symptoms):
                    matched_count += 1
            
            if matched_count > 0:
                disease_scores[disease] = {
                    'matched_count': matched_count,
                    'total_symptoms': len(symptoms),
                    'match_ratio': matched_count / len(symptoms) if symptoms else 0,
                    'symptoms': symptoms
                }
        
        # 按匹配数量和比例排序
        sorted_diseases = sorted(
            disease_scores.items(), 
            key=lambda x: (x[1]['matched_count'], x[1]['match_ratio']), 
            reverse=True
        )
        
        # 只返回前几个最匹配的疾病
        return [
            {
                "disease": disease,
                "matched_count": data['matched_count'],
                "total_symptoms": data['total_symptoms'],
                "match_ratio": data['match_ratio'],
                "all_symptoms": data['symptoms']
            }
            for disease, data in sorted_diseases[:5]
        ]
    
    def _analyze_disease_differences(self, diseases: list, context: dict) -> dict:
        """分析候选疾病的差异，生成鉴别诊断问题"""
        if len(diseases) == 0:
            return {"decision": "no_diseases", "message": "未找到匹配的疾病"}
        
        # 如果只有一个候选疾病且匹配度较高，直接返回
        if len(diseases) == 1 and diseases[0]['match_ratio'] > 0.5:
            return {
                "decision": "high_confidence",
                "primary_disease": diseases[0],
                "confidence": "high",
                "message": f"根据您的症状描述，最可能的疾病是{diseases[0]['disease']}，匹配度为{diseases[0]['match_ratio']:.1%}"
            }
        
        # 找出区分度最高的症状
        discriminating_symptom = self._find_discriminating_symptom(diseases, context)
        
        if discriminating_symptom:
            return {
                "decision": "need_more_info",
                "discriminating_symptom": discriminating_symptom,
                "possible_diseases": diseases,
                "next_question": {
                    "question_type": "symptom_confirmation",
                    "symptom": discriminating_symptom,
                    "question_text": self._generate_discrimination_question(diseases, discriminating_symptom)
                }
            }
        
        # 如果找不到好的区分症状，返回当前最佳匹配
        return {
            "decision": "moderate_confidence",
            "primary_disease": diseases[0],
            "alternative_diseases": diseases[1:],
            "message": self._generate_comparison_message(diseases)
        }
    
    def _find_discriminating_symptom(self, diseases: list, context: dict) -> str:
        """找出最能区分候选疾病的症状"""
        confirmed_symptoms = set(context["confirmed_symptoms"])
        excluded_symptoms = set(context["excluded_symptoms"])
        
        # 收集所有候选疾病的所有症状
        all_symptoms = set()
        for disease in diseases:
            all_symptoms.update(disease["all_symptoms"])
        
        # 排除已确认和已排除的症状
        candidate_symptoms = all_symptoms - confirmed_symptoms - excluded_symptoms
        
        # 计算每个症状的区分度
        symptom_scores = {}
        for symptom in candidate_symptoms:
            # 计算症状在不同疾病中的分布差异
            presence_count = sum(1 for d in diseases if symptom in d["all_symptoms"])
            absence_count = len(diseases) - presence_count
            
            # 理想区分症状：只在部分疾病中出现
            if presence_count > 0 and absence_count > 0:
                # 区分度 = |出现该症状的疾病匹配分数之和 - 未出现该症状的疾病匹配分数之和|
                presence_score = sum(d["matched_count"] for d in diseases if symptom in d["all_symptoms"])
                absence_score = sum(d["matched_count"] for d in diseases if symptom not in d["all_symptoms"])
                discrimination_score = abs(presence_score - absence_score)
                symptom_scores[symptom] = discrimination_score
        
        if symptom_scores:
            return max(symptom_scores.items(), key=lambda x: x[1])[0]
        
        return None
    
    def _generate_discrimination_question(self, diseases: list, symptom: str) -> str:
        """生成鉴别诊断问题"""
        diseases_with = [d["disease"] for d in diseases if symptom in d["all_symptoms"]]
        diseases_without = [d["disease"] for d in diseases if symptom not in d["all_symptoms"]]
        
        question = f"为了帮助进一步判断，请问您是否有'{symptom}'这个症状？\n"
        if diseases_with:
            question += f"• 如果有{symptom}，可能指向：{', '.join(diseases_with)}\n"
        if diseases_without:
            question += f"• 如果没有{symptom}，可能指向：{', '.join(diseases_without)}"
        
        return question
    
    def _generate_comparison_message(self, diseases: list) -> str:
        """生成疾病对比信息"""
        message = "根据您目前的症状，有以下几种可能性：\n\n"
        
        for i, disease in enumerate(diseases[:3], 1):
            message += f"{i}. {disease['disease']}（匹配度：{disease['matched_count']}/{disease['total_symptoms']}，{disease['match_ratio']:.1%}）\n"
            # 找出该疾病特有而其他疾病没有的症状
            unique_symptoms = set(disease["all_symptoms"])
            for other_disease in diseases:
                if other_disease != disease:
                    unique_symptoms -= set(other_disease["all_symptoms"])
            
            if unique_symptoms:
                message += f"   特征性症状：{', '.join(list(unique_symptoms)[:3])}\n"
            message += "\n"
        
        message += "您可以描述更多症状细节来帮助判断，比如症状的具体情况、发生时间等。"
        return message
    
    def _build_final_response(self, analysis_result: dict, context: dict) -> dict:
        """构建最终返回给用户的结果"""
        response = {
            "success": True,
            "conversation_round": context["conversation_round"],
            "confirmed_symptoms": context["confirmed_symptoms"],
            "analysis_result": analysis_result
        }
        
        # 根据决策类型构建不同的响应
        decision = analysis_result["decision"]
        
        if decision == "high_confidence":
            response["answer"] = analysis_result["message"]
            response["suggestions"] = [
                f"{analysis_result['primary_disease']['disease']}的典型症状有哪些？",
                f"{analysis_result['primary_disease']['disease']}应该如何治疗？"
            ]
            
        elif decision == "need_more_info":
            response["answer"] = analysis_result["next_question"]["question_text"]
            response["suggestions"] = ["有", "没有", "不确定"]
            response["expecting_confirmation"] = True
            
        elif decision == "moderate_confidence":
            response["answer"] = analysis_result["message"]
            response["suggestions"] = [
                "描述更多症状细节",
                "症状持续多久了？",
                "什么情况下症状会加重？"
            ]
        
        elif decision == "no_diseases":
            response["answer"] = analysis_result["message"]
            response["suggestions"] = [
                "请描述更多症状",
                "症状出现多长时间了？",
                "是否有其他身体不适？"
            ]
        
        return response
    
    def _build_no_result_response(self, context: dict) -> dict:
        """构建无结果响应"""
        return {
            "success": False,
            "answer": "根据您提供的信息，暂时无法确定可能的疾病。请提供更多症状描述。",
            "suggestions": [
                "请描述更多症状",
                "症状出现多长时间了？",
                "是否有其他身体不适？"
            ]
        }

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
