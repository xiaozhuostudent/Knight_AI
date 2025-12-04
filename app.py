#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
肝病智能问答系统 - Web服务
基于Flask提供RESTful API和Web界面
"""

from flask import Flask, render_template, request, jsonify
from qa_engine import QAEngine
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  # 支持中文JSON

# 初始化问答引擎
logger.info("正在初始化问答引擎...")
import os
kg_path = os.path.join(os.path.dirname(__file__), 'liver_kg.json')  # 使用肝病知识库
# 如果要使用其他知识库，修改上面这行，例如：
# kg_path = os.path.join(os.path.dirname(__file__), 'heart_disease_kg.json')  # 使用心脏病知识库
qa_engine = QAEngine(kg_path, use_bert_model=True)

# 检查实际使用的模型
if qa_engine.use_bert_model:
    logger.info("✓ 问答引擎初始化完成！使用 BERT 意图分类模型")
else:
    logger.info("✓ 问答引擎初始化完成！使用规则意图分类（BERT模型未找到或加载失败）")


@app.route('/')
def index():
    """首页"""
    return render_template('index.html')


@app.route('/api/ask', methods=['POST'])
def ask():
    """
    API接口：回答问题
    
    Request JSON:
        {
            "question": "用户问题"
        }
    
    Response JSON:
        {
            "success": true/false,
            "answer": "答案",
            "disease": "疾病名",
            "intent": "意图",
            "suggestions": ["相关问题1", ...]
        }
    """
    try:
        data = request.json
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({
                'success': False,
                'error': '请输入问题'
            }), 400
        
        logger.info(f"收到问题: {question}")
        
        # 调用问答引擎
        result = qa_engine.answer_question(question)
        
        logger.info(f"回答生成成功: {result.get('disease')} - {result.get('intent')}")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"处理问题时出错: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'服务器错误: {str(e)}'
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({
        'status': 'ok',
        'service': '肝病智能问答系统',
        'version': '1.0.0'
    })


if __name__ == '__main__':
    print("=" * 70)
    print("肝病智能问答系统 v2.0")
    print("=" * 70)
    
    # 显示系统信息
    if qa_engine.use_bert_model:
        print("\n[模型] 使用 BERT 微调模型进行意图识别")
    else:
        print("\n[模型] 使用规则方法进行意图识别（BERT模型未训练）")
    
    print(f"[知识库] {len(qa_engine.kb.get_all_diseases())} 种肝病")
    print("\n正在启动Web服务...")
    print("\n访问地址：http://localhost:5001")
    print("\n按 Ctrl+C 停止服务\n")
    print("=" * 70)
    
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5001,
        threaded=True
    )
