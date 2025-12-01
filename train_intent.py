#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练意图分类模型
使用 bert-base-chinese 进行 7 类意图分类
支持 CPU 训练，自动保存模型
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
import numpy as np
from pathlib import Path

# 设置随机种子
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

# 定义数据集类
class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_data(data_file='data/intent_train.csv'):
    """加载训练数据"""
    print(f"加载数据: {data_file}")
    df = pd.read_csv(data_file)
    
    # 标签映射
    labels = sorted(df['label'].unique())
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    
    print(f"\n类别数量: {len(labels)}")
    print(f"类别列表: {labels}")
    print(f"\n数据集大小: {len(df)} 条")
    
    # 将标签转换为数字
    df['label_id'] = df['label'].map(label2id)
    
    return df, label2id, id2label

def train_epoch(model, dataloader, optimizer, scheduler, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    predictions = []
    true_labels = []
    
    progress_bar = tqdm(dataloader, desc='Training')
    for batch in progress_bar:
        # 将数据移到设备
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # 前向传播
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        logits = outputs.logits
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # 记录
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        predictions.extend(preds)
        true_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)
    
    return avg_loss, accuracy

def evaluate(model, dataloader, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)
    
    return avg_loss, accuracy, predictions, true_labels

def main():
    # 超参数
    MODEL_NAME = 'models/bert-base-chinese'  # 使用本地下载的模型
    BATCH_SIZE = 8
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    MAX_LENGTH = 64
    OUTPUT_DIR = 'models/intent_model'
    
    print("="*60)
    print("训练意图分类模型")
    print("="*60)
    print(f"\n模型: {MODEL_NAME}")
    print(f"批大小: {BATCH_SIZE}")
    print(f"训练轮数: {EPOCHS}")
    print(f"学习率: {LEARNING_RATE}")
    
    # 设置随机种子
    set_seed(42)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    df, label2id, id2label = load_data()
    
    # 划分训练集和验证集 (9:1)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].values,
        df['label_id'].values,
        test_size=0.1,
        random_state=42,
        stratify=df['label_id'].values
    )
    
    print(f"\n训练集: {len(train_texts)} 条")
    print(f"验证集: {len(val_texts)} 条")
    
    # 加载tokenizer
    print(f"\n加载 tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    
    # 创建数据集
    train_dataset = IntentDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_dataset = IntentDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # 加载模型
    print(f"\n加载模型...")
    num_labels = len(label2id)
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels
    )
    model.to(device)
    
    # 优化器和学习率调度器
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # 训练循环
    print("\n" + "="*60)
    print("开始训练")
    print("="*60)
    
    best_val_acc = 0
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print("-"*60)
        
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
        
        # 验证
        val_loss, val_acc, val_preds, val_true = evaluate(model, val_loader, device)
        print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"✓ 验证准确率提升! 保存模型...")
            
            # 创建输出目录
            Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
            
            # 保存模型
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            
            # 保存标签映射
            import json
            with open(os.path.join(OUTPUT_DIR, 'label_map.json'), 'w', encoding='utf-8') as f:
                json.dump({
                    'label2id': label2id,
                    'id2label': id2label
                }, f, ensure_ascii=False, indent=2)
    
    # 最终评估
    print("\n" + "="*60)
    print("最终评估")
    print("="*60)
    
    # 加载最佳模型
    model = BertForSequenceClassification.from_pretrained(OUTPUT_DIR)
    model.to(device)
    
    val_loss, val_acc, val_preds, val_true = evaluate(model, val_loader, device)
    
    print(f"\n最佳验证准确率: {val_acc:.4f}")
    print("\n分类报告:")
    print(classification_report(
        val_true,
        val_preds,
        target_names=list(label2id.keys())
    ))
    
    print("\n" + "="*60)
    print(f"✓ 训练完成! 模型已保存到: {OUTPUT_DIR}")
    print("="*60)

if __name__ == "__main__":
    main()
