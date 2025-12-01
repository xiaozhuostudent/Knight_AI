#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自动生成意图分类训练数据
生成300条高质量中文训练样本，覆盖7类意图和6种肝病
"""

import csv
import random
from pathlib import Path

# 定义疾病列表
diseases = ["乙肝", "脂肪肝", "肝硬化", "肝癌", "丙肝", "酒精肝"]
disease_aliases = {
    "乙肝": ["大三阳", "小三阳", "乙型肝炎", "HBV"],
    "脂肪肝": ["非酒精性脂肪肝", "NAFLD"],
    "肝硬化": ["肝硬变", "肝纤维化"],
    "肝癌": ["原发性肝癌", "肝细胞癌", "HCC"],
    "丙肝": ["丙型肝炎", "HCV"],
    "酒精肝": ["酒精性肝病"]
}

# 定义各类意图的问法模板
intent_templates = {
    "symptom": [
        "{disease}有什么症状",
        "{disease}的症状是什么",
        "{disease}会出现哪些表现",
        "{disease}的临床表现有哪些",
        "得了{disease}会有什么感觉",
        "{disease}有哪些不舒服",
        "{disease}的症状表现",
        "{disease}会不会难受",
        "{disease}身体会有什么变化",
        "{disease}有啥不适",
        "{disease}会怎么样",
        "{disease}的早期症状",
        "{disease}晚期什么症状",
        "患{disease}后有什么反应",
        "{disease}咋知道自己得了",
    ],
    "treatment": [
        "{disease}怎么治疗",
        "{disease}如何治",
        "{disease}的治疗方法",
        "{disease}能治好吗",
        "{disease}咋治",
        "{disease}怎么办",
        "{disease}如何医治",
        "{disease}吃什么药",
        "{disease}需要吃药吗",
        "{disease}能自愈吗",
        "{disease}治疗方案",
        "{disease}最好的治疗方法",
        "{disease}可以手术吗",
        "得了{disease}怎么整",
        "{disease}用什么药好",
    ],
    "cause": [
        "{disease}是什么原因引起的",
        "{disease}的病因是什么",
        "为什么会得{disease}",
        "{disease}怎么引起的",
        "{disease}是怎么来的",
        "什么导致{disease}",
        "{disease}的发病原因",
        "{disease}咋得的",
        "为啥会有{disease}",
        "{disease}是因为什么",
        "{disease}发病机制",
        "哪些因素导致{disease}",
        "{disease}和什么有关",
        "{disease}遗传吗",
    ],
    "examination": [
        "{disease}需要做什么检查",
        "{disease}的检查项目有哪些",
        "怎么确诊{disease}",
        "{disease}做什么检查",
        "{disease}需要查什么",
        "检查{disease}做哪些项目",
        "{disease}如何诊断",
        "{disease}怎么查出来",
        "{disease}要做B超吗",
        "{disease}需要抽血吗",
        "确诊{disease}要做什么",
        "{disease}的诊断标准",
        "{disease}检查费用多少",
        "{disease}要做哪些化验",
        "{disease}筛查项目",
    ],
    "complication": [
        "{disease}有什么并发症",
        "{disease}会引起什么病",
        "{disease}的后果严重吗",
        "{disease}会恶化吗",
        "{disease}会转移吗",
        "{disease}并发症有哪些",
        "{disease}有什么危害",
        "{disease}会不会很严重",
        "得了{disease}后果",
        "{disease}能致命吗",
        "{disease}会扩散吗",
        "{disease}最坏结果",
        "{disease}会癌变吗",
        "{disease}危险吗",
        "{disease}死亡率高吗",
    ],
    "prevention": [
        "如何预防{disease}",
        "怎么预防{disease}",
        "{disease}的预防方法",
        "怎样避免{disease}",
        "{disease}怎么避免",
        "预防{disease}要注意什么",
        "{disease}预防措施",
        "咋预防{disease}",
        "{disease}如何防止",
        "{disease}要注意什么",
        "防{disease}的方法",
        "{disease}日常注意事项",
        "{disease}饮食禁忌",
        "{disease}生活注意",
        "{disease}能预防吗",
        "{disease}会传染吗",
        "{disease}会不会传染",
        "我爸爸有{disease}我会被传染吗",
        "家人有{disease}会传染吗",
        "{disease}传染吗",
        "{disease}能传染给别人吗",
        "{disease}通过什么途径传播",
        "{disease}怎么传染的",
        "接触{disease}患者会被传染吗",
        "{disease}会遗传吗",
        "{disease}会传给孩子吗",
        "和{disease}患者一起吃饭会传染吗",
        "{disease}通过血液传播吗",
        "{disease}性传播吗",
        "母婴会传{disease}吗",
    ],
    "department": [
        "{disease}挂什么科",
        "{disease}看哪个科",
        "{disease}去什么科室",
        "{disease}应该看什么科",
        "{disease}挂号挂什么",
        "{disease}哪个科室看",
        "{disease}看病挂啥科",
        "{disease}要看哪个医生",
        "{disease}属于什么科",
        "{disease}就诊科室",
        "{disease}去医院看什么科",
        "{disease}门诊挂什么",
        "{disease}专科是哪个",
        "{disease}找什么科",
        "{disease}看病流程",
    ]
}

# 口语化变体
colloquial_variants = [
    lambda t: t.replace("吗", "么"),
    lambda t: t.replace("什么", "啥"),
    lambda t: t.replace("怎么", "咋"),
    lambda t: t.replace("如何", "咋样"),
    lambda t: t + "啊",
    lambda t: t + "呢",
    lambda t: t + "？",
    lambda t: t.replace("有", "会有"),
]

def generate_samples():
    """生成训练样本"""
    samples = []
    
    # 为每个意图和疾病组合生成样本
    for intent, templates in intent_templates.items():
        for disease in diseases:
            # 使用疾病名称
            for template in templates:
                text = template.format(disease=disease)
                samples.append((text, intent))
            
            # 使用别名
            if disease in disease_aliases:
                for alias in disease_aliases[disease][:2]:  # 每个疾病取2个别名
                    for template in random.sample(templates, min(3, len(templates))):
                        text = template.format(disease=alias)
                        samples.append((text, intent))
    
    # 添加口语化变体
    original_samples = samples.copy()
    for text, label in random.sample(original_samples, min(80, len(original_samples))):
        variant_fn = random.choice(colloquial_variants)
        variant_text = variant_fn(text)
        if variant_text != text:  # 确保有变化
            samples.append((variant_text, label))
    
    # 打乱顺序
    random.shuffle(samples)
    
    return samples

def main():
    """主函数"""
    print("="*60)
    print("生成意图分类训练数据")
    print("="*60)
    
    # 创建data目录
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # 生成样本
    print("\n正在生成训练样本...")
    samples = generate_samples()
    
    # 限制到500条左右（如果超过则随机采样）
    if len(samples) > 500:
        samples = random.sample(samples, 500)
    
    print(f"✓ 生成了 {len(samples)} 条样本")
    
    # 统计各类别数量
    from collections import Counter
    label_counts = Counter(label for _, label in samples)
    print("\n类别分布:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count} 条")
    
    # 保存到CSV
    output_file = data_dir / "intent_train.csv"
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['text', 'label'])  # 写入表头
        writer.writerows(samples)
    
    print(f"\n✓ 训练数据已保存到: {output_file}")
    
    # 显示示例
    print("\n样本示例（前10条）:")
    print("-"*60)
    for i, (text, label) in enumerate(samples[:10], 1):
        print(f"{i}. [{label}] {text}")
    
    print("\n" + "="*60)
    print("✓ 数据生成完成！")
    print("="*60)

if __name__ == "__main__":
    random.seed(42)  # 设置随机种子以保证可重复性
    main()
