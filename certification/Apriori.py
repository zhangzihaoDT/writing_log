import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Mock数据 - 汽车配置购买记录（包含客户画像）
data = {
    'order_id': ['001', '002', '003', '004'],
    '年龄': [35, 28, 45, 38],
    '性别': ['男', '女', '男', '男'],
    '城市': ['北京', '上海', '深圳', '广州'],
    '真皮座椅': [1, 1, 0, 1],
    '全景天窗': [1, 0, 1, 1],
    '电动尾门': [1, 1, 0, 1],
    '四驱系统': [0, 1, 1, 1],
    'HUD抬头显示': [0, 1, 0, 1]
}
df = pd.DataFrame(data)

print("原始数据:")
print(df)
print("\n" + "="*50)

# 为Apriori算法准备数据 - 对分类变量进行二进制编码
df_encoded = df.copy()

# 年龄分组编码
df_encoded['年龄_青年'] = (df['年龄'] <= 30).astype(int)  # 30岁及以下
df_encoded['年龄_中年'] = ((df['年龄'] > 30) & (df['年龄'] <= 40)).astype(int)  # 31-40岁
df_encoded['年龄_中老年'] = (df['年龄'] > 40).astype(int)  # 40岁以上

# 性别编码
df_encoded['性别_男'] = (df['性别'] == '男').astype(int)
df_encoded['性别_女'] = (df['性别'] == '女').astype(int)

# 城市编码
df_encoded['城市_北京'] = (df['城市'] == '北京').astype(int)
df_encoded['城市_上海'] = (df['城市'] == '上海').astype(int)
df_encoded['城市_深圳'] = (df['城市'] == '深圳').astype(int)
df_encoded['城市_广州'] = (df['城市'] == '广州').astype(int)

# 为Apriori算法准备数据（去除原始的非二进制列）
df_apriori = df_encoded.drop(['order_id', '年龄', '性别', '城市'], axis=1)

print("编码后的数据（用于Apriori分析）:")
print(df_apriori)
print("\n" + "="*50)

# Step 1. 生成频繁项集
print("Step 1: 生成频繁项集 (最小支持度=0.5)")
frequent_itemsets = apriori(df_apriori, min_support=0.5, use_colnames=True)
print(frequent_itemsets)
print("\n" + "="*50)

# Step 2. 生成关联规则
print("Step 2: 生成关联规则 (最小提升度=1.0)")
if len(frequent_itemsets) > 1:
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    
    if len(rules) > 0:
        # Step 3. 按lift排序
        rules = rules.sort_values(by='lift', ascending=False)
        
        # Step 4. 查看主要结果
        print("关联规则结果:")
        print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].round(3))
        
        print("\n" + "="*50)
        print("规则解释:")
        for idx, rule in rules.iterrows():
            antecedent = list(rule['antecedents'])[0] if len(rule['antecedents']) == 1 else str(rule['antecedents'])
            consequent = list(rule['consequents'])[0] if len(rule['consequents']) == 1 else str(rule['consequents'])
            print(f"如果购买了 {antecedent} → 那么也会购买 {consequent}")
            print(f"  支持度: {rule['support']:.3f} | 置信度: {rule['confidence']:.3f} | 提升度: {rule['lift']:.3f}")
            print()
    else:
        print("没有找到满足条件的关联规则")
else:
    print("频繁项集数量不足，无法生成关联规则")
