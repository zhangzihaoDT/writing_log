# 🚗 案例：预测汽车月销量（线性回归）
# 基于 Scikit-learn + mock 数据 的完整案例，用来预测 新车销量示例。

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# 1️⃣ 构造 mock 数据
np.random.seed(42)
n_samples = 120

# 模拟特征
price = np.random.randint(100000, 400000, n_samples)        # 价格（元）
horsepower = np.random.randint(80, 250, n_samples)          # 马力
fuel_efficiency = np.random.uniform(5, 12, n_samples)       # 油耗（L/100km）
brand_score = np.random.randint(60, 95, n_samples)          # 品牌评分（市场影响力）

# 模拟销量（销量与价格负相关，与马力、品牌正相关）
sales = (
    8000 
    - 0.015 * price 
    + 12 * horsepower 
    - 100 * fuel_efficiency 
    + 80 * brand_score 
    + np.random.normal(0, 800, n_samples)  # 噪声
)

# 组装 DataFrame
df = pd.DataFrame({
    "price": price,
    "horsepower": horsepower,
    "fuel_efficiency": fuel_efficiency,
    "brand_score": brand_score,
    "sales": sales
})

print("🚘 数据样例：")
print(df.head())

# 2️⃣ 特征与目标
X = df[["price", "horsepower", "fuel_efficiency", "brand_score"]]
y = df["sales"]

# 3️⃣ 拆分训练 / 测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4️⃣ 建模
model = LinearRegression()
model.fit(X_train, y_train)

# 5️⃣ 预测与评估
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\n📊 模型评估：")
print(f"R²：{r2:.4f}")
print(f"MAE：{mae:.2f}")

# 6️⃣ 模型系数解释
print("\n⚙️ 模型系数：")
coef_df = pd.DataFrame({
    "特征": X.columns,
    "系数": model.coef_
})
print(coef_df)
print(f"截距：{model.intercept_:.2f}")

# 7️⃣ 数据探索性分析
print("\n📈 数据描述统计：")
print(df.describe())

print("\n🔗 特征相关性分析：")
correlation_matrix = df.corr()
print(correlation_matrix)

# 8️⃣ 数据可视化
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

# 创建综合数据分析图表
fig = make_subplots(
    rows=2, cols=3,
    subplot_titles=('特征相关性热力图', '销量分布', '价格 vs 销量', 
                   '马力 vs 销量', '预测值 vs 实际值', '残差图'),
    specs=[[{"type": "heatmap"}, {"type": "histogram"}, {"type": "scatter"}],
           [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}]]
)

# 1. 相关性热力图
fig.add_trace(
    go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=correlation_matrix.round(3).values,
        texttemplate="%{text}",
        textfont={"size": 10},
        showscale=True
    ),
    row=1, col=1
)

# 2. 销量分布
fig.add_trace(
    go.Histogram(
        x=df['sales'],
        nbinsx=20,
        marker_color='skyblue',
        marker_line_color='black',
        marker_line_width=1,
        opacity=0.7,
        name='销量分布'
    ),
    row=1, col=2
)

# 3. 价格vs销量散点图
fig.add_trace(
    go.Scatter(
        x=df['price'],
        y=df['sales'],
        mode='markers',
        marker=dict(color='orange', opacity=0.6),
        name='价格 vs 销量'
    ),
    row=1, col=3
)

# 4. 马力vs销量散点图
fig.add_trace(
    go.Scatter(
        x=df['horsepower'],
        y=df['sales'],
        mode='markers',
        marker=dict(color='green', opacity=0.6),
        name='马力 vs 销量'
    ),
    row=2, col=1
)

# 5. 预测值vs实际值
fig.add_trace(
    go.Scatter(
        x=y_test,
        y=y_pred,
        mode='markers',
        marker=dict(color='red', opacity=0.6),
        name='预测值 vs 实际值'
    ),
    row=2, col=2
)

# 添加理想预测线
min_val, max_val = y_test.min(), y_test.max()
fig.add_trace(
    go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='black', dash='dash', width=2),
        name='理想预测线',
        showlegend=False
    ),
    row=2, col=2
)

# 6. 残差图
residuals = y_test - y_pred
fig.add_trace(
    go.Scatter(
        x=y_pred,
        y=residuals,
        mode='markers',
        marker=dict(color='purple', opacity=0.6),
        name='残差'
    ),
    row=2, col=3
)

# 添加零线
fig.add_hline(y=0, line_dash="dash", line_color="black", row=2, col=3)

# 更新布局
fig.update_layout(
    title_text='🚗 汽车销量预测分析',
    title_x=0.5,
    height=800,
    width=1400,
    showlegend=False
)

# 更新x轴和y轴标签
fig.update_xaxes(title_text="价格（元）", row=1, col=3)
fig.update_yaxes(title_text="月销量", row=1, col=3)
fig.update_xaxes(title_text="马力", row=2, col=1)
fig.update_yaxes(title_text="月销量", row=2, col=1)
fig.update_xaxes(title_text="实际销量", row=2, col=2)
fig.update_yaxes(title_text="预测销量", row=2, col=2)
fig.update_xaxes(title_text="预测销量", row=2, col=3)
fig.update_yaxes(title_text="残差", row=2, col=3)
fig.update_xaxes(title_text="月销量", row=1, col=2)
fig.update_yaxes(title_text="频次", row=1, col=2)

# 保存图表
fig.write_html('/Users/zihao_/Documents/github/writing_log/certification/car_sales_analysis.html')
fig.write_image('/Users/zihao_/Documents/github/writing_log/certification/car_sales_analysis.png', width=1400, height=800)
fig.show()

print("\n📊 图表已保存为 'car_sales_analysis.html' 和 'car_sales_analysis.png'")

# 9️⃣ 多模型比较
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("\n🤖 多模型比较分析：")
print("="*60)

# 特征标准化（为SVM准备）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 定义多个模型
models = {
    '线性回归': LinearRegression(),
    '随机森林': RandomForestRegressor(n_estimators=100, random_state=42),
    '梯度提升': GradientBoostingRegressor(n_estimators=100, random_state=42),
    '支持向量机': SVR(kernel='rbf', C=100, gamma=0.1)
}

# 模型训练和评估
model_results = {}

for name, model in models.items():
    if name == '支持向量机':
        # SVM使用标准化数据
        model.fit(X_train_scaled, y_train)
        y_pred_model = model.predict(X_test_scaled)
    else:
        # 其他模型使用原始数据
        model.fit(X_train, y_train)
        y_pred_model = model.predict(X_test)
    
    # 计算评估指标
    r2_model = r2_score(y_test, y_pred_model)
    mae_model = mean_absolute_error(y_test, y_pred_model)
    rmse_model = np.sqrt(mean_squared_error(y_test, y_pred_model))
    mape_model = np.mean(np.abs((y_test - y_pred_model) / y_test)) * 100
    
    model_results[name] = {
        'R²': r2_model,
        'MAE': mae_model,
        'RMSE': rmse_model,
        'MAPE': mape_model
    }
    
    print(f"\n📈 {name}:")
    print(f"  R²: {r2_model:.4f}")
    print(f"  MAE: {mae_model:.2f}")
    print(f"  RMSE: {rmse_model:.2f}")
    print(f"  MAPE: {mape_model:.2f}%")

# 模型比较表格
print("\n📊 模型性能比较表：")
results_df = pd.DataFrame(model_results).T
print(results_df.round(4))

# 找出最佳模型
best_model_r2 = results_df['R²'].idxmax()
best_model_mae = results_df['MAE'].idxmin()

print(f"\n🏆 最佳模型（R²）: {best_model_r2} (R² = {results_df.loc[best_model_r2, 'R²']:.4f})")
print(f"🏆 最佳模型（MAE）: {best_model_mae} (MAE = {results_df.loc[best_model_mae, 'MAE']:.2f})")

# 🔟 特征重要性分析（随机森林）
print("\n🎯 特征重要性分析（随机森林）：")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

feature_importance = pd.DataFrame({
    '特征': X.columns,
    '重要性': rf_model.feature_importances_
}).sort_values('重要性', ascending=False)

print(feature_importance)

# 可视化特征重要性
fig_importance = go.Figure(data=[
    go.Bar(
        x=feature_importance['特征'],
        y=feature_importance['重要性'] * 100,
        marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'],
        text=[f'{imp:.2f}%' for imp in feature_importance['重要性'] * 100],
        textposition='auto',
    )
])

fig_importance.update_layout(
    title='🎯 特征重要性分析（随机森林）',
    title_x=0.5,
    xaxis_title='特征',
    yaxis_title='重要性 (%)',
    height=500,
    width=800,
    showlegend=False
)

# 保存图表
fig_importance.write_html('/Users/zihao_/Documents/github/writing_log/certification/feature_importance.html')
fig_importance.write_image('/Users/zihao_/Documents/github/writing_log/certification/feature_importance.png', width=800, height=500)
fig_importance.show()

print("\n📊 特征重要性图表已保存为 'feature_importance.html' 和 'feature_importance.png'")

# 1️⃣1️⃣ 交叉验证分析
from sklearn.model_selection import cross_val_score, KFold

print("\n🔄 交叉验证分析（5折）：")
print("="*60)

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

cv_results = {}
for name, model in models.items():
    if name == '支持向量机':
        # SVM使用标准化数据
        X_scaled = scaler.fit_transform(X)
        cv_scores = cross_val_score(model, X_scaled, y, cv=kfold, scoring='r2')
    else:
        cv_scores = cross_val_score(model, X, y, cv=kfold, scoring='r2')
    
    cv_results[name] = cv_scores
    
    print(f"\n📈 {name}:")
    print(f"  平均R²: {cv_scores.mean():.4f} (±{cv_scores.std() * 2:.4f})")
    print(f"  各折R²: {[f'{score:.4f}' for score in cv_scores]}")

# 交叉验证结果可视化
fig_cv = go.Figure()

for name, scores in cv_results.items():
    fig_cv.add_trace(go.Box(
        y=scores,
        name=name,
        boxpoints='all',
        jitter=0.3,
        pointpos=-1.8
    ))

fig_cv.update_layout(
    title='🔄 交叉验证R²分数分布',
    title_x=0.5,
    xaxis_title='模型',
    yaxis_title='R²分数',
    height=600,
    width=1000,
    showlegend=False
)

# 保存图表
fig_cv.write_html('/Users/zihao_/Documents/github/writing_log/certification/cross_validation.html')
fig_cv.write_image('/Users/zihao_/Documents/github/writing_log/certification/cross_validation.png', width=1000, height=600)
fig_cv.show()

print("\n📊 交叉验证图表已保存为 'cross_validation.html' 和 'cross_validation.png'")

# 1️⃣2️⃣ 学习曲线分析
from sklearn.model_selection import learning_curve

print("\n📚 学习曲线分析：")

# 选择最佳模型进行学习曲线分析
best_model = RandomForestRegressor(n_estimators=100, random_state=42)

train_sizes, train_scores, val_scores = learning_curve(
    best_model, X, y, cv=5, n_jobs=-1, 
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='r2'
)

# 计算均值和标准差
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

# 绘制学习曲线
fig_lc = go.Figure()

# 添加训练分数
fig_lc.add_trace(go.Scatter(
    x=train_sizes,
    y=train_mean,
    mode='lines+markers',
    name='训练分数',
    line=dict(color='blue'),
    marker=dict(size=8)
))

# 添加训练分数置信区间
fig_lc.add_trace(go.Scatter(
    x=np.concatenate([train_sizes, train_sizes[::-1]]),
    y=np.concatenate([train_mean + train_std, (train_mean - train_std)[::-1]]),
    fill='toself',
    fillcolor='rgba(0,100,80,0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip",
    showlegend=False
))

# 添加验证分数
fig_lc.add_trace(go.Scatter(
    x=train_sizes,
    y=val_mean,
    mode='lines+markers',
    name='验证分数',
    line=dict(color='red'),
    marker=dict(size=8)
))

# 添加验证分数置信区间
fig_lc.add_trace(go.Scatter(
    x=np.concatenate([train_sizes, train_sizes[::-1]]),
    y=np.concatenate([val_mean + val_std, (val_mean - val_std)[::-1]]),
    fill='toself',
    fillcolor='rgba(255,0,0,0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip",
    showlegend=False
))

fig_lc.update_layout(
    title='📚 学习曲线（随机森林）',
    title_x=0.5,
    xaxis_title='训练样本数',
    yaxis_title='R²分数',
    height=600,
    width=1000
)

# 保存图表
fig_lc.write_html('/Users/zihao_/Documents/github/writing_log/certification/learning_curve.html')
fig_lc.write_image('/Users/zihao_/Documents/github/writing_log/certification/learning_curve.png', width=1000, height=600)
fig_lc.show()

print("\n📊 学习曲线图表已保存为 'learning_curve.html' 和 'learning_curve.png'")

# 1️⃣3️⃣ 预测示例
print("\n🔮 新车销量预测示例：")
print("="*60)

# 创建新车数据示例
new_cars = pd.DataFrame({
    'price': [200000, 300000, 150000],
    'horsepower': [150, 200, 120],
    'fuel_efficiency': [8.0, 10.0, 6.5],
    'brand_score': [85, 90, 75]
})

print("新车配置：")
print(new_cars)

# 使用最佳模型进行预测
best_rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
best_rf_model.fit(X, y)
predictions = best_rf_model.predict(new_cars)

print(f"\n预测月销量：")
for i, pred in enumerate(predictions):
    print(f"  车型{i+1}: {pred:.0f} 辆/月")

print("\n" + "="*80)
print("🎉 汽车销量预测分析完成！")
print("📊 生成的图表文件：")
print("  - car_sales_analysis.html/.png: 综合数据分析")
print("  - feature_importance.html/.png: 特征重要性分析")
print("  - cross_validation.html/.png: 交叉验证结果")
print("  - learning_curve.html/.png: 学习曲线分析")
print("="*80)
