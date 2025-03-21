import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# 读取数据（添加引擎和列名检查）
train_df = pd.read_excel('Data4Regression.xlsx', sheet_name='Training Data', engine='openpyxl')
test_df = pd.read_excel('Data4Regression.xlsx', sheet_name='Test Data', engine='openpyxl')

# 数据准备（确保列名正确）
X_train = train_df['x'].values.reshape(-1, 1)
y_train = train_df['y_complex'].values
X_test = test_df['x_new'].values.reshape(-1, 1)  # 假设列名为x_new
y_test = test_df['y_new_complex'].values         # 假设列名为y_new_complex

# 定义不同次数的多项式模型
degrees = [2, 3, 5]
colors = ['red', 'green', 'purple']
linestyles = ['-', '--', ':']

# 创建画布
fig, axes = plt.subplots(1, len(degrees), figsize=(18, 5))
plt.subplots_adjust(wspace=0.3)

# 生成绘图用的连续x值
x_plot = np.linspace(
    min(X_train.min(), X_test.min()),
    max(X_train.max(), X_test.max()), 
    200
).reshape(-1, 1)

for idx, degree in enumerate(degrees):
    # 多项式特征转换
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    # 使用伪逆矩阵避免奇异矩阵问题
    try:
        w = np.linalg.inv(X_train_poly.T @ X_train_poly) @ X_train_poly.T @ y_train
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(X_train_poly.T @ X_train_poly) @ X_train_poly.T @ y_train
    
    # 计算预测值和MSE
    y_train_pred = X_train_poly @ w
    y_test_pred = X_test_poly @ w
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    
    # 生成绘图数据
    X_plot_poly = poly.transform(x_plot)
    y_plot = X_plot_poly @ w
    
    # 绘制子图
    ax = axes[idx]
    ax.scatter(X_train, y_train, s=20, color='blue', alpha=0.6, label='Train Data')
    ax.scatter(X_test, y_test, s=20, color='orange', alpha=0.6, label='Test Data')
    ax.plot(x_plot, y_plot, color=colors[idx], linestyle=linestyles[idx],
           linewidth=2, label=f'Degree {degree} Fit')
    
    # 设置标题和标签
    ax.set_title(f"Polynomial (Degree {degree})\nTrain MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()