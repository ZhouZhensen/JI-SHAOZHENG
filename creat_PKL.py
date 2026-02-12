#### 0.设置工作路径
import os

# 设置工作路径
os.chdir("D:\\医工Web")

# 获取并打印当前工作路径
current_path = os.getcwd()
print("当前工作路径:", current_path)

# 获取并打印当前工作路径
current_path = os.getcwd()
print("当前工作路径:", current_path)

#### 1.导入数据并根据结局比例82拆分数据集
# 导入pandas库，用于数据处理
import pandas as pd
# 导入numpy库，用于数值计算
import numpy as np
# 导入matplotlib.pyplot库，用于数据可视化
import matplotlib.pyplot as plt
# 从sklearn.model_selection模块导入train_test_split函数，用于数据集划分
from sklearn.model_selection import train_test_split
# 导入warnings库，用于忽略警告信息
import warnings
# 忽略与无效特征名称相关的警告信息
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
# 设置Matplotlib的全局字体为Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
# 解决Matplotlib显示负号时的乱码问题
plt.rcParams['axes.unicode_minus'] = False

# 读取CSV格式的数据集文件'Dataset.csv'并存入DataFrame
df = pd.read_csv('X_test.csv')
# 划分特征变量X（去除'target'列）
X = df.drop(['Sarcopenia'], axis=1)
# 提取目标变量y（'target'列）
y = df['Sarcopenia']
# 将数据集拆分为训练集和测试集，其中测试集占20%，随机种子设为42，并按照'target'列进行分层抽样
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=42, stratify=df['Sarcopenia'])
# 显示数据集前五行
df.head()

#### 2.使用训练集数据训练随机森林分类器
# 从sklearn.ensemble模块导入RandomForestClassifier（随机森林分类器）
from sklearn.ensemble import RandomForestClassifier

# 创建随机森林分类器实例，并设置参数
rf_classifier = RandomForestClassifier(
    n_estimators=100,         # 'n_estimators' 指定森林中树的数量，默认是100，可以根据需要调整。
    criterion='gini',         # 'criterion' 参数指定用于划分的质量指标，'gini'（默认）表示使用基尼不纯度，另一选项是 'entropy'（信息增益）。
    max_depth=None,           # 'max_depth' 限制每棵树的最大深度，'None' 表示树可以生长到纯叶子节点或样本数不足为止。
    min_samples_split=2,      # 'min_samples_split' 指定一个节点分裂所需的最小样本数，默认是2。
    min_samples_leaf=1,       # 'min_samples_leaf' 指定叶子节点所需的最小样本数，默认是1。
    min_weight_fraction_leaf=0.0, # 'min_weight_fraction_leaf' 类似于 'min_samples_leaf'，但基于总样本权重，默认是0.0。
    random_state=42,          # 'random_state' 控制随机数生成，以便结果可复现，42 是一个常用的随机种子。
    max_leaf_nodes=None,      # 'max_leaf_nodes' 限制每棵树的最大叶子节点数，'None' 表示不限制。
    min_impurity_decrease=0.0 # 'min_impurity_decrease' 指定节点分裂时所需的最小不纯度减少量，默认是0.0。
)

# 使用训练集数据训练随机森林分类器
rf_classifier.fit(X_train, y_train)

#### 3.使用训练好的streamlit run rf_predict.py随机森林分类器对测试集进行预测
# 从sklearn.metrics模块导入classification_report，用于生成分类模型的评估报告
from sklearn.metrics import classification_report

# 使用训练好的随机森林分类器对测试集进行预测
y_pred = rf_classifier.predict(X_test)

# 输出分类报告，包含精确率（precision）、召回率（recall）、F1分数（F1-score）等评价指标
print(classification_report(y_test, y_pred))

#### 9.将训练好的模型保存到 RF.pkl 文件，以便以后加载并使用
# 导入 joblib 库，用于模型的保存和加载
import joblib

# 保存训练好的随机森林模型到文件 'RF.pkl'
joblib.dump(rf_classifier, 'RF.pkl')

#### 10.将测试集特征数据保存为 CSV 文件
X_test.to_csv('X_test.csv', index=False)