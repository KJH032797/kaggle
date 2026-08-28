from statistics import LinearRegression

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# dataset
df = sns.load_dataset("penguins")
df.to_csv("penguins.csv")

# 전처리
df = df.dropna()

# 분류
features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
X = df[features]
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y,random_state=42)

# 모델
m=LogisticRegression(max_iter=5000)
m.fit(X_train,y_train)

y_pred=m.predict(X_test)

score=accuracy_score(y_test,y_pred)
print(score)

# cm=confusion_matrix(y_test,y_pred)
# print(cm)
# plt.figure(figsize=(8,6))
# sns.heatmap(cm,annot=True,fmt="g",cmap="Blues",
#             xticklabels=['Adelie','Chintrap','Gentoo'],
#             yticklabels=['Adelie','Chintrap','Gentoo'])
# plt.xlabel('real')
# plt.ylabel('predicted')
# plt.show()

# cr=classification_report(y_test,y_pred)
# print(cr)

print(m.coef_)