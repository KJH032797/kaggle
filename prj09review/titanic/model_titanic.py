from statistics import LinearRegression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, classification_report
from sklearn.model_selection import train_test_split

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

df = pd.read_csv('train.csv')


# 전처리 (결측치, 중복, 인코딩, 파생변수)
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Fare'] = df['Fare'].fillna(df['Fare'].median())

df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
# -> df['Sex']=(df['Sex']=='male').astype(int)
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

df['FamilySize'] = df['Parch'] + df['SibSp'] + 1

# 모델 (학습, 예측, 평가)
features=['Age','Embarked','Fare','Sex','FamilySize','Pclass']
X=df[features]
y=df['Survived']

X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,stratify=y,random_state=42)

m=LogisticRegression()
m.fit(X_train,y_train)

y_pred=m.predict(X_test)
score=accuracy_score(y_test,y_pred)
print('Score:',score)

sr=pd.Series(m.coef_[0],index=features).sort_values(ascending=False)
print(sr)

result=classification_report(y_test,y_pred,target_names=['사망','생존'])
print(result)


# cm=confusion_matrix(y_test,y_pred)
# plt.figure(figsize=(8,6))
# sns.heatmap(cm,annot=True,cmap='Blues',fmt="g",
#             xticklabels=['사망 예측', '생존 예측'],
#             yticklabels=['실제 사망', '실제 생존'])
#
# plt.show()