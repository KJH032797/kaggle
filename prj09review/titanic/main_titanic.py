import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터
df = sns.load_dataset("titanic")
df.to_csv("titanic.csv")

# 데이터 확인 / 결측치 처리
df.info()
print(df.isna().sum())

# 단변량

# 이변량

# 상관관계
corr=df.corr(numeric_only=True)['survived']
sns.heatmap(corr, annot=True, cmap="YlOrRd",fmt=".2f")
plt.show()