import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# 데이터
df = sns.load_dataset("tips")
df.to_csv("tips.csv")

df.info()

# 전처리
df['tip_pct'] = (df['tip'] / df['total_bill'] * 100).round(2)

# 단변량

# 이변량

# 상관관계
cor=df.corr(numeric_only=True)['tip']