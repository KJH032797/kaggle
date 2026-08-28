import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

df = sns.load_dataset("flights")
df.to_csv("flights.csv")

# df.info()

# 전처리
df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str),
                            format='%Y-%b')

df = df.sort_values('date').reset_index(drop=True)

