import matplotlib.pyplot as plt
import seaborn as sns

df = sns.load_dataset("mpg")
# df.to_csv("data/mpg.csv")

# 결측치 처리
df["horsepower"] = df["horsepower"].fillna(df["horsepower"].median())

# 단변량
# fig , axes = plt.subplots(1,2,figsize=(12,4))
# sns.histplot(df["mpg"] , bins=25, ax=axes[0])
# sns.histplot(df["weight"] , bins=25, ax=axes[1])
# plt.show()

# 이변량
# fig, axes = plt.subplots(1, 2, figsize=(15, 5))
# sns.scatterplot(data=df, x="weight", y="mpg" , ax=axes[0] , hue="origin")
# sns.lineplot(data=df, x="model_year" , y="mpg" , ax=axes[1] , marker="o")
# plt.show()

num_cols = ["mpg","cylinders","displacement","horsepower","weight","acceleration","model_year",]
cor = df[num_cols].corr()
plt.figure(figsize=(12,8))
sns.heatmap(cor,annot=True , cmap="coolwarm")
plt.show()














