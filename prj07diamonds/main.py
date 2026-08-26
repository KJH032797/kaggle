import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

df = sns.load_dataset("diamonds")
df.to_csv("data/diamonds.csv")

cut_order = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']

num_cols = ['carat', 'depth', 'table', 'price', 'x', 'y', 'z']
result = df[num_cols].corr()

print(result)

