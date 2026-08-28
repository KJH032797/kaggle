import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

df = sns.load_dataset("tips")
df.to_csv("tips.csv")

# df.info()

cor = df.corr(numeric_only=True)["tip"]
# print(cor)

df['tip_pct'] = (df['tip'] / df['total_bill'] * 100).round(2)
features = ['total_bill', 'size', 'tip_pct']

# df['time']=(df['time']=='lunch').astype(int)

X = df[features]
y = df['tip']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,
)

m=LinearRegression()
m.fit(X_train, y_train)

y_pred=m.predict(X_test)

df_coef=pd.Series(m.coef_,index=features)

mae=mean_absolute_error(y_pred,y_test)
rmse=np.sqrt(mean_squared_error(y_pred,y_test))
r2=r2_score(y_pred,y_test)

print(mae)
print(rmse)
print(r2)