import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

df = sns.load_dataset("flights")
df.to_csv("flights.csv")

# df.info()

# 전처리
df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str),
                            format='%Y-%b')

df = df.sort_values('date').reset_index(drop=True)

# 원본에 바로 적용
# df = pd.get_dummies(df, columns=['month'], drop_first=True)
# 원본 건드리지 않고 적용
mon_dum = pd.get_dummies(df['month'], drop_first=True)
# df_concat = pd.concat([df, mon_dum], axis=1)
mon_dum['t']=np.arange(len(df))

# 모델
# 분리
n_test=12
X =mon_dum
y=df['passengers']
X_train=X.iloc[:-n_test]
y_train=y.iloc[:-n_test]
X_test=X.iloc[-n_test:]
y_test=y.iloc[-n_test:]

# 학습
m=LinearRegression()
m.fit(X_train,y_train)

# 예측
y_pred=m.predict(X_test)

# 평가
mae = mean_absolute_error(y_test,y_pred)
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
r2=r2_score(y_test,y_pred)

print("MAE:",mae)
print("RMSE:",rmse)
print("R2:",r2)