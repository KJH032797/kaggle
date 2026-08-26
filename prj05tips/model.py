import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import seaborn as sns

# 데이터 준비
df = sns.load_dataset("tips")

# 데이터 전처리

# 학습용 시험용 데이터 분리
X = df[['total_bill']]
y = df['tip']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 학습
m = LinearRegression()
m.fit(X_train, y_train)

# 예측
y_pred = m.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(mae)
print(rmse)
print(r2)
