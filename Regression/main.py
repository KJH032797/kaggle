import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# print(train.info())
# 매매가 상관계수 상위 4 확인
cor=train.corr(numeric_only=True)['SalePrice'].abs().sort_values(ascending=False)
print(cor.head())

# 확인한 컬럼 선별 요소 지정
features=['OverallQual','GrLivArea','GarageCars','GarageArea']

# 결측치 처리
# 학습지
X_train = train[features].fillna(0)
# 문제지
X_test = test[features].fillna(0)
# 학습답안지
y_train=train['SalePrice']

# 모델학습(회귀Regression)
m=RandomForestRegressor(n_estimators=100,random_state=42)
m.fit(X_train, y_train)

# 예측 결과 및 저장
y_pred=m.predict(X_test)
sub = pd.DataFrame({
    'Id':test['Id'],
    'SalePrice' : y_pred,
})

sub.to_csv('data/test_submission.csv',index=False)

# 결과
# mae=mean_absolute_error()